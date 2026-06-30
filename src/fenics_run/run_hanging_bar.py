"""Hanging bar -- FEM reference side, using FEniCSx (dolfinx).

Solves the hanging-block problem (top face clamped, gravity stretches it) with a
compressible Neo-Hookean material using the SAME Lame constants as Newton, in two
element variants:

  --element tet   (default): build the mesh from Newton's shared tet mesh
                  (data/mesh.npz) and evaluate at Newton's node positions, so the
                  result can be compared to Newton NODE-FOR-NODE. -> data/fem_result.npz
  --element hex : build an independent structured hexahedral mesh of the SAME
                  block geometry. Hex8 (trilinear) is less prone to shear /
                  volumetric locking than linear tets, so this is a second,
                  more accurate FEM reference. Its nodes differ from Newton's, so
                  it is compared via the displacement profile, not node-for-node.
                  -> data/fem_result_hex.npz

Run from the repository root (after installing dolfinx, see 00_setup.ipynb):

    python -m fenics_run.run_hanging_bar                # tet (Newton's mesh)
    python -m fenics_run.run_hanging_bar --element hex  # independent hex mesh

A few dolfinx API points are version sensitive; the call sites below target the
dolfinx 0.11 API.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from common import mesh_io, params


def neo_hookean_residual(u, v, msh, mu, lmbda, body_force):
    """Weak form of compressible Neo-Hookean elastostatics with a body force.

    psi = mu/2 (Ic - 3) - mu ln J + lambda/2 (ln J)^2
    Residual = inner(grad(v), P) dx - inner(B, v) dx,   P = d psi / d F
    """
    import ufl

    d = msh.geometry.dim
    Identity = ufl.Identity(d)
    F = ufl.variable(Identity + ufl.grad(u))     # must be a ufl.variable for diff()
    C = F.T * F
    Ic = ufl.tr(C)
    J = ufl.det(F)
    psi = (mu / 2.0) * (Ic - d) - mu * ufl.ln(J) + (lmbda / 2.0) * (ufl.ln(J)) ** 2
    P = ufl.diff(psi, F)                          # 1st Piola-Kirchhoff stress
    return ufl.inner(ufl.grad(v), P) * ufl.dx - ufl.inner(body_force, v) * ufl.dx


def evaluate_at_nodes(u, msh, points):
    """Evaluate the displacement field at arbitrary points.

    Uses dolfinx's bounding-box tree so the result is returned in the order of
    ``points`` regardless of dolfinx's internal node numbering.
    """
    import dolfinx.geometry as geom

    # dolfinx geometry API: bb_tree / compute_collisions_points / compute_colliding_cells.
    tree = geom.bb_tree(msh, msh.topology.dim)
    candidates = geom.compute_collisions_points(tree, points)
    colliding = geom.compute_colliding_cells(msh, candidates, points)

    cells = np.empty(len(points), dtype=np.int32)
    for i in range(len(points)):
        c = colliding.links(i)
        if len(c) == 0:
            c = candidates.links(i)        # fall back to bbox candidates
        cells[i] = c[0]
    return u.eval(points, cells)


def fem_snes_options(rtol=1.0e-7, atol=1.0e-9, max_it=60):
    """PETSc options for a direct-LU Newton (SNES) solve -- dolfinx 0.11 style."""
    return {
        "snes_type": "newtonls",
        "snes_rtol": rtol, "snes_atol": atol, "snes_max_it": max_it,
        "ksp_type": "preonly", "pc_type": "lu",
    }


def solve_status(problem, u):
    """Solve a dolfinx 0.11 NonlinearProblem, scatter, return (n_iterations, converged).

    The SNES is read defensively (attribute name guarded) so the solve still works
    even if convergence introspection is unavailable.
    """
    problem.solve()
    u.x.scatter_forward()
    snes = getattr(problem, "solver", None)
    if snes is None:
        return -1, True
    return int(snes.getIterationNumber()), bool(snes.getConvergedReason() > 0)


def solve_hanging(msh, comm, n_steps=5):
    """Solve the clamped-top hanging block under gravity.

    Ramps gravity over ``n_steps`` load increments for nonlinear robustness.
    Returns (u, n_clamped, total_newton_iterations).
    """
    import ufl
    from dolfinx import default_scalar_type, fem
    from dolfinx.fem.petsc import NonlinearProblem
    from mpi4py import MPI

    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u = fem.Function(V, name="displacement")
    v = ufl.TestFunction(V)

    zmax = comm.allreduce(float(msh.geometry.x[:, 2].max()), op=MPI.MAX)
    tol = params.FACE_TOL_FRAC * params.CELL
    top_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], zmax, atol=tol))
    bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), top_dofs, V)

    mu = fem.Constant(msh, default_scalar_type(params.K_MU))
    lmbda = fem.Constant(msh, default_scalar_type(params.K_LAMBDA))
    g_vec = fem.Constant(msh, np.array([0.0, 0.0, -params.DENSITY * params.GRAVITY],
                                       dtype=default_scalar_type))
    load_scale = fem.Constant(msh, default_scalar_type(0.0))
    body_force = load_scale * g_vec

    residual = neo_hookean_residual(u, v, msh, mu, lmbda, body_force)
    # dolfinx 0.11: SNES-based NonlinearProblem (NewtonSolver is gone). solve()
    # updates u in place; a direct LU is plenty for these small meshes.
    problem = NonlinearProblem(
        residual, u, bcs=[bc], petsc_options_prefix="hang_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1.0e-8, "snes_atol": 1.0e-10, "snes_max_it": 50,
            "ksp_type": "preonly", "pc_type": "lu",
        },
    )

    total_its = 0
    for k in range(1, n_steps + 1):
        load_scale.value = k / n_steps
        problem.solve()
        u.x.scatter_forward()
        snes = getattr(problem, "solver", None)
        n_it = int(snes.getIterationNumber()) if snes is not None else -1
        reason = int(snes.getConvergedReason()) if snes is not None else 1
        total_its += max(n_it, 0)
        print(f"[fem] load step {k}/{n_steps}: snes_its={n_it} reason={reason}")
        if reason < 0:
            raise RuntimeError(f"FEM SNES diverged at load step {k} (reason={reason})")

    return u, len(top_dofs), total_its


def _build_tet_mesh(comm):
    """Mesh from Newton's shared tet mesh -> node-for-node comparison."""
    import basix.ufl
    import ufl
    from dolfinx import mesh as dmesh

    rest_q, tets, fixed = mesh_io.load_mesh(params.MESH_NPZ)
    tets = mesh_io.orient_tets_positive(rest_q, tets)
    coord_el = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    # dolfinx 0.11 order: create_mesh(comm, cells, e, x) -- element BEFORE coords.
    msh = dmesh.create_mesh(
        comm,
        np.ascontiguousarray(tets, dtype=np.int64),
        ufl.Mesh(coord_el),
        np.ascontiguousarray(rest_q, dtype=np.float64),
    )
    return msh, rest_q, tets, fixed


def _build_hex_mesh(comm):
    """Independent structured hex mesh of the same block geometry."""
    from dolfinx.mesh import CellType, create_box

    ox, oy, oz = params.ORIGIN
    p0 = np.array([ox, oy, oz])
    p1 = np.array([ox + params.BLOCK_LX, oy + params.BLOCK_LY, oz + params.BLOCK_LZ])
    msh = create_box(
        comm, [p0, p1],
        [params.GRID_DIM_X, params.GRID_DIM_Y, params.GRID_DIM_Z],
        CellType.hexahedron,
    )
    return msh


def main():
    parser = argparse.ArgumentParser(description="FEniCSx hanging-bar reference")
    parser.add_argument("--element", choices=["tet", "hex"], default="tet",
                        help="tet = Newton's shared mesh (node-for-node); hex = independent hex mesh")
    args = parser.parse_args()

    import dolfinx
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    print(f"[fem] dolfinx {dolfinx.__version__}, element={args.element}")
    print(params.summary())
    os.makedirs(params.DATA_DIR, exist_ok=True)

    if args.element == "tet":
        msh, rest_q, tets, fixed = _build_tet_mesh(comm)
        t0 = time.perf_counter()
        u, n_clamp, _ = solve_hanging(msh, comm)
        wall_time = time.perf_counter() - t0
        u_nodes = np.asarray(evaluate_at_nodes(u, msh, rest_q))
        final_q = rest_q + u_nodes
        out = params.FEM_NPZ
        np.savez(out, rest_q=rest_q, final_q=final_q, tet_indices=tets,
                 fixed_nodes=fixed, wall_time=wall_time)
        volume = float(mesh_io.signed_tet_volumes(rest_q, tets).sum())
        print(f"[fem] block volume = {volume:.4f} m^3, "
              f"weight = {params.DENSITY * volume * params.GRAVITY:.3f} N")
    else:
        msh = _build_hex_mesh(comm)
        t0 = time.perf_counter()
        u, n_clamp, _ = solve_hanging(msh, comm)
        wall_time = time.perf_counter() - t0
        rest_q = np.ascontiguousarray(msh.geometry.x[:, :3], dtype=np.float64)
        u_nodes = np.asarray(evaluate_at_nodes(u, msh, rest_q))
        final_q = rest_q + u_nodes
        tol = params.FACE_TOL_FRAC * params.CELL
        fixed = np.where(rest_q[:, 2] > rest_q[:, 2].max() - tol)[0]
        out = params.FEM_HEX_NPZ
        np.savez(out, rest_q=rest_q, final_q=final_q, fixed_nodes=fixed, wall_time=wall_time)

    free = np.setdiff1d(np.arange(len(rest_q)), fixed)
    tip_drop = float(-u_nodes[free, 2].min()) * 1000.0
    print(f"[fem] element={args.element}: nodes={len(rest_q)} clamped={n_clamp}")
    print(f"[fem] wrote {out}")
    print(f"[fem] max downward displacement = {tip_drop:.2f} mm")
    print(f"[fem] solve wall time = {wall_time:.3f} s")


if __name__ == "__main__":
    main()
