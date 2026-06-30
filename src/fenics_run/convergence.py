"""Convergence study -- FEM (FEniCSx) side, the hanging block.

Two classic FEM convergence checks on the hanging block:

  1. MESH h-REFINEMENT: solve on structured meshes of increasing resolution
     (params.conv_fem_resolutions()). The tip drop and strain energy converge to
     a mesh-independent limit; comparing that limit to the analytic 1-D bar tests
     how adequate the FEM model is, and how fast it converges (the discretisation
     error). cost = #DOFs and wall time.

  2. LOAD-INCREMENT sweep at the finest mesh: vary the number of gravity load
     increments in the Newton-Raphson continuation. The *converged* tip drop must
     be independent of the increment count (it only affects robustness / total
     Newton iterations) -- a direct validation that the nonlinear solve is correct.

-> data/fem_convergence.npz

Run (needs dolfinx):  python -m fenics_run.convergence
"""

from __future__ import annotations

import os
import time

import numpy as np

from common import params
from compare import energies as en
from fenics_run.run_hanging_bar import evaluate_at_nodes, solve_hanging


def _build_box(comm, nx, ny, nz, element):
    from dolfinx.mesh import CellType, create_box

    ox, oy, oz = params.ORIGIN
    p0 = np.array([ox, oy, oz])
    p1 = np.array([ox + params.BLOCK_LX, oy + params.BLOCK_LY, oz + params.BLOCK_LZ])
    cell_type = CellType.hexahedron if element == "hex" else CellType.tetrahedron
    return create_box(comm, [p0, p1], [nx, ny, nz], cell_type)


def _strain_energy(u, msh):
    """Total compressible Neo-Hookean strain energy by UFL assembly (element-agnostic)."""
    import ufl
    from dolfinx import fem
    from mpi4py import MPI

    d = msh.geometry.dim
    F = ufl.Identity(d) + ufl.grad(u)
    J = ufl.det(F)
    psi = (0.5 * params.K_MU * (ufl.tr(F.T * F) - d) - params.K_MU * ufl.ln(J)
           + 0.5 * params.K_LAMBDA * ufl.ln(J) ** 2)
    form = fem.form(psi * ufl.dx)
    return msh.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)


def _tip_drop(u, msh):
    """Downward displacement at the bottom-centre tip point [mm]."""
    ox, oy, oz = params.ORIGIN
    pt = np.array([[ox + params.BLOCK_LX / 2.0, oy + params.BLOCK_LY / 2.0, oz]])
    uz = float(np.asarray(evaluate_at_nodes(u, msh, pt))[0, 2])
    return -uz * 1000.0


def main():
    import dolfinx
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    element = params.CONV_FEM_ELEMENT
    print(f"[conv-fem] dolfinx {dolfinx.__version__}, element={element}")

    z_top = params.ORIGIN[2] + params.BLOCK_LZ
    tip_analytic = params.analytic_hanging_displacement(
        params.ORIGIN[2], z_top, params.BLOCK_LZ) * 1000.0

    # --- 1. mesh h-refinement --------------------------------------------
    res = params.conv_fem_resolutions()
    h_list, ndofs, h_tip, h_strain, h_time = [], [], [], [], []
    for (nx, ny, nz) in res:
        msh = _build_box(comm, nx, ny, nz, element)
        t0 = time.perf_counter()
        u, _, _ = solve_hanging(msh, comm, n_steps=5)
        wt = time.perf_counter() - t0
        ndof = u.function_space.dofmap.index_map.size_global * u.function_space.dofmap.index_map_bs
        tip = _tip_drop(u, msh)
        U = _strain_energy(u, msh)
        h = params.BLOCK_LZ / nz
        h_list.append(h); ndofs.append(int(ndof)); h_tip.append(tip)
        h_strain.append(U); h_time.append(wt)
        print(f"[conv-fem] {nx}x{ny}x{nz}  h={h*1000:5.1f} mm  ndof={ndof:6d}  "
              f"tip={tip:.3f} mm  U={U:.4g} J  t={wt:.2f} s")

    # --- 2. load-increment sweep at the finest mesh ----------------------
    nx, ny, nz = res[-1]
    ls_steps = np.array(params.CONV_FEM_LOADSTEPS)
    ls_tip, ls_its, ls_time = [], [], []
    for ns in ls_steps:
        msh = _build_box(comm, nx, ny, nz, element)
        t0 = time.perf_counter()
        u, _, its = solve_hanging(msh, comm, n_steps=int(ns))
        wt = time.perf_counter() - t0
        tip = _tip_drop(u, msh)
        ls_tip.append(tip); ls_its.append(int(its)); ls_time.append(wt)
        print(f"[conv-fem] load_steps={ns:2d}: tip={tip:.4f} mm  total_newton_its={its}  t={wt:.2f} s")

    os.makedirs(params.DATA_DIR, exist_ok=True)
    np.savez(
        params.FEM_CONV_NPZ,
        element=element,
        h=np.array(h_list), ndofs=np.array(ndofs), h_tip=np.array(h_tip),
        h_strain=np.array(h_strain), h_time=np.array(h_time),
        nz=np.array([r[2] for r in res]),
        load_steps=ls_steps, ls_tip=np.array(ls_tip),
        ls_its=np.array(ls_its), ls_time=np.array(ls_time),
        tip_analytic=tip_analytic,
        strain_analytic=en.analytic_hanging_strain_energy(),
    )
    print(f"[conv-fem] wrote {params.FEM_CONV_NPZ}")
    print(f"[conv-fem] finest tip = {h_tip[-1]:.3f} mm vs analytic 1-D = {tip_analytic:.3f} mm")


if __name__ == "__main__":
    main()
