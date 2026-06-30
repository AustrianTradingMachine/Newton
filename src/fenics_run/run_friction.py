"""Friction -- FEM side: a soft block sliding on a rigid floor.

A flat block rests on the floor (z = 0) under gravity. Its TOP face is dragged
tangentially (+x) in small increments. The floor enforces, in pure UFL:

  * a normal penalty           pN = kn * <-z_def>+      (one-sided contact), and
  * penalty-regularised COULOMB friction with return mapping:
        s_T   = (u_x, u_y)                tangential slip vs. the fixed floor
        t_try = kt * s_T                  elastic "stick" traction
        |t_T| = min(kt |s_T|, mu * pN)    Coulomb cap (slip once it saturates)
        t_T   = -|t_T| * s_T / |s_T|      opposes the slip

Because the floor is rigid and fixed, the slip of a contacting material point is
just its own tangential displacement, so no slip-history field is needed -- the
law is a smooth (C0) function of u and stays differentiable for Newton.

We record, per drag increment:
  * applied top drag                          [m]
  * total floor friction force (resisting +x) [N]   -> rises (stick), then
                                                       plateaus at mu * N (slip)
  * total normal force N = integral pN ds     [N]   -> should equal the weight W
  * mean bottom slip                          [m]
  * slipping area fraction                     [-]
  * cumulative frictional work                [J]   (dissipated in steady slip)

Analytic anchor: with the normal load supplied by gravity, N = W = rho*g*V, so
the friction force plateaus at mu*W -- both N~W and the plateau test the FEM.

-> data/fem_friction.npz

Run (needs dolfinx):  python -m fenics_run.run_friction

TODO[verify-on-colab]: the static free-body settle was singular at u=0 -- the block has a
free vertical rigid mode and the one-sided floor penalty is inactive at zero penetration --
which produced an all-zero F/N/slip run (data/fem_friction.npz). settle_gravity now bootstraps
an initial floor penetration; re-run on Colab to regenerate the npz and confirm N ~ W and the
friction force reaching the mu*W plateau.
"""

from __future__ import annotations

import os
import time

import numpy as np

from common import params
from fenics_run.run_hanging_bar import evaluate_at_nodes, fem_snes_options, neo_hookean_residual, solve_status


def main():
    import ufl
    from dolfinx import default_scalar_type, fem
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.mesh import CellType, create_box, locate_entities_boundary, meshtags
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    nx, ny, nz = params.FRICTION_DIM
    h = params.FRICTION_CELL
    Lx, Ly, Lz = nx * h, ny * h, nz * h
    tol = params.FACE_TOL_FRAC * h
    mu = params.FRICTION_MU

    msh = create_box(comm, [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
                     [nx, ny, nz], CellType.hexahedron)
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u = fem.Function(V, name="displacement")
    v = ufl.TestFunction(V)

    # --- top-face drag: u_x = s (ramped), u_y = 0; z free (sits on floor) -----
    # TODO[verify-on-colab]: component Dirichlet via V.sub(i).collapse()
    V0, _ = V.sub(0).collapse()
    V1, _ = V.sub(1).collapse()
    def top(x):
        return np.isclose(x[2], Lz, atol=tol)
    dofs_x = fem.locate_dofs_geometrical((V.sub(0), V0), top)
    dofs_y = fem.locate_dofs_geometrical((V.sub(1), V1), top)
    drag_bc = fem.Function(V0)            # value array set to s each step
    zero_y = fem.Function(V1)
    bc_top_x = fem.dirichletbc(drag_bc, dofs_x, V.sub(0))
    bc_top_y = fem.dirichletbc(zero_y, dofs_y, V.sub(1))
    bcs = [bc_top_x, bc_top_y]

    # --- bottom facet measure (the floor contact surface) --------------------
    fdim = msh.topology.dim - 1
    bot_facets = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], 0.0, atol=tol))
    BOT = 1
    tags = meshtags(msh, fdim, bot_facets, np.full(len(bot_facets), BOT, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)

    mu_c = fem.Constant(msh, default_scalar_type(params.K_MU))
    lmbda = fem.Constant(msh, default_scalar_type(params.K_LAMBDA))
    kn = fem.Constant(msh, default_scalar_type(params.FRICTION_KN_FACTOR * params.YOUNGS_E / h))
    kt = fem.Constant(msh, default_scalar_type(params.FRICTION_KT_FACTOR * params.YOUNGS_E / h))
    load_scale = fem.Constant(msh, default_scalar_type(0.0))      # gravity ramp
    g_vec = fem.Constant(msh, np.array([0.0, 0.0, -params.DENSITY * params.GRAVITY],
                                       dtype=default_scalar_type))
    body_force = load_scale * g_vec

    # --- floor contact + Coulomb friction (pure UFL) -------------------------
    X = ufl.SpatialCoordinate(msh)
    z_def = (X + u)[2]
    pen = ufl.max_value(-z_def, 0.0)
    pN = kn * pen                                         # normal pressure
    sT = ufl.as_vector([u[0], u[1]])                     # tangential slip
    sTn = ufl.sqrt(sT[0] ** 2 + sT[1] ** 2 + 1.0e-12)
    t_mag = ufl.min_value(kt * sTn, mu * pN)             # Coulomb cap
    tT = -t_mag * sT / sTn                                # friction traction (x, y)
    traction = ufl.as_vector([tT[0], tT[1], pN])         # + normal (up)

    bulk = neo_hookean_residual(u, v, msh, mu_c, lmbda, body_force)
    residual = bulk - ufl.inner(traction, v) * ds(BOT)

    problem = NonlinearProblem(residual, u, bcs=bcs, petsc_options_prefix="fric_",
                               petsc_options=fem_snes_options(max_it=80))

    # --- measurement forms ---------------------------------------------------
    area = comm.allreduce(fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ds(BOT))), op=MPI.SUM)
    Nform = fem.form(pN * ds(BOT))
    Ffric_form = fem.form(-tT[0] * ds(BOT))              # resisting force (+x drag)
    slip_form = fem.form(u[0] * ds(BOT))                 # mean bottom slip * area
    slipping = ufl.conditional(ufl.ge(kt * sTn, mu * pN), 1.0, 0.0)
    slipfrac_form = fem.form(slipping * ds(BOT))

    def settle_gravity():
        # The block is free in z (only the top face's x, y are fixed) and is held up solely by
        # the one-sided floor penalty, which is INACTIVE at u=0 (pen=0 -> zero contact tangent).
        # That leaves the vertical rigid-body mode unconstrained, so the Jacobian at u=0 is
        # singular and the solve never settles -- the source of an all-zero F/N/slip run.
        # Bootstrap it by starting the block penetrating the floor a few x its equilibrium
        # depth, so contact is active (pen>0, tangent kn) from the first Newton iteration.
        pen0 = params.friction_block_weight() / (float(kn.value) * area)
        u.x.array[2::3] = -2.0 * pen0          # uniform downward offset -> bottom face in contact
        u.x.scatter_forward()
        for s in (0.25, 0.5, 0.75, 1.0):
            load_scale.value = s
            solve_status(problem, u)

    drags = np.linspace(0.0, params.FRICTION_DRAG_MAX, params.FRICTION_STEPS + 1)
    weight = params.friction_block_weight()
    plateau = params.coulomb_plateau(mu)
    print(f"[friction-fem] block {Lx:.2f}x{Ly:.2f}x{Lz:.2f} m, W={weight:.2f} N, mu={mu}, "
          f"kn={float(kn.value):.3g} kt={float(kt.value):.3g} Pa/m, plateau mu*W={plateau:.2f} N")

    drag_rec, Frec, Nrec, slip_rec, frac_rec, work_rec = [], [], [], [], [], []
    work = 0.0
    prev_slip = 0.0
    t0 = time.perf_counter()
    for k, s in enumerate(drags):
        drag_bc.x.array[:] = s
        if k == 0:
            settle_gravity()
        else:
            load_scale.value = 1.0
            n_it, converged = solve_status(problem, u)
        F = comm.allreduce(fem.assemble_scalar(Ffric_form), op=MPI.SUM)
        N = comm.allreduce(fem.assemble_scalar(Nform), op=MPI.SUM)
        mean_slip = comm.allreduce(fem.assemble_scalar(slip_form), op=MPI.SUM) / area
        frac = comm.allreduce(fem.assemble_scalar(slipfrac_form), op=MPI.SUM) / area
        work += F * max(mean_slip - prev_slip, 0.0)       # frictional work increment
        prev_slip = mean_slip
        drag_rec.append(s); Frec.append(F); Nrec.append(N)
        slip_rec.append(mean_slip); frac_rec.append(frac); work_rec.append(work)
        print(f"[friction-fem] drag={s*1000:5.2f} mm  F={F:7.2f} N  N={N:7.2f} N  "
              f"slip={mean_slip*1000:5.2f} mm  slipfrac={frac:.2f}  W_fric={work:.3g} J")
    wall_time = time.perf_counter() - t0

    # deformed shear profile along height at the block centre line (for the notebook)
    zc = np.linspace(0.0, Lz, 41)
    pts = np.column_stack([np.full_like(zc, Lx / 2.0), np.full_like(zc, Ly / 2.0), zc])
    ux_line = np.asarray(evaluate_at_nodes(u, msh, pts))[:, 0]

    os.makedirs(params.DATA_DIR, exist_ok=True)
    np.savez(
        params.FEM_FRICTION_NPZ,
        drag=np.array(drag_rec), friction_force=np.array(Frec), normal_force=np.array(Nrec),
        mean_slip=np.array(slip_rec), slip_fraction=np.array(frac_rec),
        friction_work=np.array(work_rec),
        weight=weight, plateau=plateau, mu=mu, wall_time=wall_time,
        z_line=zc, ux_line=ux_line,
    )
    print(f"[friction-fem] wrote {params.FEM_FRICTION_NPZ}")
    print(f"[friction-fem] final F={Frec[-1]:.2f} N vs analytic plateau mu*W={plateau:.2f} N "
          f"(N={Nrec[-1]:.2f} N vs W={weight:.2f} N)")


if __name__ == "__main__":
    main()
