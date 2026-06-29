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
"""

from __future__ import annotations

import os
import time

import numpy as np

from common import params
from fenics_run.run_stage_a import neo_hookean_residual, evaluate_at_nodes


def main():
    import ufl
    from mpi4py import MPI
    from dolfinx import fem, default_scalar_type
    from dolfinx.mesh import create_box, CellType, locate_entities_boundary, meshtags
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.nls.petsc import NewtonSolver

    comm = MPI.COMM_WORLD
    nx, ny, nz = params.FRICTION_DIM
    h = params.FRICTION_CELL
    Lx, Ly, Lz = nx * h, ny * h, nz * h
    tol = 0.25 * h
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
    top = lambda x: np.isclose(x[2], Lz, atol=tol)
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

    problem = NonlinearProblem(residual, u, bcs=bcs)
    solver = NewtonSolver(comm, problem)
    solver.rtol = 1.0e-7
    solver.atol = 1.0e-9
    solver.max_it = 80
    solver.convergence_criterion = "incremental"

    # --- measurement forms ---------------------------------------------------
    area = comm.allreduce(fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ds(BOT))), op=MPI.SUM)
    Nform = fem.form(pN * ds(BOT))
    Ffric_form = fem.form(-tT[0] * ds(BOT))              # resisting force (+x drag)
    slip_form = fem.form(u[0] * ds(BOT))                 # mean bottom slip * area
    slipping = ufl.conditional(ufl.ge(kt * sTn, mu * pN), 1.0, 0.0)
    slipfrac_form = fem.form(slipping * ds(BOT))

    def settle_gravity():
        for s in (0.25, 0.5, 0.75, 1.0):
            load_scale.value = s
            solver.solve(u); u.x.scatter_forward()

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
            n_it, converged = solver.solve(u); u.x.scatter_forward()
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
