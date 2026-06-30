"""Dynamic drop -- FEM (FEniCSx), the literal rigid_soft_contact scenario.

A soft block rests on the ground; a rigid sphere is dropped onto it under gravity.
This is the dynamic-impact counterpart to the indentation test's quasi-static indentation, and
the FEM side of the literal example comparison against Newton XPBD
(newton_run/run_drop.py).

It stays in pure UFL/dolfinx:
  * elastodynamics with implicit Newmark-beta time integration (inertia term),
  * penalty contact against two RIGID ANALYTIC obstacles -- the ground plane z=0
    and the sphere -- so the gap is closed-form (no mesh-mesh search),
  * ~10% Kelvin-Voigt contact viscous damping for impact stability:
        p = < kn*(-g) + cd*(-gdot) >+ ,   gdot = n . (v_material - v_sphere)
  * the rigid sphere's own free-fall + contact reaction is a small staggered ODE
    (assemble the contact force, integrate the sphere) -- the only non-UFL piece,
    a handful of Python lines, no C++.

Records a time series -> data/fem_drop.npz:
  t, sphere_z, penetration, strain_energy, kinetic_energy (block), contact_force.

Run (after installing dolfinx):  python -m fenics_run.run_drop

NOTE: heavily marked TODO[verify-on-colab]; dynamic contact needs dt / damping
tuning, and the no-Dirichlet dynamic tangent relies on the mass term.
"""

from __future__ import annotations

import math
import os

import numpy as np

from common import params
from fenics_run.run_hanging_bar import fem_snes_options, solve_status


def main():
    import dolfinx
    import ufl
    from dolfinx import default_scalar_type, fem
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.mesh import CellType, create_box, locate_entities_boundary, meshtags
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    print(f"[drop-fem] dolfinx {dolfinx.__version__}")

    nx, ny, nz = params.DROP_DIM
    h = params.DROP_CELL
    Lx, Ly, Lz = nx * h, ny * h, nz * h
    R = params.DROP_SPHERE_R
    cx, cy = Lx / 2.0, Ly / 2.0
    tol = params.FACE_TOL_FRAC * h

    msh = create_box(comm, [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
                     [nx, ny, nz], CellType.hexahedron)
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    Vs = fem.functionspace(msh, ("Lagrange", 1))
    w = ufl.TestFunction(V)
    u = fem.Function(V, name="u")
    u_old = fem.Function(V)
    v_old = fem.Function(V)
    a_old = fem.Function(V)

    # facet tags: bottom (ground), top (sphere side)
    fdim = msh.topology.dim - 1
    bot = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], 0.0, atol=tol))
    top = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], Lz, atol=tol))
    BOTTOM, TOP = 1, 2
    facets = np.concatenate([bot, top])
    marks = np.concatenate([np.full(len(bot), BOTTOM, np.int32), np.full(len(top), TOP, np.int32)])
    order = np.argsort(facets)
    ft = meshtags(msh, fdim, facets[order], marks[order])
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

    # Newmark-beta (average acceleration: unconditionally stable, no numerical damping)
    beta, gamma = 0.25, 0.5
    # dt as a fem.Constant, NOT a bare numpy scalar: np.float64 * (UFL vector) lets numpy
    # hijack the product and iterate the UFL operand into a Python object array, so
    # v_new = v_old + dt*(...) silently becomes a list and ufl.dot(n_s, v_new - v_sphere)
    # below fails. See the mu/lam/rho note further down; the hanging-bar / indentation FEM
    # runs wrap their scalars the same way.
    dt = fem.Constant(msh, default_scalar_type(params.DROP_DT))
    a_new = (u - u_old - dt * v_old) / (beta * dt * dt) - (1.0 - 2.0 * beta) / (2.0 * beta) * a_old
    v_new = v_old + dt * ((1.0 - gamma) * a_old + gamma * a_new)

    # Material constants as fem.Constant for the same reason as dt above: a bare numpy
    # scalar times a UFL tensor (mu * (F - Finv_T)) gets hijacked by numpy into an object
    # array, which then breaks the residual's ufl.inner(grad(w), P). rho_val stays a plain
    # number for the body-force array literal.
    rho_val = params.DENSITY
    g = params.GRAVITY
    rho = fem.Constant(msh, default_scalar_type(rho_val))
    mu = fem.Constant(msh, default_scalar_type(params.K_MU))
    lam = fem.Constant(msh, default_scalar_type(params.K_LAMBDA))
    body_force = fem.Constant(msh, np.array([0.0, 0.0, -rho_val * g], dtype=default_scalar_type))

    # Neo-Hookean 1st Piola-Kirchhoff
    d = msh.geometry.dim
    F = ufl.Identity(d) + ufl.grad(u)
    J = ufl.det(F)
    Finv_T = ufl.inv(F).T
    P = mu * (F - Finv_T) + lam * ufl.ln(J) * Finv_T

    # contact stiffness / damping
    kn = fem.Constant(msh, default_scalar_type(params.DROP_PENALTY_FACTOR * params.YOUNGS_E / h))
    cd = fem.Constant(msh, default_scalar_type(params.DROP_DAMP_FRAC * float(kn.value) * params.DROP_DT))

    X = ufl.SpatialCoordinate(msh)
    x_def = X + u

    # ground plane z=0 (rigid): penetration below 0 pushed up (+z)
    pen_g = ufl.max_value(-x_def[2], 0.0)
    p_ground = kn * pen_g

    # sphere (rigid, kinematic falling body): Kelvin-Voigt penalty
    centre = fem.Constant(msh, np.array([cx, cy, params.DROP_SPHERE_Z0], dtype=default_scalar_type))
    v_sphere = fem.Constant(msh, np.array([0.0, 0.0, 0.0], dtype=default_scalar_type))
    dvec = x_def - centre
    dist = ufl.sqrt(ufl.dot(dvec, dvec) + 1.0e-12)
    n_s = dvec / dist
    pen_s = ufl.max_value(R - dist, 0.0)
    gdot = ufl.dot(n_s, v_new - v_sphere)                 # rate of gap change
    p_sphere = ufl.max_value(kn * pen_s - cd * gdot, 0.0)  # <kn*(-g) + cd*(-gdot)>+

    residual = (rho * ufl.inner(a_new, w) * ufl.dx
                + ufl.inner(ufl.grad(w), P) * ufl.dx
                - ufl.inner(body_force, w) * ufl.dx
                - p_ground * w[2] * ds(BOTTOM)
                - p_sphere * ufl.dot(n_s, w) * ds(TOP))

    # dynamics: the mass term regularises the tangent (no Dirichlet). dolfinx 0.11 SNES.
    problem = NonlinearProblem(residual, u, bcs=[], petsc_options_prefix="drop_",
                               petsc_options=fem_snes_options(rtol=1.0e-7, atol=1.0e-9, max_it=50))

    # update expressions (Newmark) and diagnostics
    ip = V.element.interpolation_points
    a_expr = fem.Expression(a_new, ip)
    v_expr = fem.Expression(v_new, ip)
    a_tmp, v_tmp = fem.Function(V), fem.Function(V)

    strain_form = fem.form((0.5 * mu * (ufl.tr(F.T * F) - d) - mu * ufl.ln(J)
                            + 0.5 * lam * ufl.ln(J) ** 2) * ufl.dx)
    ke_form = fem.form(0.5 * rho * ufl.dot(v_new, v_new) * ufl.dx)
    fz_on_block_form = fem.form(p_sphere * n_s[2] * ds(TOP))   # z-force sphere exerts on block (<=0)
    ips = Vs.element.interpolation_points
    pen_expr = fem.Expression(pen_s, ips)
    pen_fn = fem.Function(Vs)

    def max_pen():
        pen_fn.interpolate(pen_expr)
        loc = float(pen_fn.x.array.max()) if pen_fn.x.array.size else 0.0
        return comm.allreduce(loc, op=MPI.MAX)

    # rigid sphere ODE (vertical), staggered
    m_s = params.DROP_SPHERE_DENSITY * (4.0 / 3.0) * math.pi * R ** 3
    c_z = params.DROP_SPHERE_Z0
    v_s = 0.0
    print(f"[drop-fem] block {Lx:.2f}x{Ly:.2f}x{Lz:.2f}, sphere R={R} m={m_s:.3f} kg, "
          f"dt={params.DROP_DT} steps={params.DROP_STEPS}, kn={float(kn.value):.3g}, cd={float(cd.value):.3g}")

    hist = []  # t, sphere_z, penetration, U_strain, KE_block, contact_force
    for k in range(params.DROP_STEPS):
        centre.value = np.array([cx, cy, c_z], dtype=default_scalar_type)
        v_sphere.value = np.array([0.0, 0.0, v_s], dtype=default_scalar_type)
        n_it, converged = solve_status(problem, u)

        # contact force and sphere update (staggered ODE)
        fz_on_block = comm.allreduce(fem.assemble_scalar(fz_on_block_form), op=MPI.SUM)
        fz_on_sphere = -fz_on_block                      # Newton's third law (upward when pressed)
        v_s += params.DROP_DT * (-g + fz_on_sphere / m_s)
        c_z += params.DROP_DT * v_s

        # diagnostics
        U = comm.allreduce(fem.assemble_scalar(strain_form), op=MPI.SUM)
        ke = comm.allreduce(fem.assemble_scalar(ke_form), op=MPI.SUM)
        hist.append((k * params.DROP_DT, c_z, max_pen(), U, ke, abs(fz_on_block)))

        # advance Newmark state (compute new v,a from u BEFORE overwriting old)
        a_tmp.interpolate(a_expr)
        v_tmp.interpolate(v_expr)
        a_old.x.array[:] = a_tmp.x.array
        v_old.x.array[:] = v_tmp.x.array
        u_old.x.array[:] = u.x.array

        if k % 25 == 0:
            print(f"[drop-fem] t={k * params.DROP_DT:.3f}s  sphere_z={c_z:.3f}  "
                  f"pen={hist[-1][2] * 1000:.2f}mm  F={hist[-1][5]:.3g}N  its={n_it}"
                  + ("" if converged else "  [WARN]"))

    os.makedirs(params.DATA_DIR, exist_ok=True)
    np.savez(params.FEM_DROP_NPZ, history=np.array(hist, dtype=np.float64),
             sphere_r=R, block=(Lx, Ly, Lz))
    print(f"[drop-fem] wrote {params.FEM_DROP_NPZ}")


if __name__ == "__main__":
    main()
