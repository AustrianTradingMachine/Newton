"""Indentation -- contact prototype in FEniCSx (dolfinx), pure Python / UFL.

A rigid analytic sphere is pressed into a soft slab; non-penetration is enforced
without C++ and without a mesh-mesh search (the sphere is analytic). Two contact
methods are provided, selected per variant in params.INDENT_VARIANTS:

  method = "penalty"
      Augmented pressure p = kn * <-g>+  added to the weak form. Simple, but the
      result and the residual penetration depend on the penalty stiffness kn.

  method = "aug_lagrangian"  (Uzawa)
      Keep a multiplier field lambda (contact pressure estimate) and use the
      augmented pressure  p = <lambda - kn*g>+ . After each inner nonlinear solve
      update  lambda <- p . Iterating this OUTER loop drives the penetration to
      ~0 at a *modest* kn, i.e. it approaches the exact constraint without the
      ill-conditioning of kn -> infinity -- all still in pure UFL, no saddle-point
      system, no extra global unknowns (lambda is interpolated, not solved).

Both run the same indentation schedule as the Newton XPBD indentation run. Per step we
record contact force, strain energy, contact (penalty) energy and the max
penetration. Run:

    python -m fenics_run.run_indentation
"""

from __future__ import annotations

import os
import time

import numpy as np

from common import params
from fenics_run.run_hanging_bar import evaluate_at_nodes, fem_snes_options, neo_hookean_residual, solve_status


def _slug(cell_name, factor, method):
    tag = "aug" if method == "aug_lagrangian" else "pen"
    return f"{cell_name}_kn{factor:g}_{tag}".replace(".", "p")


def _label(cell_name, factor, method):
    tag = "AL" if method == "aug_lagrangian" else "penalty"
    return f"{cell_name} kn x{factor:g} {tag}"


def _run(cell_name, penalty_factor, method, comm):
    """Run the indentation for one (element, penalty, method) variant."""
    import ufl
    from dolfinx import default_scalar_type, fem
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.mesh import CellType, create_box, locate_entities_boundary, meshtags
    from mpi4py import MPI

    cell_type = CellType.tetrahedron if cell_name == "tet" else CellType.hexahedron
    is_aug = method == "aug_lagrangian"

    nx, ny, nz = params.INDENT_DIM
    h = params.INDENT_CELL
    Lx, Ly, Lz = nx * h, ny * h, nz * h
    R = params.INDENT_SPHERE_R
    cx, cy = Lx / 2.0, Ly / 2.0
    tol = 0.25 * h

    msh = create_box(comm, [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
                     [nx, ny, nz], cell_type)
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    Vs = fem.functionspace(msh, ("Lagrange", 1))      # scalar space for lambda / penetration
    u = fem.Function(V, name="displacement")
    v = ufl.TestFunction(V)

    bottom_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], 0.0, atol=tol))
    bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), bottom_dofs, V)

    fdim = msh.topology.dim - 1
    top_facets = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], Lz, atol=tol))
    TOP = 1
    facet_tags = meshtags(msh, fdim, top_facets, np.full(len(top_facets), TOP, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    mu = fem.Constant(msh, default_scalar_type(params.K_MU))
    lmbda = fem.Constant(msh, default_scalar_type(params.K_LAMBDA))
    kn = fem.Constant(msh, default_scalar_type(penalty_factor * params.YOUNGS_E / h))
    grav = (0.0, 0.0, -params.DENSITY * params.GRAVITY) if params.INDENT_WITH_GRAVITY else (0.0, 0.0, 0.0)
    body_force = fem.Constant(msh, np.array(grav, dtype=default_scalar_type))
    centre = fem.Constant(msh, np.array([cx, cy, Lz + R], dtype=default_scalar_type))

    # geometry: signed gap g and sphere normal at the deformed position
    X = ufl.SpatialCoordinate(msh)
    d_vec = (X + u) - centre
    dist = ufl.sqrt(ufl.dot(d_vec, d_vec) + 1.0e-12)
    gap = dist - R                                    # g (negative = penetrating)
    n_obs = d_vec / dist
    penetration_ufl = ufl.max_value(-gap, 0.0)        # <-g>+

    # multiplier field (0 for plain penalty) and augmented pressure
    lam = fem.Function(Vs, name="contact_pressure")   # stays 0 for penalty
    pressure = ufl.max_value(lam - kn * gap, 0.0)      # penalty: lam=0 -> kn*<-g>+

    bulk = neo_hookean_residual(u, v, msh, mu, lmbda, body_force)
    residual = bulk - pressure * ufl.dot(n_obs, v) * ds(TOP)

    problem = NonlinearProblem(
        residual, u, bcs=[bc],
        petsc_options_prefix=f"sb_{_slug(cell_name, penalty_factor, method)}_",
        petsc_options=fem_snes_options(rtol=1.0e-7, atol=1.0e-10, max_it=60),
    )

    force_form = fem.form(pressure * n_obs[2] * ds(TOP))
    dim = msh.geometry.dim
    Fdef = ufl.Identity(dim) + ufl.grad(u)
    Jdef = ufl.det(Fdef)
    psi = 0.5 * mu * (ufl.tr(Fdef.T * Fdef) - dim) - mu * ufl.ln(Jdef) + 0.5 * lmbda * ufl.ln(Jdef) ** 2
    strain_form = fem.form(psi * ufl.dx)
    contact_energy_form = fem.form(0.5 * kn * penetration_ufl ** 2 * ds(TOP))

    # interpolation expressions: max penetration tracker and the Uzawa update.
    # dolfinx 0.11: element.interpolation_points is an attribute (array), not a call.
    ip = Vs.element.interpolation_points
    pen_expr = fem.Expression(penetration_ufl, ip)
    lam_update = fem.Expression(pressure, ip)
    pen_func = fem.Function(Vs)
    lam_tmp = fem.Function(Vs)

    def max_penetration():
        pen_func.interpolate(pen_expr)
        local = float(pen_func.x.array.max()) if pen_func.x.array.size else 0.0
        return comm.allreduce(local, op=MPI.MAX)

    n_aug = params.INDENT_AUG_ITERS if is_aug else 1
    n_steps = params.INDENT_LOAD_STEPS
    label = _label(cell_name, penalty_factor, method)
    deltas, f_fem, e_strain, e_contact, penetration = [], [], [], [], []
    print(f"[indent-fem:{label}] slab {Lx:.2f}x{Ly:.2f}x{Lz:.2f} m, R={R} m, "
          f"kn={float(kn.value):.3g} Pa/m, {n_steps} steps, n_aug={n_aug}")
    t0 = time.perf_counter()
    for k in range(1, n_steps + 1):
        delta = params.INDENT_MAX * k / n_steps
        centre.value = np.array([cx, cy, Lz + R - delta], dtype=default_scalar_type)
        for a in range(n_aug):  # noqa: B007 -- `a` is the Uzawa count, reported after the loop
            n_it, converged = solve_status(problem, u)
            if is_aug:
                lam_tmp.interpolate(lam_update)         # p = <lam - kn*g>+
                lam.x.array[:] = lam_tmp.x.array         # lambda <- p  (Uzawa update)
                lam.x.scatter_forward()
            if max_penetration() < params.INDENT_AUG_PEN_TOL:
                break
        fz = comm.allreduce(fem.assemble_scalar(force_form), op=MPI.SUM)
        deltas.append(delta)
        f_fem.append(abs(fz))
        e_strain.append(comm.allreduce(fem.assemble_scalar(strain_form), op=MPI.SUM))
        e_contact.append(comm.allreduce(fem.assemble_scalar(contact_energy_form), op=MPI.SUM))
        penetration.append(max_penetration())
        flag = "" if converged else "  [WARN not converged]"
        aug_str = f"  aug={a + 1}" if is_aug else ""      # Uzawa count only meaningful for AL
        print(f"[indent-fem:{label}] delta={delta * 1000:6.2f} mm{aug_str}  F={abs(fz):.4g} N  "
              f"pen={penetration[-1] * 1000:.3f} mm{flag}")
    wall_time = time.perf_counter() - t0
    print(f"[indent-fem:{label}] done: {n_steps} steps in {wall_time:.2f} s "
          f"({wall_time / n_steps * 1000:.0f} ms/step)  "
          f"F_max={f_fem[-1]:.4g} N  max_pen={penetration[-1] * 1000:.3f} mm")

    line_x = np.linspace(0.0, Lx, 81)
    line_pts = np.column_stack([line_x, np.full_like(line_x, cy), np.full_like(line_x, Lz)])
    uz_line = np.asarray(evaluate_at_nodes(u, msh, line_pts))[:, 2]

    return dict(cell=cell_name, factor=penalty_factor, method=method, label=label,
                slug=_slug(cell_name, penalty_factor, method),
                deltas=np.array(deltas), f_fem=np.array(f_fem),
                e_strain=np.array(e_strain), e_contact=np.array(e_contact),
                penetration=np.array(penetration), wall_time=wall_time,
                line_x=line_x, uz_line=uz_line, cx=cx)


def main():
    import dolfinx
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    print(f"[indent-fem] dolfinx {dolfinx.__version__}, variants={params.INDENT_VARIANTS}")

    results = [_run(cell, factor, method, comm) for (cell, factor, method) in params.INDENT_VARIANTS]
    deltas = results[0]["deltas"]
    f_hertz = params.hertz_force(deltas)

    os.makedirs(params.DATA_DIR, exist_ok=True)
    save = dict(deltas=deltas, f_hertz=f_hertz, line_x=results[0]["line_x"], cx=results[0]["cx"])
    for r in results:
        save[f"f_{r['slug']}"] = r["f_fem"]
        save[f"uz_{r['slug']}"] = r["uz_line"]
        save[f"e_strain_{r['slug']}"] = r["e_strain"]
        save[f"e_contact_{r['slug']}"] = r["e_contact"]
        save[f"pen_{r['slug']}"] = r["penetration"]
        save[f"time_{r['slug']}"] = r["wall_time"]
    np.savez(params.FEM_INDENT_NPZ, **save)
    print(f"[indent-fem] wrote {params.FEM_INDENT_NPZ}")
    for r in results:
        ratio = r["f_fem"][-1] / f_hertz[-1] if f_hertz[-1] > 0 else float("nan")
        print(f"[indent-fem] {r['label']}: F={r['f_fem'][-1]:.4g} N (Hertz ratio {ratio:.3f})  "
              f"max_pen={r['penetration'][-1] * 1000:.3f} mm  wall={r['wall_time']:.2f} s")

    _plots(results, f_hertz)


def _plots(results, f_hertz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(params.FIG_DIR, exist_ok=True)

    plt.figure(figsize=(6, 5))
    for r in results:
        plt.plot(r["deltas"] * 1000, r["f_fem"], "o-", label="FEM " + r["label"])
    plt.plot(results[0]["deltas"] * 1000, f_hertz, "k--", lw=1.5, label="Hertz")
    plt.xlabel("indentation depth  [mm]"); plt.ylabel("contact force  [N]")
    plt.title("Indentation: contact force (element + penalty + AL)")
    plt.legend(); plt.grid(alpha=0.3)
    out = os.path.join(params.FIG_DIR, "indentation_force.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); print(f"[indent-fem] wrote {out}")

    plt.figure(figsize=(6, 4))
    for r in results:
        plt.plot(r["deltas"] * 1000, r["penetration"] * 1000, "o-", label="FEM " + r["label"])
    plt.xlabel("indentation depth  [mm]"); plt.ylabel("max penetration  [mm]")
    plt.title("Indentation: residual penetration (AL ~ 0 at modest kn)")
    plt.legend(); plt.grid(alpha=0.3)
    out = os.path.join(params.FIG_DIR, "indentation_penetration.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); print(f"[indent-fem] wrote {out}")

    plt.figure(figsize=(6, 4))
    for r in results:
        plt.plot((r["line_x"] - r["cx"]) * 1000, r["uz_line"] * 1000, label="FEM " + r["label"])
    plt.xlabel("x - x_centre  [mm]"); plt.ylabel("vertical displacement u_z  [mm]")
    plt.title("Indentation: deformed top-surface profile (max indentation)")
    plt.legend(); plt.grid(alpha=0.3)
    out = os.path.join(params.FIG_DIR, "indentation_profile.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); print(f"[indent-fem] wrote {out}")


if __name__ == "__main__":
    main()
