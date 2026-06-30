"""Indentation -- Newton side: a rigid sphere pressed into a soft slab.

Adapted from Newton's `examples/multiphysics/example_rigid_soft_contact.py`, but
configured to match the FEniCSx indentation benchmark (same slab geometry/material as
params.INDENT_*) and driven QUASI-STATICALLY: the sphere is a *kinematic* body
whose centre is lowered in the same indentation steps as the FEM run, and the
slab bottom is clamped.

Why kinematic + clamped: it mirrors the FEM indentation benchmark so the deformed
shapes are comparable. The difference we want to show is that the fast
positional XPBD has *no calibrated contact stiffness* -- it enforces contact as a
positional projection -- so the meaningful comparison axis is the DEFORMATION /
PENETRATION, not a contact force (XPBD does not expose one cleanly; the FEM penalty
run does).

Solver note: like the hanging bar, this runs all three Newton solvers via
``--solver xpbd|vbd|semi_implicit`` (default XPBD, the canonical run). All three drive the
SAME contact -- the kinematic sphere couples to the soft grid through the shared
soft_contact buffer (model.collide + model.soft_contact_*), the wiring Newton's own
example_rigid_soft_contact.py uses for every solver. The implicit VBD is the apples-to-
apples counterpart to the implicit FEM penalty solve; the kinematic collider makes this
the easiest contact case for a solver swap (no free rigid body to integrate).

Outputs -> data/newton_indentation{,_vbd,_semi}.npz (per --solver):
  * deltas                 indentation schedule [m]
  * line_x, uz_line        deformed top-surface dimple at max indentation
  * penetration            max sphere/soft-body overlap per indentation step [m]
  * rest_q, final_q, tet_indices, sphere_c, sphere_r   deformed mesh + sphere at
                           max indentation (for the 3D scene render)

Run on Colab (CUDA):  python -m newton_run.run_indentation [--solver vbd|semi_implicit]

Solver coverage: the kinematic-body and particle-pinning calls follow the proven
patterns in Newton's cable / rigid_soft_contact examples. All three solvers run and
record results on the pinned stack, but the VBD/SemiImplicit soft_contact path is too
soft for contact: the sphere sinks ~33 mm through the 40 mm indent, so XPBD is the only
Newton solver that geometrically resolves the indentation.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from common import params
from compare import energies as en


def _make_body_kinematic(builder, body):
    """Clear a body's mass properties so the solver treats it as kinematic.

    Mirrors `_make_body_kinematic` in Newton's cable example.
    """
    import warp as wp

    builder.body_mass[body] = 0.0
    builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = wp.mat33(0.0)
    builder.body_inv_inertia[body] = wp.mat33(0.0)


def build_model(builder_cls, color=False):
    import warp as wp

    nx, ny, nz = params.INDENT_DIM
    h = params.INDENT_CELL
    Lx, Ly, Lz = nx * h, ny * h, nz * h
    R = params.INDENT_SPHERE_R
    cx, cy = Lx / 2.0, Ly / 2.0

    # Match the FEM indentation run: gravity is OFF unless INDENT_WITH_GRAVITY (Newton default
    # is -9.81, which previously left the slab sagging while FEM had no gravity).
    g = -params.GRAVITY if params.INDENT_WITH_GRAVITY else 0.0
    builder = builder_cls(gravity=g)
    builder.default_particle_radius = 0.01
    builder.particle_max_velocity = 50.0

    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=nx, dim_y=ny, dim_z=nz,
        cell_x=h, cell_y=h, cell_z=h,
        density=params.DENSITY,
        k_mu=params.K_MU,
        k_lambda=params.K_LAMBDA,
        k_damp=params.K_DAMP,
    )

    # Kinematic rigid sphere, initially just touching the top face (delta = 0).
    sphere_cfg = newton_shape_cfg(builder_cls)
    sphere_body = builder.add_body(
        xform=wp.transform(wp.vec3(cx, cy, Lz + R), wp.quat_identity()),
        label="sphere",
    )
    builder.add_shape_sphere(sphere_body, radius=R, cfg=sphere_cfg, label="rigid_sphere")
    _make_body_kinematic(builder, sphere_body)  # zero mass -> kinematic collider

    if color:
        builder.color()        # vertex graph colouring required by the VBD solver

    return builder, sphere_body, (Lx, Ly, Lz, R, cx, cy)


def newton_shape_cfg(builder_cls):
    """Rigid-sphere contact material (XPBD branch values from the example)."""
    return builder_cls.ShapeConfig(
        density=1.0,
        ke=75.0, kd=1.0, kf=1.0e3, mu=1.0,
    )


def pin_bottom(model):
    """Clamp the slab's bottom face for every solver (zero mass + inv_mass).

    Uses the shared pin so VBD/SemiImplicit hold the clamp too, not just XPBD (VBD's
    elasticity solve fixes a vertex only at particle_mass==0). See
    newton_run._solver.pin_particles.
    """
    from newton_run._solver import pin_particles

    rest = model.particle_q.numpy()
    tol = 0.25 * params.INDENT_CELL
    bottom = np.where(rest[:, 2] < rest[:, 2].min() + tol)[0]
    pin_particles(model, bottom)
    return rest, bottom


def set_sphere_z(states, solver, sphere_body, cx, cy, cz):
    """Write the (kinematic) sphere transform into every state buffer."""
    import warp as wp

    xform = np.array([cx, cy, cz, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    for st in states:
        bq = st.body_q.numpy()
        bq[sphere_body] = xform
        st.body_q.assign(bq)
        bqd = st.body_qd.numpy()
        bqd[sphere_body] = 0.0
        st.body_qd.assign(bqd)
    # keep the solver's previous-pose buffer consistent (avoids a velocity spike)
    if hasattr(solver, "body_q_prev"):
        solver.body_q_prev = wp.clone(states[0].body_q, device=solver.device)


def main():
    from newton_run._solver import SOLVERS, make_solver, needs_coloring

    parser = argparse.ArgumentParser(description="Newton rigid-sphere indentation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--frames-per-step", type=int, default=30)
    parser.add_argument("--solver", choices=SOLVERS, default="xpbd",
                        help="xpbd = positional projection (default, canonical run); "
                             "vbd = implicit (apples-to-apples with the implicit FEM); "
                             "semi_implicit = explicit force-based")
    args = parser.parse_args()

    import newton
    import warp as wp

    wp.init()
    device = args.device or str(wp.get_device())
    print(f"[indent-newton] device = {device}, solver = {args.solver}")

    with wp.ScopedDevice(device):
        builder, sphere_body, (Lx, Ly, Lz, R, cx, cy) = build_model(
            newton.ModelBuilder, color=needs_coloring(args.solver))
        model = builder.finalize()
        rest, bottom = pin_bottom(model)
        tets = model.tet_indices.numpy()
        masses = model.particle_mass.numpy()

        # XPBD contact knobs (largely ignored for the normal force -- the point).
        model.soft_contact_ke = 75.0
        model.soft_contact_kd = 1.0
        model.soft_contact_kf = 1.0e3
        model.soft_contact_mu = 1.0

        solver = make_solver(args.solver, model, iterations=10)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        fps = params.FPS
        substeps = 32
        sim_dt = (1.0 / fps) / substeps

        # top-surface particles near the centre line (for the dimple profile)
        tol = 0.25 * params.INDENT_CELL
        top = np.where(rest[:, 2] > rest[:, 2].max() - tol)[0]
        centre_line = top[np.abs(rest[top, 1] - cy) < 0.6 * params.INDENT_CELL]
        centre_line = centre_line[np.argsort(rest[centre_line, 0])]

        n_steps = params.INDENT_LOAD_STEPS
        deltas, penetration, e_strain, ke = [], [], [], []
        uz_line = None
        wp.synchronize()
        t0 = time.perf_counter()
        for k in range(1, n_steps + 1):
            delta = params.INDENT_MAX * k / n_steps
            cz = Lz + R - delta
            set_sphere_z((state_0, state_1), solver, sphere_body, cx, cy, cz)

            for _ in range(args.frames_per_step):
                for _ in range(substeps):
                    state_0.clear_forces()
                    model.collide(state_0, contacts)
                    solver.step(state_0, state_1, control, contacts, sim_dt)
                    state_0, state_1 = state_1, state_0

            q = state_0.particle_q.numpy()
            d = q - np.array([cx, cy, cz])
            dist = np.linalg.norm(d, axis=1)
            pen = float(np.maximum(R - dist, 0.0).max())
            deltas.append(delta)
            penetration.append(pen)
            # energy diagnostics (same Neo-Hookean strain energy as the FEM run)
            e_strain.append(en.strain_energy(rest, q, tets))
            ke.append(en.kinetic_energy(masses, state_0.particle_qd.numpy()))
            uz_line = q[centre_line, 2] - rest[centre_line, 2]
            print(f"[indent-newton] delta={delta * 1000:6.2f} mm  pen={pen * 1000:.2f} mm  "
                  f"U_strain={e_strain[-1]:.4g} J  KE={ke[-1]:.2e} J")

        wp.synchronize()
        wall_time = time.perf_counter() - t0

        os.makedirs(params.DATA_DIR, exist_ok=True)
        out = params.solver_npz(params.NEWTON_INDENT_NPZ, args.solver)
        np.savez(
            out,
            deltas=np.array(deltas),
            penetration=np.array(penetration),
            e_strain=np.array(e_strain),
            ke=np.array(ke),
            line_x=rest[centre_line, 0],
            uz_line=np.asarray(uz_line),
            cx=cx,
            wall_time=wall_time,
            # deformed mesh + sphere at max indentation (for the 3D scene render)
            rest_q=rest, final_q=q, tet_indices=tets,
            sphere_c=np.array([cx, cy, cz], dtype=float), sphere_r=float(R),
            solver=args.solver,
        )
        print(f"[indent-newton] wrote {out}")
        print(f"[indent-newton] solve wall time = {wall_time:.3f} s (solver={args.solver})")


if __name__ == "__main__":
    main()
