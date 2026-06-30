"""Dynamic drop -- Newton, the literal rigid_soft_contact scenario.

Adapted from Newton's examples/multiphysics/example_rigid_soft_contact.py to match
the FEM drop benchmark (same block geometry/material and sphere as params.DROP_*):
a soft block rests on the ground and a *free* rigid sphere is dropped onto it
under gravity. This is the original dynamic-impact case (vs. the indentation
test's controlled quasi-static indentation).

Solver note: runs all three Newton solvers via ``--solver xpbd|vbd|semi_implicit``
(default XPBD). This is the HARDEST contact case: the sphere is a *free* rigid body, so
the implicit VBD must integrate it two-way itself (AVBD) -- enabled via the
``rigid_body_particle_contact_buffer_size`` knob in the shared solver factory. Only VBD
drives a genuine two-way impact on the free sphere; XPBD does not push the sphere down.
The implicit VBD is the natural counterpart to the implicit Newmark FEM, but even a VBD
run is only a *partial* fairness fix: the transient also mixes material (Newton
StVK/co-rotational vs FEM Neo-Hookean), contact model and time integration (see
compare/drop), so it is never a clean solver-only comparison.

Records a time series -> data/newton_drop{,_vbd,_semi}.npz (per --solver):
  t, sphere_z, penetration, strain_energy (block), kinetic_energy (block)
  plus rest_q, final_q, tet_indices, sphere_c at the deepest-impact frame (3D scene).

Run on Colab (CUDA):  python -m newton_run.run_drop [--solver vbd|semi_implicit]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from common import params
from compare import energies as en


def main():
    from newton_run._solver import SOLVERS, make_solver, needs_coloring

    parser = argparse.ArgumentParser(description="Newton dynamic drop")
    parser.add_argument("--device", default=None)
    parser.add_argument("--solver", choices=SOLVERS, default="xpbd",
                        help="xpbd (default) | vbd (implicit, two-way AVBD for the free "
                             "sphere) | semi_implicit (explicit)")
    args = parser.parse_args()

    import newton
    import warp as wp

    wp.init()
    device = args.device or str(wp.get_device())
    print(f"[drop-newton] device = {device}, solver = {args.solver}")

    nx, ny, nz = params.DROP_DIM
    h = params.DROP_CELL
    Lx, Ly, Lz = nx * h, ny * h, nz * h
    R = params.DROP_SPHERE_R
    cx, cy = Lx / 2.0, Ly / 2.0

    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=-params.GRAVITY)
        builder.default_particle_radius = 0.01
        builder.particle_max_velocity = 50.0

        ground_cfg = newton.ModelBuilder.ShapeConfig(ke=2.0e5, kd=1.0, kf=1.0e3, mu=0.5)
        builder.add_ground_plane(cfg=ground_cfg)

        # soft block, bottom face on the ground (z = 0)
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=nx, dim_y=ny, dim_z=nz, cell_x=h, cell_y=h, cell_z=h,
            density=params.DENSITY, k_mu=params.K_MU, k_lambda=params.K_LAMBDA, k_damp=params.K_DAMP,
        )

        # free rigid sphere dropped from above
        sphere_cfg = newton.ModelBuilder.ShapeConfig(
            density=params.DROP_SPHERE_DENSITY, ke=1.0e5, kd=1.0e-4, kf=1.0e3, mu=0.3)
        sphere_body = builder.add_body(
            xform=wp.transform(wp.vec3(cx, cy, params.DROP_SPHERE_Z0), wp.quat_identity()),
            label="sphere")
        builder.add_shape_sphere(sphere_body, radius=R, cfg=sphere_cfg, label="rigid_sphere")

        if needs_coloring(args.solver):
            builder.color()                           # vertex graph colouring for VBD
        model = builder.finalize()
        model.soft_contact_ke = 1.0e5
        model.soft_contact_kd = 1.0e-4
        model.soft_contact_kf = 1.0e3
        model.soft_contact_mu = 0.3

        # the sphere is a FREE rigid body -> VBD integrates it two-way (AVBD), which needs
        # a body<->particle contact buffer; XPBD/SemiImplicit ignore the argument.
        rigid_buffer = 256 if args.solver == "vbd" else None
        solver = make_solver(args.solver, model, iterations=10, rigid_particle_buffer=rigid_buffer)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        rest = model.particle_q.numpy()
        tets = model.tet_indices.numpy()
        masses = model.particle_mass.numpy()

        fps = params.FPS
        substeps = 32
        sim_dt = (1.0 / fps) / substeps
        n_frames = int(round(params.DROP_DURATION * fps))

        hist = []  # t, sphere_z, penetration, U_strain, KE
        lowest_z, scene_q, scene_c = float("inf"), rest, np.array([cx, cy, params.DROP_SPHERE_Z0], float)
        for frame in range(n_frames):
            for _ in range(substeps):
                state_0.clear_forces()
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0

            q = state_0.particle_q.numpy()
            c = state_0.body_q.numpy()[sphere_body][:3]
            pen = float(np.maximum(R - np.linalg.norm(q - c, axis=1), 0.0).max())
            U = en.strain_energy(rest, q, tets)
            ke = en.kinetic_energy(masses, state_0.particle_qd.numpy())
            hist.append((frame / fps, float(c[2]), pen, U, ke))
            if float(c[2]) < lowest_z:              # deepest impact = sphere centre at its lowest point
                lowest_z, scene_q, scene_c = float(c[2]), q.copy(), np.array(c, dtype=float)
            if frame % 5 == 0:
                print(f"[drop-newton] t={frame / fps:.3f}s  sphere_z={c[2]:.3f}  pen={pen * 1000:.2f}mm  "
                      f"U={U:.3g} KE={ke:.3g}")

        os.makedirs(params.DATA_DIR, exist_ok=True)
        out = params.solver_npz(params.NEWTON_DROP_NPZ, args.solver)
        np.savez(out, history=np.array(hist, dtype=np.float64),
                 sphere_r=R, block=(Lx, Ly, Lz), solver=args.solver,
                 # deformed mesh + sphere at deepest impact (for the 3D scene render)
                 rest_q=rest, final_q=scene_q, tet_indices=tets, sphere_c=scene_c)
        print(f"[drop-newton] wrote {out}")


if __name__ == "__main__":
    main()
