"""Friction -- Newton (XPBD) side: a soft block sliding on the ground.

Same scenario as fenics_run/run_friction.py: a flat soft block rests on a rigid
ground plane under gravity, and its TOP face is dragged tangentially (+x) in small
kinematic increments. Coulomb friction at the floor is set via `soft_contact_mu`.

The headline, on-message point: XPBD enforces friction as a positional projection
and exposes NO calibrated tangential force -- exactly as it exposes no calibrated
normal contact force. So the comparison axis here is the kinematic response:

  * mean bottom slip vs. top drag  -> the stick-then-slip knee, and
  * strain energy / residual KE.

The FEM run provides the actual friction force and dissipated work; this run shows
what XPBD can and cannot give.

-> data/newton_friction.npz

Run on Colab (CUDA):  python -m newton_run.run_friction

NOTE: ground-plane and particle-pinning calls are marked TODO[verify-on-colab].
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from common import params
from compare import energies as en


def main():
    parser = argparse.ArgumentParser(description="Newton XPBD sliding-block friction")
    parser.add_argument("--device", default=None)
    parser.add_argument("--frames-per-step", type=int, default=30)
    args = parser.parse_args()

    import newton
    import warp as wp

    wp.init()
    device = args.device or str(wp.get_device())
    print(f"[friction-newton] device = {device}")

    nx, ny, nz = params.FRICTION_DIM
    h = params.FRICTION_CELL
    Lx, Ly, Lz = nx * h, ny * h, nz * h
    mu = params.FRICTION_MU

    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=-params.GRAVITY)
        builder.default_particle_radius = 0.01
        builder.particle_max_velocity = 50.0
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=nx, dim_y=ny, dim_z=nz, cell_x=h, cell_y=h, cell_z=h,
            density=params.DENSITY, k_mu=params.K_MU, k_lambda=params.K_LAMBDA, k_damp=params.K_DAMP,
        )
        builder.add_ground_plane()                    # TODO[verify-on-colab]
        model = builder.finalize()

        rest = model.particle_q.numpy()
        tets = model.tet_indices.numpy()
        masses = model.particle_mass.numpy()
        tol = 0.25 * h
        top = np.where(rest[:, 2] > rest[:, 2].max() - tol)[0]
        bottom = np.where(rest[:, 2] < rest[:, 2].min() + tol)[0]
        rest_top = rest[top].copy()

        # pin the top face (kinematic drag platen): inv_mass = 0
        inv = model.particle_inv_mass.numpy()
        inv[top] = 0.0
        model.particle_inv_mass = wp.array(inv, dtype=wp.float32, device=model.device)

        # Coulomb friction at the ground; normal stiffness is uncalibrated (the point)
        model.soft_contact_ke = 1.0e3
        model.soft_contact_kd = 1.0
        model.soft_contact_kf = 1.0e3
        model.soft_contact_mu = mu

        solver = newton.solvers.SolverXPBD(model=model, iterations=10)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        fps = 60
        substeps = 32
        sim_dt = (1.0 / fps) / substeps

        def set_top_drag(s):
            for st in (state_0, state_1):
                q = st.particle_q.numpy()
                q[top] = rest_top + np.array([s, 0.0, 0.0])
                st.particle_q.assign(q.astype(np.float32))
                qd = st.particle_qd.numpy()
                qd[top] = 0.0
                st.particle_qd.assign(qd)

        drags = np.linspace(0.0, params.FRICTION_DRAG_MAX, params.FRICTION_STEPS + 1)
        weight = params.friction_block_weight()
        print(f"[friction-newton] block {Lx:.2f}x{Ly:.2f}x{Lz:.2f} m, W={weight:.2f} N, mu={mu}")

        drag_rec, slip_rec, e_strain, ke = [], [], [], []
        wp.synchronize()
        t0 = time.perf_counter()
        for s in drags:
            set_top_drag(s)
            for _ in range(args.frames_per_step):
                for _ in range(substeps):
                    state_0.clear_forces()
                    model.collide(state_0, contacts)
                    solver.step(state_0, state_1, control, contacts, sim_dt)
                    state_0, state_1 = state_1, state_0
            q = state_0.particle_q.numpy()
            slip = float(np.mean(q[bottom, 0] - rest[bottom, 0]))
            drag_rec.append(s); slip_rec.append(slip)
            e_strain.append(en.strain_energy(rest, q, tets))
            ke.append(en.kinetic_energy(masses, state_0.particle_qd.numpy()))
            print(f"[friction-newton] drag={s*1000:5.2f} mm  bottom_slip={slip*1000:5.2f} mm  "
                  f"U={e_strain[-1]:.4g} J  KE={ke[-1]:.2e} J")
        wp.synchronize()
        wall_time = time.perf_counter() - t0

        os.makedirs(params.DATA_DIR, exist_ok=True)
        np.savez(
            params.NEWTON_FRICTION_NPZ,
            drag=np.array(drag_rec), bottom_slip=np.array(slip_rec),
            e_strain=np.array(e_strain), ke=np.array(ke),
            mu=mu, weight=weight, wall_time=wall_time,
        )
        print(f"[friction-newton] wrote {params.NEWTON_FRICTION_NPZ}")
        print("[friction-newton] note: XPBD exposes no calibrated friction force; "
              "see the FEM run for force + dissipated work.")


if __name__ == "__main__":
    main()
