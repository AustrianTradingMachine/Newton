"""Effective stress-strain (3) -- Newton side, confined uniaxial strain.

Same homogeneous test as the FEM run: a small block is driven through
F = diag(1, 1, lambda) by pinning ALL boundary nodes at their affine positions
(x, y kept, z scaled by lambda) and letting the interior settle (gravity off,
SemiImplicit solver -- the force-based, differentiable one). For each lambda we
read the settled configuration and report the volume-averaged axial Neo-Hookean
stress.

Honest scope: with a fully prescribed affine boundary this primarily checks
whether Newton's solver *reproduces the prescribed homogeneous deformation* across
the strain range (the interior should settle to the affine field). The rigorous
measure of Newton's effective *constitutive* response is the differentiable
stiffness fit theta* (newton_run/diffsim_stage_a.py) and the equilibrium residual.

-> data/newton_stress_strain.npz : lambdas, sigma_newton, sigma_analytic

Run on Colab (CUDA):  python -m newton_run.run_stress_strain
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from common import params
from compare import energies as en


def main():
    parser = argparse.ArgumentParser(description="Newton uniaxial stress-strain")
    parser.add_argument("--device", default=None)
    parser.add_argument("--settle-frames", type=int, default=60)
    args = parser.parse_args()

    import warp as wp
    import newton

    wp.init()
    device = args.device or str(wp.get_device())
    print(f"[stress-newton] device = {device}")

    nx, ny, nz = params.STRESS_DIM
    h = params.STRESS_CELL
    Lx, Ly, Lz = nx * h, ny * h, nz * h

    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=0.0)        # pure material test, no gravity
        builder.default_particle_radius = 0.01
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=nx, dim_y=ny, dim_z=nz, cell_x=h, cell_y=h, cell_z=h,
            density=params.DENSITY, k_mu=params.K_MU, k_lambda=params.K_LAMBDA, k_damp=params.K_DAMP,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverSemiImplicit(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        rest = model.particle_q.numpy().astype(np.float64)
        tets = model.tet_indices.numpy()

        # all boundary nodes (any outer face) -> pinned and driven affinely
        tol = 0.25 * h
        bnd = np.where((rest[:, 0] < tol) | (rest[:, 0] > Lx - tol)
                       | (rest[:, 1] < tol) | (rest[:, 1] > Ly - tol)
                       | (rest[:, 2] < tol) | (rest[:, 2] > Lz - tol))[0]
        inv = model.particle_inv_mass.numpy()
        inv[bnd] = 0.0
        model.particle_inv_mass = wp.array(inv, dtype=wp.float32, device=model.device)

        fps = 60
        substeps = 32
        sim_dt = (1.0 / fps) / substeps

        lambdas = params.stress_lambdas()
        sigma = []
        for L in lambdas:
            # drive the pinned boundary to the affine config F = diag(1,1,L)
            target = rest[bnd].copy()
            target[:, 2] *= L
            for st in (state_0, state_1):
                q = st.particle_q.numpy()
                q[bnd] = target
                st.particle_q.assign(q.astype(np.float32))
                qd = st.particle_qd.numpy()
                qd[bnd] = 0.0
                st.particle_qd.assign(qd)

            for _ in range(args.settle_frames):
                for _ in range(substeps):
                    state_0.clear_forces()
                    model.collide(state_0, contacts)
                    solver.step(state_0, state_1, control, contacts, sim_dt)
                    state_0, state_1 = state_1, state_0

            q = state_0.particle_q.numpy().astype(np.float64)
            sig = en.mean_axial_first_piola(rest, q, tets)
            sigma.append(sig)
            print(f"[stress-newton] lambda={L:.3f}  sigma={sig:.4g} Pa")

        sigma = np.array(sigma)
        sigma_ana = en.neohookean_uniaxial_strain_stress(lambdas)
        os.makedirs(params.DATA_DIR, exist_ok=True)
        np.savez(params.NEWTON_STRESS_NPZ, lambdas=lambdas, sigma_newton=sigma, sigma_analytic=sigma_ana)
        print(f"[stress-newton] wrote {params.NEWTON_STRESS_NPZ}")


if __name__ == "__main__":
    main()
