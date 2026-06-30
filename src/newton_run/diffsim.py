"""Hanging bar -- Newton's superpower: DIFFERENTIABLE simulation for evaluation.

Newton is built on NVIDIA Warp, so the whole simulation is differentiable: we can
backpropagate a loss through the settling of the soft body and get exact gradients
w.r.t. the material parameters -- something a black-box solver cannot give.

We use this two ways, both against the FEM reference (data/fem_result.npz):

  1. SENSITIVITY (one forward+backward, robust):
       loss(theta) = || q_newton_settled(theta) - q_fem ||^2  over all nodes,
     where theta scales the Lame parameters (k_mu, k_lambda). dLoss/dtheta is the
     exact sensitivity of the Newton-vs-FEM mismatch to stiffness -- for free.

  2. INVERSE FIT (gradient descent):
     optimise theta so Newton's settled shape matches the FEM solution. The fitted
     theta* is the *effective-stiffness multiplier*: theta* > 1 means Newton is
     effectively softer than the true material (needs stiffening to match FEM),
     theta* < 1 means stiffer. This turns the qualitative "the Newton model looks
     a bit soft" into a number. NOTE: the fit runs on the differentiable
     SemiImplicit solver (below), so theta* characterises *that* solver's effective
     stiffness vs FEM -- not XPBD's, whose softness is measured directly by the
     equilibrium residual and tip ratio in compare/hanging_bar.

Pattern follows Newton's own examples/diffsim/example_diffsim_soft_body.py:
  model = builder.finalize(requires_grad=True); a full pre-allocated state
  trajectory; a wp.Tape() around forward(); tape.backward(loss); warp.optim.SGD.

Only the SemiImplicit solver is differentiable here, and it is the one this fit
uses; XPBD (the positional projection) is not differentiable.

Run on Colab (CUDA):  python -m newton_run.diffsim

NOTE: heavily marked TODO[verify-on-colab] -- this is the most Newton-version- and
autodiff-specific piece; the learning rate in particular will need tuning.
"""

import argparse
import os

import numpy as np
import warp as wp

from common import params


@wp.kernel
def assign_theta(theta: wp.array[wp.float32], k_mu0: wp.float32, k_lambda0: wp.float32,
                 tet_materials: wp.array2d[wp.float32]):
    tid = wp.tid()
    tet_materials[tid, 0] = theta[0] * k_mu0
    tet_materials[tid, 1] = theta[0] * k_lambda0


@wp.kernel
def disp_loss(q: wp.array[wp.vec3], target: wp.array[wp.vec3], loss: wp.array[wp.float32]):
    tid = wp.tid()
    d = q[tid] - target[tid]
    wp.atomic_add(loss, 0, wp.dot(d, d))


class DiffFit:
    def __init__(self, args):
        import newton

        self.args = args
        self.k_mu0 = float(params.K_MU)
        self.k_lambda0 = float(params.K_LAMBDA)

        # FEM reference (same shared mesh -> same node ordering as the soft grid)
        fem = np.load(params.FEM_NPZ)
        self.target = wp.array(fem["final_q"].astype(np.float32), dtype=wp.vec3)
        self.fixed = fem["fixed_nodes"]

        # build the same hanging block, differentiable
        builder = newton.ModelBuilder(gravity=-params.GRAVITY)  # match FEM (default is -9.81)
        builder.default_particle_radius = 0.01
        builder.add_soft_grid(
            pos=wp.vec3(*params.ORIGIN), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=params.GRID_DIM_X, dim_y=params.GRID_DIM_Y, dim_z=params.GRID_DIM_Z,
            cell_x=params.CELL, cell_y=params.CELL, cell_z=params.CELL,
            density=params.DENSITY, k_mu=params.K_MU, k_lambda=params.K_LAMBDA, k_damp=params.K_DAMP,
        )
        # TODO[verify-on-colab]: finalize(requires_grad=True) for differentiable sim
        self.model = builder.finalize(requires_grad=True)

        # clamp the top face (same as the forward hanging-bar run)
        rest = self.model.particle_q.numpy()
        tol = params.FACE_TOL_FRAC * params.CELL
        top = np.where(rest[:, 2] > rest[:, 2].max() - tol)[0]
        inv = self.model.particle_inv_mass.numpy()
        inv[top] = 0.0
        self.model.particle_inv_mass = wp.array(inv, dtype=wp.float32, device=self.model.device)

        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        self.fps = 60
        self.sim_steps = args.sim_steps
        self.sim_substeps = 16
        self.sim_dt = (1.0 / self.fps) / self.sim_substeps
        # full trajectory of states (differentiation needs every intermediate state)
        self.states = [self.model.state() for _ in range(self.sim_steps * self.sim_substeps + 1)]
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.theta = wp.array([1.0], dtype=wp.float32, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        import warp.optim
        self.optimizer = warp.optim.SGD([self.theta], lr=args.lr, nesterov=False)

    def forward(self):
        wp.launch(assign_theta, dim=self.model.tet_count,
                  inputs=(self.theta, self.k_mu0, self.k_lambda0), outputs=(self.model.tet_materials,))
        for t in range(self.sim_steps * self.sim_substeps):
            self.states[t].clear_forces()
            self.model.collide(self.states[t], self.contacts)
            self.solver.step(self.states[t], self.states[t + 1], self.control, self.contacts, self.sim_dt)
        wp.launch(disp_loss, dim=self.model.particle_count,
                  inputs=(self.states[-1].particle_q, self.target), outputs=(self.loss,))
        return self.loss

    def forward_backward(self):
        self.tape = wp.Tape()
        with self.tape:
            self.forward()
        self.tape.backward(self.loss)
        return float(self.loss.numpy()[0]), float(self.theta.grad.numpy()[0])

    def step(self):
        loss, grad = self.forward_backward()
        self.optimizer.step([self.theta.grad])
        # keep theta positive
        th = self.theta.numpy()
        th[0] = max(1e-3, th[0])
        self.theta.assign(th)
        self.tape.zero()
        self.loss.zero_()
        return loss, grad


def main():
    parser = argparse.ArgumentParser(description="Differentiable stiffness fit (hanging bar)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sim-steps", type=int, default=100, help="frames to settle per forward")
    parser.add_argument("--train-iters", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    args = parser.parse_args()

    wp.init()
    device = args.device or str(wp.get_device())
    print(f"[diffsim] device = {device}")
    if not os.path.exists(params.FEM_NPZ):
        raise FileNotFoundError(f"{params.FEM_NPZ} missing -- run fenics_run.run_hanging_bar first")

    with wp.ScopedDevice(device):
        fit = DiffFit(args)

        # 1) sensitivity at theta = 1 (one forward+backward, no optimisation needed)
        loss0, grad0 = fit.forward_backward()
        print(f"[diffsim] at theta=1: mismatch loss = {loss0:.6e} m^2,  "
              f"dLoss/dtheta = {grad0:.6e}  (exact autodiff sensitivity)")
        fit.tape.zero(); fit.loss.zero_()

        # 2) inverse fit: optimise theta so Newton matches FEM
        loss_hist, theta_hist = [], []
        for it in range(args.train_iters):
            loss, grad = fit.step()
            theta = float(fit.theta.numpy()[0])
            loss_hist.append(loss); theta_hist.append(theta)
            if it % 5 == 0 or it == args.train_iters - 1:
                print(f"[diffsim] iter {it:3d}  loss={loss:.6e}  theta={theta:.4f}  grad={grad:.3e}")

        theta_star = float(fit.theta.numpy()[0])
        os.makedirs(params.DATA_DIR, exist_ok=True)
        np.savez(params.DIFFSIM_NPZ,
                 loss_history=np.array(loss_hist), theta_history=np.array(theta_hist),
                 theta_star=theta_star, loss0=loss0, grad0=grad0, lr=args.lr)
        print(f"[diffsim] wrote {params.DIFFSIM_NPZ}")
        print(f"[diffsim] fitted effective-stiffness multiplier theta* = {theta_star:.4f}")
        print("[diffsim]   theta* > 1  => Newton effectively SOFTER than FEM (needs stiffening to match)")
        print("[diffsim]   theta* < 1  => Newton effectively STIFFER than FEM")


if __name__ == "__main__":
    main()
