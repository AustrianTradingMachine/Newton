"""Convergence study -- Newton (XPBD) side, the hanging block.

XPBD is an iterative positional projection: its effective stiffness depends on
how many solver iterations and substeps it is given. This sweep makes that
explicit by re-running the hanging block over a grid of

  * solver iterations  (at a fixed substep count), and
  * substeps           (at a fixed iteration count),

and recording, for each configuration:

  * tip vertical drop                         [mm]
  * total strain energy                       [J]
  * free-node equilibrium-residual RMS        [N]   (distance from true static
                                                     equilibrium -- the cleanest
                                                     "have we converged" measure)
  * solve wall time                           [s]

As iterations / substeps grow the residual should fall and the tip drop should
approach the implicit-FEM / analytic value -- i.e. XPBD converges towards the
true equilibrium, at a cost. This is the Newton half of the discretisation-vs-
solver-error picture; the FEM half is fenics_run/convergence.py.

-> data/newton_convergence.npz

Run on Colab (CUDA):  python -m newton_run.convergence
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from common import params
from compare import energies as en
from newton_run.run_hanging_bar import simulate


def _metrics(res):
    """Reduce one simulate() result to scalar convergence metrics."""
    rest, final, tets = res["rest_q"], res["final_q"], res["tet_indices"]
    free = np.setdiff1d(np.arange(len(rest)), res["fixed_nodes"])
    tip_drop = -(final - rest)[free, 2].min() * 1000.0
    u = en.strain_energy(rest, final, tets)
    r = en.equilibrium_residual(rest, final, tets, res["fixed_nodes"])
    return tip_drop, u, r["free_rms"], res["wall_time"]


def main():
    parser = argparse.ArgumentParser(description="Newton XPBD convergence (hanging bar)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import warp as wp
    wp.init()
    device = args.device or str(wp.get_device())
    print(f"[conv-newton] device = {device}")

    # analytic + FEM tip references (mm), for the notebook overlay
    z_top = params.ORIGIN[2] + params.BLOCK_LZ
    tip_analytic = params.analytic_hanging_displacement(
        params.ORIGIN[2], z_top, params.BLOCK_LZ) * 1000.0
    tip_fem = np.nan
    if os.path.exists(params.FEM_NPZ):
        fem = np.load(params.FEM_NPZ)
        free = np.setdiff1d(np.arange(len(fem["rest_q"])), fem["fixed_nodes"])
        tip_fem = -(fem["final_q"] - fem["rest_q"])[free, 2].min() * 1000.0

    # --- sweep iterations at fixed substeps -------------------------------
    iters = np.array(params.CONV_XPBD_ITERS)
    it_tip, it_u, it_res, it_time = [], [], [], []
    for n in iters:
        res = simulate(solver_name="xpbd", iterations=int(n),
                       substeps=params.CONV_XPBD_FIXED_SUBSTEPS,
                       device=device, verbose=False)
        tip, u, rms, wt = _metrics(res)
        it_tip.append(tip); it_u.append(u); it_res.append(rms); it_time.append(wt)
        print(f"[conv-newton] iters={n:3d} (substeps={params.CONV_XPBD_FIXED_SUBSTEPS}): "
              f"tip={tip:.2f} mm  U={u:.4g} J  res_rms={rms:.4g} N  t={wt:.2f} s")

    # --- sweep substeps at fixed iterations -------------------------------
    subs = np.array(params.CONV_XPBD_SUBSTEPS)
    sb_tip, sb_u, sb_res, sb_time = [], [], [], []
    for n in subs:
        res = simulate(solver_name="xpbd", iterations=params.CONV_XPBD_FIXED_ITERS,
                       substeps=int(n), device=device, verbose=False)
        tip, u, rms, wt = _metrics(res)
        sb_tip.append(tip); sb_u.append(u); sb_res.append(rms); sb_time.append(wt)
        print(f"[conv-newton] substeps={n:3d} (iters={params.CONV_XPBD_FIXED_ITERS}): "
              f"tip={tip:.2f} mm  U={u:.4g} J  res_rms={rms:.4g} N  t={wt:.2f} s")

    os.makedirs(params.DATA_DIR, exist_ok=True)
    np.savez(
        params.NEWTON_CONV_NPZ,
        iters=iters, it_tip=np.array(it_tip), it_strain=np.array(it_u),
        it_res_rms=np.array(it_res), it_time=np.array(it_time),
        fixed_substeps=params.CONV_XPBD_FIXED_SUBSTEPS,
        substeps=subs, sb_tip=np.array(sb_tip), sb_strain=np.array(sb_u),
        sb_res_rms=np.array(sb_res), sb_time=np.array(sb_time),
        fixed_iters=params.CONV_XPBD_FIXED_ITERS,
        tip_analytic=tip_analytic, tip_fem=tip_fem,
    )
    print(f"[conv-newton] wrote {params.NEWTON_CONV_NPZ}")


if __name__ == "__main__":
    main()
