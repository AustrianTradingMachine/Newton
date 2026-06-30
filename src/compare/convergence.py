"""(4) Convergence study -- plots for the Newton (XPBD) and FEM h-refinement sweeps.

Reads data/newton_convergence.npz and data/fem_convergence.npz (either may be
missing) and writes figures showing:

  * XPBD: tip drop & equilibrium-residual RMS vs solver iterations / substeps,
    with the FEM and analytic tip as reference lines -- XPBD converging towards
    the true equilibrium as it is given more work.
  * FEM:  tip drop & strain energy vs mesh size h (and #DOFs), converging to a
    mesh-independent limit compared against the analytic 1-D bar; plus the
    load-increment sweep (converged tip is flat; only iteration count changes).

Everything here is a single-solver sweep (XPBD only, then FEM only), so the solver
colour (orange / blue) marks the data and the analytic level is black-dashed; the
non-solver diagnostics (a swept-parameter curve, the load-step iteration count) use
NEUTRAL colours so they never read as VBD/explicit/hex. The make_* helpers return a
Figure; main() sets Agg and saves the PNGs.

Run:  python -m compare.convergence
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from common import params
from compare import style

# Backend not forced at import (import-safe); main() sets Agg before saving.


def make_newton(nw):
    """XPBD tip & residual vs iterations / substeps, + cost -> Figure."""
    iters = nw["iters"]; subs = nw["substeps"]
    tip_fem = float(nw["tip_fem"]); tip_ana = float(nw["tip_analytic"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(iters, nw["it_tip"], "o-", color=style.COLOR["xpbd"], label="XPBD tip")
    if np.isfinite(tip_fem):
        ax.axhline(tip_fem, color=style.COLOR["fem"], ls=style.ANALYTIC_LS, lw=1.2, label="FEM tet")
    ax.axhline(tip_ana, color=style.COLOR["analytic"], ls=style.ANALYTIC_LS, lw=1.2, label="analytic 1-D")
    ax.set_xlabel(f"solver iterations (substeps={int(nw['fixed_substeps'])})")
    ax.set_ylabel("tip drop [mm]"); ax.set_title("XPBD tip vs iterations")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(iters, nw["it_res_rms"], "o-", color=style.COLOR["xpbd"])
    ax.set_xlabel(f"solver iterations (substeps={int(nw['fixed_substeps'])})")
    ax.set_ylabel("free-node residual RMS [N]")
    ax.set_title("XPBD equilibrium residual vs iterations"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(subs, nw["sb_tip"], "s-", color=style.COLOR["xpbd"], label="XPBD tip")
    if np.isfinite(tip_fem):
        ax.axhline(tip_fem, color=style.COLOR["fem"], ls=style.ANALYTIC_LS, lw=1.2, label="FEM tet")
    ax.axhline(tip_ana, color=style.COLOR["analytic"], ls=style.ANALYTIC_LS, lw=1.2, label="analytic 1-D")
    ax.set_xlabel(f"substeps (iterations={int(nw['fixed_iters'])})")
    ax.set_ylabel("tip drop [mm]"); ax.set_title("XPBD tip vs substeps")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.loglog(nw["it_time"], nw["it_res_rms"], "o-", color=style.COLOR["xpbd"], label="vary iters")
    ax.loglog(nw["sb_time"], nw["sb_res_rms"], "s-", color=style.NEUTRAL[1], label="vary substeps")
    ax.set_xlabel("solve wall time [s]"); ax.set_ylabel("residual RMS [N]")
    ax.set_title("XPBD accuracy vs cost"); ax.legend(); ax.grid(alpha=0.3, which="both")

    fig.suptitle("Hanging bar convergence -- Newton (XPBD)")
    fig.tight_layout()
    return fig


def make_fem(fm):
    """FEM tip & strain energy vs mesh size h (+ #DOFs) and the load-step sweep -> Figure."""
    h = fm["h"]; tip_ana = float(fm["tip_analytic"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(h * 1000, fm["h_tip"], "o-", color=style.COLOR["fem"], label="FEM tip")
    ax.axhline(tip_ana, color=style.COLOR["analytic"], ls=style.ANALYTIC_LS, lw=1.2, label="analytic 1-D")
    ax.set_xlabel("mesh size h [mm]"); ax.set_ylabel("tip drop [mm]")
    ax.set_title("FEM tip vs mesh size (-> h=0)"); ax.legend(); ax.grid(alpha=0.3)
    ax.invert_xaxis()

    ax = axes[0, 1]
    ax.plot(h * 1000, fm["h_strain"], "o-", color=style.COLOR["fem"], label="FEM U")
    if "strain_analytic" in fm.files:
        ax.axhline(float(fm["strain_analytic"]), color=style.COLOR["analytic"], ls=style.ANALYTIC_LS,
                   lw=1.2, label="analytic 1-D")
    ax.set_xlabel("mesh size h [mm]"); ax.set_ylabel("strain energy [J]")
    ax.set_title("FEM strain energy vs mesh size"); ax.legend(); ax.grid(alpha=0.3)
    ax.invert_xaxis()

    ax = axes[1, 0]
    ax.semilogx(fm["ndofs"], fm["h_tip"], "o-", color=style.COLOR["fem"])
    ax.set_xlabel("# DOFs"); ax.set_ylabel("tip drop [mm]")
    ax.set_title("FEM tip vs problem size"); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    ax.plot(fm["load_steps"], fm["ls_tip"], "o-", color=style.COLOR["fem"], label="tip drop [mm]")
    ax.set_xlabel("# gravity load increments"); ax.set_ylabel("tip drop [mm]", color=style.COLOR["fem"])
    ax2 = ax.twinx()
    ax2.plot(fm["load_steps"], fm["ls_its"], "s--", color=style.NEUTRAL[1], label="total Newton its")
    ax2.set_ylabel("total Newton iterations", color=style.NEUTRAL[1])
    ax.set_title("FEM load-increment sweep (tip flat = solve OK)"); ax.grid(alpha=0.3)

    fig.suptitle("Hanging bar convergence -- FEM (FEniCSx, h-refinement)")
    fig.tight_layout()
    return fig


def main():
    matplotlib.use("Agg")
    os.makedirs(params.FIG_DIR, exist_ok=True)
    nw = np.load(params.NEWTON_CONV_NPZ) if os.path.exists(params.NEWTON_CONV_NPZ) else None
    fm = np.load(params.FEM_CONV_NPZ) if os.path.exists(params.FEM_CONV_NPZ) else None
    if nw is None and fm is None:
        raise FileNotFoundError("run newton_run.convergence and/or fenics_run.convergence first")
    if nw is not None:
        fig = make_newton(nw)
        out = os.path.join(params.FIG_DIR, "convergence_newton.png")
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"[conv] wrote {out}")
        print(f"[conv] XPBD residual: {nw['it_res_rms'][0]:.3g} N @ {int(nw['iters'][0])} iters "
              f"-> {nw['it_res_rms'][-1]:.3g} N @ {int(nw['iters'][-1])} iters")
    if fm is not None:
        fig = make_fem(fm)
        out = os.path.join(params.FIG_DIR, "convergence_fem.png")
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"[conv] wrote {out}")
        print(f"[conv] FEM tip converges to {fm['h_tip'][-1]:.3f} mm "
              f"(analytic 1-D {float(fm['tip_analytic']):.3f} mm); "
              f"load-step spread = {fm['ls_tip'].max()-fm['ls_tip'].min():.2e} mm")


if __name__ == "__main__":
    main()
