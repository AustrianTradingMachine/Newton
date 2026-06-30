"""Friction -- plots for the sliding-block test, FEM vs Newton solvers.

Reads data/fem_friction.npz and whichever Newton solver runs are present
(data/newton_friction{,_vbd,_semi}.npz -- run_friction --solver writes one each; any
may be missing) and writes:

  * friction force vs top drag (FEM) with the analytic Coulomb plateau mu*W and
    the normal force N (~ weight) -- the stick-then-slip curve,
  * cumulative frictional work + slip fraction (FEM),
  * the kinematic response every solver shares: mean bottom slip vs top drag (XPBD exposes
    the slip but no calibrated friction force; the implicit VBD is the apples-to-apples
    counterpart to the implicit FEM).

The make_* helpers build and return a Figure (no save/show) with compare.style colours --
solver curves use the solver colours; FEM diagnostics (normal force, work, slip fraction)
use NEUTRAL colours so they never read as a solver. main() sets Agg and saves the PNGs.

Note: all three solvers' ground-contact runs (XPBD, VBD, SemiImplicit) record results;
the plots gracefully overlay whichever runs are present on disk.

Run:  python -m compare.friction
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from common import params
from compare import style

# Backend not forced at import (import-safe); main() sets Agg before saving.


def make_force(fem):
    """FEM friction force + normal force vs the analytic Coulomb plateau mu*W -> Figure."""
    d = fem["drag"] * 1000.0
    fig = plt.figure(figsize=(6, 5))
    plt.plot(d, fem["friction_force"], "o-", color=style.COLOR["fem"], label="FEM friction force")
    plt.plot(d, fem["normal_force"], "^-", color=style.NEUTRAL[2], alpha=0.8, label="FEM normal force N")
    plt.axhline(float(fem["plateau"]), color=style.COLOR["analytic"], ls=style.ANALYTIC_LS, lw=1.5,
                label="analytic mu*W")
    plt.axhline(float(fem["weight"]), color=style.NEUTRAL[1], ls=style.REF_LS, lw=1.2, label="weight W")
    plt.xlabel("top drag [mm]"); plt.ylabel("force [N]")
    plt.title("Friction: stick -> slip plateau (FEM vs Coulomb mu*W)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    return fig


def make_work(fem):
    """FEM dissipated frictional work + slipping-area fraction (twin axis) -> Figure."""
    d = fem["drag"] * 1000.0
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(d, fem["friction_work"], "o-", color=style.NEUTRAL[0], label="frictional work")
    ax1.set_xlabel("top drag [mm]"); ax1.set_ylabel("cumulative friction work [J]", color=style.NEUTRAL[0])
    ax2 = ax1.twinx()
    ax2.plot(d, fem["slip_fraction"], "s--", color=style.NEUTRAL[1], alpha=0.8)
    ax2.set_ylabel("slipping area fraction", color=style.NEUTRAL[1]); ax2.set_ylim(-0.05, 1.05)
    ax1.set_title("Friction: dissipated work & stick/slip transition (FEM)")
    fig.tight_layout()
    return fig


def make_slip(fem, newtons):
    """Shared kinematic response: mean bottom slip vs top drag (FEM + every Newton solver) -> Figure."""
    fig = plt.figure(figsize=(6, 5))
    if fem is not None:
        plt.plot(fem["drag"] * 1000.0, fem["mean_slip"] * 1000.0, "o-", color=style.COLOR["fem"],
                 label="FEM mean bottom slip")
    for label, d, color, *rest in newtons:
        marker = rest[0] if rest else "o"
        plt.plot(d["drag"] * 1000.0, d["bottom_slip"] * 1000.0, color=color, marker=marker,
                 ls="-", label=f"{label} bottom slip")
    plt.plot([0, params.FRICTION_DRAG_MAX * 1000.0], [0, params.FRICTION_DRAG_MAX * 1000.0],
             color=style.COLOR["analytic"], ls=style.REF_LS, lw=1, label="full slip (slip = drag)")
    plt.xlabel("top drag [mm]"); plt.ylabel("mean bottom slip [mm]")
    plt.title("Friction: bottom slip vs drag (stick below, slip above the knee)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    return fig


def main():
    matplotlib.use("Agg")
    os.makedirs(params.FIG_DIR, exist_ok=True)
    fem = np.load(params.FEM_FRICTION_NPZ) if os.path.exists(params.FEM_FRICTION_NPZ) else None
    newtons = style.load_newton_runs(params.NEWTON_FRICTION_NPZ)
    if fem is None and not newtons:
        raise FileNotFoundError("run fenics_run.run_friction and/or newton_run.run_friction first")

    figures = []
    if fem is not None:
        figures += [(make_force(fem), "friction_force.png"), (make_work(fem), "friction_work.png")]
    figures.append((make_slip(fem, newtons), "friction_slip.png"))
    for fig, name in figures:
        out = os.path.join(params.FIG_DIR, name)
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"[friction] wrote {out}")

    if fem is not None:
        print(f"[friction] FEM peak friction force = {fem['friction_force'].max():.2f} N "
              f"vs analytic mu*W = {float(fem['plateau']):.2f} N "
              f"(N = {fem['normal_force'][-1]:.2f} N vs W = {float(fem['weight']):.2f} N)")
    for label, _d, *_ in newtons:
        print(f"[friction] {label}: bottom slip reported; no calibrated friction force (by design).")


if __name__ == "__main__":
    main()
