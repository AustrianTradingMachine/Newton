"""Friction -- plots for the sliding-block test, FEM vs Newton solvers.

Reads data/fem_friction.npz and whichever Newton solver runs are present
(data/newton_friction{,_vbd,_semi}.npz -- run_friction --solver writes one each; any
may be missing) and writes:

  * friction force vs top drag (FEM) with the analytic Coulomb plateau mu*W and
    the normal force N (~ weight) -- the stick-then-slip curve,
  * cumulative frictional work (FEM),
  * the kinematic response solvers share: mean bottom slip vs top drag (XPBD exposes
    the slip but no calibrated friction force; the implicit VBD is the apples-to-apples
    counterpart to the implicit FEM).

Caveat: whether the VBD/SemiImplicit ground-contact runs exist depends on a recent
Newton (TODO[verify-on-colab]); if only XPBD is present, only it is overlaid.

Run:  python -m compare.friction
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

from common import params

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Newton solver runs to overlay if their npz is present:  label -> (solver key, colour, marker)
NEWTON_RUNS = (
    ("Newton XPBD", "xpbd", "tab:orange", "s"),
    ("Newton VBD", "vbd", "tab:red", "D"),
    ("Newton explicit", "semi_implicit", "tab:purple", "^"),
)


def _load_newtons():
    """Whichever Newton friction solver runs exist on disk: [(label, data, colour, marker)]."""
    out = []
    for label, solver, color, marker in NEWTON_RUNS:
        path = params.solver_npz(params.NEWTON_FRICTION_NPZ, solver)
        if os.path.exists(path):
            out.append((label, np.load(path), color, marker))
    return out


def main():
    os.makedirs(params.FIG_DIR, exist_ok=True)
    fem = np.load(params.FEM_FRICTION_NPZ) if os.path.exists(params.FEM_FRICTION_NPZ) else None
    newtons = _load_newtons()
    if fem is None and not newtons:
        raise FileNotFoundError("run fenics_run.run_friction and/or newton_run.run_friction first")

    # --- friction force + Coulomb plateau (FEM) --------------------------
    if fem is not None:
        d = fem["drag"] * 1000.0
        plt.figure(figsize=(6, 5))
        plt.plot(d, fem["friction_force"], "o-", color="tab:blue", label="FEM friction force")
        plt.plot(d, fem["normal_force"], "^-", color="tab:green", alpha=0.7, label="FEM normal force N")
        plt.axhline(float(fem["plateau"]), color="k", ls="--", lw=1.5, label="analytic mu*W")
        plt.axhline(float(fem["weight"]), color="grey", ls=":", lw=1.2, label="weight W")
        plt.xlabel("top drag [mm]"); plt.ylabel("force [N]")
        plt.title("Friction: stick -> slip plateau (FEM vs Coulomb mu*W)")
        plt.legend(); plt.grid(alpha=0.3)
        out = os.path.join(params.FIG_DIR, "friction_force.png")
        plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
        print(f"[friction] wrote {out}")

        # frictional work + slip fraction
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(d, fem["friction_work"], "o-", color="tab:red", label="frictional work")
        ax1.set_xlabel("top drag [mm]"); ax1.set_ylabel("cumulative friction work [J]", color="tab:red")
        ax2 = ax1.twinx()
        ax2.plot(d, fem["slip_fraction"], "s--", color="tab:purple", alpha=0.7)
        ax2.set_ylabel("slipping area fraction", color="tab:purple"); ax2.set_ylim(-0.05, 1.05)
        plt.title("Friction: dissipated work & stick/slip transition (FEM)")
        out = os.path.join(params.FIG_DIR, "friction_work.png")
        fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
        print(f"[friction] wrote {out}")

    # --- shared kinematic response: bottom slip vs drag ------------------
    plt.figure(figsize=(6, 5))
    if fem is not None:
        plt.plot(fem["drag"] * 1000.0, fem["mean_slip"] * 1000.0, "o-", color="tab:blue", label="FEM mean bottom slip")
    for label, d, color, marker in newtons:
        plt.plot(d["drag"] * 1000.0, d["bottom_slip"] * 1000.0, color=color, marker=marker,
                 ls="-", label=f"{label} bottom slip")
    plt.plot([0, params.FRICTION_DRAG_MAX * 1000.0], [0, params.FRICTION_DRAG_MAX * 1000.0],
             "k:", lw=1, label="full slip (slip = drag)")
    plt.xlabel("top drag [mm]"); plt.ylabel("mean bottom slip [mm]")
    plt.title("Friction: bottom slip vs drag (stick below, slip above the knee)")
    plt.legend(); plt.grid(alpha=0.3)
    out = os.path.join(params.FIG_DIR, "friction_slip.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
    print(f"[friction] wrote {out}")

    if fem is not None:
        print(f"[friction] FEM peak friction force = {fem['friction_force'].max():.2f} N "
              f"vs analytic mu*W = {float(fem['plateau']):.2f} N "
              f"(N = {fem['normal_force'][-1]:.2f} N vs W = {float(fem['weight']):.2f} N)")
    for label, _d, _c, _m in newtons:
        print(f"[friction] {label}: bottom slip reported; no calibrated friction force (by design).")


if __name__ == "__main__":
    main()
