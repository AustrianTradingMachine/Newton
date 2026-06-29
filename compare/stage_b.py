"""Stage B comparison -- FEniCSx (FEM) penalty contact vs Newton (XPBD).

Overlays the deformed top-surface dimple at maximum indentation for every FEM
variant (the tet penalty sweep + hex) and, if present, the Newton XPBD result.
Also plots Newton's penetration vs. indentation.

The key point: the FEM penalty run yields a calibrated contact-force curve
(stage_b_force.png), while XPBD enforces contact positionally and exposes no
comparable force -- so here the solvers are compared on DEFORMATION, which is the
quantity both expose cleanly.

Run from the repository root (after Stage B FEM and/or Newton have produced npz):

    python -m compare.stage_b
"""

from __future__ import annotations

import os

import numpy as np

from common import params

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fem_variants(fem):
    """Return [(slug, uz_array)] for every FEM variant stored in the npz."""
    return [(k[3:], fem[k]) for k in fem.files if k.startswith("uz_")]


def main():
    os.makedirs(params.FIG_DIR, exist_ok=True)

    if not os.path.exists(params.FEM_STAGEB_NPZ):
        raise FileNotFoundError(
            f"{params.FEM_STAGEB_NPZ} missing -- run `python -m fenics_run.run_stage_b` first")
    fem = np.load(params.FEM_STAGEB_NPZ, allow_pickle=False)
    newton = (np.load(params.NEWTON_STAGEB_NPZ, allow_pickle=False)
              if os.path.exists(params.NEWTON_STAGEB_NPZ) else None)

    # ---- dimple overlay --------------------------------------------------
    plt.figure(figsize=(6, 5))
    fem_x = fem["line_x"] - float(fem["cx"])
    for slug, uz in _fem_variants(fem):
        plt.plot(fem_x * 1000, uz * 1000, label=f"FEM {slug}")
    if newton is not None:
        nx = newton["line_x"] - float(newton["cx"])
        plt.plot(nx * 1000, newton["uz_line"] * 1000, "k-o", ms=3, lw=1.6,
                 label="Newton XPBD")
    plt.xlabel("x - x_centre  [mm]")
    plt.ylabel("vertical displacement u_z  [mm]")
    plt.title("Stage B: deformed dimple -- FEM vs Newton (XPBD)")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(params.FIG_DIR, "stage_b_compare_profile.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"[compare-B] wrote {out}")

    # ---- Newton penetration vs indentation -------------------------------
    if newton is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(newton["deltas"] * 1000, newton["penetration"] * 1000, "k-o", ms=3)
        plt.xlabel("indentation depth  [mm]")
        plt.ylabel("max sphere/body penetration  [mm]")
        plt.title("Stage B: Newton XPBD penetration vs. indentation")
        plt.grid(alpha=0.3)
        out = os.path.join(params.FIG_DIR, "stage_b_newton_penetration.png")
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        print(f"[compare-B] wrote {out}")
        print(f"[compare-B] Newton max penetration at full indentation = "
              f"{newton['penetration'][-1] * 1000:.2f} mm")
    else:
        print("[compare-B] no Newton result (data/newton_stage_b.npz) -- FEM-only plot")


if __name__ == "__main__":
    main()
