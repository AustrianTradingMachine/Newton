"""Indentation comparison -- FEniCSx (FEM) penalty contact vs Newton (XPBD).

Overlays the deformed top-surface dimple at maximum indentation for every FEM
variant (the tet penalty sweep + hex) and, if present, the Newton XPBD result.
Also plots Newton's penetration vs. indentation.

The key point: the FEM penalty run yields a calibrated contact-force curve
(indentation_force.png), while XPBD enforces contact positionally and exposes no
comparable force -- so here the solvers are compared on DEFORMATION, which is the
quantity both expose cleanly.

Run from the repository root (after the indentation FEM and/or Newton have produced npz):

    python -m compare.indentation
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

from common import params

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fem_variants(fem):
    """Return [(slug, uz_array)] for every FEM variant stored in the npz."""
    return [(k[3:], fem[k]) for k in fem.files if k.startswith("uz_")]


def main():
    os.makedirs(params.FIG_DIR, exist_ok=True)

    if not os.path.exists(params.FEM_INDENT_NPZ):
        raise FileNotFoundError(
            f"{params.FEM_INDENT_NPZ} missing -- run `python -m fenics_run.run_indentation` first")
    fem = np.load(params.FEM_INDENT_NPZ, allow_pickle=False)
    newton = (np.load(params.NEWTON_INDENT_NPZ, allow_pickle=False)
              if os.path.exists(params.NEWTON_INDENT_NPZ) else None)

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
    plt.title("Indentation: deformed dimple -- FEM vs Newton (XPBD)")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(params.FIG_DIR, "indentation_compare_profile.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"[compare-indent] wrote {out}")

    # ---- Newton penetration vs indentation -------------------------------
    if newton is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(newton["deltas"] * 1000, newton["penetration"] * 1000, "k-o", ms=3)
        plt.xlabel("indentation depth  [mm]")
        plt.ylabel("max sphere/body penetration  [mm]")
        plt.title("Indentation: Newton XPBD penetration vs. indentation")
        plt.grid(alpha=0.3)
        out = os.path.join(params.FIG_DIR, "indentation_newton_penetration.png")
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        print(f"[compare-indent] wrote {out}")
        print(f"[compare-indent] Newton max penetration at full indentation = "
              f"{newton['penetration'][-1] * 1000:.2f} mm")
    else:
        print("[compare-indent] no Newton result (data/newton_indentation.npz) -- FEM-only plot")


if __name__ == "__main__":
    main()
