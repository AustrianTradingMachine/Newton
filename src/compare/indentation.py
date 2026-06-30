"""Indentation comparison -- FEniCSx (FEM) penalty contact vs Newton solvers.

Overlays the deformed top-surface dimple at maximum indentation for every FEM
variant (the tet penalty sweep + hex) and whichever Newton solver runs are present
on disk (XPBD / VBD / SemiImplicit -- run_indentation --solver writes one npz each).
Also plots each Newton solver's penetration vs. indentation.

The key point: the FEM penalty run yields a calibrated contact-force curve
(indentation_force.png), while the fast positional XPBD enforces contact positionally
and exposes no comparable force -- so here the solvers are compared on DEFORMATION,
which is the quantity all expose cleanly. The implicit VBD is the apples-to-apples
counterpart to the implicit FEM.

Caveat: whether the VBD/SemiImplicit contact runs exist depends on a recent Newton
(TODO[verify-on-colab]); if only XPBD is present, only it is overlaid. Part of the
residual Newton-vs-FEM dimple gap is constitutive (Newton StVK vs FEM Neo-Hookean),
growing outside small strain -- it is not a pure solver gap.

The make_* helpers build and return a Figure (no save/show), so the notebook (20_contact)
imports the SAME functions and renders inline; main() sets Agg and saves the PNGs. Colours
come from compare.style so every solver/variant matches across all contact plots.

Run from the repository root (after the indentation FEM and/or Newton have produced npz):

    python -m compare.indentation
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from common import params
from compare import style

# Backend is NOT forced at import (20_contact imports the make_* helpers and renders inline);
# main() sets Agg before saving.


def _fem_variants(fem):
    """Return [(slug, uz_array)] for every FEM variant stored in the npz (canonical order)."""
    return [(k[3:], fem[k]) for k in fem.files if k.startswith("uz_")]


def make_dimple(fem, newtons):
    """Deformed top-surface dimple: every FEM variant + every present Newton solver -> Figure."""
    fig = plt.figure(figsize=(6, 5))
    fem_x = fem["line_x"] - float(fem["cx"])
    for i, (slug, uz) in enumerate(_fem_variants(fem)):
        plt.plot(fem_x * 1000, uz * 1000, color=style.fem_variant_color(i), label=f"FEM {slug}")
    for label, d, color, *rest in newtons:
        marker = rest[0] if rest else "o"
        nx = d["line_x"] - float(d["cx"])
        plt.plot(nx * 1000, d["uz_line"] * 1000, color=color, marker=marker, ms=3, lw=1.6, label=label)
    plt.xlabel("x - x_centre  [mm]")
    plt.ylabel("vertical displacement u_z  [mm]")
    plt.title("Indentation: deformed dimple -- FEM vs Newton solvers")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def make_newton_penetration(newtons):
    """Each present Newton solver's penetration vs indentation depth -> Figure (None if none)."""
    if not newtons:
        return None
    fig = plt.figure(figsize=(6, 4))
    for label, d, color, *rest in newtons:
        marker = rest[0] if rest else "o"
        plt.plot(d["deltas"] * 1000, d["penetration"] * 1000, color=color, marker=marker, ms=3, label=label)
    plt.xlabel("indentation depth  [mm]")
    plt.ylabel("max sphere/body penetration  [mm]")
    plt.title("Indentation: Newton penetration vs. indentation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    matplotlib.use("Agg")   # headless for the pipeline; 20_contact imports the make_* helpers
    os.makedirs(params.FIG_DIR, exist_ok=True)

    if not os.path.exists(params.FEM_INDENT_NPZ):
        raise FileNotFoundError(
            f"{params.FEM_INDENT_NPZ} missing -- run `python -m fenics_run.run_indentation` first")
    fem = np.load(params.FEM_INDENT_NPZ, allow_pickle=False)
    newtons = style.load_newton_runs(params.NEWTON_INDENT_NPZ)

    for fig, name in (
        (make_dimple(fem, newtons), "indentation_compare_profile.png"),
        (make_newton_penetration(newtons), "indentation_newton_penetration.png"),
    ):
        if fig is None:
            continue
        out = os.path.join(params.FIG_DIR, name)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"[compare-indent] wrote {out}")

    if newtons:
        for label, d, *_ in newtons:
            print(f"[compare-indent] {label} max penetration at full indentation = "
                  f"{d['penetration'][-1] * 1000:.2f} mm")
    else:
        print("[compare-indent] no Newton result (data/newton_indentation*.npz) -- FEM-only dimple")


if __name__ == "__main__":
    main()
