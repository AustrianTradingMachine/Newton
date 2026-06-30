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

# Newton solver runs to overlay if their npz is present:  label -> (solver key, colour, marker)
NEWTON_RUNS = (
    ("Newton XPBD", "xpbd", "tab:orange", "o"),
    ("Newton VBD", "vbd", "tab:red", "s"),
    ("Newton explicit", "semi_implicit", "tab:purple", "^"),
)


def _fem_variants(fem):
    """Return [(slug, uz_array)] for every FEM variant stored in the npz."""
    return [(k[3:], fem[k]) for k in fem.files if k.startswith("uz_")]


def _load_newtons():
    """Whichever Newton indentation solver runs exist on disk: [(label, data, colour, marker)]."""
    out = []
    for label, solver, color, marker in NEWTON_RUNS:
        path = params.solver_npz(params.NEWTON_INDENT_NPZ, solver)
        if os.path.exists(path):
            out.append((label, np.load(path, allow_pickle=False), color, marker))
    return out


def main():
    os.makedirs(params.FIG_DIR, exist_ok=True)

    if not os.path.exists(params.FEM_INDENT_NPZ):
        raise FileNotFoundError(
            f"{params.FEM_INDENT_NPZ} missing -- run `python -m fenics_run.run_indentation` first")
    fem = np.load(params.FEM_INDENT_NPZ, allow_pickle=False)
    newtons = _load_newtons()

    # ---- dimple overlay --------------------------------------------------
    plt.figure(figsize=(6, 5))
    fem_x = fem["line_x"] - float(fem["cx"])
    for slug, uz in _fem_variants(fem):
        plt.plot(fem_x * 1000, uz * 1000, label=f"FEM {slug}")
    for label, d, color, marker in newtons:
        nx = d["line_x"] - float(d["cx"])
        plt.plot(nx * 1000, d["uz_line"] * 1000, color=color, marker=marker, ms=3, lw=1.6,
                 label=label)
    plt.xlabel("x - x_centre  [mm]")
    plt.ylabel("vertical displacement u_z  [mm]")
    plt.title("Indentation: deformed dimple -- FEM vs Newton solvers")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(params.FIG_DIR, "indentation_compare_profile.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"[compare-indent] wrote {out}")

    # ---- Newton penetration vs indentation -------------------------------
    if newtons:
        plt.figure(figsize=(6, 4))
        for label, d, color, marker in newtons:
            plt.plot(d["deltas"] * 1000, d["penetration"] * 1000, color=color, marker=marker,
                     ms=3, label=label)
        plt.xlabel("indentation depth  [mm]")
        plt.ylabel("max sphere/body penetration  [mm]")
        plt.title("Indentation: Newton penetration vs. indentation")
        plt.legend()
        plt.grid(alpha=0.3)
        out = os.path.join(params.FIG_DIR, "indentation_newton_penetration.png")
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        print(f"[compare-indent] wrote {out}")
        for label, d, _, _ in newtons:
            print(f"[compare-indent] {label} max penetration at full indentation = "
                  f"{d['penetration'][-1] * 1000:.2f} mm")
    else:
        print("[compare-indent] no Newton result (data/newton_indentation*.npz) -- FEM-only plot")


if __name__ == "__main__":
    main()
