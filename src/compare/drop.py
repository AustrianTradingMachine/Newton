"""Dynamic drop comparison -- Newton solvers vs FEM (Newmark).

Overlays the transient time series of the literal rigid_soft_contact scenario:
sphere height, penetration, block strain & kinetic energy (and the FEM contact
force). This contrasts Newton's solvers against implicit FEM (Newmark) elastodynamics +
contact, picking up whichever Newton solver runs are present
(data/newton_drop{,_vbd,_semi}.npz). The implicit VBD is the natural counterpart to the
implicit Newmark FEM -- but it is the hardest contact case (a *free* rigid sphere VBD must
integrate two-way; TODO[verify-on-colab]), so its run may be absent on an older Newton.

CAVEAT: even with an implicit (VBD) curve, the transient gap is NOT solver-only -- the
sides also differ in material (Newton StVK/co-rotational vs FEM Neo-Hookean), contact
model (soft_contact penalty + in-solver free rigid body vs analytic-obstacle penalty +
Kelvin-Voigt + a staggered sphere ODE) and time integration, and the material difference
grows once impact strains leave the small-strain regime. So VBD-vs-Newmark is a *partial*
fairness fix, not a clean solver-only comparison.

History columns:
  Newton: t, sphere_z, penetration, U_strain, KE
  FEM:    t, sphere_z, penetration, U_strain, KE, contact_force

The make_* helpers build and return a Figure (no save/show), so the notebook (25_dynamic)
imports the SAME functions and renders inline; main() sets Agg and saves the PNGs.

Run:  python -m compare.drop
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from common import params
from compare import style

# Backend is NOT forced at import (so 25_dynamic can import the make_* helpers and render
# inline); main() sets Agg before saving.


def make_scene(nd, label):
    """3-D scene of the deepest impact for one Newton run -> Figure (None if no mesh saved)."""
    if "tet_indices" not in nd.files:
        return None
    from compare import scene

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    norm, lab = scene.render(ax, nd["rest_q"], nd["final_q"], nd["tet_indices"],
                             ghost_rest=False, title=f"{label} - sphere drop (deepest impact)")
    scene.add_sphere(ax, nd["sphere_c"], float(nd["sphere_r"]))
    scene.add_colorbar(fig, ax, norm, lab)
    ax.view_init(elev=16, azim=-60)
    fig.tight_layout()
    return fig


def make_series(newtons, fem_hist, col, ylabel, title, scale=1.0):
    """Overlay one history column for every present Newton solver (+ FEM Newmark) -> Figure."""
    fig = plt.figure(figsize=(6, 4))
    for label, nd, color, *_ in newtons:
        hh = nd["history"]
        plt.plot(hh[:, 0], hh[:, col] * scale, label=label, color=color)
    if fem_hist is not None:
        plt.plot(fem_hist[:, 0], fem_hist[:, col] * scale,
                 label=f"{style.LABEL['fem']} Newmark", color=style.COLOR["fem"])
    plt.xlabel("time [s]"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    return fig


def make_contact_force(fem_hist):
    """FEM contact force over time -> Figure (None if no FEM force column present)."""
    if fem_hist is None or fem_hist.shape[1] <= 5:
        return None
    fig = plt.figure(figsize=(6, 4))
    plt.plot(fem_hist[:, 0], fem_hist[:, 5], color=style.COLOR["fem"],
             label=f"{style.LABEL['fem']} Newmark")
    plt.xlabel("time [s]"); plt.ylabel("contact force [N]")
    plt.title("Drop: contact force (FEM only -- Newton exposes none)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    return fig


def main():
    matplotlib.use("Agg")   # headless for the pipeline; 25_dynamic imports the make_* helpers
    os.makedirs(params.FIG_DIR, exist_ok=True)
    newtons = style.load_newton_runs(params.NEWTON_DROP_NPZ)
    fe = np.load(params.FEM_DROP_NPZ)["history"] if os.path.exists(params.FEM_DROP_NPZ) else None
    if not newtons and fe is None:
        raise FileNotFoundError("run newton_run.run_drop and/or fenics_run.run_drop first")

    figures = []
    if newtons:
        figures.append((make_scene(newtons[0][1], newtons[0][0]), "drop_scene.png"))
    figures += [
        (make_series(newtons, fe, 1, "sphere centre height [m]",
                     "Drop: sphere trajectory (impact & rebound)"), "drop_sphere_z.png"),
        (make_series(newtons, fe, 2, "max penetration [mm]",
                     "Drop: sphere/block penetration", scale=1000.0), "drop_penetration.png"),
        (make_series(newtons, fe, 3, "block strain energy [J]",
                     "Drop: block internal energy"), "drop_strain_energy.png"),
        (make_series(newtons, fe, 4, "block kinetic energy [J]",
                     "Drop: block kinetic energy"), "drop_kinetic_energy.png"),
        (make_contact_force(fe), "drop_contact_force.png"),
    ]
    for fig, name in figures:
        if fig is None:
            continue
        out = os.path.join(params.FIG_DIR, name)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"[drop] wrote {out}")

    # short numeric summary
    summaries = [(label, nd["history"]) for label, nd, *_ in newtons]
    if fe is not None:
        summaries.append(("FEM Newmark", fe))
    for name, hh in summaries:
        print(f"[drop] {name}: min sphere_z={hh[:, 1].min():.3f} m, "
              f"max penetration={hh[:, 2].max() * 1000:.2f} mm, "
              f"peak strain energy={hh[:, 3].max():.4g} J")


if __name__ == "__main__":
    main()
