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


def scene_run(newtons, fem_hist=None):
    """Pick the Newton run to render in 3-D: the deepest *genuine* impact.

    The deepest *raw* impact can be an unstable blow-up (it descends furthest precisely because
    it is exploding) -- e.g. the explicit drop sinks lowest but its energy diverges -- so we
    restrict to runs that genuinely impacted (penetrated the block and stayed bounded, per
    impact_table) and render the one whose sphere sits lowest. If none qualify, fall back to the
    lowest stored scene sphere so the panel still renders something."""
    genuine = {r["label"] for r in impact_table(newtons, fem_hist) if r["genuine"]}
    runs = [(float(nd["sphere_c"][2]), nd, label)
            for label, nd, *_ in newtons if "sphere_c" in nd.files and label in genuine]
    if not runs:   # nothing genuine -> render the lowest stored sphere so the figure is not blank
        runs = [(float(nd["sphere_c"][2]), nd, label)
                for label, nd, *_ in newtons if "sphere_c" in nd.files]
    if not runs:
        return (newtons[0][1], newtons[0][0]) if newtons else (None, None)
    _z, nd, label = min(runs, key=lambda r: r[0])
    return nd, label


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


def make_series(newtons, fem_hist, col, ylabel, title, scale=1.0, logy=False):
    """Overlay one history column for every present Newton solver (+ FEM Newmark) -> Figure.

    logy=True puts the y-axis on a log scale (impact spikes span orders of magnitude;
    non-positive samples are masked by matplotlib).
    """
    fig = plt.figure(figsize=(6, 4))
    for label, nd, color, *_ in newtons:
        hh = nd["history"]
        plt.plot(hh[:, 0], hh[:, col] * scale, label=label, color=color)
    if fem_hist is not None:
        plt.plot(fem_hist[:, 0], fem_hist[:, col] * scale,
                 label=f"{style.LABEL['fem']} Newmark", color=style.COLOR["fem"])
    if logy:
        plt.yscale("log")
    plt.xlabel("time [s]"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3, which="both"); plt.tight_layout()
    return fig


def drop_touch_z():
    """Sphere-centre height at first contact with the flat block top (z = Lz + R)."""
    return params.DROP_DIM[2] * params.DROP_CELL + params.DROP_SPHERE_R


def impact_table(newtons, fem_hist):
    """Per-run impact summary from the saved histories (pure post-processing, no re-sim).

    Penetration alone is misleading on the drop: a solver reads ~0 mm BOTH when it resolves
    contact well AND when the sphere never reaches the block, or when the run goes unstable.
    So we classify each run by direct evidence:

      * ``impacted`` -- the sphere actually overlapped block material (max penetration > 0);
      * ``stable``   -- the strain energy stayed bounded (within 5x the FEM peak, when an FEM
                        reference is present), i.e. the run did not blow up;
      * ``reached``  -- the sphere descended to the nominal contact height (z <= Lz + R); a
                        descriptive flag (a genuine impact can rest just above it, since the
                        block surface bulges up to meet the sphere).

    A penetration is a meaningful contact measurement only when ``genuine`` = impacted and
    stable. Returns a list of dicts: label, min_z, max_pen_mm, peak_U, impacted, stable,
    reached, genuine.
    """
    touch = drop_touch_z()
    fem_U = float(fem_hist[:, 3].max()) if fem_hist is not None else None
    runs = [(label, nd["history"]) for label, nd, *_ in newtons]
    if fem_hist is not None:
        runs.append((f"{style.LABEL['fem']} Newmark", fem_hist))
    rows = []
    for label, hh in runs:
        zmin = float(hh[:, 1].min())
        peak_U = float(hh[:, 3].max())
        pen_mm = float(hh[:, 2].max()) * 1000.0
        impacted = pen_mm > 1.0e-3
        stable = fem_U is None or peak_U <= 5.0 * fem_U
        rows.append(dict(label=label, min_z=zmin, max_pen_mm=pen_mm, peak_U=peak_U,
                         impacted=impacted, stable=stable, reached=zmin <= touch + 1.0e-9,
                         genuine=impacted and stable))
    return rows


def make_penetration(newtons, fem_hist):
    """Penetration vs time for the runs that genuinely impact -> Figure.

    Non-impacting (sphere never reached the block) or unstable (energy blow-up) runs are
    excluded from the curve -- their ~0 mm is an artefact, not good contact -- and reported
    by impact_table() instead. Linear axis: the meaningful penetrations are a few mm.
    """
    genuine = {r["label"] for r in impact_table(newtons, fem_hist) if r["genuine"]}
    fig = plt.figure(figsize=(6, 4))
    for label, nd, color, *_ in newtons:
        if label in genuine:
            hh = nd["history"]
            plt.plot(hh[:, 0], hh[:, 2] * 1000.0, label=label, color=color)
    flabel = f"{style.LABEL['fem']} Newmark"
    if fem_hist is not None and flabel in genuine:
        plt.plot(fem_hist[:, 0], fem_hist[:, 2] * 1000.0, label=flabel, color=style.COLOR["fem"])
    plt.xlabel("time [s]"); plt.ylabel("max penetration [mm]")
    plt.title("Drop: penetration where a genuine impact occurs")
    if genuine:
        plt.legend()
    plt.grid(alpha=0.3); plt.tight_layout()
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
        figures.append((make_scene(*scene_run(newtons, fe)), "drop_scene.png"))
    figures += [
        (make_series(newtons, fe, 1, "sphere centre height [m]",
                     "Drop: sphere trajectory (impact & rebound)"), "drop_sphere_z.png"),
        (make_penetration(newtons, fe), "drop_penetration.png"),
        (make_series(newtons, fe, 3, "block strain energy [J]",
                     "Drop: block internal energy", logy=True), "drop_strain_energy.png"),
        (make_series(newtons, fe, 4, "block kinetic energy [J]",
                     "Drop: block kinetic energy", logy=True), "drop_kinetic_energy.png"),
        (make_contact_force(fe), "drop_contact_force.png"),
    ]
    for fig, name in figures:
        if fig is None:
            continue
        out = os.path.join(params.FIG_DIR, name)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"[drop] wrote {out}")

    # impact summary: penetration is only meaningful where a genuine impact occurred
    print(f"[drop] first contact at sphere centre z <= Lz+R = {drop_touch_z():.3f} m "
          f"(drop from z0 = {params.DROP_SPHERE_Z0:.2f} m)")
    for r in impact_table(newtons, fe):
        if r["genuine"]:
            note = "genuine two-way impact"
        elif not r["reached"]:
            note = "sphere never reached the block -> pen=0 is trivial, not good contact"
        else:
            note = "unstable / energy blow-up -> penetration reading not meaningful"
        print(f"[drop] {r['label']:14s} min sphere_z={r['min_z']:.3f} m  "
              f"max penetration={r['max_pen_mm']:6.2f} mm  peak strain energy={r['peak_U']:.4g} J  [{note}]")


if __name__ == "__main__":
    main()
