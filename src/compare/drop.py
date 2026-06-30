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

Run:  python -m compare.drop
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

from common import params

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Newton solver runs to overlay if their npz is present:  label -> (solver key, colour)
NEWTON_RUNS = (
    ("Newton XPBD", "xpbd", "tab:orange"),
    ("Newton VBD", "vbd", "tab:red"),
    ("Newton explicit", "semi_implicit", "tab:purple"),
)


def _load(path):
    return np.load(path)["history"] if os.path.exists(path) else None


def _load_newtons():
    """[(label, npz, colour)] for whichever drop solver runs exist (history + scene fields)."""
    out = []
    for label, solver, color in NEWTON_RUNS:
        path = params.solver_npz(params.NEWTON_DROP_NPZ, solver)
        if os.path.exists(path):
            out.append((label, np.load(path), color))
    return out


def main():
    os.makedirs(params.FIG_DIR, exist_ok=True)
    newtons = _load_newtons()
    fe = _load(params.FEM_DROP_NPZ)
    if not newtons and fe is None:
        raise FileNotFoundError("run newton_run.run_drop and/or fenics_run.run_drop first")

    # 3D scene of the deepest impact, from the canonical (first present) Newton run
    if newtons:
        scene_label, nd, _ = newtons[0]
        if "tet_indices" in nd.files:
            from compare import scene
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d")
            norm, lab = scene.render(ax, nd["rest_q"], nd["final_q"], nd["tet_indices"],
                                     ghost_rest=False,
                                     title=f"{scene_label} - sphere drop (deepest impact)")
            scene.add_sphere(ax, nd["sphere_c"], float(nd["sphere_r"]))
            scene.add_colorbar(fig, ax, norm, lab)
            ax.view_init(elev=16, azim=-60)
            out = os.path.join(params.FIG_DIR, "drop_scene.png")
            plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
            print(f"[drop] wrote {out}")

    def plot(col, ylabel, title, fname, scale=1.0):
        plt.figure(figsize=(6, 4))
        for label, nd, color in newtons:
            hh = nd["history"]
            plt.plot(hh[:, 0], hh[:, col] * scale, label=label, color=color)
        if fe is not None:
            plt.plot(fe[:, 0], fe[:, col] * scale, label="FEM Newmark", color="tab:blue")
        plt.xlabel("time [s]"); plt.ylabel(ylabel); plt.title(title)
        plt.legend(); plt.grid(alpha=0.3)
        out = os.path.join(params.FIG_DIR, fname)
        plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
        print(f"[drop] wrote {out}")

    plot(1, "sphere centre height [m]", "Drop: sphere trajectory (impact & rebound)", "drop_sphere_z.png")
    plot(2, "max penetration [mm]", "Drop: sphere/block penetration", "drop_penetration.png", scale=1000.0)
    plot(3, "block strain energy [J]", "Drop: block internal energy", "drop_strain_energy.png")
    plot(4, "block kinetic energy [J]", "Drop: block kinetic energy", "drop_kinetic_energy.png")

    if fe is not None and fe.shape[1] > 5:
        plt.figure(figsize=(6, 4))
        plt.plot(fe[:, 0], fe[:, 5], color="tab:blue")
        plt.xlabel("time [s]"); plt.ylabel("contact force [N]")
        plt.title("Drop: FEM contact force (XPBD exposes none)")
        plt.grid(alpha=0.3)
        out = os.path.join(params.FIG_DIR, "drop_contact_force.png")
        plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
        print(f"[drop] wrote {out}")

    # short numeric summary
    summaries = [(label, nd["history"]) for label, nd, _ in newtons]
    if fe is not None:
        summaries.append(("FEM Newmark", fe))
    for name, hh in summaries:
        print(f"[drop] {name}: min sphere_z={hh[:, 1].min():.3f} m, "
              f"max penetration={hh[:, 2].max() * 1000:.2f} mm, "
              f"peak strain energy={hh[:, 3].max():.4g} J")


if __name__ == "__main__":
    main()
