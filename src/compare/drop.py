"""Dynamic drop comparison -- Newton (XPBD) vs FEM (Newmark).

Overlays the transient time series of the literal rigid_soft_contact scenario:
sphere height, penetration, block strain & kinetic energy (and the FEM contact
force). This contrasts Newton's fast positional XPBD against implicit FEM (Newmark)
elastodynamics + contact. CAVEAT: the transient gap is NOT solver-only -- the two sides
also differ in material (Newton StVK/co-rotational vs FEM Neo-Hookean), contact model
(soft_contact penalty + in-solver free rigid body vs analytic-obstacle penalty +
Kelvin-Voigt + a staggered sphere ODE) and time integration, and the material difference
grows once impact strains leave the small-strain regime. The implicit VBD (the natural
counterpart to the implicit Newmark FEM) is unavailable for this rigid free-body scene
(XPBD-only; see newton_run/run_drop).

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


def _load(path):
    return np.load(path)["history"] if os.path.exists(path) else None


def main():
    os.makedirs(params.FIG_DIR, exist_ok=True)
    nw = _load(params.NEWTON_DROP_NPZ)
    fe = _load(params.FEM_DROP_NPZ)
    if nw is None and fe is None:
        raise FileNotFoundError("run newton_run.run_drop and/or fenics_run.run_drop first")

    # 3D scene of the deepest impact, if the Newton run saved the deformed mesh
    nd = np.load(params.NEWTON_DROP_NPZ) if os.path.exists(params.NEWTON_DROP_NPZ) else None
    if nd is not None and "tet_indices" in nd.files:
        from compare import scene
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        norm, lab = scene.render(ax, nd["rest_q"], nd["final_q"], nd["tet_indices"],
                                 ghost_rest=False, title="Newton XPBD - sphere drop (deepest impact)")
        scene.add_sphere(ax, nd["sphere_c"], float(nd["sphere_r"]))
        scene.add_colorbar(fig, ax, norm, lab)
        ax.view_init(elev=16, azim=-60)
        out = os.path.join(params.FIG_DIR, "drop_scene.png")
        plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
        print(f"[drop] wrote {out}")

    def plot(col, ylabel, title, fname, scale=1.0):
        plt.figure(figsize=(6, 4))
        if nw is not None:
            plt.plot(nw[:, 0], nw[:, col] * scale, label="Newton XPBD", color="tab:orange")
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
    for name, hh in (("Newton", nw), ("FEM", fe)):
        if hh is not None:
            print(f"[drop] {name}: min sphere_z={hh[:, 1].min():.3f} m, "
                  f"max penetration={hh[:, 2].max() * 1000:.2f} mm, "
                  f"peak strain energy={hh[:, 3].max():.4g} J")


if __name__ == "__main__":
    main()
