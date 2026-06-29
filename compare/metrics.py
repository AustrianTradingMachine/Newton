"""Stage A -- compare Newton (XPBD) against FEniCSx (FEM) and analytic theory.

Newton and the tet-FEM run share node ordering (same mesh, FEM evaluated at
Newton's nodes), so they are compared NODE-FOR-NODE (RMS / max / relative error).

The hex-FEM run (optional, data/fem_result_hex.npz) uses an independent mesh, so
it cannot be compared node-for-node; it is added to the displacement *profile*
and the tip-displacement table to show the tet-vs-hex element effect.

Figures are written to figures/. Run from the repository root:

    python -m compare.metrics
"""

from __future__ import annotations

import os

import numpy as np

from common import params

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load():
    newton = np.load(params.NEWTON_NPZ, allow_pickle=False)
    fem = np.load(params.FEM_NPZ, allow_pickle=False)
    assert newton["rest_q"].shape == fem["rest_q"].shape, "mesh mismatch (Newton vs tet-FEM)"
    hexfem = np.load(params.FEM_HEX_NPZ, allow_pickle=False) if os.path.exists(params.FEM_HEX_NPZ) else None
    return newton, fem, hexfem


def displacement(d):
    return d["final_q"] - d["rest_q"]


def _tip_drop(d):
    """Max downward (−z) displacement over the free nodes, in mm."""
    free = np.setdiff1d(np.arange(len(d["rest_q"])), d["fixed_nodes"])
    return -displacement(d)[free, 2].min() * 1000.0


def report(newton, fem, hexfem):
    rest = newton["rest_q"]
    free = np.setdiff1d(np.arange(len(rest)), newton["fixed_nodes"])

    un = displacement(newton)
    uf = displacement(fem)

    tip_newton = _tip_drop(newton)
    tip_fem = _tip_drop(fem)
    z_top = rest[newton["fixed_nodes"], 2].mean()
    tip_analytic = params.analytic_hanging_displacement(
        rest[:, 2].min(), z_top, params.BLOCK_LZ) * 1000.0

    # node-for-node error Newton vs tet-FEM (the matched reference)
    err = np.linalg.norm(un - uf, axis=1)
    ref_scale = np.linalg.norm(uf[free], axis=1)
    rms = float(np.sqrt(np.mean(err[free] ** 2)) * 1000.0)
    emax = float(err[free].max() * 1000.0)
    rel = float(np.mean(err[free] / np.maximum(ref_scale, 1e-9)) * 100.0)

    lines = [
        "============== Stage A: Newton (XPBD) vs FEniCSx (FEM) ==============",
        params.summary().rstrip(),
        "-" * 70,
        f"Tip vertical displacement   Newton    : {tip_newton:8.2f} mm",
        f"Tip vertical displacement   FEM tet    : {tip_fem:8.2f} mm  (node-for-node ref)",
    ]
    if hexfem is not None:
        lines.append(f"Tip vertical displacement   FEM hex    : {_tip_drop(hexfem):8.2f} mm  (independent mesh)")
    lines += [
        f"Tip vertical displacement   analytic   : {tip_analytic:8.2f} mm  (1-D bar)",
        "-" * 70,
        f"Newton-vs-FEM(tet) full-field error RMS: {rms:8.3f} mm",
        f"Newton-vs-FEM(tet) full-field error max: {emax:8.3f} mm",
        f"Newton-vs-FEM(tet) mean relative error : {rel:8.2f} %",
        f"Newton/FEM(tet) tip ratio              : {tip_newton / tip_fem:8.3f}",
        "=" * 70,
    ]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(params.FIG_DIR, "stage_a_report.txt"), "w") as fh:
        fh.write(text + "\n")


def plot_profile(newton, fem, hexfem):
    """Downward displacement vs original height z, all meshes overlaid."""
    rest = newton["rest_q"]
    z_top = rest[newton["fixed_nodes"], 2].mean()
    za = np.linspace(rest[:, 2].min(), rest[:, 2].max(), 100)
    ua = params.analytic_hanging_displacement(za, z_top, params.BLOCK_LZ) * 1000.0

    plt.figure(figsize=(6, 5))
    plt.scatter(newton["rest_q"][:, 2], -displacement(newton)[:, 2] * 1000.0,
                s=8, alpha=0.4, label="Newton XPBD", color="tab:orange")
    plt.scatter(fem["rest_q"][:, 2], -displacement(fem)[:, 2] * 1000.0,
                s=8, alpha=0.4, label="FEM tet", color="tab:blue")
    if hexfem is not None:
        plt.scatter(hexfem["rest_q"][:, 2], -displacement(hexfem)[:, 2] * 1000.0,
                    s=8, alpha=0.4, label="FEM hex", color="tab:green")
    plt.plot(za, ua, "k--", lw=1.5, label="analytic 1-D bar")
    plt.xlabel("original height z  [m]")
    plt.ylabel("downward displacement  [mm]")
    plt.title("Stage A: displacement profile")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(params.FIG_DIR, "stage_a_profile.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"[compare] wrote {out}")


def plot_settling(newton):
    hist = newton["history"]
    if hist.size == 0:
        return
    t, tip_z, ke = hist[:, 0], hist[:, 1], hist[:, 2]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(t, tip_z, color="tab:orange", label="tip z")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("tip height z [m]", color="tab:orange")
    ax2 = ax1.twinx()
    ax2.semilogy(t, np.maximum(ke, 1e-12), color="tab:green", alpha=0.6)
    ax2.set_ylabel("kinetic energy [J]", color="tab:green")
    plt.title("Stage A: Newton settling")
    out = os.path.join(params.FIG_DIR, "stage_a_settling.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[compare] wrote {out}")


def plot_error_hist(newton, fem):
    rest = newton["rest_q"]
    free = np.setdiff1d(np.arange(len(rest)), newton["fixed_nodes"])
    err = np.linalg.norm(displacement(newton) - displacement(fem), axis=1)[free] * 1000.0
    plt.figure(figsize=(6, 4))
    plt.hist(err, bins=40, color="tab:purple", alpha=0.8)
    plt.xlabel("|u_Newton - u_FEM(tet)| per node  [mm]")
    plt.ylabel("count")
    plt.title("Stage A: Newton-vs-FEM(tet) node error")
    out = os.path.join(params.FIG_DIR, "stage_a_error_hist.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"[compare] wrote {out}")


def main():
    os.makedirs(params.FIG_DIR, exist_ok=True)
    newton, fem, hexfem = _load()
    report(newton, fem, hexfem)
    plot_profile(newton, fem, hexfem)
    plot_settling(newton)
    plot_error_hist(newton, fem)


if __name__ == "__main__":
    main()
