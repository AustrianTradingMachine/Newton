"""Hanging bar -- compare the Newton solvers against FEniCSx (FEM) and analytic theory.

Every Newton solver (XPBD, VBD, explicit/SemiImplicit) and the tet-FEM run share
node ordering -- they are built on the same mesh and the FEM solution is evaluated
at Newton's nodes -- so each Newton solver is compared to FEM(tet) NODE-FOR-NODE
(RMS / max / relative displacement error).

The hex-FEM run (optional, data/fem_result_hex.npz) uses an independent mesh, so
it cannot be matched node-for-node; it is shown on the displacement *profile* and
in the tip-displacement table to expose the tet-vs-hex element effect.

Whichever Newton runs are present on disk are picked up automatically. Figures and
a text report are written to figures/. Run from the repository root:

    python -m compare.hanging_bar
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

from common import params

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Newton solver runs to include if their npz is present:  label -> (path, colour)
NEWTON_RUNS = (
    ("Newton XPBD", params.NEWTON_NPZ, "tab:orange"),
    ("Newton VBD", params.NEWTON_VBD_NPZ, "tab:red"),
    ("Newton explicit", params.NEWTON_SEMI_NPZ, "tab:purple"),
)


def _load():
    fem = np.load(params.FEM_NPZ, allow_pickle=False)
    newtons = []
    for label, path, color in NEWTON_RUNS:
        if os.path.exists(path):
            d = np.load(path, allow_pickle=False)
            assert d["rest_q"].shape == fem["rest_q"].shape, f"mesh mismatch ({label} vs tet-FEM)"
            newtons.append((label, d, color))
    if not newtons:
        raise FileNotFoundError("no Newton hanging-bar result found -- run newton_run.run_hanging_bar first")
    hexfem = np.load(params.FEM_HEX_NPZ, allow_pickle=False) if os.path.exists(params.FEM_HEX_NPZ) else None
    return newtons, fem, hexfem


def displacement(d):
    return d["final_q"] - d["rest_q"]


def _tip_drop(d):
    """Max downward (-z) displacement over the free nodes, in mm."""
    free = np.setdiff1d(np.arange(len(d["rest_q"])), d["fixed_nodes"])
    return -displacement(d)[free, 2].min() * 1000.0


def _node_error(d, fem, free):
    """Node-for-node displacement error of a Newton run vs tet-FEM: (RMS, max [mm], mean rel [%])."""
    err = np.linalg.norm(displacement(d) - displacement(fem), axis=1)
    ref = np.linalg.norm(displacement(fem)[free], axis=1)
    return (float(np.sqrt(np.mean(err[free] ** 2)) * 1000.0),
            float(err[free].max() * 1000.0),
            float(np.mean(err[free] / np.maximum(ref, 1e-9)) * 100.0))


def report(newtons, fem, hexfem):
    rest = fem["rest_q"]
    free = np.setdiff1d(np.arange(len(rest)), fem["fixed_nodes"])
    z_top = rest[fem["fixed_nodes"], 2].mean()
    tip_fem = _tip_drop(fem)
    tip_analytic = params.analytic_hanging_displacement(
        rest[:, 2].min(), z_top, params.BLOCK_LZ) * 1000.0

    lines = [
        "============== Hanging bar: Newton solvers vs FEniCSx (FEM) ==============",
        params.summary().rstrip(),
        "-" * 72,
        "Tip vertical displacement (downward) [mm]:",
    ]
    for label, d, _ in newtons:
        lines.append(f"    {label:16s}: {_tip_drop(d):8.2f}")
    lines.append(f"    {'FEM tet':16s}: {tip_fem:8.2f}   (node-for-node reference)")
    if hexfem is not None:
        lines.append(f"    {'FEM hex':16s}: {_tip_drop(hexfem):8.2f}   (independent mesh)")
    lines.append(f"    {'analytic 1-D':16s}: {tip_analytic:8.2f}   (self-weight bar)")
    lines += [
        "-" * 72,
        "Node-for-node error vs FEM(tet):   RMS / max [mm]   mean rel [%]   tip ratio",
    ]
    for label, d, _ in newtons:
        rms, emax, rel = _node_error(d, fem, free)
        lines.append(f"    {label:16s}: {rms:7.3f} / {emax:7.3f}      {rel:7.2f}      {_tip_drop(d) / tip_fem:6.3f}")
    lines.append("=" * 72)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(params.FIG_DIR, "hanging_bar_report.txt"), "w") as fh:
        fh.write(text + "\n")


def plot_profile(newtons, fem, hexfem):
    """Downward displacement vs original height z, all solvers/meshes overlaid."""
    rest = fem["rest_q"]
    z_top = rest[fem["fixed_nodes"], 2].mean()
    za = np.linspace(rest[:, 2].min(), rest[:, 2].max(), 100)
    ua = params.analytic_hanging_displacement(za, z_top, params.BLOCK_LZ) * 1000.0

    plt.figure(figsize=(6, 5))
    for label, d, color in newtons:
        plt.scatter(d["rest_q"][:, 2], -displacement(d)[:, 2] * 1000.0,
                    s=8, alpha=0.4, label=label, color=color)
    plt.scatter(fem["rest_q"][:, 2], -displacement(fem)[:, 2] * 1000.0,
                s=8, alpha=0.4, label="FEM tet", color="tab:blue")
    if hexfem is not None:
        plt.scatter(hexfem["rest_q"][:, 2], -displacement(hexfem)[:, 2] * 1000.0,
                    s=8, alpha=0.4, label="FEM hex", color="tab:green")
    plt.plot(za, ua, "k--", lw=1.5, label="analytic 1-D bar")
    plt.xlabel("original height z  [m]")
    plt.ylabel("downward displacement  [mm]")
    plt.title("Hanging bar: displacement profile")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(params.FIG_DIR, "hanging_bar_profile.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"[compare] wrote {out}")


def plot_settling(newtons):
    """Settling history (tip height + kinetic energy) of the canonical XPBD run."""
    d = next((d for label, d, _ in newtons if label == "Newton XPBD"), newtons[0][1])
    hist = d["history"]
    if hist.size == 0:
        return
    t, tip_z, ke = hist[:, 0], hist[:, 1], hist[:, 2]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(t, tip_z, color="tab:orange")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("tip height z [m]", color="tab:orange")
    ax2 = ax1.twinx()
    ax2.semilogy(t, np.maximum(ke, 1e-12), color="tab:green", alpha=0.6)
    ax2.set_ylabel("kinetic energy [J]", color="tab:green")
    plt.title("Hanging bar: Newton settling (XPBD)")
    out = os.path.join(params.FIG_DIR, "hanging_bar_settling.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[compare] wrote {out}")


def plot_error_hist(newtons, fem):
    """Per-node displacement error of each Newton solver vs tet-FEM."""
    free = np.setdiff1d(np.arange(len(fem["rest_q"])), fem["fixed_nodes"])
    plt.figure(figsize=(6, 4))
    for label, d, color in newtons:
        err = np.linalg.norm(displacement(d) - displacement(fem), axis=1)[free] * 1000.0
        plt.hist(err, bins=40, alpha=0.6, label=f"{label} vs FEM tet", color=color)
    plt.xlabel("per-node |u_Newton - u_FEM(tet)|  [mm]")
    plt.ylabel("count")
    plt.title("Hanging bar: node-for-node error")
    plt.legend()
    out = os.path.join(params.FIG_DIR, "hanging_bar_error_hist.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"[compare] wrote {out}")


def main():
    os.makedirs(params.FIG_DIR, exist_ok=True)
    newtons, fem, hexfem = _load()
    report(newtons, fem, hexfem)
    plot_profile(newtons, fem, hexfem)
    plot_settling(newtons)
    plot_error_hist(newtons, fem)


if __name__ == "__main__":
    main()
