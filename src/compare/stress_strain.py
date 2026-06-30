"""Effective stress-strain (3) -- compare FEM, Newton and analytic Neo-Hookean.

Plots the axial 1st Piola stress vs. stretch for the confined uniaxial-strain test
and reports the deviation from the closed form, highlighting the large-strain end
(where the hanging bar's small-strain equivalence no longer holds).

Run:  python -m compare.stress_strain
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

from common import params
from compare import energies as en

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    os.makedirs(params.FIG_DIR, exist_ok=True)
    fem = np.load(params.FEM_STRESS_NPZ) if os.path.exists(params.FEM_STRESS_NPZ) else None
    nw = np.load(params.NEWTON_STRESS_NPZ) if os.path.exists(params.NEWTON_STRESS_NPZ) else None
    if fem is None and nw is None:
        raise FileNotFoundError("run fenics_run.run_stress_strain and/or newton_run.run_stress_strain first")

    lam = (fem if fem is not None else nw)["lambdas"]
    ana = en.neohookean_uniaxial_strain_stress(lam)

    plt.figure(figsize=(6, 5))
    plt.plot(lam, ana / 1e3, "k--", lw=1.5, label="analytic Neo-Hookean")
    if fem is not None:
        plt.plot(lam, fem["sigma_fem"] / 1e3, "o-", color="tab:blue", label="FEM")
    if nw is not None:
        plt.plot(lam, nw["sigma_newton"] / 1e3, "s-", color="tab:orange", label="Newton (SemiImplicit)")
    plt.axhline(0, color="grey", lw=0.6); plt.axvline(1, color="grey", lw=0.6)
    plt.xlabel("stretch  lambda"); plt.ylabel("axial 1st Piola stress  [kPa]")
    plt.title("Effective stress-strain (confined uniaxial)")
    plt.legend(); plt.grid(alpha=0.3)
    out = os.path.join(params.FIG_DIR, "stress_strain.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
    print(f"[stress] wrote {out}")

    def report(name, sig):
        rel = np.abs(sig - ana) / (np.abs(ana) + 1.0)
        print(f"[stress] {name}: max rel. deviation vs analytic = {rel.max() * 100:.2f}% "
              f"(at lambda={lam[np.argmax(rel)]:.2f}); at lambda={lam[-1]:.2f}: "
              f"{sig[-1] / 1e3:.3g} vs {ana[-1] / 1e3:.3g} kPa")

    if fem is not None:
        report("FEM   ", fem["sigma_fem"])
    if nw is not None:
        report("Newton", nw["sigma_newton"])


if __name__ == "__main__":
    main()
