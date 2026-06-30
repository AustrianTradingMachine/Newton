"""Effective stress-strain (3) -- compare FEM, Newton and analytic Neo-Hookean.

Plots the axial 1st Piola stress vs. stretch for the confined uniaxial-strain test
and reports the deviation from the closed form, highlighting the large-strain end
(where the hanging bar's small-strain equivalence no longer holds).

make_stress_curve builds and returns the Figure (no save/show) using compare.style
colours; main() sets Agg and saves the PNG.

Run:  python -m compare.stress_strain
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from common import params
from compare import energies as en
from compare import style

# Backend not forced at import (import-safe); main() sets Agg before saving.


def make_stress_curve(fem, nw):
    """Axial stress vs stretch: analytic Neo-Hookean + FEM + Newton (SemiImplicit) -> Figure."""
    lam = (fem if fem is not None else nw)["lambdas"]
    ana = en.neohookean_uniaxial_strain_stress(lam)

    fig = plt.figure(figsize=(6, 5))
    plt.plot(lam, ana / 1e3, color=style.COLOR["analytic"], ls=style.ANALYTIC_LS, lw=1.5,
             label="analytic Neo-Hookean")
    if fem is not None:
        plt.plot(lam, fem["sigma_fem"] / 1e3, "o-", color=style.COLOR["fem"], label=style.LABEL["fem"])
    if nw is not None:
        plt.plot(lam, nw["sigma_newton"] / 1e3, "s-", color=style.COLOR["semi_implicit"],
                 label=style.LABEL["semi_implicit"])
    plt.axhline(0, color="0.7", lw=0.6); plt.axvline(1, color="0.7", lw=0.6)
    plt.xlabel("stretch  lambda"); plt.ylabel("axial 1st Piola stress  [kPa]")
    plt.title("Effective stress-strain (confined uniaxial)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    matplotlib.use("Agg")
    os.makedirs(params.FIG_DIR, exist_ok=True)
    fem = np.load(params.FEM_STRESS_NPZ) if os.path.exists(params.FEM_STRESS_NPZ) else None
    nw = np.load(params.NEWTON_STRESS_NPZ) if os.path.exists(params.NEWTON_STRESS_NPZ) else None
    if fem is None and nw is None:
        raise FileNotFoundError("run fenics_run.run_stress_strain and/or newton_run.run_stress_strain first")

    fig = make_stress_curve(fem, nw)
    out = os.path.join(params.FIG_DIR, "stress_strain.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[stress] wrote {out}")

    lam = (fem if fem is not None else nw)["lambdas"]
    ana = en.neohookean_uniaxial_strain_stress(lam)

    def report(name, sig):
        # max ABSOLUTE deviation in kPa -- a relative metric divides by ~0 at the lambda=1
        # stress zero-crossing and reports a meaningless ~600% there; absolute is faithful.
        abs_dev = np.abs(sig - ana)
        i = int(np.argmax(abs_dev))
        print(f"[stress] {name}: max abs. deviation vs analytic = {abs_dev[i] / 1e3:.3g} kPa "
              f"(at lambda={lam[i]:.2f}); at lambda={lam[-1]:.2f}: "
              f"{sig[-1] / 1e3:.3g} vs {ana[-1] / 1e3:.3g} kPa")

    if fem is not None:
        report("FEM   ", fem["sigma_fem"])
    if nw is not None:
        report("Newton", nw["sigma_newton"])


if __name__ == "__main__":
    main()
