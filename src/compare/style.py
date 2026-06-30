"""Unified plotting style -- one source of truth for the colour of every simulation and
reference, shared by all compare overlays AND the analysis notebooks, so the same solver is
always the same colour and the comparison reads at a glance.

Reserved colours (never reuse for an unrelated quantity):
    Newton XPBD = orange,  Newton VBD = red,  Newton explicit = purple,
    FEM = blue,  FEM hex = green,  analytic / closed-form = black (drawn dashed).

For a NON-solver diagnostic line (residual, dissipated work, slip fraction, a swept
parameter, ...) use a NEUTRAL colour below so it never reads as one of the solvers.

Both sides import this: the compare CLIs (`from compare import style`) and the notebooks
(`from compare import style`). It pulls in no matplotlib, so it is import-safe everywhere.
"""

from __future__ import annotations

# canonical colour per solver / reference key
COLOR = {
    "xpbd": "tab:orange",
    "vbd": "tab:red",
    "semi_implicit": "tab:purple",
    "fem": "tab:blue",
    "fem_hex": "tab:green",
    "analytic": "black",
}

# canonical display label per key
LABEL = {
    "xpbd": "Newton XPBD",
    "vbd": "Newton VBD",
    "semi_implicit": "Newton explicit",
    "fem": "FEM",
    "fem_hex": "FEM hex",
    "analytic": "analytic",
}

# per-Newton-solver marker for overlay scatter / lines
MARKER = {"xpbd": "o", "vbd": "s", "semi_implicit": "^"}

# line styles for reference (non-simulation) lines
ANALYTIC_LS = "--"   # the closed-form / analytic curve or level
REF_LS = ":"         # a secondary reference (weight, a static target level, ...)

# neutral colours for NON-solver diagnostic quantities (so they never read as a solver)
NEUTRAL = ("tab:brown", "tab:gray", "tab:cyan", "tab:olive", "tab:pink")

# distinct blue-family colours for the FEM CONTACT variants (coarse -> accurate = light ->
# dark), chosen so they never collide with the qualitative solver colours (orange/red/purple)
# or the hex green when Newton solvers are overlaid. Use via fem_variant_color(i); a given
# variant keeps the same colour across every contact plot. (Sampled from ColorBrewer Blues.)
FEM_VARIANT_COLORS = ("#9ecae1", "#4292c6", "#2171b5", "#08519c", "#08306b")


def fem_variant_color(i):
    """Colour of the i-th FEM contact variant (cycles if there are more variants than colours)."""
    return FEM_VARIANT_COLORS[i % len(FEM_VARIANT_COLORS)]

# the three Newton solvers in canonical display order: (key, --solver arg; None == xpbd default)
NEWTON_SOLVERS = (("xpbd", None), ("vbd", "vbd"), ("semi_implicit", "semi_implicit"))


def load_newton_runs(base_npz):
    """[(label, npz, colour, marker)] for whichever per-solver runs of ``base_npz`` exist.

    Centralises "load every present Newton solver, in canonical order, with canonical
    colour/label/marker" so every overlay shows all solvers that have data -- and only
    those -- with identical styling. ``base_npz`` is the XPBD/canonical path; the VBD and
    explicit paths are derived via ``params.solver_npz``.
    """
    import os

    import numpy as np

    from common import params

    out = []
    for key, arg in NEWTON_SOLVERS:
        path = params.solver_npz(base_npz, arg)
        if os.path.exists(path):
            out.append((LABEL[key], np.load(path), COLOR[key], MARKER[key]))
    return out
