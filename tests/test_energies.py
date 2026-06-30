"""Validation of the shared energy diagnostics (``compare/energies.py``).

These tests back the two correctness claims the comparison relies on, so they can
be checked rather than asserted:

  * the internal nodal forces are the *exact* gradient of the strain energy,
    ``f = -dU/dx`` -- verified against a central finite difference of the energy;
  * the volume-averaged axial first Piola-Kirchhoff stress for a confined
    uniaxial deformation ``F = diag(1, 1, lambda)`` reproduces the closed-form
    compressible Neo-Hookean law to machine precision.

Plus two structural invariants (energy zero at the rest state; internal forces
self-equilibrated). Everything is pure numpy -- no GPU, Warp or FEM dependency --
so it runs anywhere:

    pytest tests/                  # or, without pytest:
    python tests/test_energies.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# allow running as a plain script (and under pytest) without installing the package:
# the importable packages live under <repo>/src in the src/ layout.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from compare import energies as en  # noqa: E402


def _unit_cube_tets():
    """8 corner nodes of the unit cube and its 6-tet Freudenthal decomposition."""
    rest = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=float)
    tets = np.array([
        [0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
        [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7],
    ], dtype=np.int64)
    return rest, tets


def test_nodal_forces_match_energy_finite_difference():
    """f = -dU/dx: the internal nodal forces equal the central FD of the energy."""
    rest, tets = _unit_cube_tets()
    rng = np.random.default_rng(0)
    final = rest + 0.05 * rng.standard_normal(rest.shape)   # small generic deformation

    f = en.nodal_forces(rest, final, tets)                  # analytic -dU/dx

    h = 1.0e-6
    f_fd = np.zeros_like(final)
    for i in range(final.shape[0]):
        for k in range(3):
            up = final.copy(); up[i, k] += h
            dn = final.copy(); dn[i, k] -= h
            dU = en.strain_energy(rest, up, tets) - en.strain_energy(rest, dn, tets)
            f_fd[i, k] = -dU / (2.0 * h)

    err = float(np.max(np.abs(f - f_fd)))
    scale = float(np.max(np.abs(f_fd))) + 1.0
    assert err / scale < 1.0e-6, f"nodal force vs finite-difference mismatch: {err / scale:.2e}"


def test_uniaxial_stress_matches_closed_form():
    """Volume-averaged axial 1st Piola for F = diag(1,1,lambda) == the closed form."""
    rest, tets = _unit_cube_tets()
    for lam in (0.7, 0.9, 1.0, 1.25, 1.5):
        final = rest.copy()
        final[:, 2] *= lam                                  # affine: F = diag(1,1,lam) per tet
        sig_fem = en.mean_axial_first_piola(rest, final, tets)
        sig_ana = float(en.neohookean_uniaxial_strain_stress(lam))
        assert abs(sig_fem - sig_ana) <= 1.0e-12 * (abs(sig_ana) + 1.0), \
            f"lambda={lam}: averaged P_zz {sig_fem:.6g} != closed form {sig_ana:.6g}"


def test_strain_energy_zero_at_rest():
    """No deformation -> zero strain energy and zero uniaxial stress at lambda = 1."""
    rest, tets = _unit_cube_tets()
    assert abs(en.strain_energy(rest, rest.copy(), tets)) < 1.0e-9
    assert abs(float(en.neohookean_uniaxial_strain_stress(1.0))) < 1.0e-12


def test_internal_forces_self_equilibrated():
    """Internal elastic forces sum to zero (no net self-force on the body)."""
    rest, tets = _unit_cube_tets()
    rng = np.random.default_rng(1)
    final = rest + 0.05 * rng.standard_normal(rest.shape)
    f = en.nodal_forces(rest, final, tets)
    assert float(np.max(np.abs(f.sum(axis=0)))) < 1.0e-9


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS  {_name}")
    print("All energy validation tests passed.")
