# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **VBD (Vertex Block Descent)** as a third Newton solver for the hanging bar
  (`run_hanging_bar --solver vbd`), alongside XPBD and SemiImplicit.
- **Validation tests** (`tests/test_energies.py`) backing the diagnostics: nodal
  forces vs. a finite-difference of the energy, and the uniaxial Neo-Hookean stress
  vs. its closed form (to machine precision). Run with `pytest tests/`.
- **Documentation set** under `docs/` — `METHOD.md`, `CONTACT.md`, `EXPERIMENTS.md`,
  `STATUS.md` — and a slimmer README with the layered `analytic → FEM → Newton`
  reference chain.
- **3D scene rendering** (`compare/scene.py`, matplotlib, headless-safe) and real
  renders for the indentation and drop scenarios.
- Project tooling: continuous integration (ruff + `pytest tests/` + build),
  pre-commit hooks, `.python-version`, and this changelog.

### Changed
- **`src/` package layout** — `common`, `newton_run`, `fenics_run`, `compare` now
  live under `src/` (import names and `python -m` commands unchanged; requires
  `pip install -e .`).
- Scenarios renamed from opaque "Stage A/B" to descriptive names (`hanging_bar`,
  `indentation`, `drop`, `friction`, `stress_strain`, `convergence`), including all
  figure/log artifact names.
- Indentation contact reworked into a coarse → accurate gradation
  (tet penalty → stiffer penalty → tet Augmented-Lagrangian → locking-free hex AL).
- Documented GPU target as Google Colab **A100 (high-RAM)**; fixed the fem-on-colab
  installer URL in `requirements.txt`.

## [0.1.0] - 2026-06-29

### Added
- Initial Newton (XPBD) vs. FEniCSx (FEM) soft-body comparison: hanging bar,
  rigid-sphere indentation, dynamic drop, Coulomb friction, confined uniaxial
  material test, convergence study, and the differentiable θ\* stiffness fit, all
  driven from a single source of truth (`common/params.py`).
