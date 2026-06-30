# How accurate is NVIDIA Newton, really? — Newton vs. FEniCSx (FEM)

A quantitative, apples-to-apples comparison of one **deformable soft body** simulated two ways:

- **NVIDIA Newton** (on Warp, CUDA) — three solvers: **XPBD** (fast positional projection), **VBD** (implicit), **SemiImplicit** (explicit, differentiable);
- **FEniCSx / dolfinx** — **implicit FEM**, used as the reference solve.

Same mesh, same material parameters (Lamé μ, λ), same gravity — *only the solver differs* (the FEM side uses a compressible Neo-Hookean law, Newton an StVK/co-rotational one at the same μ, λ — equal at small strain). The goal is to make it **measurable** how far the fast game/robotics solver deviates from an accurate FEM solve, and exactly *why*.

> **Hanging bar (the one test with a known answer):** FEM lands within a few percent of the analytic value (the residual is the 3-D/Poisson correction the 1-D bar omits), and the implicit **VBD** is expected to track it; the fast **XPBD** settles softer — because it *projects* positions rather than *solving* the force balance (it leaves a finite equilibrium residual). The numbers and their provenance are in **[docs/STATUS.md](docs/STATUS.md)**.

### The references are layered: analytic → FEM → Newton

Where a **closed-form solution** exists, it anchors *both* sides — so a skeptic can
follow the whole trust chain, not just take FEM on faith:

- the **1-D self-weight bar** (hanging bar) — tip elongation `ρgL²/2E` (an
  *approximate* anchor; it omits Poisson contraction and 3-D effects);
- the **confined uniaxial Neo-Hookean stress law** (material test) — matched **to
  machine precision** by a test in [`tests/`](tests/test_energies.py);
- the **Coulomb `μ·W` plateau** and `N = W` (friction);
- the **Hertz** sphere-on-half-space force (indentation, an *approximate* anchor for a
  finite soft slab).

The FEM (the reference) is itself checked against these analytic solutions in the
simple cases; the fast Newton solvers are then scored against the FEM on the *same*
mesh and material. So the comparison is **analytic → FEM → Newton**, with each link
testable.

## The scenarios (named for what they do)

| scenario | what it does | the point |
|---|---|---|
| **hanging bar** | a soft bar stretches under self-weight | the only case with a *closed-form* (analytic) reference → score every solver against it, with the FEM solve as the reference |
| **indentation** | a rigid sphere is pressed into a soft slab | FEM gives a calibrated contact-force curve; the fast XPBD gives deformation, not a force (VBD/explicit selectable via `--solver`) |
| **drop** | a sphere is dropped onto a block (dynamic impact) | transient impact; FEM Newmark + contact vs. Newton solvers (implicit VBD is the natural match; see [docs/CONTACT.md](docs/CONTACT.md)) |
| **friction** | a block is dragged on a rigid floor | FEM friction force + dissipated work vs. analytic `μ·W`; XPBD slip only |
| **material test** | confined uniaxial squeeze/stretch | stress vs. stretch into large strain (constitutive fidelity) |
| **convergence** | swept budgets / meshes | discretisation error (FEM) vs. solver-budget error (XPBD) |

Each scenario has `newton_run/run_<x>.py`, `fenics_run/run_<x>.py` and `compare/<x>.py` (the overlay). The analysis lives in the `10/20/30/40_*` notebooks, each written as a **10-minute read for a skeptical expert** (verdict first, then the mechanism), with a rendered 3D scene of what is being simulated. The contact scenarios (indentation / drop / friction) run all three Newton solvers via `--solver` on the same `soft_contact` scene, so the implicit **VBD** — not just XPBD — is the apples-to-apples partner for the implicit FEM; [docs/CONTACT.md](docs/CONTACT.md) covers the version caveats and what "no calibrated contact force" does and does not mean.

## The three Newton solvers

- **XPBD** — positional projection; fast; leaves a finite equilibrium residual → reads slightly soft. The canonical run (writes the shared mesh).
- **VBD** — Vertex Block Descent; *implicit* (minimises the backward-Euler objective); converges toward the FEM-like solution.
- **SemiImplicit** — explicit, force-based; the *differentiable* one → used by the `diffsim` θ\* fit and the material test.

## Platform

Newton needs a **CUDA GPU** — that is the only requirement, so the repo runs on **any CUDA-capable machine**. The easiest zero-setup option is **Google Colab** (free GPU; an A100 / high-RAM runtime is comfortable but not required). FEniCSx is CPU (installed via fem-on-colab on the same instance, or conda-forge locally). The repo is public, so Colab opens it directly (File → Open notebook → GitHub) — see `00_setup.ipynb`.

> If the two large stacks clash in one runtime (numpy/petsc/mpi), run the Newton and FEM sides in **separate notebooks** — they only exchange `data/*.npz`.

## Quickstart

```bash
pip install -e .                                   # registers common/newton_run/fenics_run/compare
pytest tests/                                      # validate the diagnostics (no GPU needed)

python -m newton_run.run_hanging_bar               # XPBD (CUDA); add --solver vbd | semi_implicit
python -m fenics_run.run_hanging_bar --element tet  # FEM on Newton's mesh (node-for-node)
python -m fenics_run.run_hanging_bar --element hex  # FEM, independent hex mesh
python -m compare.hanging_bar                       # overlay + figures/ + a text report
```

Then the other scenarios: `run_indentation`, `run_drop`, `run_friction`, `run_stress_strain`, `convergence` (each on both `newton_run` / `fenics_run`, then `compare.<scenario>`), and `newton_run.diffsim`. The `00_setup.ipynb` notebook installs everything and runs the whole pipeline on any CUDA GPU (Colab is the easy path); each scenario appends an OK/ERR line + key results to `logs/summary.txt` — a running health report.

## Layout

```
src/common/params.py        single source of truth (geometry, material, gravity, paths, per-scenario params)
src/common/mesh_io.py        shared mesh Newton <-> FEM (tet orientation)
src/common/runlog.py        notebook pipeline-stage runner (live stream + logs/summary.txt)
src/newton_run/   run_hanging_bar · run_indentation · run_drop · run_friction · run_stress_strain · convergence · diffsim · _solver (shared solver factory)
src/fenics_run/   run_hanging_bar · run_indentation · run_drop · run_friction · run_stress_strain · convergence
src/compare/      hanging_bar · indentation · drop · friction · stress_strain · convergence · energies · scene
tests/        test_energies.py   (finite-difference force check + machine-precision stress check)
10_hanging_bar · 20_contact · 25_dynamic · 30_convergence · 40_friction   (analysis notebooks)
00_setup.ipynb     install + run the whole pipeline (any CUDA GPU; Colab = easy path)
```

`compare/energies.py` (pure numpy) computes the diagnostics identically for both solvers — strain energy, Jacobian (det F), internal nodal forces `f = −∂U/∂x`, and the **equilibrium residual** `f_internal + f_gravity` (the sharpest "solve vs. project" measure). Its correctness claims are **backed by `tests/test_energies.py`**. `compare/scene.py` renders the deformed body in 3D (matplotlib, headless-safe), solver-agnostic.

## Documentation

- **[docs/METHOD.md](docs/METHOD.md)** — problem setup, material matching, element variants (tet/hex locking), the three solvers, the energy/residual diagnostics.
- **[docs/CONTACT.md](docs/CONTACT.md)** — the indentation contact law (penalty / Augmented Lagrangian, no C++) and the dynamic drop.
- **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** — friction, the material test, the convergence study, and the differentiable θ\* fit.
- **[docs/STATUS.md](docs/STATUS.md)** — what is verified vs. still being confirmed on Colab, with the result numbers and their provenance.
- **[CLAUDE.md](CLAUDE.md)** — concise project guide / conventions.

## Development

```bash
pip install -e ".[dev]"     # ruff, pytest, build
ruff check .                 # lint (config lives in pyproject.toml)
pytest tests/                # the pure-numpy validation tests (no GPU/FEM)
pre-commit install           # optional: ruff + hygiene hooks on each commit
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs ruff + `pytest tests/` + a package build on every push. It deliberately does **not** run the GPU (Newton) or FEM (dolfinx) stages — those have no runner on GitHub Actions and are exercised on a CUDA GPU (Colab or local) via `00_setup.ipynb`. The `src/` layout means the packages must be installed (`pip install -e .`) before `python -m …` resolves them.
