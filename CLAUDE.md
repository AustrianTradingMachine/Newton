# CLAUDE.md — project guide

## What this project is

A quantitative, apples-to-apples comparison of one **deformable soft body** simulated two ways:

- **NVIDIA Newton** (on Warp, CUDA) — three solvers: **XPBD** (fast positional projection), **VBD** (implicit), **SemiImplicit** (explicit, differentiable);
- **FEniCSx / dolfinx** — **implicit FEM** (the reference solve).

The goal is to make it **measurable** how far Newton's fast solvers deviate from an accurate FEM solve of the *same* physical problem — same mesh, same material parameters (Lamé μ/λ; FEM uses a Neo-Hookean law, Newton an StVK/co-rotational one, equal at small strain), same gravity; only the solver differs — with closed-form analytic solutions as analytic references where they exist.

Everything is Python-driven. Newton needs a CUDA GPU → runs on **any CUDA-capable machine**; the easiest is **Google Colab** (free GPU, A100/high-RAM comfortable). FEniCSx is CPU (installed via fem-on-colab, or conda-forge locally). The repo is public on GitHub so Colab opens it directly.

## Hard conventions

- **English only** — all code, comments, docstrings, markdown, notebooks and commit messages. (Chat with the user may be German.)
- **`common/params.py` is the single source of truth** — geometry, material (Lamé μ/λ), gravity, every file path and every per-scenario parameter. Both sides import from it, so they describe the identical physical problem.
- **Scenarios are named by what they do** (never "Stage A/B"):

  | module stem | scenario | what it does |
  |---|---|---|
  | `hanging_bar` | **closed-form** | a soft bar stretches under self-weight; a closed-form answer exists, so every solver is scored against an analytic reference |
  | `indentation` | contact | a rigid sphere is pressed into a soft slab |
  | `drop` | dynamic impact | a sphere is dropped onto a block resting on the ground |
  | `friction` | Coulomb friction | a block is dragged across a rigid floor |
  | `stress_strain` | material test | confined uniaxial squeeze/stretch, stress vs stretch |
  | `convergence` | error study | does more solver budget / a finer mesh close the gap |

  Each scenario has `newton_run/run_<x>.py`, `fenics_run/run_<x>.py` and `compare/<x>.py` (the overlay). `compare/energies.py` is the shared, pure-numpy diagnostics library.

## Layout

```
src/common/params.py                single source of truth
src/common/mesh_io.py               shared mesh Newton <-> FEM (tet orientation)
src/common/runlog.py                notebook pipeline-stage runner (live stream + logs/summary.txt)
src/newton_run/run_hanging_bar.py   three solvers (--solver xpbd|vbd|semi_implicit); writes the shared mesh
src/newton_run/run_indentation.py   sphere indentation (--solver xpbd|vbd|semi_implicit)
src/newton_run/run_drop.py          dynamic drop (--solver xpbd|vbd|semi_implicit)
src/newton_run/run_friction.py      sliding block (--solver xpbd|vbd|semi_implicit)
src/newton_run/_solver.py           shared Newton solver factory (make_solver/needs_coloring)
src/newton_run/run_stress_strain.py SemiImplicit uniaxial material test
src/newton_run/convergence.py       XPBD convergence (iterations / substeps sweep)
src/newton_run/diffsim.py           differentiable theta* stiffness fit (SemiImplicit)
src/fenics_run/run_hanging_bar.py   FEM static solve (--element tet|hex)
src/fenics_run/run_indentation.py   FEM penalty / Augmented-Lagrangian contact
src/fenics_run/run_drop.py          FEM Newmark elastodynamics + contact
src/fenics_run/run_friction.py      FEM Coulomb friction (force + dissipated work)
src/fenics_run/run_stress_strain.py FEM uniaxial material test
src/fenics_run/convergence.py       FEM h-refinement + load-step sweep
src/compare/{hanging_bar,indentation,drop,friction,stress_strain,convergence}.py   overlays -> figures/
src/compare/energies.py             pure-numpy diagnostics (strain energy, nodal forces, residual, Jacobian)
src/compare/scene.py                3-D scene render of the deformed body (matplotlib, headless-safe)
tests/test_energies.py          validates the diagnostics (no GPU/FEM needed)
00_setup.ipynb            install + run the whole pipeline (any CUDA GPU); each stage -> logs/summary.txt (OK/ERR health report)
10_hanging_bar.ipynb            hanging-bar analysis: XPBD / VBD / explicit vs FEM tet/hex vs analytic
15_material.ipynb            material test: stress vs stretch (FEM / SemiImplicit vs analytic Neo-Hookean)
20_contact.ipynb            contact: force vs Hertz, dimple, penalty-vs-AL penetration
25_dynamic.ipynb            dynamic drop: sphere trajectory, penetration, block energies (Newton vs Newmark FEM)
30_convergence.ipynb            discretisation error vs solver error
40_friction.ipynb               Coulomb friction force / work / slip
```

Data flow: `newton_run/run_hanging_bar` produces the **shared mesh** (`data/mesh.npz`) → `fenics_run/run_hanging_bar` evaluates its FEM solution at Newton's nodes → results land in `data/*.npz` → `compare/*` writes figures to `figures/`. `data/`, `figures/`, `logs/` are intentionally tracked, so a reference Colab run ships with the repo (only scratch dirs like `out/` are git-ignored).

## The three Newton solvers (hanging-bar scenario)

- **XPBD** — positional projection; fast; leaves a finite **equilibrium residual** (its settled state is a projection, not a force balance), so it reads slightly soft. The canonical run that writes the shared mesh.
- **VBD** — Vertex Block Descent; **implicit** (minimises the backward-Euler objective by block coordinate descent over a coloured vertex graph; needs `builder.color()`). In principle converges toward the FEM-like implicit solution *as iterations grow*, but on the slender hanging bar at the budget used (50 iters) it still settles ~3× too soft — block Gauss-Seidel needs many iterations to propagate the clamp down a 17-layer bar (observed tip ≈ 140 mm vs FEM ≈ 43 mm). The implicit VBD is not the accurate partner here; quantifying the budget needed is the convergence study's job.
- **SemiImplicit** — explicit, force-based; the **differentiable** one → used by `diffsim.py` (the θ* effective-stiffness fit) and the Newton material test.

## Honesty rules (keep the repo defensible)

- The diagnostics' correctness claims are **backed by `tests/test_energies.py`** (`pytest tests/` or `python tests/test_energies.py`): nodal forces = −dU/dx via a finite-difference check, and the uniaxial closed form to machine precision. Do not write "validated" for anything not covered there.
- The θ* differentiable fit runs on **SemiImplicit** and characterises *that* solver vs FEM — **not XPBD**. XPBD's softness is measured by the equilibrium residual / tip ratio.
- Colab result numbers are **observations** (Python 3.12, dolfinx 0.11.0), recorded with provenance — not eternal "verified" claims. `TODO[verify-on-colab]` marks what still needs a GPU run.
- The **contact scenarios (indentation/drop/friction) now wire all three solvers** via `--solver xpbd|vbd|semi_implicit` on the shared `soft_contact` path (as Newton's `example_rigid_soft_contact.py` does for every solver) — XPBD is the canonical default. So the implicit **VBD** is the apples-to-apples partner for the implicit FEM; running XPBD alone was a wiring choice, not a Newton limit. **But** whether the *pinned* Newton supports the VBD soft/rigid-contact path (AVBD, `rigid_body_particle_contact_buffer_size`) is unverified (`TODO[verify-on-colab]`): on an older pin the VBD/SemiImplicit contact runs error and only XPBD records a result — don't claim VBD-contact works without a Colab run. Attribute "no calibrated contact force" to **XPBD**, not Newton generally. The **drop** is the hardest case (free sphere → two-way AVBD) and only a *partial* fairness fix vs. implicit Newmark — its transient also mixes material/contact-model/time-integration, so don't present it as a clean solver-only gap.

## Running

```
pip install -e .                                   # registers common/newton_run/fenics_run/compare
python -m newton_run.run_hanging_bar               # XPBD (CUDA); add --solver vbd | semi_implicit
python -m fenics_run.run_hanging_bar --element tet  # FEM on Newton's mesh (node-for-node)
python -m compare.hanging_bar                       # overlay + figures/
pytest tests/                                       # validate diagnostics (no GPU needed)
```

On a CUDA GPU (e.g. Colab), run `00_setup.ipynb`. To update a running checkout:
`git fetch --depth 1 origin main && git reset --hard origin/main`.

## Environment notes

- Newton/Warp and the fem-on-colab dolfinx stack are large and can clash (numpy / petsc / mpi). If one runtime misbehaves, run the two sides in **separate notebooks** — they only exchange `data/*.npz`.
- dolfinx 0.11 API in use: `create_mesh(comm, cells, element, x)`, SNES-based `NonlinearProblem(..., petsc_options_prefix=...)`, and `element.interpolation_points` as an **attribute**. fem-on-colab installer: `fenicsx-install-release-real.sh`.
