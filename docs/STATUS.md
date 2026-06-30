# STATUS — what is verified, what is observed, what is still open

This file keeps the project **defensible**: it states exactly which claims are backed
by a test that runs anywhere, which are *observations* recorded with provenance, and
which paths are still being confirmed on a GPU. We do not write "validated" for
anything outside the first tier.

## Environment

| | |
|---|---|
| GPU runtime | **any CUDA GPU** (Newton/Warp needs CUDA); easiest is Google Colab — Tier-2 numbers below were measured on Colab A100 (high-RAM) |
| FEM runtime | CPU, **dolfinx 0.11.0** via fem-on-colab (`fenicsx-install-release-real.sh`) |
| Python | 3.12 (Colab) |
| Newton/Warp | `pip install "newton[examples]"` |
| local dev | no GPU and no dolfinx — only the pure-numpy tier (below) runs locally |

If the Newton/Warp and dolfinx stacks clash in one runtime (numpy / petsc / mpi), run
the two sides in **separate notebooks**; they communicate only through `data/*.npz`.

## Tier 1 — verified, runs anywhere (no GPU, no FEM)

These are backed by code that runs on any machine, so they are the claims the
comparison actually *rests* on. CI-equivalent: `pytest tests/` (or
`python tests/test_energies.py`).

- **`f = −∂U/∂x`** — the internal nodal forces equal a central finite difference of
  the Neo-Hookean strain energy (relative error < 10⁻⁶).
  `test_nodal_forces_match_energy_finite_difference`.
- **Closed-form uniaxial stress** — the volume-averaged axial 1st Piola stress for
  `F = diag(1,1,λ)` reproduces the analytic compressible Neo-Hookean law to machine
  precision (≤ 10⁻⁹) for λ ∈ {0.7, 0.9, 1.0, 1.25, 1.5}.
  `test_uniaxial_stress_matches_closed_form`.
- **Structural invariants** — zero strain energy at the rest state; internal forces
  self-equilibrate (Σf = 0). `test_strain_energy_zero_at_rest`,
  `test_internal_forces_self_equilibrated`.

Also verified locally without a GPU/FEM: every module `py_compile`s, the notebooks are
valid JSON (`nbformat`), and `compare/scene.py` renders synthetic tet grids headless.

## Tier 2 — observed on Colab, recorded with provenance

These are **measurements**, not eternal claims. **Provenance:** a full Colab run on
**2026-06-30**, Colab A100 (high-RAM), **dolfinx 0.11.0.post0**, committed with this repo
(`data/*.npz`, `figures/*`, `logs/summary.txt`). Solver budgets: XPBD 32 substeps × 10 iters;
VBD 10 substeps × 50 iters; SemiImplicit 32 substeps. They are **not** a substitute for a
Tier-1 test, and are re-recorded per run.

**Hanging bar — tip vertical drop** (max downward displacement over free nodes; source
`figures/hanging_bar_report.txt`):

| solver / reference | tip [mm] | ratio vs FEM tet |
|---|---|---|
| FEM tet | 43.20 | 1.00 (node-for-node reference) |
| FEM hex | 43.40 | — (independent mesh) |
| analytic 1-D | 44.30 | — (self-weight bar) |
| Newton XPBD | 158.98 | 3.68 |
| Newton VBD | 139.53 | 3.23 |
| Newton SemiImplicit | 84.99 | 1.97 |

The FEM few-percent gap to the analytic bar is the 3-D/Poisson correction; the hex
(locking-free) result sits closest to analytic, as expected. All three Newton solvers settle
far softer at this budget — XPBD because it projects positions and never reaches a force
balance (finite equilibrium residual), VBD because its block Gauss-Seidel has not converged on
this slender bar at 50 iters, SemiImplicit (closest) being force-based. The differentiable θ\*
fit is a *separate* tool characterising **SemiImplicit** vs. FEM, **not** XPBD.

**FEM validated against the analytic anchors** (same run):

- **friction:** plateau F = 74.97 N vs Coulomb μ·W = 75.32 N, and N = 250.69 N vs weight
  W = 251.05 N (both < 0.5 %).
- **material (uniaxial stress):** FEM matches the closed-form Neo-Hookean stress to < 0.3 %
  across λ ∈ [0.7, 1.5].
- **convergence:** the FEM tip refines 40.05 → 43.68 mm toward a mesh-independent limit near
  the analytic 44.30 mm.

> Reproduce Tier 2 yourself: run `00_setup.ipynb` §1–§4 (setup), then the scenarios §5–§11
> — each one streams its result and appends to `logs/summary.txt` (the OK/ERR health report) —
> then the `10/15/20/25/30/40_*` notebooks.

## Tier 3 — open / still being confirmed on Colab

Paths marked `TODO[verify-on-colab]` in the source. They follow the public Newton /
dolfinx examples, but the exact API names or numerical constants can shift between
versions and have **not** been re-run on the current stack. The marked files:

| file | what is unconfirmed |
|---|---|
| `newton_run/run_hanging_bar.py` | `particle_inv_mass` pinning attribute (Warp/sim convention) |
| `newton_run/run_indentation.py` | kinematic-body call + `particle_inv_mass` pinning |
| `newton_run/run_friction.py` | `add_ground_plane` + particle pinning |
| `newton_run/diffsim.py` | `finalize(requires_grad=True)`, Tape/optimiser, **learning rate needs tuning** |
| `fenics_run/run_hanging_bar.py` | dolfinx geometry API (`bb_tree` / `compute_collisions_points` / `compute_colliding_cells`) |
| `fenics_run/run_friction.py` | component Dirichlet via `V.sub(i).collapse()` |
| `fenics_run/run_drop.py` | dynamic contact: dt / damping tuning, the no-Dirichlet dynamic tangent |
| contact scenarios `--solver vbd / semi_implicit` | indentation/drop/friction now wire **all three** solvers on the shared `soft_contact` path (as Newton's own `example_rigid_soft_contact.py` does); unconfirmed is whether the **pinned** Newton supports the VBD soft/rigid-contact path (AVBD + `rigid_body_particle_contact_buffer_size`). On an older pin the VBD/SemiImplicit contact runs error and only XPBD records a result. The **drop** (free sphere, two-way AVBD) is the most version-sensitive; see CONTACT.md. |

The dynamic **drop** and the **θ\* fit** are the least settled numerically and should
be treated as work-in-progress until they have a clean Colab run.

## Honesty rules (the standard this repo holds itself to)

- "Validated" / "to machine precision" is used **only** for Tier-1 claims backed by
  `tests/test_energies.py`.
- The differentiable **θ\* fit characterises the SemiImplicit solver vs. FEM, not
  XPBD.** XPBD's softness is measured directly by the equilibrium residual and tip
  ratio. (See [EXPERIMENTS.md](EXPERIMENTS.md) §4.)
- Hertz (indentation) and the 1-D bar (hanging) are **approximate analytic anchors**
  for finite, soft, 3-D bodies — the FEM solve is the reference the fast solvers are
  scored against; the closed-form uniaxial stress and the Coulomb `μ·W` plateau are
  **exact** anchors that also check the FEM itself.
- Colab numbers are **observations with provenance**, re-recorded per run — never
  presented as fixed truths.

## Reproducibility checklist

1. `pip install -e .` — registers `common` / `newton_run` / `fenics_run` / `compare`.
2. `pytest tests/` — Tier 1, no GPU needed.
3. `00_setup.ipynb` on a CUDA GPU (Colab A100/high-RAM, or any CUDA machine) — installs Newton + dolfinx, runs
   the scenarios §5–§11; each stage appends to `logs/summary.txt` (the OK/ERR health log).
4. `10/15/20/25/30/40_*.ipynb` — the verdict-first analysis notebooks.

See [METHOD.md](METHOD.md), [CONTACT.md](CONTACT.md), and
[EXPERIMENTS.md](EXPERIMENTS.md) for what each number means.
