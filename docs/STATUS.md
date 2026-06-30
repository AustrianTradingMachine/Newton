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
| Newton XPBD | 158.86 | 3.68 |
| Newton VBD | 139.53 | 3.23 |
| Newton SemiImplicit | 84.99 | 1.97 |

The FEM few-percent gap to the analytic bar is the 3-D/Poisson correction; the hex
(locking-free) result sits closest to analytic, as expected. All three Newton solvers settle
far softer at this budget — XPBD because it projects positions and never reaches a force
balance (finite equilibrium residual), VBD because its block Gauss-Seidel has not converged on
this slender bar at 50 iters, SemiImplicit (closest) being force-based. The differentiable θ\*
fit is a *separate* tool characterising **SemiImplicit** vs. FEM, **not** XPBD.

**FEM checked against the analytic anchors** (same run):

- **friction:** plateau F = 74.97 N vs Coulomb μ·W = 75.32 N, and N = 250.69 N vs weight
  W = 251.05 N (both < 0.5 %).
- **material (uniaxial stress):** FEM reproduces the closed-form Neo-Hookean stress to ~1e-12
  relative (max abs dev 5.24e-12 kPa at λ=0.75) across λ ∈ [0.7, 1.5]. This agreement is
  *expected*, not a constitutive test of FEM: the confined test prescribes an affine deformation
  gradient F (material-independent) and both sides evaluate the same Neo-Hookean 1st-Piola
  formula. The StVK-vs-Neo-Hookean signal is the θ\* fit, not this test.
- **convergence:** the FEM tip refines 40.05 → 43.68 mm toward a mesh-independent limit near
  the analytic 44.30 mm.

> Reproduce Tier 2 yourself: run `00_setup.ipynb` §1–§4 (setup), then the scenarios §5–§11
> — each one streams its result and appends to `logs/summary.txt` (the OK/ERR health report) —
> then the `10/15/20/25/30/40_*` notebooks.

## Tier 3 — what the latest run confirmed, and what is still open

Paths that were previously marked `TODO[verify-on-colab]` in the source were **all re-run on
the 2026-06-30 Colab stack** (Warp 1.14.0 / dolfinx 0.11.0.post0) and produced the committed
data/figures cited in Tier 2 — so the open question has moved from "does the API even run" to
"is the result numerically settled". What the run established:

- **All three Newton solvers run every scenario.** The hanging bar (`particle_inv_mass`
  pinning), indentation (kinematic body + pinning), friction (`add_ground_plane` + pinning) and
  the differentiable θ\* fit (`finalize(requires_grad=True)`, Tape/optimiser) all produced finite
  results; the θ\* fit converged (θ\* = 1.79, loss 0.275 → 2.1e-4).
- **The FEM scenario scripts run.** The tet node-for-node reference (43.20 mm, via the dolfinx
  geometry API), FEM friction (component Dirichlet, F = 74.97 N) and the FEM Newmark drop
  (genuine impact, pen 1.99 mm) all produced results.
- **The VBD/SemiImplicit soft-contact path runs on the pinned Newton.** Indentation, drop and
  friction all record VBD and SemiImplicit results — the AVBD + `rigid_body_particle_contact_buffer_size`
  path is no longer the "only XPBD records a result" risk; in the drop, VBD genuinely impacts the
  free sphere two-way (pen 9.35 mm).

What is **still numerically open** (quality, not wiring):

| item | what is unsettled |
|---|---|
| **SemiImplicit drop** | numerically unstable at this dt/substep budget — the energy blows up (peak strain energy 7536 J vs FEM 3.4 J); the only clearly-broken numeric this run |
| **FEM drop** (`fenics_run/run_drop.py`) | dt / damping are not tuned to convergence; the transient numbers are observations, not a settled benchmark |
| **VBD / SemiImplicit indentation contact** | the `soft_contact` penalty is too soft — the sphere sinks ~33 mm through the 40 mm indent (strain energy ~0.1 J vs XPBD's ~13 J), so **XPBD** is the only Newton solver that geometrically resolves the indentation; the soft-contact stiffness for the implicit/explicit solvers is unsettled |
| **θ\* fit** (`newton_run/diffsim.py`) | one converged fit characterising the **SemiImplicit** solver vs FEM (θ\* = 1.79); not cross-validated across budgets/scenarios |

The `TODO[verify-on-colab]` markers that used to flag these paths have been retired from the
source (the run executed them); what remains is the **numerical tuning** above, not whether the
API runs.

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
