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

These are **measurements**, not eternal claims. They were produced on the Colab/A100 +
dolfinx 0.11.0 stack and are reported with their context. They are **not** a
substitute for a Tier-1 test, and the most recent set predates the file/structure
rename — so they are **pending a clean re-run** on the renamed pipeline.

- **Hanging bar, FEM tip deflection** (earlier session): tet ≈ 43.2 mm, hex ≈ 43.4 mm,
  vs. the analytic 1-D bar ≈ 44.3 mm — i.e. a few-percent gap consistent with the
  3-D/Poisson correction the 1-D formula omits. The hex (locking-free) result sitting
  closest to analytic is the expected ordering.
- **Hanging bar, XPBD softness:** an early run came out far too soft because the
  dynamic solver was under-damped and never settled (it was ringing, not balanced).
  Adding per-frame velocity damping (`SETTLE_VEL_DAMP = 0.97`) drains the transient KE
  so the measured state is the static equilibrium. Whether the *remaining* XPBD
  softness is genuine positional-solver compliance is exactly what the convergence
  study (the equilibrium-residual RMS vs. budget, and the tip ratio) is there to
  quantify — on a fresh GPU run with the current code. (The differentiable θ\* fit is a
  *separate* tool that characterises the **SemiImplicit** solver vs. FEM, **not** XPBD;
  see the honesty rules below.)

> Reproduce Tier 2 yourself: run `00_setup.ipynb` §1–§4 (setup), then the scenarios §5–§13
> — each one streams its result and appends to `logs/summary.txt` (the OK/ERR health report) —
> then the `10/20/30/40_*` notebooks. Numbers will be re-recorded here with the run date once the renamed
> pipeline has been executed on Colab.

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
| contact scenarios (XPBD-only) | whether VBD/SemiImplicit can drive Newton's rigid-body / ground-plane `soft_contact` path (indentation/drop/friction use XPBD only; see CONTACT.md) |

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
   the scenarios §5–§13; each stage appends to `logs/summary.txt` (the OK/ERR health log).
4. `10/20/30/40_*.ipynb` — the verdict-first analysis notebooks.

See [METHOD.md](METHOD.md), [CONTACT.md](CONTACT.md), and
[EXPERIMENTS.md](EXPERIMENTS.md) for what each number means.
