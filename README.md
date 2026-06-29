# Newton (XPBD) vs. FEniCSx (FEM) — Soft-Body Comparison

Comparison of a deformable body simulated with **NVIDIA Newton** (the fast XPBD
solver) on one side and a **real FEM solver, FEniCSx / dolfinx**, on the other.
Goal: make it visible and **quantitative** how far Newton's fast, position-based
approximation deviates from an accurate implicit FEM solve of the same problem.
Everything is driven from Python — preprocessing, simulation, postprocessing and
evaluation.

> **Platform:** Newton needs a CUDA GPU → we run on **Google Colab (T4 GPU)**.
> FEniCSx is CPU and runs on the same instance (installed via fem-on-colab) or
> locally via conda-forge. The repo is public on GitHub, so Colab opens it
> directly (File → Open notebook → GitHub). See `00_setup_colab.ipynb`.

## Two-stage layout

| Stage | Scenario | Contact? | Character |
|-------|----------|----------|-----------|
| **A** | Hanging block (top clamped, gravity stretches it) | no | clean, **quantitative**, with analytic reference; **both Newton solvers** (XPBD + explicit) |
| **B** | Rigid sphere indents the soft slab | yes | FEM penalty contact (dolfinx, Hertz-anchored) **vs. Newton XPBD** — compared on deformation |
| **Drop** | Rigid sphere **dropped** onto a soft block on the ground | yes | the literal example: Newton XPBD **vs. FEM Newmark elastodynamics** + penalty contact — dynamic transient |
| **Friction** | Soft block dragged on a rigid floor | yes | Coulomb friction: FEM friction force + dissipated work vs. analytic `mu·W`; XPBD slip only |
| **Convergence** | Stage A, swept | — | discretisation error vs solver error: XPBD iters/substeps + FEM h-refinement |

**Why is stage A hanging and not a horizontal cantilever?**
The material of the Newton example is very soft (E ≈ 28 kPa). A horizontal
cantilever would fully collapse under its own weight (deflection ≫ length, fully
nonlinear) — not a clean benchmark. The **hanging** configuration stays at small
strain (~3 %), is stable, and has a closed-form reference solution (axial
self-weight bar). Bending/contact follows in stage B.

## Project structure

```
common/params.py        # single source of truth: geometry, material (Lame mu/lambda), gravity
common/mesh_io.py        # shared mesh Newton<->FEniCSx, tet orientation
newton_run/run_stage_a.py   # XPBD + explicit: build block, settle, export particle_q + mesh (--solver)
newton_run/run_stage_b.py   # XPBD: rigid-sphere indentation (kinematic sphere, clamped slab)
newton_run/convergence_stage_a.py  # XPBD convergence: sweep iterations & substeps
newton_run/run_friction.py  # XPBD: sliding block on ground (soft_contact_mu) -> slip
newton_run/diffsim_stage_a.py  # DIFFERENTIABLE: backprop through Newton to fit effective stiffness vs FEM
fenics_run/run_stage_a.py   # dolfinx: Neo-Hookean static solve (--element tet|hex)
fenics_run/run_stage_b.py   # dolfinx: rigid-sphere penalty contact (tet & hex + penalty sweep)
fenics_run/convergence_stage_a.py  # FEM convergence: mesh h-refinement + load-increment sweep
fenics_run/run_friction.py  # dolfinx: sliding block, penalty-regularised Coulomb friction (force + work)
compare/metrics.py          # Stage A: node matching, error (RMS/max/relative), plots
compare/stage_b.py          # Stage B: FEM-vs-Newton dimple overlay + penetration
compare/convergence.py      # Convergence: XPBD + FEM h-refinement plots
compare/friction.py         # Friction: force vs Coulomb plateau, dissipated work, slip
compare/energies.py         # strain / kinetic / gravitational energies (numpy, solver-agnostic)
newton_run/example_rigid_soft_contact.py  # Dynamic drop: literal XPBD example (free falling sphere)
fenics_run/run_drop.py      # Dynamic drop: FEM Newmark elastodynamics + penalty contact (sphere + ground)
compare/drop.py             # Dynamic drop: Newton-vs-FEM transient overlay
fenics_run/run_stress_strain.py  # Material test: FEM uniaxial sigma(lambda)
newton_run/run_stress_strain.py  # Material test: Newton uniaxial sigma(lambda)
compare/stress_strain.py    # Material test: FEM/Newton/analytic stress-strain overlay
00_setup_colab.ipynb        # Colab: install + clone repo + run all stages; §15 runs everything + log summary
10_stage_a_analysis.ipynb   # Stage A: deflection, profile, energies, FEM-vs-analytic validity, both solvers
20_stage_b_analysis.ipynb   # Stage B evaluation: force, dimple, strain/contact energy, penetration
30_convergence_analysis.ipynb  # Convergence: discretisation vs solver error
40_friction_analysis.ipynb  # Friction: force vs Coulomb plateau, dissipated work, slip
```

Data flow: `newton_run/run_stage_a` produces **the mesh** (`data/mesh.npz`) →
`fenics_run/run_stage_a` uses exactly that mesh and evaluates its solution at
Newton's node positions → both results land in `data/*.npz` → `compare` computes
the errors and writes plots to `figures/`.

## Install

The project is an editable package, so the modules import from anywhere (no
`sys.path` hacks):

```bash
pip install -e .          # pulls numpy + matplotlib; registers the packages
```

The heavy simulators are installed separately: Newton via `pip install
"newton[examples]"`, FEniCSx via fem-on-colab (Colab) or conda-forge (local) —
see `00_setup_colab.ipynb` / `requirements.txt`.

## Run locally (in order)

```bash
python -m common.params              # show parameters/material
python -m newton_run.run_stage_a                # Newton XPBD  (CUDA GPU recommended)
python -m newton_run.run_stage_a --solver semi_implicit  # second Newton solver (explicit)
python -m fenics_run.run_stage_a --element tet  # FEM, Newton's mesh (node-for-node)
python -m fenics_run.run_stage_a --element hex  # FEM, independent hex mesh
python -m compare.metrics                       # Stage A comparison + plots in figures/
python -m fenics_run.run_stage_b                # Stage B FEM contact (tet & hex + penalty sweep)
python -m newton_run.run_stage_b                # Stage B Newton XPBD indentation (CUDA GPU)
python -m compare.stage_b                       # Stage B FEM-vs-Newton overlay -> figures/
python -m newton_run.convergence_stage_a        # (4) XPBD convergence: iters & substeps (CUDA)
python -m fenics_run.convergence_stage_a        # (4) FEM convergence: h-refinement + load steps
python -m compare.convergence                   # convergence plots -> figures/
python -m fenics_run.run_friction               # friction: FEM Coulomb (force + dissipated work)
python -m newton_run.run_friction               # friction: Newton XPBD sliding block (CUDA)
python -m compare.friction                      # friction overlay -> figures/
python -m newton_run.diffsim_stage_a            # (optional) differentiable stiffness fit vs FEM (CUDA)
python -m newton_run.example_rigid_soft_contact # dynamic drop, Newton XPBD (CUDA)
python -m fenics_run.run_drop                   # dynamic drop, FEM Newmark + contact
python -m compare.drop                          # dynamic drop comparison -> figures/
python -m fenics_run.run_stress_strain          # material test: FEM uniaxial sigma(lambda)
python -m newton_run.run_stress_strain          # material test: Newton uniaxial sigma(lambda) (CUDA)
python -m compare.stress_strain                 # stress-strain overlay -> figures/
```

## Run on Google Colab

The repo is public on GitHub, so Colab runs it directly — no upload dance:

1. Colab → **File → Open notebook → GitHub** → enter the repo URL → open
   `00_setup_colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Run the cells: §2–§4 install Newton + dolfinx (via fem-on-colab) and clone the
   repo; the stage cells follow (the explicit Stage A solver, convergence and
   friction are in §14). **§15 runs the whole pipeline and prints a compact
   OK / ERR summary** of every stage (per-stage logs in `logs/`, full summary in
   `logs/summary.txt`) — handy instead of hunting per-cell outputs.

To pull the latest fixes inside a running notebook (more reliable than `git pull`
on the shallow clone):

```bash
!cd /content/Newton && git fetch --depth 1 origin main && git reset --hard origin/main
```

> **Environment note:** Newton/Warp and the fem-on-colab dolfinx stack are large
> and can clash (numpy/petsc/mpi versions). If a single runtime misbehaves, run
> the two sides in **separate notebooks** — they only exchange the `data/*.npz`
> files, so they never need to coexist.

## Evaluation notebooks

The setup notebooks install everything and run the sims (producing all
`data/*.npz`). The actual *evaluation* lives in four dedicated notebooks that load
those arrays and quantify the differences:

* **`10_stage_a_analysis.ipynb`** — deflection (Newton XPBD / **Newton explicit** /
  FEM tet / FEM hex / analytic), an explicit **FEM-vs-analytic validity check**
  (§1b: is the FEM itself adequate?), displacement profile, **internal (strain)
  energy** (same Neo-Hookean density on the same mesh → directly comparable),
  released gravitational PE, settling kinetic energy, **computation time**,
  **volume / Jacobian (det F) distribution**, the **dissipation / energy
  budget**, the **equilibrium residual** (out-of-balance force: FEM ≈ 0 since
  it solved R(u)=0, XPBD ≠ 0 — the most direct "solve vs. project" measure), and
  the **differentiable stiffness fit** (Newton backprops through the sim to fit
  its effective stiffness `θ*` to FEM — see *Newton's superpower* below).
* **`20_stage_b_analysis.ipynb`** — FEM contact force vs. indentation (+ Hertz),
  deformed dimple (FEM vs Newton), **internal (strain) energy**, **contact
  (penalty) energy**, Newton penetration and settle-KE diagnostic,
  **computation time** per variant, and **residual penetration penalty-vs-AL**.
* **`30_convergence_analysis.ipynb`** — the **(4) convergence study**: XPBD tip /
  equilibrium residual vs solver iterations & substeps (it approaches the FEM /
  analytic value as the budget grows), FEM tip & strain energy under **mesh
  h-refinement** (with the load-increment sweep validating the nonlinear solve).
* **`40_friction_analysis.ipynb`** — **friction**: FEM friction force vs the
  analytic Coulomb plateau `mu·W`, the normal reaction (≈ weight), the dissipated
  frictional work and the stick→slip transition; the shared slip curve (FEM vs
  XPBD — XPBD slides but exposes no calibrated friction force).

`compare/energies.py` (pure numpy, solver-agnostic) computes all of this on
identical footing for both solvers: compressible Neo-Hookean strain energy, lumped
nodal masses, gravitational PE and kinetic energy, per-tet **Jacobian** `det F` and
volume change, internal **nodal forces** `f = −∂U/∂x` (first Piola-Kirchhoff,
validated against a finite-difference of the energy), and the **equilibrium
residual** `f_internal + f_gravity`. The residual is the sharpest mechanistic
differentiator: the implicit FEM solve drives it to ~0, while XPBD's positional
projection leaves a finite out-of-balance force at the free nodes.

**Computation time** is recorded for every solve (`wall_time` in the npz, GPU
work synchronised before stopping the clock) and plotted in both notebooks — the
project's core trade-off (Newton fast on GPU vs. accurate-but-slower implicit
FEM) is now *measured*, not just asserted.

**Hourglassing** does not occur here: linear tets (Tet4) have no zero-energy
modes (they lock instead), and dolfinx integrates Hex8 **fully** (no reduced
integration), so there is nothing to stabilise. It would only become relevant if
a reduced-integration hex variant were added (which could be done to *demonstrate*
it).

**Convergence study (built).** `30_convergence_analysis.ipynb` quantifies
*discretisation* error separately from the solver difference:
`newton_run/convergence_stage_a.py` sweeps XPBD **iterations** and **substeps**
(the positional projection's equilibrium-residual RMS falls and the tip drop
approaches the implicit-FEM / analytic value as the budget grows — XPBD's
effective stiffness is solver-budget dependent), while
`fenics_run/convergence_stage_a.py` does FEM **mesh h-refinement** (tip & strain
energy converge monotonically to a budget-independent limit) plus a
**load-increment sweep** whose converged tip must be flat (a check on the
nonlinear solve). That separation — *solver* error vs *discretisation* error — is
the essence of the Newton-vs-FEM difference.

**Friction (built).** See *Friction — Coulomb sliding block* below: the FEM run
exposes the friction force and dissipated work and validates them against the
analytic Coulomb plateau `mu·W`; XPBD slides the block but exposes no calibrated
friction force.

## Newton's superpower: differentiable evaluation

`newton_run/diffsim_stage_a.py` exploits that Newton is built on differentiable
Warp: it backpropagates a loss through the *entire* settling simulation. The loss
is the node-for-node mismatch between Newton's settled block and the FEM reference
(`data/fem_result.npz`, same shared mesh), differentiated w.r.t. a stiffness
multiplier `theta` that scales the Lamé parameters.

* **Sensitivity** — one forward+backward gives the exact `dLoss/dθ` at `θ=1`: how
  fast the Newton-vs-FEM mismatch changes with stiffness, for free.
* **Inverse fit** — gradient descent (`warp.optim.SGD`) on `θ` so Newton matches
  FEM. The fitted **`θ*`** is the *effective-stiffness multiplier*: `θ* > 1` ⇒
  Newton effectively softer than the true material (needs stiffening to match
  FEM), `θ* < 1` ⇒ stiffer. This turns the qualitative "the solver looks a bit
  soft" into a number obtained by differentiating through the physics.

It follows Newton's own `examples/diffsim/example_diffsim_soft_body.py`:
`finalize(requires_grad=True)`, a full pre-allocated state trajectory, a
`wp.Tape()` around the forward pass, `tape.backward(loss)`, and `warp.optim`.
Differentiability is proven for the **SemiImplicit** solver (used here), not XPBD;
the result is shown in §9 of `10_stage_a_analysis.ipynb`. The learning rate and
step count are the main `TODO[verify-on-colab]` knobs.

## Material matching (apples-to-apples)

Newton's `add_soft_grid` uses Lame parameters `k_mu = mu`, `k_lambda = lambda`.
dolfinx's **compressible Neo-Hookean uses the same (mu, lambda) directly**, so no
conversion is needed:

```
psi = mu/2 (Ic - 3) - mu ln(J) + lambda/2 (ln J)^2
```

At stage-A small strain (~3 %) Neo-Hookean ≈ St-Venant-Kirchhoff ≈ linear
elasticity for identical (mu, lambda), so the comparison is fair. (`E`, `nu` are
also reported for reference, via `lame_to_E_nu` in `common/params.py`.)

**Caveat (energy/residual diagnostics).** `compare/energies.py` evaluates the
*same* Neo-Hookean from each solver's node positions for both Newton and FEM. This
is exact at Stage A small strain, where Newton's internal tet model and
Neo-Hookean coincide. At large strain the strain-energy and equilibrium-residual
numbers would then also reflect any difference in the *underlying energy model*,
not only the solver — keep that in mind before over-reading them outside the
small-strain regime. Gravity is set explicitly on both sides (`gravity=-GRAVITY`
on the Newton builder) so the two never silently disagree.

## Element variants (tet vs. hex)

Both stages run in two element flavours to expose the **element effect**: linear
tetrahedra (Tet4) are prone to **shear and volumetric locking** and behave too
stiff, while trilinear hexahedra (Hex8) are markedly more accurate per element.

* **Stage A — `--element tet`** (default): the mesh is Newton's *exact* tet mesh
  (linear P1, corner nodes only, constant strain per element). This matches
  Newton's own discretisation, so the result is compared to Newton
  **node-for-node** and the tet stiffness *cancels* (both sides share it).
* **Stage A — `--element hex`**: an independent structured Hex8 mesh of the same
  block geometry. Its nodes differ from Newton's, so it is overlaid on the
  displacement **profile** (and the tip table), not compared node-for-node. With
  less locking it typically deflects a little more (softer) and sits closer to
  the analytic bar — it is the more accurate FEM reference.
* **Stage B**: runs the variants in `STAGEB_VARIANTS` — a **tet penalty-strength
  sweep** (kn×5 vs kn×50) plus a **hex** reference — and overlays their
  contact-force / indentation curves against Hertz. Two effects show at once:
  penalty strength (stiffer `kn` ⇒ higher force, less penetration) and element
  locking (tet stiffer than hex).

All variants use **degree-1 Lagrange** (corner nodes, no edge nodes). Adding
mid-edge nodes would mean degree-2 (Tet10 / Hex20) — a one-line change to the
function-space degree — at the cost of breaking Stage A's same-discretisation
match with Newton.

## Stage B — contact, implemented in dolfinx (no C++)

`fenics_run/run_stage_b.py` is a standalone prototype that presses a **rigid
analytic sphere** into a soft slab and resolves the contact with a penalty
formulation written entirely in **Python / UFL**. It shows how to add a *custom
contact law* to the FEM solver without dropping to C++. It runs the variants in
`STAGEB_VARIANTS` — a **tet penalty-strength sweep (kn×5 vs kn×50)**, a **hex**
reference, and an **Augmented-Lagrangian (Uzawa)** variant at the same modest kn —
and overlays them (see *Element variants* above and *Contact methods* below).

### Setup

A slab (default `1.0 × 1.0 × 0.4 m`, `20 × 20 × 8` cells) is **clamped on its
bottom face** (zero displacement — this removes the rigid-body modes and makes
the quasi-static problem well-posed). A rigid sphere of radius `R` sits above the
top face; its centre `c` is lowered step by step so the prescribed indentation
depth `δ` ramps from `0` to `δ_max`. The material is the same compressible
Neo-Hookean as Stage A (Lame `mu`, `lambda`). Gravity is off by default to
isolate the contact response for the Hertz comparison.

### Contact formulation (penalty)

Non-penetration against a rigid obstacle is the Signorini / KKT condition

```
gap g ≥ 0        pressure p ≥ 0        g · p = 0
```

— an *inequality* constraint: the body may not enter the sphere, contact
pressure is compressive only, and pressure acts only where the gap is closed. We
enforce it with a **penalty regularisation** — wherever the body penetrates the
sphere we add a restoring pressure proportional to the penetration depth:

```
x = X + u                      deformed position  (reference X + displacement u)
g = |x − c| − R                signed gap to the sphere surface
n = (x − c)/|x − c|            outward sphere normal at the contact point
p = kn · ⟨−g⟩₊                 penalty pressure   (⟨·⟩₊ = max(·, 0))
```

This adds a surface term to the Stage-A Neo-Hookean residual `R_bulk`:

```
R(u; v) = R_bulk(u; v) − ∫_top  kn ⟨−g⟩₊ (n · v) dS = 0
```

The **Macaulay bracket** `⟨·⟩₊` is the key: it switches the contact contribution
on/off *pointwise*, so the **active set** (which surface points actually touch)
is handled by the integrand and the integration domain stays fixed (the whole
candidate top face). dolfinx/UFL auto-differentiates the full residual —
bulk + contact — into the consistent Newton tangent.

### Contact methods: penalty vs. Augmented Lagrangian

Each variant chooses a `method`:

* **`penalty`** — pressure `p = kn ⟨−g⟩₊`. Simple, but the result and the residual
  penetration depend on `kn` (the `kn×5` vs `kn×50` sweep shows this).
* **`aug_lagrangian`** (Uzawa) — keep a multiplier field `λ` (contact-pressure
  estimate) and use the augmented pressure `p = ⟨λ − kn g⟩₊`. After each inner
  nonlinear solve, update `λ ← p`. Iterating this **outer loop** drives the
  penetration toward zero at a *modest* `kn` — approaching the exact
  non-penetration constraint without the ill-conditioning of `kn → ∞`.

The Augmented Lagrangian stays **pure UFL**: `λ` is a scalar Lagrange `Function`
that is *interpolated* (`λ ← ⟨λ − kn g⟩₊`), not solved for, so there is **no
saddle-point system and no extra global unknowns** — the practical step up from
penalty without the machinery a true Lagrange-multiplier / mortar method needs
(boundary multiplier space, indefinite solver, active-set loop). The default
config contrasts `tet kn×5 penalty` against `tet kn×5 AL` at identical `kn`, and
`stage_b_penetration.png` shows the AL penetration collapsing to ~0.

### Why no C++ / no contact search

Because the obstacle is a **rigid analytic surface**, the gap `g` is a
closed-form expression of the deformed position, evaluated at quadrature points
of the candidate boundary. There is **no mesh-to-mesh search**, no closest-point
projection, no contact-pair bookkeeping — the single most painful part of general
contact codes — which is exactly why everything fits into a UFL form. C++ (or the
`dolfinx_contact` extension) only becomes necessary for **deformable–deformable
or self-contact**, where the contacting surfaces must be re-discovered every
iteration.

### Solution procedure

Quasi-static, frictionless, incremental: for each of `STAGEB_LOAD_STEPS`
indentation increments we move the sphere centre, solve the nonlinear problem
with dolfinx's `NewtonSolver`, and assemble the resulting vertical contact force.
Ramping the indentation (instead of applying `δ_max` in one shot) keeps Newton in
its convergence basin as new nodes enter contact.

### Parameters (`common/params.py`)

| Name | Default | Meaning |
|------|---------|---------|
| `STAGEB_DIM` | `(20, 20, 8)` | slab resolution in cells |
| `STAGEB_CELL` | `0.05 m` | cell size → slab `1.0 × 1.0 × 0.4 m` |
| `STAGEB_SPHERE_R` | `0.30 m` | rigid sphere radius |
| `STAGEB_INDENT_MAX` | `0.04 m` | maximum indentation depth |
| `STAGEB_LOAD_STEPS` | `20` | indentation increments |
| `STAGEB_WITH_GRAVITY` | `False` | add self-weight (off for the Hertz comparison) |
| `STAGEB_VARIANTS` | `(("tet",5,"penalty"),…,("tet",5,"aug_lagrangian"))` | `(element, factor, method)`; `kn = factor · E / cell` |
| `STAGEB_AUG_ITERS` | `8` | Uzawa multiplier updates per indentation step |
| `STAGEB_AUG_PEN_TOL` | `1e-5` | stop Uzawa when max penetration < this [m] |

### Outputs & Hertz reference

Two figures land in `figures/`:

* `stage_b_force.png` — contact force vs. indentation depth for **each variant
  (tet kn-sweep + hex)**, overlaid with the **Hertz** solution for a rigid sphere
  on an elastic half-space, `F = (4/3) E* √R · δ^{3/2}`, `E* = E / (1 − ν²)`.
* `stage_b_penetration.png` — max residual penetration vs. indentation per
  variant (penalty kn-dependent; **Augmented Lagrangian ~ 0** at modest kn).
* `stage_b_profile.png` — the deformed top-surface dimple (per variant) along a
  line through the contact centre.

Hertz is an **approximate anchor, not ground truth**: it assumes small strain, a
frictionless rigid sphere and an infinite half-space. Our slab is finite and the
material soft, so expect the FEM curve to follow the `δ^{3/2}` trend and agree at
small `δ`, then deviate as the contact radius grows relative to the slab
thickness.

### Convergence & tuning

Penalty contact with a basic Newton solver is the most likely place to need
tuning — the `max(·,0)` kink and the changing active set make the tangent
non-smooth:

* **Penalty stiffness `kn`** — too low ⇒ visible penetration; too high ⇒
  ill-conditioning and Newton stalls. The default `STAGEB_VARIANTS` sweeps a tet
  pair (factor 5 vs 50), so the effect is visible: the stiffer `kn` gives a
  higher force with less penetration, approaching the exact non-penetration limit
  (`kn → ∞`). The **Augmented-Lagrangian** variant sidesteps this trade-off —
  near-zero penetration at modest `kn` (see *Contact methods*) — at the cost of
  `STAGEB_AUG_ITERS` extra inner solves per step.
* **Load steps** — increase `STAGEB_LOAD_STEPS` if an increment fails.
* **Solver** — if the basic `NewtonSolver` stalls, switch to a PETSc **SNES**
  with line search (noted in the code); markedly more robust for contact.

### Deliberately deferred

* **Coulomb friction** — needs a tangential return-mapping algorithm and
  per-point history (stick/slip state).
* **Dynamic impact** — needs a time integrator plus contact stabilisation
  (energy oscillations at impact are a known difficulty).

Both are genuine additions, not one-liners; the prototype isolates the contact
mechanics first.

### Newton (XPBD) contact comparison

`newton_run/run_stage_b.py` runs the **same indentation in Newton's XPBD solver**:
the slab is an `add_soft_grid` body clamped on the bottom, and the sphere is a
**kinematic** rigid body whose centre is lowered in the same steps as the FEM run
(pattern from Newton's cable example: `is_kinematic` / `body_inv_mass = 0`, sphere
pose written into the state each step).

The deliberate point: **XPBD has no calibrated contact stiffness** — it enforces
non-penetration as a *positional projection* and does not expose a contact force
comparable to the FEM penalty pressure. So the solvers are compared on what both
expose cleanly — the **deformation**: `compare/stage_b.py` overlays the deformed
top-surface dimple (FEM variants vs. Newton) and plots Newton's penetration vs.
indentation. The FEM force-vs-indentation curve (`stage_b_force.png`) has no XPBD
counterpart — and that absence *is* the difference between a fast positional
solver and an implicit FEM contact solve.

Caveat: the two sides use independent meshes (Newton's tet soft-grid vs. the FEM
box), so the comparison is via the **profile**, not node-for-node. A `soft_contact_ke`
sweep on Newton's *SemiImplicit* solver (which does use a penalty stiffness) would
be the closest Newton-side analogue of the FEM penalty sweep, if a force curve on
the Newton side is wanted later.

## Effective stress-strain — material fidelity

Stages A/B run at small strain, where Neo-Hookean ≈ StVK ≈ linear and the material
models coincide. This test pushes into the **large-strain** regime to check the
*constitutive* fidelity. A small block is driven through the homogeneous
deformation `F = diag(1, 1, λ)` (confined uniaxial strain) by an affine boundary,
and the axial stress is compared to the closed form
`σ(λ) = μ(λ − 1/λ) + λ_lame·ln(λ)/λ` (validated to machine precision in
`compare/energies.py`).

* **FEM** (`fenics_run/run_stress_strain.py`) assembles the block's own
  volume-averaged 1st Piola stress over a λ sweep → it reproduces the exact
  Neo-Hookean curve into large strain (a verification; deviations would flag a
  large-strain discretisation issue / inverted elements).
* **Newton** (`newton_run/run_stress_strain.py`, SemiImplicit) pins the boundary
  to the affine field and settles the interior. **Honest scope:** with a fully
  prescribed affine boundary this mainly checks whether Newton *reproduces the
  homogeneous deformation* across the strain range; the rigorous measure of
  Newton's effective *stiffness* is the differentiable `θ*` fit and the
  equilibrium residual.

`compare/stress_strain.py` overlays both against analytic and reports the
deviation, emphasising the large-strain end (`figures/stress_strain.png`).

## Friction — Coulomb sliding block

A flat soft block rests on a rigid floor (`z = 0`) under gravity; its **top face
is dragged** tangentially (`+x`) in small increments while Coulomb friction at the
floor resists. With the normal load supplied by gravity, the steady-slip friction
force has a closed form — `F = μ·N = μ·W = μ·ρ·g·V` — so the whole contact model
has an **analytic anchor**.

* **FEM** (`fenics_run/run_friction.py`) — pure UFL. A one-sided normal penalty
  `pN = kn⟨−z⟩₊` plus **penalty-regularised Coulomb friction with return mapping**:
  ```
  s_T   = (u_x, u_y)                tangential slip vs. the fixed floor
  |t_T| = min(kt·|s_T|, μ·pN)       elastic stick, capped by the Coulomb limit
  t_T   = −|t_T| · s_T/|s_T|        opposing the slip
  ```
  Because the floor is rigid and fixed, the slip of a contacting point is just its
  own tangential displacement, so **no slip-history field is needed** and the law
  stays a smooth (C⁰) function of `u` — differentiable for Newton. The run records
  the friction force (rises during **stick**, then plateaus at `μ·W` during
  **slip**), the normal force (≈ `W`), the dissipated frictional work and the
  slipping-area fraction.
* **Newton** (`newton_run/run_friction.py`) — the block sits on `add_ground_plane()`
  with `soft_contact_mu = μ`; the top face is dragged kinematically. XPBD enforces
  friction as a positional projection and **exposes no calibrated tangential
  force** — exactly as it exposes no normal contact force. So it reports the
  **slip** (the stick→slip knee is visible) but not the force.

`compare/friction.py` and `40_friction_analysis.ipynb` overlay the FEM friction
force against the analytic `μ·W` plateau, the dissipated work, and the shared
bottom-slip-vs-drag curve. Friction parameters (`FRICTION_*` in `common/params.py`)
and the `min/max` Coulomb-law convergence are `TODO[verify-on-colab]`.

## Two Newton solvers in Stage A (XPBD vs. explicit)

Stage A runs with **both** Newton solver families so each can be checked against
FEM and analytic: `--solver xpbd` (default, the positional/PBD projection that
also writes the shared mesh) and `--solver semi_implicit` (the explicit,
force-based integrator — the one Warp can differentiate, also used by the diffsim
and stress-strain runs). They build the *same* grid, so both are compared
node-for-node against FEM, and both appear in `10_stage_a_analysis.ipynb`
(deflection, profile, strain energy, equilibrium residual, timing). XPBD and the
explicit solver reaching the same settled state — and the same residual — is a
cross-check that the Newton-side result is solver-robust, not an artefact of one
integrator.

## Dynamic drop — the literal example

Stages A/B are clean, controlled sub-problems. The **drop** reproduces the actual
`rigid_soft_contact` scenario: a soft block rests on the ground and a **free rigid
sphere is dropped** onto it under gravity (dynamic impact). It is the dynamic
counterpart to Stage B's quasi-static indentation.

* **Newton** (`newton_run/example_rigid_soft_contact.py`) — adapted from Newton's
  own example: soft grid on a ground plane, a free dynamic sphere body, XPBD,
  gravity on; logs the transient (sphere height, penetration, block strain &
  kinetic energy).
* **FEM** (`fenics_run/run_drop.py`) — stays **pure UFL/dolfinx**:
  * implicit **Newmark-β** elastodynamics (the inertia term `∫ ρ a·w dx`),
  * penalty contact against **two rigid analytic obstacles** — the ground plane
    `z=0` and the sphere — so both gaps are closed-form (no mesh-mesh search),
  * **~10% Kelvin-Voigt contact damping** for impact stability:
    `p = ⟨ kn(−g) + cd(−ġ) ⟩₊`, `ġ = n·(v_material − v_sphere)`,
  * the sphere's own free-fall + contact reaction is a small **staggered ODE**
    (assemble the contact force, integrate the sphere) — the only non-UFL piece,
    a handful of Python lines, **no C++**. The dynamic mass term regularises the
    tangent, so no Dirichlet BC is needed.

`compare/drop.py` overlays the two transients (sphere trajectory incl. rebound,
penetration, block energies; the FEM contact force has no XPBD counterpart). This
is the most experimental piece — `dt`, penalty `kn` and the damping fraction are
the `TODO[verify-on-colab]` knobs, and dynamic contact is where impact stability
must be watched.

## Verified on Colab — and still to confirm

The repo runs on **Google Colab (Python 3.12, dolfinx 0.11.0)**. Confirmed working:
**FEM Stage A** (tet 43.2 / hex 43.4 mm vs analytic 44.3 — validated), the dolfinx
0.11 port (`create_mesh(comm, cells, e, x)`, SNES-based `NonlinearProblem`,
`bb_tree`/`compute_collisions_points`/`compute_colliding_cells`), the fem-on-colab
install URL (`fenicsx-install-release-real.sh`), and the Newton
settle-to-equilibrium velocity damping. Still being confirmed (`TODO[verify-on-colab]`):

1. **Newton (Stage A):** XPBD's large effective compliance for the long hanging
   block — confirm via the convergence study that the tip stiffens toward FEM as
   iterations/substeps grow (vs a stiffness bug); and the `--solver semi_implicit`
   explicit run reaching the same settled state.
2. **Newton (Stage B):** the kinematic-sphere drive (`_make_body_kinematic`, writing
   `body_q` into the state each step) and the particle pinning.
3. **FEM Stage B / stress-strain:** the penalty/Uzawa and affine-boundary solves
   under the new SNES `NonlinearProblem` (facet tagging + `ufl.max_value`).
4. **Drop (FEM):** Newmark update via `fem.Expression`, the no-Dirichlet dynamic
   tangent (mass-regularised), and impact stability (`dt` / `kn` / damping).
5. **Diffsim:** `finalize(requires_grad=True)`, the Tape/backward, and the SGD lr.
6. **Friction (FEM):** the component Dirichlet BCs (`V.sub(i).collapse()`) and the
   `min/max` Coulomb-law Newton convergence; **Newton:** `add_ground_plane()` +
   `soft_contact_mu`.
