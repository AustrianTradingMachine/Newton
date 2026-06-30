# CONTACT — indentation and the dynamic drop

Real soft-body work is mostly contact, and contact is where the two solvers differ
most sharply. Both contact scenarios are implemented **entirely in Python / UFL** on
the FEM side — no C++ extension, no mesh-to-mesh collision search — because every
obstacle is a **rigid analytic shape** whose gap to the deformed surface is
closed-form. That keeps the whole contact law inside the variational form and
auditable.

The headline both scenarios make: **FEM yields a calibrated contact force; XPBD does
not.** XPBD enforces non-penetration by positional projection and exposes no
comparable force, so the honest common axis is the deformation, and the force curve
is something the fast solver simply cannot provide.

> **A note on fairness.** Each contact scenario can run **all three** Newton solvers via
> `--solver xpbd|vbd|semi_implicit` (default XPBD, the canonical run) — so the **implicit
> VBD** is the apples-to-apples partner for the implicit FEM, not just XPBD. All three drive
> the *same* contact: each step calls `model.collide` once into the shared `soft_contact`
> buffer (`model.soft_contact_ke/kd/kf/mu`), and `solver.step(..., contacts, dt)` reads it —
> exactly the wiring Newton's own `example_rigid_soft_contact.py` uses when switched between
> solvers (our contact runs are adapted from it). So running XPBD alone was a **wiring
> choice**, not a Newton limitation.
>
> The honest caveat is now about **version**, not capability. The rich VBD soft/rigid-contact
> path (VBD integrating rigid bodies itself via AVBD; the `rigid_body_particle_contact_buffer_size`
> knob) landed on recent Newton `main`; the repo's pinned version may predate it, in which
> case the VBD/SemiImplicit contact runs **error** and only XPBD records a result. This is
> **unverified locally** (no GPU here) — `TODO[verify-on-colab]`: one CUDA run checks
> `newton.__version__` and that `example_rigid_soft_contact.py --solver vbd` runs.
>
> Two consequences to read honestly regardless: (i) the **drop** is the hardest case — its
> *free* sphere needs VBD's two-way AVBD (the most version-sensitive path), and the FEM side
> is *implicit* Newmark, so even a successful VBD run is only a **partial** fairness fix; the
> transient still mixes material, contact model and time integration. (ii) part of any residual
> gap is **constitutive** (Newton's StVK/co-rotational vs FEM's Neo-Hookean), which grows once
> strains leave the small-strain regime — not pure solver error. And "no calibrated contact
> force" remains specifically an **XPBD** limitation, not a general Newton one.

---

## 1. Indentation — a rigid sphere pressed into a soft slab

Quasi-static: a clamped slab, a rigid sphere lowered onto its top face in equal
steps. Geometry and schedule (`common/params.py`, `INDENT_*`):

| quantity | value |
|---|---|
| slab | 1.0 × 1.0 × 0.4 m (`INDENT_DIM = 20×20×8`, `INDENT_CELL = 0.05`) |
| clamp | bottom face (z = 0) |
| sphere radius `R` | 0.30 m |
| max indentation | 0.04 m over 20 equal steps (`INDENT_MAX`, `INDENT_LOAD_STEPS`) |
| gravity | **off** (`INDENT_WITH_GRAVITY = False`) — isolates contact for the Hertz anchor |

### The two FEM contact laws (`fenics_run/run_indentation.py`)

The sphere is analytic: the signed gap `g = ‖(X+u) − c‖ − R` and the surface normal
`n = (X+u−c)/‖·‖` are closed-form UFL expressions evaluated on the top facets.

- **penalty** — an augmented pressure `p = kn·⟨−g⟩₊` is added to the weak form, with
  `kn = factor · E / cell`. Simple, but both the result and the residual penetration
  depend on the penalty stiffness `kn`.
- **aug_lagrangian (Uzawa)** — keep a contact-pressure multiplier field `λ` and use
  `p = ⟨λ − kn·g⟩₊`; after each inner nonlinear solve, update `λ ← p`. Iterating this
  **outer loop** drives the penetration toward zero at a *modest* `kn`, approaching
  the exact non-penetration constraint without the ill-conditioning of `kn → ∞`. It
  stays in pure UFL — `λ` is an interpolated field, not an extra global unknown, so
  there is **no saddle-point system** (`INDENT_AUG_ITERS = 8`,
  `INDENT_AUG_PEN_TOL = 10⁻⁵ m`).

### The gradation: coarse → accurate

`INDENT_VARIANTS` is a deliberate ladder from approximate to accurate contact. Because
`create_box` splits each hex cell into tets on the *same* grid, the tet and hex meshes
share **identical nodes** — so only the element type and the contact method change:

| # | variant | what limits its accuracy |
|---|---|---|
| 1 | tet, kn×5, penalty | soft penalty → visible penetration, **plus** tet locking |
| 2 | tet, kn×50, penalty | stiffer penalty → less penetration, still tet locking |
| 3 | tet, kn×10, AL | Uzawa drives penetration ≈ 0 at modest kn, still tet locking |
| 4 | **hex, kn×10, AL** | AL (penetration ≈ 0) on **locking-free hex** → the most accurate (same AL + kn as #3 → isolates the element/locking effect) |

Per step the run records the **total contact force** (∫ p·n_z ds), the strain energy,
the penalty (contact) energy `½ kn ⟨−g⟩₊²`, and the **max residual penetration**. The
penetration plot is the clearest axis: penalty leaves a kn-dependent gap; AL collapses
it to ≈ 0.

### The analytic anchor: Hertz

The dashed reference is the Hertzian normal force for a rigid sphere on an elastic
half-space (`params.hertz_force`):

```
F = (4/3) E* √R · δ^{3/2},     1/E* = (1 − ν²)/E.
```

**Honest scope:** Hertz assumes small strain, frictionless contact, contact radius
`a ≪ R`, and an infinite half-space. Our slab is **finite and soft**, so Hertz is an
*approximate anchor*, not ground truth — the FEM curves follow the δ^{3/2} trend at
small depth and deviate as the contact radius grows relative to the slab. The run
prints each variant's force/Hertz ratio so the deviation is explicit.

### The Newton side (`newton_run/run_indentation.py`, all three solvers)

The same slab and schedule, but the sphere is a **kinematic rigid body** lowered in
the same steps and the slab bottom is pinned (inverse mass → 0). `--solver` selects
XPBD / VBD / SemiImplicit on this identical scene; the kinematic collider (no free body
to integrate) makes it the easiest contact case for the swap. XPBD's contact stiffness
knobs (`soft_contact_ke/kd/kf`) are set but are **largely ignored for the normal force**
— that is exactly the point: XPBD has no calibrated contact force to report. So the
comparison axis is the **deformed dimple** (top-surface displacement through the contact
centre) and the **penetration**, not a force. The run also saves the deformed mesh +
sphere at maximum indentation so
[`compare/scene.py`](../src/compare/scene.py) can render the real dimple in 3-D.

`python -m compare.indentation` overlays the FEM force curves (+ Hertz), the dimples,
and the penetration gradation; [`20_indentation.ipynb`](../20_indentation.ipynb)
tells the story verdict-first.

---

## 2. Drop — a sphere dropped onto a block (dynamic impact)

The literal `rigid_soft_contact`-style scenario and the dynamic counterpart to the
quasi-static indentation: a soft block rests on the ground (z = 0) and a **free**
rigid sphere falls onto it under gravity. Geometry (`DROP_*`):

| quantity | value |
|---|---|
| block | 0.8 × 0.8 × 0.5 m (`DROP_DIM = 8×8×5`, `DROP_CELL = 0.1`) |
| sphere | R = 0.25 m, density 500 kg/m³ |
| drop height | centre starts at z = 1.10 m (≈ 0.35 m free fall) |
| time stepping | dt = 1×10⁻³ s, 400 steps (0.4 s) |

### The FEM side (`fenics_run/run_drop.py`)

Pure UFL/dolfinx **elastodynamics**:

- **Implicit Newmark-β** time integration (β = 0.25, γ = 0.5 — the average-acceleration
  scheme: unconditionally stable, no algorithmic damping). The inertia term
  `ρ ⟨a_new, w⟩` regularises the tangent, so no Dirichlet clamp is needed.
- **Penalty contact against two rigid analytic obstacles** — the ground plane z = 0
  and the falling sphere — each with a closed-form gap.
- **~10 % Kelvin–Voigt contact damping** for impact stability:
  `p = ⟨kn(−g) + cd(−ġ)⟩₊`, with `ġ = n·(v_material − v_sphere)`,
  `cd = DROP_DAMP_FRAC · kn · dt` (`DROP_DAMP_FRAC = 0.10`,
  `kn = DROP_PENALTY_FACTOR · E / h`, `DROP_PENALTY_FACTOR = 20`).
- The **rigid sphere's own motion** is a small staggered ODE: assemble the contact
  force on the sphere, integrate free-fall + reaction (`m_s = ρ_s · 4/3 π R³`),
  advance its centre. This is the only non-UFL piece — a handful of Python lines, no
  C++.

It records a time series → `data/fem_drop.npz`: sphere height, penetration, block
strain energy, block kinetic energy, contact force.

### The Newton side (`newton_run/run_drop.py`, all three solvers)

The same block/sphere with `add_ground_plane` and a free rigid sphere
(`add_shape_sphere`), stepped with the `--solver`-selected solver. It records the same
time-series diagnostics and keeps the **deepest-impact frame** (deformed mesh + sphere
centre) for the 3-D render. `compare/drop.py` overlays whichever solver runs are present
and writes `drop_scene.png`. The **free** sphere makes the implicit VBD path the hardest /
most version-sensitive here (two-way AVBD body integration; `TODO[verify-on-colab]`).

**Honesty note.** The dynamic contact path is the least settled numerically: dt,
penalty stiffness and damping all need tuning, and the FEM drop file flags its dynamic
path `TODO[verify-on-colab]` (`fenics_run/run_drop.py`). The structure (Newmark +
penalty + staggered rigid ODE) is standard; the specific constants are **observations
pending a clean GPU/Colab run**, not verified results. See [STATUS.md](STATUS.md).

---

Friction (a third contact scenario — a block dragged across a rigid floor, with a
Coulomb return-mapping law and the analytic `μ·W` plateau) is documented with the
other experiments in [EXPERIMENTS.md](EXPERIMENTS.md).
