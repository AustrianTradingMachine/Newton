# METHOD — the hanging bar, the material, and the diagnostics

This is the methodological core of the comparison: the one physical problem, how
the two codes are made to describe *exactly* it, the three Newton solvers, the FEM
reference, and the pure-numpy diagnostics that turn "looks soft" into numbers.
Everything here is defined once in [`common/params.py`](../src/common/params.py) — the
single source of truth both sides import.

## 1. The problem: a soft bar hanging under its own weight

A soft block is **clamped on its top face** (max-z) and hangs downward; gravity
(−z) stretches it. We deliberately picked this as the flagship because it is the
one configuration with a **closed-form reference**, so every solver can be scored
against an answer rather than only against each other.

| quantity | symbol | value | source |
|---|---|---|---|
| block size | `BLOCK_LX × LY × LZ` | 0.6 × 0.6 × 1.6 m | `GRID_DIM = 6×6×16`, `CELL = 0.1 m` |
| long (hanging) axis | z | clamped face = top (max-z) | `FIXED_FACE = "top"` |
| density | ρ | 100 kg/m³ | `DENSITY` |
| Lamé parameters | μ, λ | 1.0×10⁴, 5.0×10⁴ Pa | `K_MU`, `K_LAMBDA` |
| → Young's modulus, Poisson | E, ν | ≈ 2.83×10⁴ Pa, ≈ 0.417 | `lame_to_E_nu()` |
| gravity | g | 9.80665 m/s² (−z) | `GRAVITY` |

Gravity is set **explicitly** on the Newton builder (`gravity=-GRAVITY`) because
Newton's default is −9.81; this matches FEM to the same constant. The resulting
tip strain is ≈ 2.8 % — small enough that linear bar theory is a meaningful anchor,
large enough that the nonlinear terms are exercised.

### The closed-form anchor

For a 1-D bar of length L hanging from a clamp, the downward displacement of a
material point originally at depth `s = z_top − z` below the clamp is

```
u(s) = (ρ g / E) · (L·s − s²/2),     tip elongation u(L) = ρ g L² / (2E).
```

With the numbers above the tip elongation is **≈ 44.3 mm**
(`params.analytic_hanging_displacement`, printed by `params.summary()`). The
matching 1-D strain energy is `U = ρ²g²A L³ / (6E)`
(`energies.analytic_hanging_strain_energy`).

**Honest scope:** this 1-D solution ignores Poisson contraction and 3-D effects,
so it is a *sanity anchor*, not the ground truth — the FEM solve is the reference
the fast solvers are scored against. The few-percent gap between the analytic bar
and the 3-D FEM result is exactly the Poisson/3-D correction, and the convergence
study quantifies it (see [EXPERIMENTS.md](EXPERIMENTS.md)).

## 2. Same problem, two codes

The point of the project is that the *only* difference between the two simulations
is the solver. Two mechanisms enforce that:

- **One shared mesh.** The canonical Newton XPBD run builds the soft grid, finalises
  the model, and exports the rest node coordinates, the tet connectivity and the
  clamped-node indices to `data/mesh.npz`
  ([`common/mesh_io.py`](../src/common/mesh_io.py)). The FEM tet run *consumes those exact
  arrays* (`create_mesh(comm, cells, element, x)`) and evaluates its solution at
  Newton's node positions, so the two results are comparable **node-for-node**. Tets
  are defensively re-oriented to positive volume for dolfinx
  (`orient_tets_positive`).
- **One material.** Newton's `add_soft_grid` uses a co-rotational tetrahedral FEM
  model parameterised by Lamé (μ, λ). FEniCSx uses a **compressible Neo-Hookean**
  strain-energy density with the *same* (μ, λ):

  ```
  ψ = μ/2 (I_C − 3) − μ ln J + λ/2 (ln J)²,     P = ∂ψ/∂F      (1st Piola–Kirchhoff)
  ```

  The same ψ is used by the pure-numpy diagnostics (§4), so Newton's settled state
  and FEM's solution are evaluated on identical footing.

### The two FEM element variants

`fenics_run/run_hanging_bar.py --element {tet|hex}`:

- **tet** (default) — built from Newton's shared mesh, **genuine simplex** linear
  (P1) tetrahedra (`basix` `Lagrange`/`tetrahedron`/1, full Galerkin quadrature),
  node-for-node with Newton. These are *real* 4-node constant-strain simplices — **not**
  the collapsed/degenerate-hexahedron tets with reduced integration + hourglass control
  that explicit codes (e.g. LS-DYNA) often use for "tets". Real P1 simplex tets are known
  to be **over-stiff in bending and near-incompressibility (locking)** because a
  constant-strain element is too poor to represent those modes, so this is the honest
  like-for-like element, not the most accurate one. → `data/fem_result.npz`
- **hex** — an *independent* structured hexahedral mesh of the same block geometry
  (`create_box`, trilinear Hex8, full 2×2×2 integration — no reduced-integration
  hourglass modes). Hex8 is far less prone to shear/volumetric locking, so it is a
  **second, more accurate FEM reference**. It uses the **same** 6×6×16 cell grid and the
  **same** 833 nodes / 2499 DOFs as the tet run (only the cell *type* and split differ),
  so it is the *same fineness* — it just costs more per element (8 Gauss points, denser
  LU). Its nodes differ in *position* from Newton's, so it is compared via the
  displacement profile, not node-for-node. → `data/fem_result_hex.npz`

Reporting both tells the reader how much of any Newton–FEM gap is *solver* and how
much is *element choice* — the hex result brackets the tet result from the accurate
side.

### The FEM solve itself

Compressible Neo-Hookean elastostatics, solved with dolfinx 0.11's SNES-based
`NonlinearProblem` (Newton line-search, direct LU; these meshes are small). Gravity
is **ramped over 5 load increments** for nonlinear robustness; the converged
solution is independent of that count (the convergence study checks exactly this).
Per-step SNES iteration counts and the converged reason are printed.

## 3. The three Newton solvers

`newton_run/run_hanging_bar.py --solver {xpbd|vbd|semi_implicit}`. All three settle
the *same* grid to the *same* static equilibrium in principle; they differ in how
they get there, which is the whole point.

| solver | class | kind | budget | writes |
|---|---|---|---|---|
| **XPBD** | `SolverXPBD` | positional projection (fast) | 32 substeps × 10 iters | `mesh.npz` + `newton_result.npz` |
| **VBD** | `SolverVBD` | **implicit** (Vertex Block Descent) | 10 substeps × 10 iters | `newton_result_vbd.npz` |
| **SemiImplicit** | `SolverSemiImplicit` | explicit, force-based, **differentiable** | 32 substeps | `newton_result_semi.npz` |

- **XPBD** is the canonical run (it owns the shared mesh). It enforces constraints
  by **projecting positions**, not by solving a force balance, so its settled state
  leaves a finite **equilibrium residual** (§4) and reads slightly soft. This is the
  expected behaviour of a fast positional solver, and the residual is how we measure
  it — not a bug.
- **VBD** minimises the backward-Euler objective by block coordinate descent over a
  coloured vertex graph (it requires `builder.color()`, which `build_model` runs only
  for this solver). Being implicit, it converges toward the same solution the implicit
  FEM finds — it is the genuine volumetric soft-body counterpart to XPBD.
- **SemiImplicit** is the explicit, force-based integrator. It is the **one Warp can
  differentiate**, so it is what the θ\* stiffness fit (`diffsim.py`) and the material
  test use. It is not used to claim XPBD's accuracy.

### Settling to a *static* answer

A dynamic solver hit with full gravity at t=0 rings (under-damped). To measure the
*static* equilibrium rather than a snapshot of an oscillation, the settle loop
applies a per-frame velocity damping `SETTLE_VEL_DAMP = 0.97` that drains transient
kinetic energy. At rest v ≈ 0, so the factor does nothing to the equilibrium itself
— it only removes the transient. The loop stops when KE < `SETTLE_KE_TOL` (1×10⁻⁶ J)
after at least `MIN_SETTLE_FRAMES`, capped at `MAX_FRAMES`.

## 4. The diagnostics — and what actually backs them

[`compare/energies.py`](../src/compare/energies.py) is **pure numpy** and computes the
same quantities for *either* solver from `(rest_q, final_q, tets)`:

- **Strain energy** `U` — the Neo-Hookean ψ integrated over the rest tets.
- **Jacobian** `J = det F` per tet and the total **volume change** — how much the
  solver compresses/inflates the material.
- **Internal nodal forces** `f = −∂U/∂x` from the 1st Piola stress
  `P = μ(F − F⁻ᵀ) + λ ln(J) F⁻ᵀ`, assembled to nodes by `H = −V·P·Dm⁻ᵀ`.
- **Equilibrium residual** `r = f_internal + f_gravity` per node. This is the
  sharpest "solve vs. project" measure: for a true static equilibrium (FEM) the
  free-node residual ≈ 0; for XPBD it is finite and measures how far the *projected*
  state is from the *material's* force balance. At the clamped nodes, `−Σ r_z` is the
  support reaction, which must equal the total weight `ρgV` (≈ 565 N for this block)
  — a built-in consistency check.

### The honesty backbone: `tests/test_energies.py`

The two correctness claims the comparison rests on are **tested, not asserted**
(`pytest tests/`, or `python tests/test_energies.py` — pure numpy, no GPU/FEM needed):

1. `test_nodal_forces_match_energy_finite_difference` — the analytic `f = −∂U/∂x`
   matches a **central finite difference** of the strain energy (relative error
   < 10⁻⁶).
2. `test_uniaxial_stress_matches_closed_form` — the volume-averaged axial 1st Piola
   stress for `F = diag(1,1,λ)` reproduces the closed-form compressible Neo-Hookean
   law **to machine precision** (≤ 10⁻⁹) across λ ∈ [0.7, 1.5].

Plus two structural invariants: zero strain energy at the rest state, and internal
forces that sum to zero (no net self-force). These four tests are the only things in
the repo described as "validated"; anything depending on a GPU or a FEM install is
reported as an **observation with provenance** (see [STATUS.md](STATUS.md)).

## 5. Reading the result

`python -m compare.hanging_bar` overlays whichever Newton results exist (XPBD / VBD /
explicit) against FEM tet/hex and the analytic bar, writes the figures, and prints a
text report: the tip vertical deflection per solver (next to the FEM tet/hex and
analytic-1-D references) and, per Newton solver, the node-for-node error vs. FEM tet
(RMS / max [mm], mean relative [%]) and the XPBD/FEM tip ratio. The strain-energy,
Jacobian and equilibrium-residual diagnostics from `compare/energies.py` are surfaced
in the analysis notebook rather than this CLI report. The 3-D scene
([`compare/scene.py`](../src/compare/scene.py), matplotlib, headless-safe) renders the
deformed bar coloured by displacement so the reader sees the stretch directly. The
notebook [`10_hanging_bar.ipynb`](../10_hanging_bar.ipynb) walks through it
verdict-first.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the convergence study and the differentiable
θ\* fit that quantify the XPBD and SemiImplicit gaps, and
[CONTACT.md](CONTACT.md) for the indentation and drop scenarios.
