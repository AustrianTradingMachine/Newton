# EXPERIMENTS — friction, material test, convergence, and the differentiable fit

Four supporting studies. Each isolates one axis along which a fast solver can drift
from accurate FEM, and each carries its own reference — three of them an **analytic
one**, so the FEM side is itself checked, not just trusted.

---

## 1. Friction — a block dragged on a rigid floor (Coulomb)

A flat block rests on the floor (z = 0) under gravity; its top face is dragged
tangentially (+x) in small increments. Coulomb friction at the floor resists: the
tangential force rises while the contact sticks, then plateaus at `μ·N` once it slips.

| quantity | value |
|---|---|
| block | 0.8 × 0.8 × 0.4 m (`FRICTION_DIM = 8×8×4`, `FRICTION_CELL = 0.1`) |
| friction coefficient `μ` | 0.3 |
| drag | 0–0.02 m over 20 steps |
| normal/tangential penalty | `kn = kt = 20 · E / h` |

### The analytic anchor

The normal load is supplied by gravity, so `N = W = ρ g V`. For this block
`W ≈ 251 N`, and the steady-slip friction force plateaus at **`μ·W ≈ 75.3 N`**
(`params.coulomb_plateau`). Two things are therefore checked against closed form: the
integrated normal force `N` must equal the weight `W`, and the force plateau must
equal `μ·W`.

### The FEM side (`fenics_run/run_friction.py`)

Pure UFL, with a penalty-regularised Coulomb **return mapping**:

```
pN    = kn·⟨−z_def⟩₊                          one-sided normal pressure
s_T   = (u_x, u_y)                            tangential slip vs. the fixed floor
|t_T| = min(kt·|s_T|, μ·pN)                   elastic stick, capped at the Coulomb limit
t_T   = −|t_T|·s_T/|s_T|                       traction opposes the slip
```

Because the floor is rigid and fixed, a contacting point's slip is just its own
tangential displacement, so **no slip-history field is needed** — the law is a smooth
(C⁰) function of `u` and stays differentiable. Per drag step the run records the
friction force (rises then plateaus), the normal force `N` (≈ `W`), the mean bottom
slip, the slipping-area fraction, and the **cumulative frictional work** dissipated in
steady slip. → `data/fem_friction.npz`.

### The Newton side (`newton_run/run_friction.py`, all three solvers)

`run_friction.py --solver xpbd|vbd|semi_implicit` drags the block on the shared
ground-contact scene (default XPBD; VBD/SemiImplicit version-gated `TODO[verify-on-colab]`).
Coulomb friction is set via `soft_contact_mu`, the top face is pinned and dragged.
As with normal contact, the fast positional **XPBD exposes no calibrated tangential
force** — so the axis here is the kinematic response: mean bottom slip vs. top drag (the
stick-then-slip knee) and the strain energy / residual KE. The FEM run supplies the force
and the dissipated work that XPBD cannot. `compare/friction.py` overlays them;
[`40_friction.ipynb`](../40_friction.ipynb) is the walkthrough.

---

## 2. Material test — confined uniaxial stress vs. stretch

A homogeneous constitutive check, independent of geometry: a small cube
(0.4³ m, `STRESS_DIM = 4×4×4`) is driven through `F = diag(1, 1, λ)` by prescribing
the affine displacement `u = (F − I)X` on the **entire** boundary (lateral motion
confined → uniaxial *strain*). λ is swept over **17 points in [0.7, 1.5]**
(compression to tension), into the large-strain regime where Neo-Hookean, StVK and a
fast solver's response diverge.

### The analytic anchor (closed form, exact)

For confined uniaxial strain the axial 1st Piola–Kirchhoff stress is closed-form
(`energies.neohookean_uniaxial_strain_stress`):

```
P_zz(λ) = μ (λ − 1/λ) + λ_lame · ln(λ) / λ,     P_zz(1) = 0,
small-strain slope = λ_lame + 2μ   (the oedometric modulus).
```

This is **the** ground truth here, and it is the one backed to machine precision by
`tests/test_energies.py` (see [METHOD.md](METHOD.md) §4).

### Both sides

- **FEM** (`fenics_run/run_stress_strain.py`): prescribe the affine boundary, assemble
  the block's volume-averaged `P_zz`, compare to the closed form. The run prints the
  **max relative deviation vs. analytic** — a direct measure of constitutive fidelity
  into large strain. → `data/fem_stress_strain.npz`.
- **Newton** (`newton_run/run_stress_strain.py`): the **SemiImplicit** solver, all
  boundary nodes pinned to their affine positions, interior settled (gravity off);
  read the volume-averaged axial stress. **Honest scope:** with a fully prescribed
  affine boundary this primarily checks that Newton *reproduces the prescribed
  homogeneous deformation* across the strain range — the interior should settle to the
  affine field. The rigorous measure of Newton's effective *constitutive* response is
  the θ\* fit (§4) and the equilibrium residual, not this test alone.

---

## 3. Convergence — does more compute close the gap?

The study that separates **discretisation error** (FEM) from **solver-budget error**
(XPBD), both against the analytic bar and each other. → `data/{newton,fem}_convergence.npz`.

### FEM (`fenics_run/convergence.py`)

1. **Mesh h-refinement** on structured hex meshes (`CONV_FEM_NZ = (4, 8, 12, 16, 24)`,
   cells kept ~cubic; hex chosen because it locks less → cleaner convergence). Tip
   drop and strain energy converge to a **mesh-independent limit**; the gap from that
   limit to the analytic 1-D bar is the genuine 3-D/Poisson correction.
2. **Load-increment sweep** at the finest mesh (`CONV_FEM_LOADSTEPS = (1, 2, 4, 8)`):
   the *converged* tip drop must be independent of the number of gravity increments
   (they affect only Newton-Raphson robustness / total iterations) — a direct
   self-consistency check on the nonlinear solve.

Records `h`, #DOFs, tip, strain, wall time per resolution.

### Newton XPBD (`newton_run/convergence.py`)

XPBD's effective stiffness depends on its iteration and substep budget, so the sweep
makes that explicit:

- iterations `(1, 2, 4, 8, 16, 32)` at fixed 32 substeps, and
- substeps `(4, 8, 16, 32, 64)` at fixed 10 iterations.

For each it records tip drop, strain energy, the **free-node equilibrium-residual RMS**
(the cleanest "have we converged?" measure — distance from the true force balance),
and wall time. As the budget grows the residual falls and the tip approaches the
implicit-FEM / analytic value: **XPBD converges toward the true equilibrium, at a
cost.** The notebook [`30_convergence.ipynb`](../30_convergence.ipynb) overlays both
halves against the analytic and FEM tip references.

---

## 4. The differentiable θ\* fit

Newton is built on NVIDIA Warp, so the whole simulation is **differentiable**: we can
backpropagate a loss through the settling of the soft body and get exact gradients
w.r.t. material parameters — something a black-box solver cannot give. Used two ways
against the FEM reference (`newton_run/diffsim.py`):

1. **Sensitivity** (one forward + backward, robust): with
   `loss(θ) = ‖q_newton_settled(θ) − q_fem‖²` over all nodes and θ scaling the Lamé
   parameters, `dLoss/dθ` is the exact sensitivity of the Newton–FEM mismatch to
   stiffness — computed in a single autodiff pass via `wp.Tape`.
2. **Inverse fit** (SGD): optimise θ so Newton's settled shape matches FEM. The fitted
   **θ\* is an effective-stiffness multiplier** — θ\* > 1 means Newton is effectively
   *softer* than the true material (needs stiffening to match FEM); θ\* < 1 means
   stiffer. This turns "the model looks a bit soft" into a number.

**Honest scope (important).** The fit runs on the differentiable **SemiImplicit**
solver — Warp differentiates that integrator, not XPBD's positional projection. So
**θ\* characterises the SemiImplicit solver's effective stiffness vs. FEM, not XPBD's.**
XPBD's softness is measured separately and directly, by the equilibrium residual and
the tip ratio in the convergence study and `compare/hanging_bar`. This piece is also
the most Warp-version- and autodiff-specific (`finalize(requires_grad=True)`,
`warp.optim.SGD`, learning rate); it is marked `TODO[verify-on-colab]` and the learning
rate in particular needs tuning. See [STATUS.md](STATUS.md).

---

See [METHOD.md](METHOD.md) for the shared problem, material and diagnostics, and
[CONTACT.md](CONTACT.md) for the indentation and drop contact scenarios.
