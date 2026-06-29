"""Single source of truth for the Newton-vs-FEniCSx comparison.

Both the Newton (XPBD) run and the FEniCSx (FEM) run import their geometry,
material and gravity from here, so the two simulations describe *exactly* the
same physical problem and only differ in the solver.

Stage A = "hanging soft block":
    A soft block is fixed on its TOP face (max-z) and hangs under its own
    weight. Gravity (-z) stretches it. This is a stable, small-strain (~3 %)
    benchmark with a closed-form axial reference solution, so we can compare
    Newton's fast XPBD projection against an accurate implicit FEM solve AND
    against analytic beam theory.

The material is the same St-Venant-Kirchhoff / co-rotational tetrahedral model
that Newton's `add_soft_grid` uses, parameterised by Lame mu (k_mu) and
Lame lambda (k_lambda). FEniCSx's compressible Neo-Hookean uses the SAME
(mu, lambda) directly; we also report (E, nu) for reference.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Paths (everything resolved relative to the repository root)
# --------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIG_DIR = os.path.join(REPO_ROOT, "figures")
# Shared artefacts written by the Newton run and consumed downstream
MESH_NPZ = os.path.join(DATA_DIR, "mesh.npz")            # rest mesh + fixed nodes
NEWTON_NPZ = os.path.join(DATA_DIR, "newton_result.npz")  # XPBD settled state
FEM_NPZ = os.path.join(DATA_DIR, "fem_result.npz")        # FEniCSx static solution (tet, shared mesh)
FEM_HEX_NPZ = os.path.join(DATA_DIR, "fem_result_hex.npz")  # FEniCSx static solution (independent hex mesh)
FEM_STAGEB_NPZ = os.path.join(DATA_DIR, "fem_stage_b.npz")  # Stage B contact result (tet + hex)
NEWTON_STAGEB_NPZ = os.path.join(DATA_DIR, "newton_stage_b.npz")  # Stage B Newton XPBD contact result
DIFFSIM_NPZ = os.path.join(DATA_DIR, "diffsim_stage_a.npz")  # differentiable stiffness fit (Stage A)
NEWTON_DROP_NPZ = os.path.join(DATA_DIR, "newton_drop.npz")  # dynamic drop, Newton XPBD
FEM_DROP_NPZ = os.path.join(DATA_DIR, "fem_drop.npz")        # dynamic drop, FEM (Newmark)
NEWTON_STRESS_NPZ = os.path.join(DATA_DIR, "newton_stress_strain.npz")  # uniaxial material test
FEM_STRESS_NPZ = os.path.join(DATA_DIR, "fem_stress_strain.npz")        # uniaxial material test
NEWTON_SEMI_NPZ = os.path.join(DATA_DIR, "newton_result_semi.npz")  # Stage A, explicit (SemiImplicit) solver
NEWTON_CONV_NPZ = os.path.join(DATA_DIR, "newton_convergence.npz")  # Stage A convergence (XPBD iters/substeps)
FEM_CONV_NPZ = os.path.join(DATA_DIR, "fem_convergence.npz")        # Stage A convergence (mesh + load steps)
FEM_FRICTION_NPZ = os.path.join(DATA_DIR, "fem_friction.npz")       # sliding-block Coulomb friction (FEM)
NEWTON_FRICTION_NPZ = os.path.join(DATA_DIR, "newton_friction.npz")  # sliding-block friction (Newton XPBD)

# --------------------------------------------------------------------------
# Geometry  (block built from a regular grid of hexahedral cells -> tets)
# Long axis is z so the block hangs and stretches vertically.
# --------------------------------------------------------------------------
GRID_DIM_X = 6      # cells in x
GRID_DIM_Y = 6      # cells in y
GRID_DIM_Z = 16     # cells in z (vertical, the hanging direction)
CELL = 0.1          # cell edge length [m]  -> block = 0.6 x 0.6 x 1.6 m

# Origin (min corner) of the block in world space
ORIGIN = (0.0, 0.0, 0.0)

# Which face is clamped. "top" = the max-z face is held fixed; the block hangs.
FIXED_FACE = "top"

BLOCK_LX = GRID_DIM_X * CELL
BLOCK_LY = GRID_DIM_Y * CELL
BLOCK_LZ = GRID_DIM_Z * CELL          # hanging length L

# --------------------------------------------------------------------------
# Material  (identical numbers to Newton's rigid_soft_contact example)
# --------------------------------------------------------------------------
DENSITY = 100.0       # [kg/m^3]
K_MU = 1.0e4          # Lame mu  (shear modulus) [Pa]
K_LAMBDA = 5.0e4      # Lame lambda             [Pa]
K_DAMP = 1.0          # internal damping coefficient


def lame_to_E_nu(mu: float = K_MU, lam: float = K_LAMBDA) -> tuple[float, float]:
    """Convert Lame parameters (mu, lambda) to Young's modulus E and Poisson nu."""
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    return E, nu


YOUNGS_E, POISSON_NU = lame_to_E_nu()

# --------------------------------------------------------------------------
# Physics / simulation
# --------------------------------------------------------------------------
GRAVITY = 9.80665     # [m/s^2], along -z; set explicitly on the Newton builder
                      # (gravity=-GRAVITY) so it matches FEM (Newton's default is -9.81)

# Newton XPBD time stepping
FPS = 60
SIM_SUBSTEPS = 32
XPBD_ITERATIONS = 10
# Explicit (SemiImplicit, force-based) solver -- the SECOND Newton solver we run
# in Stage A so both solver families can be checked against FEM + analytic.
# CFL for this soft block: c = sqrt((lambda+2mu)/rho) ~ 26 m/s, cell 0.1 m ->
# dt < ~3.8 ms; (1/60)/32 ~ 0.52 ms is comfortably stable.
SIM_SUBSTEPS_EXPLICIT = 32
MAX_FRAMES = 600                 # hard cap on the settling loop
SETTLE_KE_TOL = 1.0e-6           # kinetic energy threshold to call it settled
MIN_SETTLE_FRAMES = 60           # never stop before this many frames

# Tolerance (as a fraction of CELL) for picking nodes that lie on a face
FACE_TOL_FRAC = 0.25

# --------------------------------------------------------------------------
# Stage B -- rigid-sphere indentation (contact prototype)
# A flat slab is clamped on its bottom face; a rigid analytic sphere is pushed
# into the top face by a prescribed indentation depth, with penalty contact.
# Wide & flat so the Hertz half-space solution is a reasonable analytic anchor.
# --------------------------------------------------------------------------
STAGEB_DIM = (20, 20, 8)        # cells (nx, ny, nz)
STAGEB_CELL = 0.05              # cell edge [m]  -> slab 1.0 x 1.0 x 0.4 m
STAGEB_SPHERE_R = 0.30          # rigid sphere radius [m]
STAGEB_INDENT_MAX = 0.04        # max indentation depth [m]  (small vs R)
STAGEB_LOAD_STEPS = 20          # incremental indentation steps
STAGEB_WITH_GRAVITY = False     # isolate contact for the Hertz comparison

# Contact variants run in Stage B: (element, penalty factor, method).
#   kn = factor * E / cell  [Pa/m]
#   method = "penalty"        -> single penalty solve per step (kn-dependent)
#            "aug_lagrangian" -> Uzawa outer loop drives penetration -> 0 at modest kn
# Includes a TET penalty-strength sweep, a HEX reference, and an Augmented
# Lagrangian variant at the same modest kn as the soft penalty (direct contrast).
STAGEB_VARIANTS = (
    ("tet", 5.0, "penalty"),
    ("tet", 50.0, "penalty"),
    ("hex", 10.0, "penalty"),
    ("tet", 5.0, "aug_lagrangian"),
)
STAGEB_AUG_ITERS = 8            # Uzawa multiplier updates per indentation step
STAGEB_AUG_PEN_TOL = 1.0e-5     # stop Uzawa when max penetration < this [m]


# --------------------------------------------------------------------------
# Dynamic drop -- the literal `rigid_soft_contact`-style scenario:
# a soft block rests on the ground and a rigid sphere is dropped onto it.
# Gravity ON; impact is stabilised with ~10% Kelvin-Voigt contact damping.
# --------------------------------------------------------------------------
DROP_DIM = (8, 8, 5)            # block cells -> 0.8 x 0.8 x 0.5 m, bottom on ground (z=0)
DROP_CELL = 0.1
DROP_SPHERE_R = 0.25            # rigid sphere radius [m]
DROP_SPHERE_DENSITY = 500.0    # sphere density [kg/m^3]
DROP_SPHERE_Z0 = 1.10          # initial sphere centre height [m] (~0.35 m free fall)
DROP_DT = 1.0e-3               # FEM time step [s]
DROP_STEPS = 400               # FEM steps (0.4 s)
DROP_DURATION = DROP_DT * DROP_STEPS
DROP_PENALTY_FACTOR = 20.0     # contact kn = factor * E / cell  [Pa/m]
DROP_DAMP_FRAC = 0.10          # ~10% Kelvin-Voigt contact viscous damping (cd = frac*kn*dt)


# --------------------------------------------------------------------------
# Uniaxial-strain material test (effective stress-strain curve)
# A small block is driven through F = diag(1, 1, lambda) via an affine boundary;
# the macroscopic axial stress sigma(lambda) tests constitutive fidelity into the
# large-strain regime (where Neo-Hookean, StVK and XPBD's response diverge).
# --------------------------------------------------------------------------
STRESS_DIM = (4, 4, 4)         # small cube, 0.4^3 m (homogeneous test, size irrelevant)
STRESS_CELL = 0.1
STRESS_LAMBDA_MIN = 0.7        # compression
STRESS_LAMBDA_MAX = 1.5        # tension
STRESS_LAMBDA_N = 17


def stress_lambdas():
    import numpy as np
    return np.linspace(STRESS_LAMBDA_MIN, STRESS_LAMBDA_MAX, STRESS_LAMBDA_N)


# --------------------------------------------------------------------------
# (4) Convergence study (Stage A hanging block)
# Shows discretisation error vs solver error, and XPBD's iteration/substep
# dependent effective stiffness, against the FEM and analytic references.
# --------------------------------------------------------------------------
# Newton (XPBD): sweep solver iterations and substeps. More iterations / smaller
# substeps -> the positional projection approaches the true static equilibrium.
CONV_XPBD_ITERS = (1, 2, 4, 8, 16, 32)        # swept at CONV_XPBD_FIXED_SUBSTEPS
CONV_XPBD_SUBSTEPS = (4, 8, 16, 32, 64)       # swept at CONV_XPBD_FIXED_ITERS
CONV_XPBD_FIXED_SUBSTEPS = 32
CONV_XPBD_FIXED_ITERS = 10

# FEM: structured-mesh h-refinement (cells along the long z axis; nx, ny derived
# to keep cells ~cubic) and a Newton load-increment sweep at the finest mesh.
CONV_FEM_NZ = (4, 8, 12, 16, 24)
CONV_FEM_LOADSTEPS = (1, 2, 4, 8)
CONV_FEM_ELEMENT = "hex"          # hex8: less locking -> cleaner h-convergence


def conv_fem_resolutions():
    """List of (nx, ny, nz) structured-mesh resolutions for the FEM h-sweep.

    nx, ny follow nz scaled by the block aspect ratio so the cells stay ~cubic.
    """
    out = []
    for nz in CONV_FEM_NZ:
        nx = max(2, round(nz * BLOCK_LX / BLOCK_LZ))
        ny = max(2, round(nz * BLOCK_LY / BLOCK_LZ))
        out.append((int(nx), int(ny), int(nz)))
    return out


# --------------------------------------------------------------------------
# Friction -- sliding soft block on a rigid floor (Coulomb friction)
# A flat block rests on the floor (z = 0) under gravity; its TOP face is dragged
# tangentially (+x). Coulomb friction at the floor resists: the tangential force
# rises (stick) then plateaus at mu * N (slip). With the normal load supplied by
# gravity, N = weight = rho*g*V, so the plateau mu*rho*g*V is an analytic anchor.
# --------------------------------------------------------------------------
FRICTION_DIM = (8, 8, 4)          # cells -> 0.8 x 0.8 x 0.4 m flat block
FRICTION_CELL = 0.1
FRICTION_MU = 0.3                 # Coulomb friction coefficient
FRICTION_DRAG_MAX = 0.02          # max tangential drag of the top face [m]
FRICTION_STEPS = 20               # incremental drag steps
FRICTION_KN_FACTOR = 20.0         # normal penalty   kn = factor * E / cell  [Pa/m]
FRICTION_KT_FACTOR = 20.0         # tangential penalty kt = factor * E / cell [Pa/m]


def friction_block_weight():
    """Total weight of the friction block  W = rho * g * V  [N]."""
    nx, ny, nz = FRICTION_DIM
    vol = (nx * FRICTION_CELL) * (ny * FRICTION_CELL) * (nz * FRICTION_CELL)
    return DENSITY * GRAVITY * vol


def coulomb_plateau(mu: float = FRICTION_MU):
    """Analytic steady-slip friction force  F = mu * W  [N] (Coulomb limit)."""
    return mu * friction_block_weight()


def hertz_force(delta, R: float = STAGEB_SPHERE_R,
                E: float = YOUNGS_E, nu: float = POISSON_NU):
    """Hertzian normal force for a rigid sphere on an elastic half-space.

        F = (4/3) E* sqrt(R) delta^(3/2),     1/E* = (1 - nu^2)/E   (rigid indenter)

    Valid for small strain, frictionless, a << R, half-space. Our slab is finite
    and the material is soft, so this is an approximate anchor, not ground truth.
    """
    import numpy as np

    Estar = E / (1.0 - nu * nu)
    return (4.0 / 3.0) * Estar * np.sqrt(R) * np.maximum(np.asarray(delta, float), 0.0) ** 1.5


def analytic_hanging_displacement(z, z_top: float, L: float,
                                  rho: float = DENSITY, g: float = GRAVITY,
                                  E: float = YOUNGS_E):
    """Closed-form *axial* self-weight solution for a 1-D hanging bar.

    A bar of length L is fixed at z_top and hangs downward. For a material
    point originally at height ``z`` the downward displacement is

        u(s) = (rho*g / E) * (L*s - s^2 / 2),   s = z_top - z   (depth below clamp)

    Tip (s = L) elongation = rho*g*L^2 / (2 E).

    This ignores Poisson contraction and 3-D effects, so it is a sanity anchor,
    not the ground truth -- the FEM solve is. Returns downward displacement
    magnitude (positive = moved in -z).
    """
    import numpy as np

    s = z_top - np.asarray(z, dtype=float)
    return (rho * g / E) * (L * s - 0.5 * s * s)


def summary() -> str:
    E, nu = lame_to_E_nu()
    tip = analytic_hanging_displacement(ORIGIN[2], ORIGIN[2] + BLOCK_LZ, BLOCK_LZ)
    return (
        f"Block            : {BLOCK_LX:.2f} x {BLOCK_LY:.2f} x {BLOCK_LZ:.2f} m "
        f"({GRID_DIM_X}x{GRID_DIM_Y}x{GRID_DIM_Z} cells)\n"
        f"Density          : {DENSITY:.1f} kg/m^3\n"
        f"Lame (mu, lambda): {K_MU:.3e}, {K_LAMBDA:.3e} Pa\n"
        f"-> Young's E, nu : {E:.3e} Pa, {nu:.4f}\n"
        f"Gravity          : {GRAVITY:.5f} m/s^2 (-z)\n"
        f"Analytic tip elongation (1-D bar): {tip * 1000:.2f} mm\n"
    )


if __name__ == "__main__":
    print(summary())
