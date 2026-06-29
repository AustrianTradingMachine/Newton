"""Energy diagnostics shared by both solvers (pure numpy).

The strain energy is evaluated from node positions + tetrahedra with the SAME
compressible Neo-Hookean density that dolfinx uses, so Newton and FEM(tet) -- which
share the Stage A mesh -- are compared on identical footing:

    F   = D_s D_m^{-1}        per-tet deformation gradient
    I_C = tr(F^T F)           first invariant
    J   = det F
    psi = mu/2 (I_C - 3) - mu ln J + lambda/2 (ln J)^2
    E   = sum_tets  psi * V_rest

Also provides nodal masses (from the rest mesh + density), gravitational
potential energy and kinetic energy.
"""

from __future__ import annotations

import numpy as np

from common import params


def _edge_matrices(q, tets):
    """Per-tet 3x3 edge matrix [x1-x0, x2-x0, x3-x0]."""
    x0 = q[tets[:, 0]]
    return np.stack((q[tets[:, 1]] - x0, q[tets[:, 2]] - x0, q[tets[:, 3]] - x0), axis=-1)


def tet_rest_volumes(rest_q, tets):
    return np.abs(np.linalg.det(_edge_matrices(rest_q, tets))) / 6.0


def strain_energy(rest_q, final_q, tets, mu=params.K_MU, lam=params.K_LAMBDA,
                  per_tet=False):
    """Total compressible Neo-Hookean strain energy [J]."""
    Dm = _edge_matrices(rest_q, tets)
    Ds = _edge_matrices(final_q, tets)
    F = Ds @ np.linalg.inv(Dm)
    C = np.transpose(F, (0, 2, 1)) @ F
    Ic = np.trace(C, axis1=1, axis2=2)
    J = np.linalg.det(F)
    Js = np.clip(J, 1e-9, None)                 # guard against inverted tets
    lnJ = np.log(Js)
    psi = 0.5 * mu * (Ic - 3.0) - mu * lnJ + 0.5 * lam * lnJ * lnJ
    vol = np.abs(np.linalg.det(Dm)) / 6.0
    e = psi * vol
    return e if per_tet else float(e.sum())


def _deformation_gradient(rest_q, final_q, tets):
    """Per-tet deformation gradient F, the inverse rest-edge matrix, and rest volume."""
    Dm = _edge_matrices(rest_q, tets)
    Dm_inv = np.linalg.inv(Dm)
    F = _edge_matrices(final_q, tets) @ Dm_inv
    vol = np.abs(np.linalg.det(Dm)) / 6.0
    return F, Dm_inv, vol


# -------------------------------------------------------------------- (2) volume / Jacobian
def jacobians(rest_q, final_q, tets):
    """Per-tet volume ratio J = det(F) (1 = volume preserved, <1 = compressed)."""
    F, _, _ = _deformation_gradient(rest_q, final_q, tets)
    return np.linalg.det(F)


def volume_change(rest_q, final_q, tets):
    """Relative total volume change (V_def - V_rest) / V_rest."""
    F, _, vol = _deformation_gradient(rest_q, final_q, tets)
    v_rest = vol.sum()
    v_def = float((np.linalg.det(F) * vol).sum())
    return (v_def - v_rest) / v_rest


# -------------------------------------------------------------------- (1) equilibrium residual
def nodal_forces(rest_q, final_q, tets, mu=params.K_MU, lam=params.K_LAMBDA):
    """Internal elastic nodal forces f = -dU/dx (compressible Neo-Hookean).

    First Piola-Kirchhoff  P = mu (F - F^-T) + lambda ln(J) F^-T,
    tet node forces  H = -V P Dm^-T  (columns = forces on nodes 1,2,3; node 0 = -sum).
    """
    F, Dm_inv, vol = _deformation_gradient(rest_q, final_q, tets)
    J = np.clip(np.linalg.det(F), 1e-9, None)
    F_invT = np.transpose(np.linalg.inv(F), (0, 2, 1))
    P = mu * (F - F_invT) + (lam * np.log(J))[:, None, None] * F_invT
    H = -vol[:, None, None] * (P @ np.transpose(Dm_inv, (0, 2, 1)))   # (Nt, 3, 3)
    f = np.zeros_like(rest_q, dtype=float)
    np.add.at(f, tets[:, 1], H[:, :, 0])
    np.add.at(f, tets[:, 2], H[:, :, 1])
    np.add.at(f, tets[:, 3], H[:, :, 2])
    np.add.at(f, tets[:, 0], -(H[:, :, 0] + H[:, :, 1] + H[:, :, 2]))
    return f


def equilibrium_residual(rest_q, final_q, tets, fixed_nodes,
                         mu=params.K_MU, lam=params.K_LAMBDA, g=params.GRAVITY):
    """Out-of-balance force  r = f_internal + f_gravity  per node.

    For a true static equilibrium (FEM) r ~ 0 at the free nodes; for XPBD the
    free-node residual measures how far the settled state is from the true
    material's equilibrium. At the clamped nodes -sum(r_z) is the support
    reaction, which should equal the total weight.
    """
    f_int = nodal_forces(rest_q, final_q, tets, mu, lam)
    m = node_masses(rest_q, tets)
    f_grav = np.zeros_like(f_int)
    f_grav[:, 2] = -m * g
    r = f_int + f_grav
    free = np.setdiff1d(np.arange(len(rest_q)), fixed_nodes)
    return {
        "residual": r,
        "free": free,
        "free_norm": float(np.linalg.norm(r[free])),
        "free_rms": float(np.sqrt(np.mean(np.sum(r[free] ** 2, axis=1)))),
        "reaction_z": float(-r[fixed_nodes, 2].sum()),
        "weight": float(m.sum() * g),
    }


# -------------------------------------------------------------------- (3) stress-strain
def neohookean_uniaxial_strain_stress(lam, mu=params.K_MU, lam_lame=params.K_LAMBDA):
    """Analytic axial 1st Piola-Kirchhoff stress for confined uniaxial strain.

    Deformation gradient F = diag(1, 1, lambda) (lateral motion fully confined):
        P_zz(lambda) = mu (lambda - 1/lambda) + lam_lame ln(lambda) / lambda

    P_zz(1) = 0; small-strain slope = lam_lame + 2 mu (the oedometric modulus).
    """
    lam = np.asarray(lam, dtype=float)
    return mu * (lam - 1.0 / lam) + lam_lame * np.log(lam) / lam


def mean_axial_first_piola(rest_q, final_q, tets, mu=params.K_MU, lam=params.K_LAMBDA, axis=2):
    """Volume-averaged axial 1st Piola stress P[axis, axis] over the block [Pa]."""
    F, _, vol = _deformation_gradient(rest_q, final_q, tets)
    J = np.clip(np.linalg.det(F), 1e-9, None)
    F_invT = np.transpose(np.linalg.inv(F), (0, 2, 1))
    P = mu * (F - F_invT) + (lam * np.log(J))[:, None, None] * F_invT
    return float((P[:, axis, axis] * vol).sum() / vol.sum())


def node_masses(rest_q, tets, density=params.DENSITY):
    """Lumped nodal masses: each tet distributes density*V equally to its 4 nodes."""
    vol = tet_rest_volumes(rest_q, tets)
    m = np.zeros(len(rest_q))
    np.add.at(m, tets.reshape(-1), np.repeat(density * vol / 4.0, 4))
    return m


def gravitational_pe(q, masses, g=params.GRAVITY, axis=2):
    """Gravitational potential energy Sum m g z  [J] (z = q[:, axis])."""
    return float(np.sum(masses * g * q[:, axis]))


def kinetic_energy(masses, vel):
    """Kinetic energy 1/2 Sum m |v|^2  [J]."""
    return float(0.5 * np.sum(masses * np.sum(np.asarray(vel) ** 2, axis=1)))


def analytic_hanging_strain_energy(L=params.BLOCK_LZ, area=None,
                                   rho=params.DENSITY, g=params.GRAVITY,
                                   E=params.YOUNGS_E):
    """Strain energy of the 1-D self-weight bar:  U = rho^2 g^2 A L^3 / (6 E).

    A = cross-section area (defaults to the block's x*y footprint).
    """
    if area is None:
        area = params.BLOCK_LX * params.BLOCK_LY
    return rho ** 2 * g ** 2 * area * L ** 3 / (6.0 * E)
