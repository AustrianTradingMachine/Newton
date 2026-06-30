"""Effective stress-strain (3) -- FEM side, confined uniaxial strain.

A small block is driven through the homogeneous deformation F = diag(1, 1, lambda)
by prescribing the affine displacement u = (F - I) X on the *entire* boundary
(lateral motion confined -> uniaxial strain). For each lambda we assemble the
block's own volume-averaged axial 1st Piola stress and compare it to the closed
form. This verifies that the FEM Neo-Hookean reproduces the exact constitutive
law into the large-strain regime (where the hanging bar's small-strain equivalence ends).

-> data/fem_stress_strain.npz : lambdas, sigma_fem, sigma_analytic

Run (needs dolfinx):  python -m fenics_run.run_stress_strain
"""

from __future__ import annotations

import os

import numpy as np

from common import params
from compare import energies as en
from fenics_run.run_hanging_bar import fem_snes_options, solve_status


def main():
    import dolfinx
    import ufl
    from dolfinx import default_scalar_type, fem
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.mesh import CellType, create_box, exterior_facet_indices, locate_dofs_topological
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    print(f"[stress-fem] dolfinx {dolfinx.__version__}")

    nx, ny, nz = params.STRESS_DIM
    h = params.STRESS_CELL
    Lx, Ly, Lz = nx * h, ny * h, nz * h
    msh = create_box(comm, [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
                     [nx, ny, nz], CellType.hexahedron)
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u = fem.Function(V, name="u")
    w = ufl.TestFunction(V)

    # affine displacement on the whole boundary: u = (F - I) X, F = diag(1,1,lambda)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    bfacets = exterior_facet_indices(msh.topology)
    bdofs = locate_dofs_topological(V, fdim, bfacets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, bdofs)

    lam_c = fem.Constant(msh, default_scalar_type(1.0))
    X = ufl.SpatialCoordinate(msh)
    bc_expr = fem.Expression(ufl.as_vector((0.0 * X[0], 0.0 * X[1], (lam_c - 1.0) * X[2])),
                             V.element.interpolation_points)

    # fem.Constant, not bare numpy scalars: numpy hijacks np.float64 * (UFL tensor) into
    # an object array, which breaks the ufl.inner(grad(w), P) form below. (Same fix as the
    # drop run; the hanging-bar / indentation FEM runs wrap these the same way.) Note lam
    # here is Lame lambda -- distinct from lam_c, the stretch control above.
    mu = fem.Constant(msh, default_scalar_type(params.K_MU))
    lam = fem.Constant(msh, default_scalar_type(params.K_LAMBDA))
    d = msh.geometry.dim
    F = ufl.Identity(d) + ufl.grad(u)
    J = ufl.det(F)
    Finv_T = ufl.inv(F).T
    P = mu * (F - Finv_T) + lam * ufl.ln(J) * Finv_T
    residual = ufl.inner(ufl.grad(w), P) * ufl.dx

    problem = NonlinearProblem(residual, u, bcs=[bc], petsc_options_prefix="stress_",
                               petsc_options=fem_snes_options(rtol=1.0e-8, max_it=50))

    vol = Lx * Ly * Lz
    stress_form = fem.form(P[2, 2] * ufl.dx)

    lambdas = params.stress_lambdas()
    sigma_fem = []
    for L in lambdas:
        lam_c.value = float(L)
        u_bc.interpolate(bc_expr)
        solve_status(problem, u)
        sig = comm.allreduce(fem.assemble_scalar(stress_form), op=MPI.SUM) / vol
        sigma_fem.append(sig)
        print(f"[stress-fem] lambda={L:.3f}  sigma={sig:.4g} Pa")

    sigma_fem = np.array(sigma_fem)
    sigma_ana = en.neohookean_uniaxial_strain_stress(lambdas)
    os.makedirs(params.DATA_DIR, exist_ok=True)
    np.savez(params.FEM_STRESS_NPZ, lambdas=lambdas, sigma_fem=sigma_fem, sigma_analytic=sigma_ana)
    rel = np.max(np.abs(sigma_fem - sigma_ana) / (np.abs(sigma_ana) + 1.0))
    print(f"[stress-fem] wrote {params.FEM_STRESS_NPZ}  (max rel. dev. vs analytic = {rel:.2e})")


if __name__ == "__main__":
    main()
