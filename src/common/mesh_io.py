"""Helpers to pass the *identical* tetrahedral mesh between Newton and FEniCSx.

The Newton run is the single producer of the mesh: it builds the soft grid,
finalises the model and exports the rest-state node coordinates, the tet
connectivity and the set of fixed (clamped) node indices. The FEniCSx run
consumes exactly those arrays, so both solvers discretise the same geometry
and any difference in the result is due to the solver, not the mesh.
"""

from __future__ import annotations

import numpy as np


def save_mesh(path: str, rest_q: np.ndarray, tet_indices: np.ndarray,
              fixed_nodes: np.ndarray) -> None:
    np.savez(
        path,
        rest_q=np.asarray(rest_q, dtype=np.float64),
        tet_indices=np.asarray(tet_indices, dtype=np.int64),
        fixed_nodes=np.asarray(fixed_nodes, dtype=np.int64),
    )


def load_mesh(path: str):
    d = np.load(path)
    return d["rest_q"], d["tet_indices"], d["fixed_nodes"]


def signed_tet_volumes(q: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Signed volume of each tetrahedron (positive = right-handed)."""
    x0 = q[tets[:, 0]]
    x1 = q[tets[:, 1]]
    x2 = q[tets[:, 2]]
    x3 = q[tets[:, 3]]
    return np.linalg.det(np.stack((x1 - x0, x2 - x0, x3 - x0), axis=-1)) / 6.0


def orient_tets_positive(q: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return a copy of ``tets`` with every element positively oriented.

    FEM codes (FEniCSx/dolfinx included) expect positively oriented tets.
    Newton's tets are already positively oriented, but we enforce it defensively
    by swapping the last two nodes of any element with negative signed volume.
    """
    tets = np.array(tets, dtype=np.int64, copy=True)
    vol = signed_tet_volumes(q, tets)
    flip = vol < 0.0
    if np.any(flip):
        tets[flip, 2], tets[flip, 3] = tets[flip, 3], tets[flip, 2].copy()
    return tets
