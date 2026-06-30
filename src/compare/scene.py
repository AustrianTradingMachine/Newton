"""Lightweight 3D scene rendering of a deformed soft body (matplotlib only).

Both the Newton and the FEM runs save the same arrays (rest_q, final_q,
tet_indices), so this renders *either* solver's result identically -- no GPU, no
extra dependency, headless-safe on Colab. It draws the deformed body's surface
coloured by displacement, optionally ghosting the undeformed shape behind it, so a
reader instantly sees *what the scenario is* (a bar stretching, a slab dimpling).

(Newton ships a real-time viewer and dolfinx pairs with PyVista for prettier,
off-screen renders; this keeps the analysis notebooks self-contained.)
"""

from __future__ import annotations

import numpy as np


def boundary_faces(tets):
    """Triangular surface faces: those that belong to exactly one tetrahedron."""
    tets = np.asarray(tets)
    faces = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]],
                            tets[:, [0, 3, 2]], tets[:, [1, 2, 3]]], axis=0)
    key = np.sort(faces, axis=1)
    _uniq, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    return faces[idx[counts == 1]]


def _set_equal_aspect(ax, pts):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ctr = 0.5 * (lo + hi)
    r = max(0.5 * float((hi - lo).max()), 1e-6)
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def render(ax, rest_q, final_q, tets, color_by="uz", cmap="viridis",
           ghost_rest=True, title=None):
    """Draw the deformed surface on a 3D axis, coloured by displacement.

    color_by : 'uz' -> downward displacement [mm];  'mag' -> |u| [mm].
    ghost_rest : draw the undeformed shape faintly behind the deformed one.
    Returns (norm, label) so the caller can add a colorbar.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rest_q = np.asarray(rest_q, float)
    final_q = np.asarray(final_q, float)
    faces = boundary_faces(tets)
    disp = final_q - rest_q
    val = (-disp[:, 2] * 1000.0) if color_by == "uz" else (np.linalg.norm(disp, axis=1) * 1000.0)
    fval = val[faces].mean(axis=1)
    vmin, vmax = float(fval.min()), float(fval.max())
    norm = colors.Normalize(vmin, vmax if vmax > vmin else vmin + 1e-9)

    if ghost_rest:
        ax.add_collection3d(Poly3DCollection(rest_q[faces], facecolor="grey",
                                             alpha=0.06, linewidths=0))
    poly = Poly3DCollection(final_q[faces], alpha=0.95, linewidths=0.15,
                            edgecolor=(0, 0, 0, 0.25))
    poly.set_facecolor(plt.get_cmap(cmap)(norm(fval)))
    ax.add_collection3d(poly)

    _set_equal_aspect(ax, np.vstack([rest_q, final_q]) if ghost_rest else final_q)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    if title:
        ax.set_title(title)
    return norm, ("downward u_z [mm]" if color_by == "uz" else "|u| [mm]")


def add_sphere(ax, center, radius, color="tab:red", n=18, alpha=0.5):
    """Overlay a wireframe sphere (for the indentation / drop contact scenes)."""
    center = np.asarray(center, float)
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, lw=0.4, alpha=alpha)


def frame(ax, pts):
    """Public equal-aspect framing of a 3D axis from a set of points."""
    _set_equal_aspect(ax, np.asarray(pts, float))


def draw_box(ax, lo, hi, color="tab:blue", alpha=0.12, frame_pts=None):
    """Draw a translucent box (e.g. the undeformed slab/block) from its corners.

    Used for *setup schematics* in scenarios whose npz stores only line profiles,
    not the full deformed mesh.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    ax.add_collection3d(Poly3DCollection([v[i] for i in f], facecolor=color,
                                         alpha=alpha, edgecolor="k", linewidths=0.4))
    _set_equal_aspect(ax, v if frame_pts is None else np.asarray(frame_pts, float))


def add_colorbar(fig, ax, norm, label, cmap="viridis"):
    """Attach a colorbar to a 3D scene axis."""
    from matplotlib import cm
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, label=label)
