"""Stage A -- Newton side.

Builds the hanging soft block, simulates it until it settles under gravity, and
records the rest mesh, the settled deformed node positions and the
tip-displacement-over-time history.

Two Newton solvers can be used, so BOTH solver families can be checked against
the FEM reference and analytic bar theory:

    python -m newton_run.run_stage_a                       # XPBD (positional) -- default
    python -m newton_run.run_stage_a --solver semi_implicit  # explicit, force-based
    python -m newton_run.run_stage_a --device cpu

The XPBD run is the canonical one: it writes the shared mesh (data/mesh.npz) that
the FEM side consumes, plus data/newton_result.npz. The explicit run writes
data/newton_result_semi.npz on the SAME grid (node-for-node comparable) but does
not overwrite the shared mesh.

NOTE: a few Warp/Newton API points are marked TODO[verify-on-colab]. They match
the public examples, but exact attribute names can shift between Newton versions.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from common import params
from common import mesh_io
from compare import energies as en


def build_model(builder_cls):
    """Create the soft-grid block. Geometry/material come from params."""
    # gravity set explicitly from params (Newton's default is -9.81; we match FEM)
    builder = builder_cls(gravity=-params.GRAVITY)
    builder.default_particle_radius = 0.01
    builder.particle_max_velocity = 50.0

    import warp as wp

    builder.add_soft_grid(
        pos=wp.vec3(*params.ORIGIN),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=params.GRID_DIM_X,
        dim_y=params.GRID_DIM_Y,
        dim_z=params.GRID_DIM_Z,
        cell_x=params.CELL,
        cell_y=params.CELL,
        cell_z=params.CELL,
        density=params.DENSITY,
        k_mu=params.K_MU,
        k_lambda=params.K_LAMBDA,
        k_damp=params.K_DAMP,
    )
    return builder.finalize()


def pick_fixed_nodes(rest_q: np.ndarray) -> np.ndarray:
    """Indices of nodes lying on the clamped face (default: top / max-z)."""
    tol = params.FACE_TOL_FRAC * params.CELL
    z = rest_q[:, 2]
    if params.FIXED_FACE == "top":
        return np.where(z > z.max() - tol)[0]
    if params.FIXED_FACE == "bottom":
        return np.where(z < z.min() + tol)[0]
    raise ValueError(f"unsupported FIXED_FACE={params.FIXED_FACE!r}")


def pin_nodes(model, fixed_nodes: np.ndarray) -> None:
    """Clamp nodes by zeroing their inverse mass (immovable in XPBD)."""
    import warp as wp

    # TODO[verify-on-colab]: attribute name `particle_inv_mass` (warp.sim convention)
    inv_mass = model.particle_inv_mass.numpy()
    inv_mass[fixed_nodes] = 0.0
    model.particle_inv_mass = wp.array(inv_mass, dtype=wp.float32, device=model.device)


def kinetic_energy(model, state) -> float:
    mass = model.particle_mass.numpy()
    vel = state.particle_qd.numpy()
    return float(0.5 * np.sum(mass * np.sum(vel * vel, axis=1)))


def pick_tip_node(rest_q: np.ndarray) -> int:
    """Free-end node closest to the block's central axis (for the time series)."""
    cx = 0.5 * (rest_q[:, 0].min() + rest_q[:, 0].max())
    cy = 0.5 * (rest_q[:, 1].min() + rest_q[:, 1].max())
    zmin = rest_q[:, 2].min()
    tol = params.FACE_TOL_FRAC * params.CELL
    bottom = np.where(rest_q[:, 2] < zmin + tol)[0]
    d = (rest_q[bottom, 0] - cx) ** 2 + (rest_q[bottom, 1] - cy) ** 2
    return int(bottom[np.argmin(d)])


def _make_solver(solver_name, model, iterations):
    """Map a solver name to a Newton solver instance."""
    import newton

    if solver_name == "xpbd":
        return newton.solvers.SolverXPBD(model=model, iterations=iterations)
    if solver_name in ("semi_implicit", "explicit", "semi"):
        # explicit, force-based integrator (the one Warp can differentiate)
        return newton.solvers.SolverSemiImplicit(model)
    raise ValueError(f"unknown solver {solver_name!r}")


def simulate(solver_name="xpbd", iterations=params.XPBD_ITERATIONS,
             substeps=params.SIM_SUBSTEPS, max_frames=params.MAX_FRAMES,
             settle_tol=params.SETTLE_KE_TOL, min_frames=params.MIN_SETTLE_FRAMES,
             device=None, verbose=True):
    """Run one hanging-block settle and return a result dict (numpy).

    Caller is responsible for ``wp.init()`` (so this can be looped cheaply by the
    convergence study). Returns rest/settled positions, mesh, history, timing.
    """
    import warp as wp
    import newton

    device = device or str(wp.get_device())
    with wp.ScopedDevice(device):
        model = build_model(newton.ModelBuilder)

        rest_q = model.particle_q.numpy().astype(np.float64)
        tet_indices = model.tet_indices.numpy().astype(np.int64)
        fixed_nodes = pick_fixed_nodes(rest_q)
        tip = pick_tip_node(rest_q)
        if verbose:
            print(f"[newton] nodes={len(rest_q)} tets={len(tet_indices)} "
                  f"fixed={len(fixed_nodes)} tip_node={tip}")

        pin_nodes(model, fixed_nodes)

        solver = _make_solver(solver_name, model, iterations)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        frame_dt = 1.0 / params.FPS
        sim_dt = frame_dt / substeps

        history = []  # (time, tip_z, kinetic_energy, strain_energy)
        z_top = rest_q[fixed_nodes, 2].mean()

        wp.synchronize()
        t0 = time.perf_counter()
        for frame in range(max_frames):
            for _ in range(substeps):
                state_0.clear_forces()
                # No contacts in Stage A, but collide() is cheap and keeps the
                # loop identical in shape to Stage B.
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0

            q_now = state_0.particle_q.numpy()
            ke = kinetic_energy(model, state_0)
            tip_z = float(q_now[tip, 2])
            u_strain = en.strain_energy(rest_q, q_now, tet_indices)
            history.append((frame * frame_dt, tip_z, ke, u_strain))
            if verbose and frame % 20 == 0:
                print(f"  frame {frame:4d}  tip_z={tip_z:+.5f}  KE={ke:.3e}")
            if frame >= min_frames and ke < settle_tol:
                if verbose:
                    print(f"[newton] settled at frame {frame} (KE={ke:.3e})")
                break

        wp.synchronize()
        wall_time = time.perf_counter() - t0
        final_q = state_0.particle_q.numpy().astype(np.float64)

    return dict(
        rest_q=rest_q, final_q=final_q, tet_indices=tet_indices,
        fixed_nodes=fixed_nodes, tip_node=tip,
        history=np.array(history, dtype=np.float64), z_top=z_top,
        wall_time=wall_time, n_substeps=substeps, iterations=iterations,
        solver=solver_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Newton hanging-block (Stage A)")
    parser.add_argument("--device", default=None, help="warp device, e.g. cuda:0 or cpu")
    parser.add_argument("--solver", choices=["xpbd", "semi_implicit"], default="xpbd",
                        help="xpbd = positional (default, writes shared mesh); "
                             "semi_implicit = explicit force-based solver")
    args = parser.parse_args()

    import warp as wp

    wp.init()
    device = args.device or str(wp.get_device())
    print(f"[newton] device = {device}, solver = {args.solver}")
    print(params.summary())

    substeps = params.SIM_SUBSTEPS if args.solver == "xpbd" else params.SIM_SUBSTEPS_EXPLICIT
    res = simulate(solver_name=args.solver, iterations=params.XPBD_ITERATIONS,
                   substeps=substeps, device=device, verbose=True)

    rest_q, final_q = res["rest_q"], res["final_q"]
    tip = res["tip_node"]

    # ---- persist results -------------------------------------------------
    os.makedirs(params.DATA_DIR, exist_ok=True)
    if args.solver == "xpbd":
        # the canonical run owns the shared mesh consumed by the FEM side
        mesh_io.save_mesh(params.MESH_NPZ, rest_q, res["tet_indices"], res["fixed_nodes"])
        out = params.NEWTON_NPZ
        print(f"[newton] wrote {params.MESH_NPZ}")
    else:
        out = params.NEWTON_SEMI_NPZ

    np.savez(
        out,
        rest_q=rest_q,
        final_q=final_q,
        tet_indices=res["tet_indices"],
        fixed_nodes=res["fixed_nodes"],
        tip_node=tip,
        history=res["history"],
        z_top=res["z_top"],
        wall_time=res["wall_time"],
        n_substeps=res["n_substeps"],
        xpbd_iterations=res["iterations"],
        solver=args.solver,
    )
    tip_drop = (rest_q[tip, 2] - final_q[tip, 2]) * 1000.0
    print(f"[newton] wrote {out}")
    print(f"[newton] tip vertical drop = {tip_drop:.2f} mm")
    print(f"[newton] solve wall time = {res['wall_time']:.3f} s "
          f"(solver={args.solver}, substeps={substeps}, iters={params.XPBD_ITERATIONS})")


if __name__ == "__main__":
    main()
