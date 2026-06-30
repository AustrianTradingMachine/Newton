"""Hanging bar -- Newton side.

Builds the hanging soft block, simulates it until it settles under gravity, and
records the rest mesh, the settled deformed node positions and the
tip-displacement-over-time history.

Three Newton solvers can be used, so every solver family can be checked against
the same FEM reference and analytic bar theory on the *same* grid:

    python -m newton_run.run_hanging_bar                        # XPBD: positional projection (default)
    python -m newton_run.run_hanging_bar --solver vbd           # VBD: implicit block coordinate descent
    python -m newton_run.run_hanging_bar --solver semi_implicit # explicit, force-based
    python -m newton_run.run_hanging_bar --device cpu

The XPBD run is the canonical one: it writes the shared mesh (data/mesh.npz) that
the FEM side consumes, plus data/newton_result.npz. The VBD and explicit runs
write data/newton_result_vbd.npz / data/newton_result_semi.npz on the SAME grid
(node-for-node comparable) but do not overwrite the shared mesh.

VBD needs a vertex graph colouring (builder.color()); build_model applies it for
the VBD solver only. Pinning (inverse mass = 0) and the settle loop are shared by
all three solvers.

The Warp/Newton API points used here match the public examples and run on the
pinned Newton/Warp stack.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from common import mesh_io, params
from compare import energies as en


def build_model(builder_cls, color=False):
    """Create the soft-grid block. Geometry/material come from params.

    ``color=True`` runs the particle graph colouring (``builder.color()``) that
    the VBD solver needs -- its block coordinate descent updates one colour of
    independent vertices at a time. It is a no-op for XPBD and the explicit
    solver, so it is only requested for VBD.
    """
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
    if color:
        builder.color()        # vertex graph colouring required by the VBD solver
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
    """Clamp the clamped-face nodes for every solver (zero mass + inv_mass).

    Delegates to the shared pin so XPBD *and* VBD hold the clamp identically -- VBD's
    elasticity solve keys off particle_mass==0, not inv_mass, so an inv_mass-only pin
    let the bar sink under VBD. See newton_run._solver.pin_particles.
    """
    from newton_run._solver import pin_particles

    pin_particles(model, fixed_nodes)


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
    """Map a solver name to a Newton solver instance (shared factory).

    Uses the same factory the contact scenarios use, so the three-solver story is
    identical everywhere. The hanging bar has no collider, so no rigid-particle
    buffer is needed here.
    """
    from newton_run._solver import make_solver

    return make_solver(solver_name, model, iterations)


def simulate(solver_name="xpbd", iterations=params.XPBD_ITERATIONS,
             substeps=params.SIM_SUBSTEPS, max_frames=params.MAX_FRAMES,
             settle_tol=params.SETTLE_KE_TOL, min_frames=params.MIN_SETTLE_FRAMES,
             vel_damp=params.SETTLE_VEL_DAMP, device=None, verbose=True):
    """Run one hanging-block settle and return a result dict (numpy).

    Caller is responsible for ``wp.init()`` (so this can be looped cheaply by the
    convergence study). Returns rest/settled positions, mesh, history, timing.
    """
    import newton
    import warp as wp

    device = device or str(wp.get_device())
    with wp.ScopedDevice(device):
        model = build_model(newton.ModelBuilder, color=(solver_name == "vbd"))

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
                # No contacts in the hanging bar, but collide() is cheap and keeps the
                # loop identical in shape to the contact runs.
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0

            # quasi-static relaxation: drain kinetic energy so the block settles
            # to its STATIC equilibrium instead of ringing under the sudden load
            if vel_damp < 1.0:
                qd = state_0.particle_qd.numpy()
                qd *= vel_damp
                state_0.particle_qd.assign(qd)

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
    parser = argparse.ArgumentParser(description="Newton hanging-bar")
    parser.add_argument("--device", default=None, help="warp device, e.g. cuda:0 or cpu")
    parser.add_argument("--solver", choices=["xpbd", "vbd", "semi_implicit"], default="xpbd",
                        help="xpbd = positional projection (default, writes the shared mesh); "
                             "vbd = implicit block coordinate descent; "
                             "semi_implicit = explicit force-based solver")
    args = parser.parse_args()

    import warp as wp

    wp.init()
    device = args.device or str(wp.get_device())
    print(f"[newton] device = {device}, solver = {args.solver}")
    print(params.summary())

    # per-solver substep / iteration budget. The settled static equilibrium is the
    # same for all three; these only set how each integrator is driven there.
    solver_steps = {
        "xpbd": (params.SIM_SUBSTEPS, params.XPBD_ITERATIONS),
        "vbd": (params.SIM_SUBSTEPS_VBD, params.VBD_ITERATIONS),
        "semi_implicit": (params.SIM_SUBSTEPS_EXPLICIT, params.XPBD_ITERATIONS),
    }
    substeps, iters = solver_steps[args.solver]
    res = simulate(solver_name=args.solver, iterations=iters,
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
    elif args.solver == "vbd":
        out = params.NEWTON_VBD_NPZ
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
        solver_iterations=res["iterations"],
        solver=args.solver,
    )
    tip_drop = (rest_q[tip, 2] - final_q[tip, 2]) * 1000.0
    print(f"[newton] wrote {out}")
    print(f"[newton] tip vertical drop = {tip_drop:.2f} mm")
    print(f"[newton] solve wall time = {res['wall_time']:.3f} s "
          f"(solver={args.solver}, substeps={substeps}, iters={iters})")


if __name__ == "__main__":
    main()
