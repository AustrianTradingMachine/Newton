"""Shared Newton solver factory.

Maps a solver name to a Newton solver instance, so every scenario -- the flagship
hanging bar AND the contact scenarios (indentation, drop, friction) -- selects
XPBD / VBD / SemiImplicit from one place. This is what lets the contact comparison
be apples-to-apples: the implicit VBD is the natural counterpart to the implicit
FEM solve, not just the fast positional XPBD.

All three solvers consume the SAME contact data. Each step calls model.collide(...)
once, filling the shared contacts buffer (model.soft_contact_ke/kd/kf/mu), and
solver.step(..., contacts, dt) reads it -- exactly the wiring Newton's own
examples/multiphysics/example_rigid_soft_contact.py uses when switched between
``--solver xpbd | vbd | semi_implicit`` on a soft grid + rigid sphere + ground.

TODO[verify-on-colab]: the rich VBD soft/rigid-contact path -- VBD integrating rigid
bodies itself via AVBD (two-way coupling for a *free* collider) and the
``rigid_body_particle_contact_buffer_size`` argument -- requires a recent Newton.
The repo's pinned version may predate it (older VBD was cloth/soft self-contact
only), in which case the VBD/SemiImplicit contact runs will error and only XPBD
records a result. That is the honest, version-gated state until a CUDA run confirms
it; see docs/CONTACT.md.
"""

from __future__ import annotations

SOLVERS = ("xpbd", "vbd", "semi_implicit")


def pin_particles(model, fixed_nodes) -> None:
    """Clamp the given particles so EVERY solver treats them as immovable.

    The subtlety that makes VBD work: XPBD reads ``particle_inv_mass`` everywhere, but
    VBD's elasticity solve fixes a vertex only when ``particle_mass == 0`` -- it ignores
    inv_mass there. Zeroing inv_mass alone pins the clamp for XPBD but NOT for VBD (the
    clamped face is then dragged by neighbour forces and the body sinks under gravity
    without ever settling). So we zero BOTH mass and inv_mass -- the same pin Newton's own
    ``examples/softbody/example_softbody_hanging.py`` produces via
    ``add_soft_grid(fix_*=True)`` (mass = 0 at the fixed face -> finalize maps it to
    inv_mass = 0). Harmless to XPBD/SemiImplicit, which already key off inv_mass. Written
    in place with ``.assign()`` so a solver constructed afterwards reads the update.

    TODO[verify-on-colab]: attribute names particle_inv_mass / particle_mass are from
    Newton main and may shift between versions.
    """
    inv_mass = model.particle_inv_mass.numpy()
    inv_mass[fixed_nodes] = 0.0
    model.particle_inv_mass.assign(inv_mass)

    mass = model.particle_mass.numpy()
    mass[fixed_nodes] = 0.0
    model.particle_mass.assign(mass)


def needs_coloring(solver: str) -> bool:
    """Whether the solver needs a vertex-graph colouring (``builder.color()``).

    Only VBD does: its block coordinate descent updates one colour of mutually
    independent vertices at a time. XPBD and SemiImplicit ignore it.
    """
    return solver == "vbd"


def make_solver(solver: str, model, iterations: int = 10, rigid_particle_buffer: int | None = None):
    """Instantiate a Newton solver by name on a finalized ``model``.

    ``rigid_particle_buffer`` sizes VBD's body<->particle contact list and is only
    needed when the collider is a *free* rigid body that VBD must integrate two-way
    (the drop's falling sphere); leave it ``None`` for kinematic/static colliders
    (indentation's prescribed sphere, friction's ground plane) and for the hanging
    bar (no collider). It is passed only when given, so older Newton builds that
    lack the argument are unaffected.
    """
    import newton

    if solver == "xpbd":
        return newton.solvers.SolverXPBD(model=model, iterations=iterations)
    if solver == "vbd":
        # IMPLICIT block coordinate descent. Particle self-contact is OFF here -- the
        # contact we want is against EXTERNAL geometry (rigid sphere / ground plane),
        # which flows through the shared soft_contact buffer, not particle self-contact.
        kwargs = dict(
            model=model,
            iterations=iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )
        if rigid_particle_buffer is not None:
            # TODO[verify-on-colab]: VBD two-way rigid coupling buffer (free collider).
            kwargs["rigid_body_particle_contact_buffer_size"] = rigid_particle_buffer
        return newton.solvers.SolverVBD(**kwargs)
    if solver in ("semi_implicit", "explicit", "semi"):
        # explicit, force-based integrator (the differentiable one); resolves
        # particle-vs-shape contact as explicit penalty forces from the same buffer.
        return newton.solvers.SolverSemiImplicit(model)
    raise ValueError(f"unknown solver {solver!r}")
