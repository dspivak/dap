"""The cotangent functor and the dynamics functors ``Phi_intg`` (cor.functor).

The dynamics functor of the framework (thm.dynamics_functor) is the composite

    sarr --Phi'_interpsm--> Para(cot, poly) --Psi_intg--> org

of the smooth polynomial interpretation (interpretation.py) with an integrator
semantics (integrator.py). Specializing the integrator to the configuration and
phase integrators gives the two functors of interest (eqn.our_phis):

    Phiconf, Phiphase : sarr -> org.

This module also exposes the cotangent functor ``cot`` on ``R^d`` (def.cot),
used to interpret interfaces.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .integrator import (
    Integrator,
    configuration_integrator,
    damped_phase_integrator,
    phase_integrator,
)
from .interpretation import smooth_interpretation
from .org import OrgMorphism
from .polynomial import Cot, DirichletProduct, PolyMap, Poly
from .rvect import SmoothMap


# ---------------------------------------------------------------------------
# cot : mfd -> poly (def.cot), restricted to R^d.
# ---------------------------------------------------------------------------


def cot_object(d: int) -> Cot:
    """Apply def.cot on objects: ``R^d |-> Cot(d)``."""
    return Cot(d)


def cot_map(f: SmoothMap, src_dim: int, tgt_dim: int, label: str = "") -> PolyMap:
    """Apply def.cot on morphisms: forward part ``f``, backward part ``(T_x f)^T``.

    The cotangent pullback at ``x`` is computed with ``jax.vjp``.
    """

    def position_action(x: Array) -> Array:
        return f(x)

    def direction_action(x: Array, xi_target: Array) -> Array:
        _, vjp_fn = jax.vjp(f, x)
        (xi_source,) = vjp_fn(xi_target)
        return xi_source

    return PolyMap(
        src=Cot(src_dim),
        tgt=Cot(tgt_dim),
        position_action=position_action,
        direction_action=direction_action,
        label=label or "cot(f)",
    )


def phi_box_poly(d_out: int, d_in: int) -> Poly:
    """``Phi(<R^{d_in}/R^{d_out}>)`` as a poly object (eqn.Phi_on_obs).

    Positions are pairs ``(out_m, omega)`` and directions ``(xi, in_m)``; we use
    a tagged ``DirichletProduct(Cot(d_out), Cot(d_in))`` to record the
    dimensions and thread the covector-field data through the step closures.
    """
    return DirichletProduct(Cot(d_out), Cot(d_in))


# ---------------------------------------------------------------------------
# Phi_intg : sarr -> org (cor.functor).
# ---------------------------------------------------------------------------


def Phi(arr: SmoothArrangement, integrator: Integrator) -> OrgMorphism:
    """The dynamics functor ``Phi_intg`` applied to ``arr`` (cor.functor).

    Composes the integrator-free interpretation ``Phi'_interpsm(arr)`` with the
    given integrator. The readout and returned direction are the same for every
    integrator (they are the interpretation's, evaluated at the parameter
    position the integrator presents); only the state space and update differ.
    """

    Q = arr.Q
    interp = smooth_interpretation(arr)
    src_p = phi_box_poly(arr.out_dim_M, arr.in_dim_M)
    tgt_p = phi_box_poly(arr.out_dim_N, arr.in_dim_N)

    def step(state):
        q = integrator.position(Q, state)
        position_action, direction_action = interp(q)

        def act_positions(in_pos):
            out_m, omega_M = in_pos
            return position_action(out_m, omega_M)

        def act_directions(in_pos, in_dir):
            out_m, omega_M = in_pos
            xi_N, in_n = in_dir
            _, xi_M, in_m = direction_action(out_m, omega_M, xi_N, in_n)
            return (xi_M, in_m)

        act = PolyMap(
            src=src_p,
            tgt=tgt_p,
            position_action=act_positions,
            direction_action=act_directions,
            label=f"Phi_{integrator.label}({arr.label})",
        )

        def fiber(in_pos):
            out_m, omega_M = in_pos
            out_pos = position_action(out_m, omega_M)

            def at_pos(in_dir):
                xi_N, in_n = in_dir
                xi_Q, xi_M, in_m = direction_action(out_m, omega_M, xi_N, in_n)
                out_dir = (xi_M, in_m)
                new_state = integrator.step(Q, state, xi_Q)
                return out_dir, new_state

            return out_pos, at_pos

        return act, fiber

    return OrgMorphism(
        src_poly=src_p,
        tgt_poly=tgt_p,
        state=integrator.init(Q),
        step=step,
    )


def Phiconf(arr: SmoothArrangement) -> OrgMorphism:
    """Configuration dynamics ``Phiconf`` (sec.configuration_dynamics)."""
    return Phi(arr, configuration_integrator())


def Phiphase(arr: SmoothArrangement) -> OrgMorphism:
    """Phase-space dynamics ``Phiphase`` (sec.phase_dynamics)."""
    return Phi(arr, phase_integrator())


def Phidamped(arr: SmoothArrangement, damping: float = 0.1) -> OrgMorphism:
    """Damped phase dynamics: heavy-ball *momentum* (damped_phase_integrator).

    EXTENSION (beyond the paper) -- see ``damped_phase_integrator``: its dissipation
    1-form is commented out in the published paper. The same arrangement read by a
    phase integrator with friction ``damping``; on a learner this is momentum-based
    training, converging where the conservative ``Phiphase`` would only oscillate.
    """
    return Phi(arr, damped_phase_integrator(damping))
