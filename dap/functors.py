"""The cotangent functor and the dynamics functors ``Phi_intg`` (cor.functor).

The dynamics functor of the framework (thm.dynamics_functor) is the composite

    sarr --Phi'_interpsm--> Para(cot, poly) --Psi_intg--> pc

of the smooth polynomial interpretation (interpretation.py) with an integrator
semantics (integrator.py). Specializing the integrator to the configuration and
phase integrators gives the two functors of interest (eqn.our_phis):

    Phiconf, Phiphase : sarr -> pc.

This module also exposes the cotangent functor ``cot`` on ``R^d`` (def.cot),
used to interpret interfaces, and ``phi_laxator``, the comparison map of the
*lax* monoidal structure of ``Phi`` on objects, by which the paper's second pass
combines several coalgebras before post-composing a wiring
(sec.spring_second_pass).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .integrator import (
    Integrator,
    configuration_integrator,
    damped_phase_integrator,
    gyro_phase_integrator,
    phase_integrator,
)
from .interpretation import smooth_interpretation
from .pc import PCMorphism
from .pcK import PCMorphismK, pcK_from_integrator
from .polynomial import Cot, DirichletProduct, Poly, PolyMap, nest_left, unnest_left
from .rk4 import rk4_gyro_integrator, rk4_integrator
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
# The lax monoidal comparison of Phi on objects (prop.pc, sec.spring_second_pass).
# ---------------------------------------------------------------------------


def phi_laxator(boxes: Sequence[Tuple[int, int]], label: str = "lax") -> PolyMap:
    """The laxator ``Phi(b_1) (x) ... (x) Phi(b_K) -> Phi(b_1 (x) ... (x) b_K)``.

    ``boxes`` lists the interfaces as ``(d_out, d_in)`` pairs. This is the
    comparison map of the *lax* monoidal structure by which the paper's second pass
    "combines the K coalgebras" before post-composing the wiring
    (sec.spring_second_pass): ``Phi`` sends a box ``<R^{d_in}|R^{d_out}>`` to
    positions ``R^{d_out} x Omega(R^{d_in})`` and directions ``(R^{d_out})^* x R^{d_in}``
    (eqn.Phi_on_obs), so on directions the comparison is the reassociating
    isomorphism, but on positions it is only a map one way:

        Omega(R^{a_1}) x ... x Omega(R^{a_K})  -->  Omega(R^{a_1 + ... + a_K}),
        (omega_1, ..., omega_K) |-> (z |-> (omega_1(z_1), ..., omega_K(z_K))),

    the direct sum of covector fields, which is not surjective -- a field on a
    product is not a tuple of fields on its factors. Hence *lax*, not strong.

    The source is left-nested to match ``PCMorphism.parallel``, which folds
    pairwise; the target is the flat ``phi_box_poly(sum d_out, sum d_in)`` that
    ``Phi`` assigns to the tensored interface.
    """
    K = len(boxes)
    if K < 1:
        raise ValueError("phi_laxator: need at least one box")
    outs = [d_out for d_out, _ in boxes]
    ins = [d_in for _, d_in in boxes]
    o_cuts = [sum(outs[:k]) for k in range(K + 1)]
    i_cuts = [sum(ins[:k]) for k in range(K + 1)]

    def position_action(i):
        pairs = unnest_left(i, K)                    # [(out_m_k, omega_k)]

        def omega(z: Array) -> Array:            # the direct sum of the K fields
            return jnp.concatenate(
                [w(z[i_cuts[k]:i_cuts[k + 1]]) for k, (_, w) in enumerate(pairs)]
            )

        return jnp.concatenate([m for m, _ in pairs]), omega

    def direction_action(i, d):
        xi, in_m = d                             # xi : (R^{sum d_out})^*, in_m : R^{sum d_in}
        return nest_left([
            (xi[o_cuts[k]:o_cuts[k + 1]], in_m[i_cuts[k]:i_cuts[k + 1]])
            for k in range(K)
        ])

    src = phi_box_poly(*boxes[0])
    for d_out, d_in in boxes[1:]:
        src = DirichletProduct(src, phi_box_poly(d_out, d_in))

    return PolyMap(
        src=src,
        tgt=phi_box_poly(sum(outs), sum(ins)),
        position_action=position_action,
        direction_action=direction_action,
        label=label,
    )


# ---------------------------------------------------------------------------
# Phi_intg : sarr -> pc (cor.functor).
# ---------------------------------------------------------------------------


def Phi(arr: SmoothArrangement, integrator: Integrator) -> PCMorphism:
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

    return PCMorphism(
        src_poly=src_p,
        tgt_poly=tgt_p,
        state=integrator.init(Q),
        step=step,
    )


def Phiconf(arr: SmoothArrangement) -> PCMorphism:
    """Configuration dynamics ``Phiconf`` (sec.configuration_dynamics)."""
    return Phi(arr, configuration_integrator())


def Phiphase(arr: SmoothArrangement) -> PCMorphism:
    """Phase-space dynamics ``Phiphase`` (sec.phase_dynamics)."""
    return Phi(arr, phase_integrator())


def Phidamped(arr: SmoothArrangement, damping: float = 0.1) -> PCMorphism:
    """Damped phase dynamics: heavy-ball *momentum* (damped_phase_integrator).

    EXTENSION -- see ``damped_phase_integrator``: it wires the paper's damping 1-form
    ``ex.damping_one_form`` (present, but "plays no further role" there) into the phase
    integrator. The same arrangement read by a phase integrator with friction
    ``damping``; on a learner this is momentum-based training, converging where the
    conservative ``Phiphase`` would only oscillate.
    """
    return Phi(arr, damped_phase_integrator(damping))


def Phigyro(arr: SmoothArrangement, damping: float = 0.0, gamma: float = 0.0, J=None,
            drag: float = 0.0, gyro_block=None) -> PCMorphism:
    """Gyroscopic phase dynamics: phase + damping + skew + optional quadratic drag.

    EXTENSION (beyond the paper) -- see ``gyro_phase_integrator``. The same
    arrangement read by a phase integrator carrying, besides the kinetic 1-form,
    a damping term ``c``, a velocity-perpendicular gyroscopic term ``gamma * J``, and
    (with ``gyro_block`` set) a per-gyro quadratic-drag 1-form ``drag * |v| v``. This is
    the integrator behind the gyro surrogate (gravity + springs in ``U``, mass in the
    sharp, damping/precession/drag in the 1-form); see ``dap/gyroscope.py``.
    """
    return Phi(arr, gyro_phase_integrator(damping, gamma, J, drag, gyro_block))


def Phirk4(arr: SmoothArrangement, h: float = 0.1) -> PCMorphismK:
    """Classical RK4 dynamics of a (closed) arrangement, as an ``pc^(4)`` morphism.

    The ``K = 4`` analog of ``Phileap`` (which is ``pc^(2)``): the RK4 four-stage
    integrator (``rk4.rk4_integrator``) pushed through ``pcK.pcK_from_integrator``.
    It steps the gradient flow ``ẋ = -sharpR_x(dU)`` with classical RK4 (step ``h``),
    so its global error is ``O(h^4)``. Like ``pc^(2)``, whether ``sarr → pc^(4)`` is
    a *functor* (rmk.multistage) is left open; this is the datatype plus an instance.
    """
    return pcK_from_integrator(arr, rk4_integrator(h))


def Phirk4gyro(arr: SmoothArrangement, h: float = 0.1, drag: float = 0.0,
               gamma: float = 0.0, J=None, gyro_block=None) -> PCMorphismK:
    """RK4 gyro phase dynamics: the faithful machine's integrator, as an ``pc^(4)`` morphism.

    The ``K = 4`` analog of ``Phigyro`` -- the same forces (springs and rod gravity in
    ``U``, quadratic drag and per-gyro precession in the 1-form) but stepped with classical
    RK4 (``rk4.rk4_gyro_integrator``) on the phase state, matching the blog's integrator
    rather than symplectic Euler. Non-symplectic; ``sarr → pc^(4)`` functoriality is left
    open, as for every ``pc^(K)``.
    """
    return pcK_from_integrator(arr, rk4_gyro_integrator(h, drag, gamma, J, gyro_block))
