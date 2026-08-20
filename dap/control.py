"""Impedance (P-)control as a virtual spring.

EXTENSION (beyond the paper) -- an application of the existing machinery, no
new formal content. An impedance controller of stiffness ``k`` is a *stateless*
box whose entire content is the potential

    U_c(q_rep, s) = (k/2) |q_rep - s|^2,

a virtual spring hung between the plant's reported position ``q_rep`` and a
setpoint ``s``. The plant is the particle pattern of sec.spring_first_pass with
zero potential of its own (the simplest honest choice: all closed-loop forces
come from the controller); the coupling is a wiring; and the closed-loop
behavior is whatever the dynamics functors make of the composite:

* ``Phiconf``  -- proportional setpoint tracking, ``q <- q - sharpR_q(k(q - s))``;
* ``Phiphase`` -- the classical stiffness-only impedance law: undamped
  oscillation about ``s`` (semi-implicit Euler on the harmonic oscillator).

The setpoint is NOT baked into the arrangement. The composite is an *open*
morphism ``<R^0|R^0> -> <R^d|R^d>`` whose outer input port carries ``s``; the
environment supplies it per step through ``PCMorphism.run_one``'s
``in_dir_from`` closure, exactly as the training environment streams data
``(x, lambda)`` in ``learning.train`` (ex.training_data). Changing the value
passed at step ``t`` changes the setpoint at step ``t`` -- a genuine run-time
input.

Boundary accounting: the composite's outer *position* also emits the covector
field ``omega_N(s) = dU_c/ds = -k(q_rep - s)`` on the setpoint port
(eqn.omegaprime) -- the reaction of the virtual spring on whoever holds its far
end. It is equal and opposite to the covector ``xi_Q = dU_c/dq = k(q_rep - s)``
that drives the plant, because both are gradients of the one shared potential.
``run_tracking`` below *discards* it: the setpoint holder is treated as
infinitely stiff, and the work it does against the spring is untracked.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .interpretation import trivial_omega
from .pc import PCMorphism
from .rvect import ReactiveVectorSpace, trivial
from .wiring import compose_seq, parallel_arrangements


# ---------------------------------------------------------------------------
# The three boxes: plant, controller, wiring.
# ---------------------------------------------------------------------------


def plant_box(Q: ReactiveVectorSpace) -> SmoothArrangement:
    """The plant ``Part_plant : <R^0|R^0> -> <R^d|R^0>``, ``d = Q.dim``.

    Parameter the given reactive vector space ``Q`` (the sharp carries the
    timestep/mass: ``euclidean(d, dt)`` for descent, ``diagonal([1/m]*d)`` for
    phase dynamics, per eqn.learning_sharp / sec.spring_first_pass); output map
    the identity readout ``q``; no input ports; zero potential.
    """
    d = Q.dim
    return SmoothArrangement(
        Q=Q,
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=d,
        in_dim_N=0,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: jnp.array(0.0),
        label="Part_plant",
    )


def impedance_controller(k: float, d: int = 1) -> SmoothArrangement:
    """The stateless controller ``Part_ctrl : <R^0|R^0> -> <R^0|R^{2d}>``.

    Trivial parameter ``R^0`` (stateless: no state, no dynamics of its own),
    no output ports, and ``2d`` input ports read as ``(q_rep, s)``; the entire
    controller is the virtual-spring potential

        U_c(q_rep, s) = (k/2) |q_rep - s|^2.
    """

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        q_rep, s = n_in[:d], n_in[d:]
        return 0.5 * k * jnp.sum((q_rep - s) ** 2)

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=0,
        in_dim_N=2 * d,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=U,
        label=f"Part_ctrl(k={k:g})",
    )


def impedance_wire(d: int = 1) -> SmoothArrangement:
    """The wiring ``<R^{2d}|R^d> -> <R^d|R^d>`` closing the loop.

    A static arrangement (trivial parameter, ``U = 0``) routing, on the model
    of ``chain_wire``:

    * outer output   <- plant's reported position (``out_f = id`` on ``m_out``);
    * controller port 1 <- plant's reported position (feedback);
    * controller port 2 <- outer input (the setpoint ``s``).

    ``M = plant_iface (x) ctrl_iface`` has out-ports ``R^d`` (plant) and
    in-ports ``R^{2d}`` (controller), so ``in_f(m_out, n_in) = (m_out, n_in)``.
    """

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return m_out

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.concatenate([m_out, n_in])

    def U(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.array(0.0)

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=d,
        in_dim_M=2 * d,
        out_dim_N=d,
        in_dim_N=d,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label=f"impedance_wire(d={d})",
    )


def impedance_arrangement(Q: ReactiveVectorSpace, k: float) -> SmoothArrangement:
    """``wire(Part_plant, Part_ctrl) : <R^0|R^0> -> <R^d|R^d>`` by genuine composition.

        compose_seq( plant_box(Q) || impedance_controller(k, d), impedance_wire(d) ).

    The composite parameter is ``Q (+) R^0 = Q`` (the plant's state; the
    controller contributes none), and the composite potential *emerges* from
    ``compose_seq`` as ``U(q, s) = (k/2)|q - s|^2`` -- it is not written by
    hand. The outer input port carries the setpoint ``s``; the outer output
    port reports the plant position (at whichever parameter position the
    integrator presents: ``q`` for ``conf``, ``q~ = q + sharpR_q(xi)`` for
    ``phase``).
    """
    d = Q.dim
    inner = parallel_arrangements(plant_box(Q), impedance_controller(k, d))
    wired = compose_seq(inner, impedance_wire(d))
    return replace(wired, label=f"impedance(d={d}, k={k:g})")


# ---------------------------------------------------------------------------
# Running the closed loop: the setpoint as a per-step run-time input.
# ---------------------------------------------------------------------------

_IN_POS = (jnp.zeros(0), trivial_omega(0))  # the trivial <R^0|R^0> source position


def run_tracking(
    O: PCMorphism, state: Any, setpoints: Sequence
) -> Tuple[List[Array], List[Any]]:
    """Step the coalgebra once per entry of ``setpoints``, threading the state.

    At step ``t`` the environment closes the boundary with the in-direction
    ``(xi_N, in_n) = (0, s_t)``: no external covector pushes on the reported-
    position port, and the setpoint port carries ``s_t`` -- supplied from
    outside at run time, so a mid-run change of setpoint is just a change in
    the list. Returns ``(outs, states)`` with ``outs[t]`` the reported outer
    position after reading ``s_t`` and ``states`` of length ``len(setpoints)+1``
    (including the initial state).

    The emitted covector field ``omega_N`` on the setpoint port (see module
    docstring) is discarded here: the setpoint holder's work is untracked.
    """
    outs: List[Array] = []
    states: List[Any] = [state]
    for s in setpoints:
        s_arr = jnp.atleast_1d(jnp.asarray(s, dtype=float))
        in_dir = (jnp.zeros(s_arr.shape[0]), s_arr)
        out_pos, _out_dir, state = O.with_state(state).run_one(
            _IN_POS, lambda _op, d=in_dir: d
        )
        outs.append(out_pos[0])
        states.append(state)
    return outs, states
