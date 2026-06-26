"""Integrators (def.integrator) and the two we use: configuration and phase.

By the framework of ch.framework, once the polynomial interpretation
``Phi'_interpsm`` (interpretation.py) has turned an arrangement into a
``cot(Q)``-parameterized polynomial map, the remaining choice is an
*integrator*: a state space ``S = F(Q)`` together with a rule that updates the
current state from an incoming parameter covector ``xi_Q in Q^*``. Each
integrator induces a semantics ``Psi_intg : Para(cot, poly) -> org``
(prop.integrator_to_org), and composing with the interpretation gives a
dynamics functor ``Phi_intg : sarr -> org`` (cor.functor).

We implement an integrator as three callables bound to the parameter reactive
vector space ``Q`` at runtime:

* ``init(Q)``           -- the initial state.
* ``position(Q, s)``    -- the parameter position ``q in Q`` to run the
                           (integrator-free) interpretation at.
* ``step(Q, s, xi_Q)``  -- the new state, given the parameter covector ``xi_Q``.

The interpretation depends only on ``position(Q, s)``; everything that differs
between configuration and phase dynamics lives in ``init``/``position``/``step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp

from .rvect import ReactiveVectorSpace


@dataclass(frozen=True)
class Integrator:
    """A cot-integrator (def.integrator), as a triple of callables."""

    init: Callable[[ReactiveVectorSpace], Any]
    position: Callable[[ReactiveVectorSpace, Any], Any]
    step: Callable[[ReactiveVectorSpace, Any, Any], Any]
    label: str = ""


def configuration_integrator() -> Integrator:
    """The configuration integrator ``conf = (|-|, (-1) o chi)`` (sec.config_integrator).

    State space ``S = |Q|``; an incoming covector descends the state,
    ``q |-> q - sharpR_q(xi_Q)`` (eqn.conf_integrator, eqn.state_update_gradient).
    The minus is the integrator's: it makes the dynamics descend.
    """

    return Integrator(
        init=lambda Q: jnp.zeros(Q.dim),
        position=lambda Q, s: s,
        step=lambda Q, s, xi_Q: s - Q.apply_sharp(s, xi_Q),
        label="conf",
    )


def phase_integrator() -> Integrator:
    """The phase integrator (sec.phase_integrators): the instance nu_{sigma^1, beta}.

    State space ``S = |T^*Q| = Q (+) Q^*``, written ``s = (q, xi)`` for position
    and momentum. The readout is the *presented position* (the exponential
    readout sigma^1, eqn.presented_position):

        q~ = q + sharpR_q(xi),

    the position the step moves to. Forces are evaluated there, which is what
    makes the step symplectic (semi-implicit Euler) rather than explicit. With
    the velocity read off the stored momentum, one step of Hamilton's equations
    (eqn.phase_update) is

        (q, xi) |-> (q + sharpR_q(xi), xi - xi_Q),   xi_Q computed at q~.
    """

    return Integrator(
        init=lambda Q: (jnp.zeros(Q.dim), jnp.zeros(Q.dim)),
        position=lambda Q, s: s[0] + Q.apply_sharp(s[0], s[1]),
        step=lambda Q, s, xi_Q: (s[0] + Q.apply_sharp(s[0], s[1]), s[1] - xi_Q),
        label="phase",
    )


def damped_phase_integrator(damping: float = 0.0) -> Integrator:
    """The phase integrator with the paper's *damping* 1-form added.

    The damping 1-form ``ex.damping_one_form`` -- ``zeta(q, xi) = (xi, 0)`` -- is
    *present* in the paper (where it "plays no further role"); that example already
    records that for a coefficient ``c in [0,1]`` the scalar multiple ``c*zeta``
    "removes a fraction c of the momentum at each step". This integrator is that
    line put to work: the only step beyond the paper's formal development is wiring
    the 1-form into an integrator to make an optimizer (README "Extensions").

    The phase readout combines the kinetic 1-form ``beta`` with the *damping*
    1-form ``c * zeta``, ``zeta(q, xi) = (xi, 0)`` (a monoidal 1-form by
    prop.one_forms_vector_space). Fed through the symplectic sharp it adds
    ``-c * xi`` to the momentum, so the Hamilton step (eqn.phase_update) gains a
    friction term:

        (q, xi) |-> (q + sharpR_q(xi),  (1 - c) * xi - xi_Q),   xi_Q at q~.

    At ``c = 0`` this is the conservative phase integrator (oscillates forever);
    at ``0 < c < 1`` it dissipates a fraction ``c`` of the momentum each step --
    heavy-ball *momentum* that converges; as ``c -> 1`` it collapses toward plain
    descent (the configuration integrator). The integrator alone interpolates
    between the optimizers, exactly as it does between wave and heat dynamics.
    """

    c = float(damping)
    return Integrator(
        init=lambda Q: (jnp.zeros(Q.dim), jnp.zeros(Q.dim)),
        position=lambda Q, s: s[0] + Q.apply_sharp(s[0], s[1]),
        step=lambda Q, s, xi_Q: (
            s[0] + Q.apply_sharp(s[0], s[1]),
            (1.0 - c) * s[1] - xi_Q,
        ),
        label=f"damped({c:g})",
    )


def gyro_phase_integrator(
    damping: float = 0.0, gamma: float = 0.0, J=None
) -> Integrator:
    """The phase integrator with a *damping* and a *gyroscopic* 1-form added.

    EXTENSION (beyond the paper). Like ``damped_phase_integrator`` this lives in
    the readout slot ``nu_{sigma^1, omega}`` (sec.phase_integrators), but adds, to
    the kinetic 1-form ``beta`` and the damping 1-form ``c*zeta``, a third,
    *skew* 1-form -- the gyroscopic term of a spinning rotor. With a complex
    structure ``J`` (a block-diagonal 90-degree rotation, one ``[[0,-1],[1,0]]``
    block per 2-D gyro), define

        omega_gyro(q, xi) = (gamma * J @ sharpR_q(xi),  0) : Q^* (+) Q.

    Fed through the symplectic sharp (which sends ``(a, b) |-> (b, -a)``,
    eqn.canonical_sharp) it adds ``-gamma * J @ sharpR_q(xi)`` to the momentum: a
    force perpendicular to the velocity (``(J v) . v = 0``, rmk.symplectic_perpendicular).
    In continuous time such a skew force does no work -- gyroscopic precession rotates
    the velocity without changing its length -- and, unlike the potential, it is
    velocity-dependent, so it cannot be a ``dU``. Beware: this *explicit* discretization
    of the skew term is not symplectic; it mildly injects energy each step and is meant
    to be run with the damping ``c`` (as in ``dap/gyroscope.py``), which keeps it
    bounded. Combined with damping the Hamilton step (eqn.phase_update) becomes

        (q, xi) |-> ( q + sharpR_q(xi),
                      (1 - c) * xi - xi_Q - gamma * J @ sharpR_q(xi) ),   xi_Q at q~.

    Caveat (why it is beyond the paper): ``omega_gyro`` is a monoidal 1-form only
    over the subcategory of ``rvect`` whose isomorphisms commute with ``J`` -- the
    rotor picks out an orientation, breaking the full reactive symmetry the paper's
    1-forms (``beta``, ``zeta``) enjoy. It is exactly the minimal example that lives
    in ``rvect`` but outside the *harmonic* (symmetric-sharp) regime of
    def.arrangement_terminology.
    """

    # damping ``c`` and gyroscopic ``gamma`` may be traced (they are trainable in
    # the gyroscope emulation), so we do not coerce them to ``float`` here; only
    # the *presence* of ``J`` is a static (Python-time) choice.
    c = damping
    g = gamma
    Jm = None if J is None else jnp.asarray(J, float)

    def step(Q: ReactiveVectorSpace, s, xi_Q):
        q, xi = s
        v = Q.apply_sharp(q, xi)  # velocity sharpR_q(xi)
        new_xi = (1.0 - c) * xi - xi_Q
        if Jm is not None:
            new_xi = new_xi - g * (Jm @ v)  # gyroscopic precession (skew, no work)
        return (q + v, new_xi)

    return Integrator(
        init=lambda Q: (jnp.zeros(Q.dim), jnp.zeros(Q.dim)),
        position=lambda Q, s: s[0] + Q.apply_sharp(s[0], s[1]),
        step=step,
        label="gyro",
    )


@dataclass(frozen=True)
class Integrator2:
    """A *two-stage* cot-integrator: operationally ``Store∘S ⇒ cot^{◁2}`` (rmk.multistage).

    The multi-stage form the remark proposes (``upd: Store∘S ⇒ p^{◁2}`` with
    ``p = cot``): two emit/receive rounds before the state updates. As callables:

    * ``init(Q)``                       -- the initial state.
    * ``read1(Q, state)``               -- parameter position emitted in round 1.
    * ``advance(Q, state, xi_Q1)``      -- consume the round-1 covector; return the
                                           intermediate ("inner coalgebra state").
    * ``read2(Q, mid)``                 -- parameter position emitted in round 2.
    * ``finish(Q, state, mid, xi_Q2)``  -- consume the round-2 covector; new state.

    ``org2.org2_from_integrator`` turns one of these into an ``org^(2)`` morphism,
    exactly as ``functors.Phi`` turns a 1-stage ``Integrator`` into an ``org`` one.
    """

    init: Callable[[ReactiveVectorSpace], Any]
    read1: Callable[[ReactiveVectorSpace, Any], Any]
    advance: Callable[[ReactiveVectorSpace, Any, Any], Any]
    read2: Callable[[ReactiveVectorSpace, Any], Any]
    finish: Callable[[ReactiveVectorSpace, Any, Any, Any], Any]
    label: str = ""
