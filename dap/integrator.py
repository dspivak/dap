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
* ``position(s)``       -- the parameter position ``q in Q`` to run the
                           (integrator-free) interpretation at.
* ``step(Q, s, xi_Q)``  -- the new state, given the parameter covector ``xi_Q``.

The interpretation depends only on ``position(s)``; everything that differs
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
    position: Callable[[Any], Any]
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
        position=lambda s: s,
        step=lambda Q, s, xi_Q: s - Q.apply_sharp(s, xi_Q),
        label="conf",
    )


def phase_integrator() -> Integrator:
    """The phase integrator ``phase = (|T^*-|, upd_phase)`` (sec.phase_integrators).

    State space ``S = |T^*Q| = Q (+) Q^*``, written ``s = (q, xi)`` for position
    and momentum. One explicit symplectic-Euler step of Hamilton's equations
    (eqn.phase_update, eqn.state_update):

        (q, xi) |-> (q + sharpR_q(xi), xi - xi_Q).

    The position moves by the velocity ``sharpR_q(xi)`` read off the stored
    momentum; the momentum moves by subtracting the incoming gradient ``xi_Q``.
    """

    return Integrator(
        init=lambda Q: (jnp.zeros(Q.dim), jnp.zeros(Q.dim)),
        position=lambda s: s[0],
        step=lambda Q, s, xi_Q: (s[0] + Q.apply_sharp(s[0], s[1]), s[1] - xi_Q),
        label="phase",
    )
