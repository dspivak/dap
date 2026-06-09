"""Leapfrog (velocity Verlet) as a two-stage integrator (rmk.org_N).

Leapfrog is now ONE ``Integrator2`` — a single ``Store∘S ⇒ cot^{◁2}`` natural
transformation, the multi-stage integrator the remark proposes — pushed through
the general ``org2.org2_from_integrator`` to yield an ``org^(2)`` morphism. This
is exactly parallel to the 1-stage story (a 1-stage ``Integrator`` + ``functors.Phi``
gives an ``org`` morphism). The kick/drift/kick lives once, in the integrator:

    read1(q, ξ)   = q
    advance       : ξ½ = ξ − ½ ξ_Q(q),   q' = q + sharpR_q(ξ½)     (mid = (q', ξ½))
    read2(q', ξ½) = q'
    finish        : ξ' = ξ½ − ½ ξ_Q(q')                            (new = (q', ξ'))

so the two force evaluations ξ_Q(q) and ξ_Q(q') are the two ``org^(2)`` rounds.
Constant (mass-like) ``sharpR`` — the physics case.
"""

from __future__ import annotations

import jax.numpy as jnp

from .arrangement import SmoothArrangement
from .integrator import Integrator2
from .org2 import OrgMorphism2, org2_from_integrator


def leapfrog_integrator() -> Integrator2:
    """The leapfrog (velocity Verlet) two-stage integrator, ``Store∘S ⇒ cot^{◁2}``."""

    def advance(Q, s, xi_Q1):
        q, xi = s
        xi_half = xi - 0.5 * xi_Q1            # half-kick
        return (q + Q.apply_sharp(q, xi_half), xi_half)   # drift -> mid = (q', ξ½)

    def finish(Q, s, mid, xi_Q2):
        q2, xi_half = mid
        return (q2, xi_half - 0.5 * xi_Q2)    # half-kick -> (q', ξ')

    return Integrator2(
        init=lambda Q: (jnp.zeros(Q.dim), jnp.zeros(Q.dim)),
        read1=lambda s: s[0],
        advance=advance,
        read2=lambda mid: mid[0],
        finish=finish,
        label="leapfrog",
    )


def Phileap(arr: SmoothArrangement) -> OrgMorphism2:
    """Leapfrog dynamics of a (closed) arrangement, as an ``org^(2)`` morphism."""
    return org2_from_integrator(arr, leapfrog_integrator())
