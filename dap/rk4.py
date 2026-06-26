"""Classical RK4 as a four-stage integrator ``org^(4)`` (rmk.multistage, ell = 4).

RK4 is ONE ``IntegratorK`` (K = 4) -- a single ``Store∘S ⇒ cot^{◁4}`` rule, the
``ell = 4`` case of the multi-stage integrator the remark proposes -- pushed through
``orgK.orgK_from_integrator`` to an ``org^(4)`` morphism, exactly parallel to the
two-stage leapfrog story (``leapfrog.py``). It integrates the first-order flow

    ẋ = f(x),      f(x) = -sharpR_x(ξ_Q(x)),

i.e. the same gradient flow the configuration integrator steps (eqn.conf_integrator),
but with the classical four-stage Runge--Kutta update instead of explicit Euler. The
per-round "force" ``ξ_Q`` is the framework covector ``dU`` at the emitted position
(computed by the interpretation's backward pass); applying the reactive sharp and the
descent sign gives the rate ``k = -sharpR(ξ_Q)``. The four rounds are the four force
evaluations

    read₁ = x                 → k₁ = f(x)
    read₂ = x + ½h·k₁          → k₂ = f(read₂)
    read₃ = x + ½h·k₂          → k₃ = f(read₃)
    read₄ = x + h·k₃           → k₄ = f(read₄)
    finish: x' = x + (h/6)(k₁ + 2k₂ + 2k₃ + k₄).

For ``ẋ = -A x`` these stages collect to ``x' = [I - hA + (hA)²/2 - (hA)³/6 +
(hA)⁴/24] x``, the degree-4 Taylor polynomial of ``e^{-hA}`` -- the signature of
genuine RK4, with global error ``O(h⁴)`` (``test_orgk.py`` checks the ``h⁴`` rate).

RK4 is **non-symplectic** -- fine here: it is what the multi-stage construction
gives, and what the gyroscope blog uses; symplecticity is not required.
"""

from __future__ import annotations

import jax.numpy as jnp

from .integrator import IntegratorK


def rk4_integrator(h: float = 0.1) -> IntegratorK:
    """The classical RK4 four-stage integrator, ``Store∘S ⇒ cot^{◁4}`` (step ``h``).

    The macro-state is the position ``x``; intermediates carry ``x`` together with the
    rates ``kᵢ`` collected so far. Each ``advance`` evaluates the rate at the *same*
    position its ``read`` emitted, so a position-dependent ``sharpR`` is honored.
    """
    half = 0.5 * h

    def rate(Q, pos, xi_Q):
        # f(pos) = -sharpR_pos(ξ_Q): the reactive sharp of the covector, descending.
        return -Q.apply_sharp(pos, xi_Q)

    def read1(Q, x):
        return x

    def read2(Q, s):
        x, ks = s
        return x + half * ks[0]

    def read3(Q, s):
        x, ks = s
        return x + half * ks[1]

    def read4(Q, s):
        x, ks = s
        return x + h * ks[2]

    def advance1(Q, x, xi_Q):
        k1 = rate(Q, read1(Q, x), xi_Q)
        return (x, (k1,))

    def advance2(Q, s, xi_Q):
        x, ks = s
        k2 = rate(Q, read2(Q, s), xi_Q)
        return (x, ks + (k2,))

    def advance3(Q, s, xi_Q):
        x, ks = s
        k3 = rate(Q, read3(Q, s), xi_Q)
        return (x, ks + (k3,))

    def finish(Q, s, xi_Q):
        x, ks = s
        k1, k2, k3 = ks
        k4 = rate(Q, read4(Q, s), xi_Q)
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return IntegratorK(
        init=lambda Q: jnp.zeros(Q.dim),
        reads=(read1, read2, read3, read4),
        advances=(advance1, advance2, advance3, finish),
        label=f"rk4(h={h:g})",
    )
