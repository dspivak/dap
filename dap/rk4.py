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

from .integrator import IntegratorK, quadratic_drag_kick


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


def rk4_gyro_integrator(
    h: float = 0.1, drag: float = 0.0, gamma: float = 0.0, J=None, gyro_block=None
) -> IntegratorK:
    """Classical RK4 on the gyro *phase* system, an ``org^(4)`` integrator (rmk.multistage).

    The faithful-machine integrator: it carries the *same* forces as
    ``gyro_phase_integrator`` (springs and rod gravity in the potential ``U``, quadratic
    drag and gyroscopic precession in the 1-form) but steps them with the blog's
    classical four-stage Runge--Kutta instead of symplectic Euler. The phase vector
    field on the state ``(q, xi)`` is

        qdot  = sharpR_q(xi) =: v
        xidot = -dU(q) - drag * F(v) - gamma (.) (J v),

    with ``dU`` the framework covector ``xi_Q`` evaluated by the interpretation at each
    stage's position (so it includes any open-port forcing), ``F`` the per-gyro quadratic
    drag (``quadratic_drag_kick``), and ``gamma`` a scalar or per-component vector. The
    four ``org^(4)`` rounds are the four force evaluations ``k1..k4``; ``finish`` does
    ``(q,xi) + (h/6)(k1 + 2k2 + 2k3 + k4)``.

    Non-symplectic (RK4 is); ``drag``/``gamma`` may be traced, ``J``/``gyro_block`` are
    Python-time structural choices. State and forces mirror ``gyro_phase_integrator``;
    only the time-stepping differs.
    """
    half = 0.5 * h
    Jm = None if J is None else jnp.asarray(J, float)

    def deriv(Q, q_s, xi_s, xi_Q):
        """The phase derivative ``(qdot, xidot)`` at a stage state, given ``xi_Q = dU``."""
        v = Q.apply_sharp(q_s, xi_s)
        k_xi = -xi_Q
        if gyro_block is not None:
            k_xi = k_xi - drag * quadratic_drag_kick(v, gyro_block)
        if Jm is not None:
            k_xi = k_xi - gamma * (Jm @ v)
        return (v, k_xi)

    def axpy(base, a, k):
        (q, xi), (kq, kxi) = base, k
        return (q + a * kq, xi + a * kxi)

    def read1(Q, base):
        return base[0]

    def read2(Q, mid):
        base, ks = mid
        return axpy(base, half, ks[0])[0]

    def read3(Q, mid):
        base, ks = mid
        return axpy(base, half, ks[1])[0]

    def read4(Q, mid):
        base, ks = mid
        return axpy(base, h, ks[2])[0]

    def advance1(Q, base, xi_Q):
        k1 = deriv(Q, base[0], base[1], xi_Q)
        return (base, (k1,))

    def advance2(Q, mid, xi_Q):
        base, ks = mid
        s = axpy(base, half, ks[0])
        return (base, ks + (deriv(Q, s[0], s[1], xi_Q),))

    def advance3(Q, mid, xi_Q):
        base, ks = mid
        s = axpy(base, half, ks[1])
        return (base, ks + (deriv(Q, s[0], s[1], xi_Q),))

    def finish(Q, mid, xi_Q):
        base, ks = mid
        s = axpy(base, h, ks[2])
        k4 = deriv(Q, s[0], s[1], xi_Q)
        k1, k2, k3 = ks
        q0, xi0 = base
        new_q = q0 + (h / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        new_xi = xi0 + (h / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        return (new_q, new_xi)

    return IntegratorK(
        init=lambda Q: (jnp.zeros(Q.dim), jnp.zeros(Q.dim)),
        reads=(read1, read2, read3, read4),
        advances=(advance1, advance2, advance3, finish),
        label=f"rk4_gyro(h={h:g})",
    )
