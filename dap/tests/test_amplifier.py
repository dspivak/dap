"""Tests for ``amplifier`` -- the audit of the Black-feedback-amplifier remark.

The claims under test (all through ``Phiconf`` of the genuine ``compose_seq``
composite; nothing hand-rolled):

1.  *One-step exactness.*  One ``Phiconf`` step of the composite is exactly
    ``v <- v + dt kappa (A g(e_src - beta v) - v)`` (``w`` frozen at 0).
2.  *Black's gain formula as an equilibrium.*  The framework run converges to
    ``v* = A/(1 + A beta) e_src`` for ``A in {1e2, 1e3, 1e4}``, ``beta = 0.1``,
    approaching ``1/beta = 10``; the emitted source reaction vanishes there.
3.  *Desensitization.*  For ``A: 1e3 -> 1e4`` the raw gain change is
    ``dG/G = (dA/A1)/(1 + A2 beta) = 9/1001 ~ 0.899%`` -- note: NOT below 0.1%
    -- while the sensitivity ``(dG/G)/(dA/A) = 1/(1 + A2 beta) ~ 0.0999% < 0.1%``,
    exact for the linear stage.
4.  *Distortion suppression* (quasi-static, matched peak output).  Measured:
    open-loop THD 2.755e-2, closed-loop THD 3.039e-4, improvement factor 90.7
    against the classical prediction ``1 + A beta = 101``; the shortfall is the
    varying loop gain ``1 + A beta g'(e)`` over the excursion, so the honest
    bound is the derived bracket ``[1 + A beta g'_min, 1 + A beta]``.
5.  *Stability boundary.*  The iteration's error factor is exactly
    ``1 - dt kappa (1 + A beta)``: convergent just inside
    ``dt kappa (1 + A beta) = 2``, divergent just outside.  No unconditional
    convergence.

Mechanism audit (which unilaterality operates):

6.  The stage's reaction is degenerate and non-symmetric (logic.py's kernel
    sharp scaled by ``dt``); output-port covectors land in its kernel exactly
    (load rejection), and so does the loop back-action.
7.  The source-port reaction covector IS emitted (``omega_N = kappa r A g'``,
    A-large off equilibrium) and merely discarded by the runner -- the
    ``control.py`` boundary situation, at the source only.
8.  The audit's negative: the *bilateral* (reciprocal, gradient) stage has the
    SAME closed-loop equilibrium -- Black's formula does not need unilaterality
    -- but its update feels the loop factor ``(1 + A beta)`` and it diverges at
    a step size where the unilateral loop is stable (boundary
    ``dt kappa (1 + A beta)^2 = 2`` instead of ``dt kappa (1 + A beta) = 2``).
"""

import jax.numpy as jnp
import numpy as np

from dap.amplifier import (
    KERNEL,
    amp_state,
    black_amplifier,
    settle,
    stepper,
)

BETA = 0.1


def _tanh_g(s):
    return lambda x: s * jnp.tanh(x / s)


# ---------------------------------------------------------------------------
# 6.  The mechanism: the sharp, load rejection, and the emitted source reaction.
# ---------------------------------------------------------------------------


def test_sharp_is_degenerate_and_nonsymmetric():
    """def.rvect asks only for ``Q -> vect(Q^*, Q)``: no symmetry, no invertibility."""
    S = KERNEL
    assert not np.allclose(S, S.T)
    assert np.linalg.matrix_rank(np.asarray(S)) == 1


def test_output_load_rejection():
    """A covector at the output port pulls back to ``(xi_N, 0)`` -- in ``ker(sharp)``.

    The load cannot move the stage: the transistor-style kernel condition on
    the output side, exact for any load.
    """
    one = stepper(black_amplifier(A=50.0, beta=BETA, kappa=0.7, dt=0.03))
    s0, e = jnp.array([0.3, 0.0]), jnp.array([0.5])
    _, _, s_unloaded = one(s0, e, jnp.zeros(1))
    _, _, s_loaded = one(s0, e, jnp.array([7.3]))
    np.testing.assert_array_equal(np.asarray(s_loaded), np.asarray(s_unloaded))


def test_source_reaction_emitted_and_discarded():
    """The drive port emits ``omega_N(e_src) = kappa * r * A`` (linear stage).

    Off equilibrium it is A-large -- the runner discards it (the ``control.py``
    boundary situation): the source is treated as infinitely stiff.  This is
    the part of the unilaterality story that is NOT the kernel condition.
    """
    A, kappa, dt = 50.0, 0.7, 0.03
    one = stepper(black_amplifier(A, BETA, kappa, dt))
    v0, e = 0.3, 0.5
    _, react, _ = one(jnp.array([v0, 0.0]), jnp.array([e]), jnp.zeros(1))
    r = A * (e - BETA * v0) - v0
    np.testing.assert_allclose(float(react), kappa * r * A, rtol=1e-13)


# ---------------------------------------------------------------------------
# 1.  One-step exactness.
# ---------------------------------------------------------------------------


def test_one_step_exactness_linear():
    """One ``Phiconf`` step is ``v + dt*kappa*(A*(e_src - beta*v) - v)`` exactly.

    The derived constant is ``kappa' = dt * kappa`` (``dt`` lives in the sharp,
    ``kappa`` in the potential); ``w`` stays frozen at 0.
    """
    A, kappa, dt = 50.0, 0.7, 0.03
    one = stepper(black_amplifier(A, BETA, kappa, dt))
    v0, e = 0.3, 0.5
    out, _, s1 = one(jnp.array([v0, 0.0]), jnp.array([e]), jnp.zeros(1))
    expected = v0 + dt * kappa * (A * (e - BETA * v0) - v0)
    assert abs(float(s1[0]) - expected) < 1e-13
    assert float(s1[1]) == 0.0          # w frozen
    assert float(out) == v0             # the report is the pre-step state


def test_one_step_exactness_nonlinear():
    """With a saturating ``g``: ``v + dt*kappa*(A*g(e_src - beta*v) - v)`` exactly."""
    A, kappa, dt, s = 1000.0, 1.0, 0.008, 0.01
    one = stepper(black_amplifier(A, BETA, kappa, dt, g=_tanh_g(s)))
    v0, e = 2.0, 0.4
    _, _, s1 = one(jnp.array([v0, 0.0]), jnp.array([e]), jnp.zeros(1))
    expected = v0 + dt * kappa * (A * s * np.tanh((e - BETA * v0) / s) - v0)
    assert abs(float(s1[0]) - expected) < 1e-13


def test_bilateral_one_step_feels_the_loop():
    """The reciprocal control's update carries the loop factor ``(1 + A beta)``.

    Gradient descent on the same composite potential:
    ``v <- v + dt*kappa*(1 + A*beta)*(A*(e - beta*v) - v)``.  The contrast with
    ``test_one_step_exactness_linear`` is the kernel condition made visible:
    unilaterality changes the *dynamics*, not the fixed-point equation.
    """
    A, kappa, dt = 50.0, 0.7, 1e-4
    one = stepper(black_amplifier(A, BETA, kappa, dt, unilateral=False))
    v0, e = 0.3, 0.5
    _, _, s1 = one(jnp.array([v0]), jnp.array([e]), jnp.zeros(1))
    expected = v0 + dt * kappa * (1 + A * BETA) * (A * (e - BETA * v0) - v0)
    assert abs(float(s1[0]) - expected) < 1e-12


# ---------------------------------------------------------------------------
# 2.  Black's gain formula as an equilibrium of the composite system.
# ---------------------------------------------------------------------------


def test_black_gain_formula():
    e_src, kappa = 2.0, 1.0
    gains = []
    for A in (1e2, 1e3, 1e4):
        dt = 0.5 / (kappa * (1.0 + A * BETA))   # error factor 1/2 per step
        one = stepper(black_amplifier(A, BETA, kappa, dt))
        v, react, _ = settle(one, amp_state(), e_src, 300)
        G = float(v) / e_src
        np.testing.assert_allclose(G, A / (1.0 + A * BETA), rtol=1e-12)
        assert abs(float(react)) < 1e-6         # the source feels nothing at equilibrium
        gains.append(G)
    # monotonically approaching 1/beta = 10 from below
    assert gains[0] < gains[1] < gains[2] < 1.0 / BETA
    assert abs(gains[2] - 1.0 / BETA) < 1e-3 * (1.0 / BETA)


# ---------------------------------------------------------------------------
# 3.  Desensitization (Black's theorem).
# ---------------------------------------------------------------------------


def test_desensitization():
    """A 10x change of A moves the gain by 0.899% raw; sensitivity 0.0999% < 0.1%.

    Exact for the linear stage: ``dG/G1 = (dA/A1)/(1 + A2 beta) = 9/1001`` and
    ``(dG/G)/(dA/A) = 1/(1 + A2 beta) = 1/1001``.  HONESTY: the raw gain change
    for the factor-10 change of A is 0.899%, NOT below 0.1%; it is the
    *sensitivity* (per unit fractional change of A) that is 1/(1+A*beta) ~ 0.1%.
    """
    e_src, kappa = 2.0, 1.0
    A1, A2 = 1e3, 1e4
    G = {}
    for A in (A1, A2):
        dt = 0.5 / (kappa * (1.0 + A * BETA))
        one = stepper(black_amplifier(A, BETA, kappa, dt))
        v, _, _ = settle(one, amp_state(), e_src, 400)
        G[A] = float(v) / e_src
    dA_rel = (A2 - A1) / A1                    # = 9
    raw = (G[A2] - G[A1]) / G[A1]
    S = raw / dA_rel
    np.testing.assert_allclose(raw, dA_rel / (1.0 + A2 * BETA), rtol=1e-9)
    np.testing.assert_allclose(S, 1.0 / (1.0 + A2 * BETA), rtol=1e-9)
    assert S < 1e-3                            # the < 0.1% claim, as a sensitivity
    assert raw > 1e-3                          # ... and NOT as a raw gain change


# ---------------------------------------------------------------------------
# 4.  Distortion suppression (why the amplifier mattered).
# ---------------------------------------------------------------------------


def _thd(y):
    """Total harmonic distortion of one period sampled at len(y) points."""
    Y = np.fft.rfft(y) / len(y)
    return float(np.sqrt(np.sum(np.abs(Y[2:]) ** 2)) / np.abs(Y[1]))


def test_distortion_suppression():
    """Quasi-static THD, open vs closed loop at matched peak output.

    Stage nonlinearity ``g(e) = s*tanh(e/s)``, driven so the stage input swings
    to ``0.6 s`` (``g'`` down to 0.712 at the peaks).  Measured (64 samples,
    equilibrium per sample through the framework):

        THD_open   ~ 2.755e-2
        THD_closed ~ 3.039e-4
        improvement ~ 90.7   vs   1 + A*beta = 101.

    The classical factor ``1 + A beta`` is the constant-``g'`` first-order
    prediction; with ``g'`` varying over the excursion the honest statement is
    the derived bracket ``1 + A beta g'_min <= improvement <= 1 + A beta``
    (here ``[72.2, 101]``).  We assert the bracket, not the folklore number.
    """
    A, kappa, s = 1000.0, 1.0, 0.01
    g = _tanh_g(s)
    b = 0.6 * s                                # stage-input peak
    Vp = A * s * np.tanh(b / s)                # matched output peak
    u0 = b + BETA * Vp                         # closed-loop drive peak (exact match)
    N = 64
    theta = 2 * np.pi * np.arange(N) / N       # theta[16] = pi/2: the peak is sampled

    # closed loop: settle the framework run at each drive level (quasi-static)
    one_cl = stepper(black_amplifier(A, BETA, kappa, dt=0.008 / kappa, g=g))
    state, y_cl = amp_state(), []
    for th in theta:
        v, _, state = settle(one_cl, state, u0 * np.sin(th), 600)
        y_cl.append(float(v))

    # open loop: same stage, beta = 0, driven for the same peak output
    one_ol = stepper(black_amplifier(A, 0.0, kappa, dt=0.8 / kappa, g=g))
    state, y_ol = amp_state(), []
    for th in theta:
        v, _, state = settle(one_ol, state, b * np.sin(th), 100)
        y_ol.append(float(v))

    y_cl, y_ol = np.array(y_cl), np.array(y_ol)
    np.testing.assert_allclose(y_cl.max(), Vp, rtol=1e-8)   # matched peak output
    np.testing.assert_allclose(y_ol.max(), Vp, rtol=1e-8)

    thd_ol, thd_cl = _thd(y_ol), _thd(y_cl)
    improvement = thd_ol / thd_cl

    # measured numbers (regression-pinned loosely so a silent change surfaces)
    np.testing.assert_allclose(thd_ol, 2.755e-2, rtol=0.02)
    np.testing.assert_allclose(thd_cl, 3.039e-4, rtol=0.02)

    # the honest claim: within the derived bracket around (1 + A*beta)
    gp_min = 1.0 - np.tanh(b / s) ** 2                      # g' at the peak
    assert 1.0 + A * BETA * gp_min <= improvement <= 1.0 + A * BETA
    np.testing.assert_allclose(improvement, 90.66, rtol=0.02)


# ---------------------------------------------------------------------------
# 5.  Stability boundary: no unconditional convergence.
# ---------------------------------------------------------------------------


def test_stability_boundary():
    """Error factor exactly ``1 - dt*kappa*(1 + A*beta)``: divergence past 2.

    Just inside ``dt*kappa*(1+A*beta) = 1.95``: converges (factor -0.95).
    Just outside ``= 2.05``: diverges (factor -1.05).  So the remark cannot
    claim unconditional convergence to the equilibrium.
    """
    A, kappa, e_src = 1e3, 1.0, 2.0
    vstar = A / (1.0 + A * BETA) * e_src
    for prod, diverges in ((1.95, False), (2.05, True)):
        dt = prod / (kappa * (1.0 + A * BETA))
        one = stepper(black_amplifier(A, BETA, kappa, dt))
        lam = 1.0 - dt * kappa * (1.0 + A * BETA)
        state = jnp.array([vstar + 1.0, 0.0])
        _, _, state = one(state, jnp.array([e_src]), jnp.zeros(1))
        # the one-step error amplification is exactly |lam|
        np.testing.assert_allclose(abs(float(state[0]) - vstar), abs(lam), rtol=1e-12)
        for _ in range(399):
            _, _, state = one(state, jnp.array([e_src]), jnp.zeros(1))
        err = abs(float(state[0]) - vstar)
        # 1e-12 above is the exact per-step claim; over 400 steps float rounding
        # accumulates to ~4e-6 relative, so the long-run cross-check gets 1e-5.
        np.testing.assert_allclose(err, abs(lam) ** 400, rtol=1e-5)
        assert (err > 1e6) if diverges else (err < 1e-8)


def test_bilateral_diverges_where_unilateral_converges():
    """Reciprocity makes the loop stiffer by the loop gain.

    At ``dt*kappa*(1+A*beta) = 1.95`` the unilateral loop converges
    (previous test); the bilateral loop's factor is ``1 - 1.95*(1+A*beta)``,
    catastrophically divergent.  Its own boundary is
    ``dt*kappa*(1+A*beta)^2 = 2``.
    """
    A, kappa, e_src = 1e3, 1.0, 2.0
    dt = 1.95 / (kappa * (1.0 + A * BETA))
    one = stepper(black_amplifier(A, BETA, kappa, dt, unilateral=False))
    state = jnp.array([A / (1.0 + A * BETA) * e_src + 1.0])
    for _ in range(6):
        _, _, state = one(state, jnp.array([e_src]), jnp.zeros(1))
    assert abs(float(state[0])) > 1e10


# ---------------------------------------------------------------------------
# 8.  The audit's negative: the equilibrium statement does not need unilaterality.
# ---------------------------------------------------------------------------


def test_bilateral_same_equilibrium():
    """The reciprocal (gradient) closed loop converges to the SAME Black gain.

    Fixed points of both dynamics are the zero locus of the composite
    potential's differential, ``A g(e_src - beta v) = v``; unilaterality plays
    no role in it.  "Equilibria of the composite system" is honest -- but
    attributing the *insensitivity* to unilaterality is not.
    """
    A, kappa, e_src = 1e3, 1.0, 2.0
    dt = 0.5 / (kappa * (1.0 + A * BETA) ** 2)   # inside the bilateral boundary
    one = stepper(black_amplifier(A, BETA, kappa, dt, unilateral=False))
    v, _, _ = settle(one, amp_state(unilateral=False), e_src, 300)
    np.testing.assert_allclose(float(v) / e_src, A / (1.0 + A * BETA), rtol=1e-12)
