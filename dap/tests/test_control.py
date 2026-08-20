"""Impedance (P-)control as a virtual spring (dap/control.py) -- audit tests.

The claim under test: a stateless controller box whose potential is
``U_c(q_rep, s) = (k/2)(q_rep - s)^2``, wired to a zero-potential plant, with
the setpoint ``s`` a genuine run-time input on the outer boundary, gives

* under ``Phiconf``:  geometric convergence of the plant state to ``s``;
* under ``Phiphase``: undamped oscillation about ``s`` at frequency
  ``~ sqrt(k/m)`` (the classical stiffness-only impedance controller).

Timestep convention: the framework has no separate ``dt``; it lives in the
sharp (eqn.learning_sharp: ``euclidean(1, dt)`` makes one conf step
``q <- q + dt * (-k(q - s))``) and, for phase dynamics, one coalgebra step is
one unit of time with ``sharp = xi/m`` (as in test_wave_equation), so "small
dt" means small ``k/m``.
"""

import jax.numpy as jnp
import numpy as np

from dap.control import impedance_arrangement, run_tracking
from dap.functors import Phiconf, Phiphase
from dap.interpretation import trivial_omega
from dap.rvect import diagonal, euclidean

_IN_POS = (jnp.zeros(0), trivial_omega(0))


def _one_step(O, state, s):
    """One coalgebra step with setpoint ``s`` and zero boundary covector."""
    in_dir = (jnp.zeros(1), jnp.array([float(s)]))
    return O.with_state(state).run_one(_IN_POS, lambda _op: in_dir)


# ---------------------------------------------------------------------------
# 1. One-step exactness under Phiconf.
# ---------------------------------------------------------------------------


def test_conf_one_step_is_exact_virtual_spring_descent():
    """One Phiconf step equals ``q + dt * (-k (q - s))`` to numerical precision.

    Plant potential is zero (the honest simplest choice), so the *entire*
    parameter covector is the controller's ``xi_Q = dU_c/dq = k(q - s)`` and
    the conf integrator applies ``q <- q - sharpR_q(xi_Q)`` with
    ``sharpR = dt * I`` (eqn.conf_integrator).
    """
    dt, k = 0.07, 1.3
    arr = impedance_arrangement(euclidean(1, dt), k)
    O = Phiconf(arr)

    rng = np.random.default_rng(20260820)
    for _ in range(5):
        q0 = jnp.array([rng.standard_normal()])
        s = float(rng.standard_normal())
        out_pos, _out_dir, new_q = _one_step(O, q0, s)
        expected = q0 + dt * (-k * (q0 - s))
        np.testing.assert_allclose(np.asarray(new_q), np.asarray(expected), atol=1e-12)
        # conf readout is the current parameter position q itself
        np.testing.assert_allclose(np.asarray(out_pos[0]), np.asarray(q0), atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Step response: convergence, and a mid-run setpoint change.
# ---------------------------------------------------------------------------


def test_conf_step_response_tracks_runtime_setpoint():
    """Constant setpoint => exact geometric convergence, ratio (1 - dt k) per
    step; changing the setpoint mid-run (same coalgebra, same threaded state,
    only the per-step input changes) re-converges to the new value."""
    dt, k = 0.05, 2.0          # contraction factor 1 - dt k = 0.9
    rho = 1.0 - dt * k
    s1, s2, T = 1.5, -0.7, 120
    arr = impedance_arrangement(euclidean(1, dt), k)
    O = Phiconf(arr)

    setpoints = [s1] * T + [s2] * T
    _, states = run_tracking(O, jnp.array([0.0]), setpoints)
    q = np.array([float(st[0]) for st in states])  # q_0 .. q_{2T}

    # phase 1: exact geometric contraction toward s1 at every step
    e1 = q[: T + 1] - s1
    np.testing.assert_allclose(e1[1:], rho * e1[:-1], atol=1e-12)
    assert abs(q[T] - s1) < 1e-4          # |e| = 1.5 * 0.9^120 ~ 4.7e-6

    # phase 2: the SAME run re-converges to the new setpoint s2
    e2 = q[T: 2 * T + 1] - s2
    np.testing.assert_allclose(e2[1:], rho * e2[:-1], atol=1e-12)
    assert abs(q[2 * T] - s2) < 1e-4


# ---------------------------------------------------------------------------
# 3. Phiphase: undamped oscillation about the setpoint at ~ sqrt(k/m).
# ---------------------------------------------------------------------------


def test_phase_oscillates_about_setpoint_at_sqrt_k_over_m():
    """Same construction, phase integrator: semi-implicit Euler on the
    harmonic oscillator ``m qdd = -k(q - s)``.

    One step is ``q <- q + p/m``, ``p <- p - k(q~ - s)`` (forces at the
    presented position ``q~``), whose exact rotation angle per step is
    ``theta = arccos(1 - k/(2m))``; ``k_eff = k`` (the only stiffness in the
    construction is the controller's), so for small ``k/m`` the period is
    ``~ 2 pi / sqrt(k/m)`` steps. Tolerances: measured period vs the exact
    discrete angle to 0.5% (zero-crossing interpolation error over ~38
    averaged periods is far below this); vs the continuum ``sqrt(k/m)`` to 1%
    (the discretization offset is ``theta/sqrt(k/m) - 1 ~ k/(24 m) ~ 0.17%``).
    Amplitude bounds +-15%: symplectic Euler preserves a modified quadratic
    invariant, so the amplitude wobbles by O(theta) ~ 20% at most per phase of
    the modified ellipse but has no secular drift; we check both no decay and
    no growth between the first and last quarters of the run.
    """
    m, k = 1.0, 0.04           # k_eff/m = 0.04, theta ~ 0.2 rad/step
    s_star, A = 0.5, 1.0
    T = 1200                   # ~38 periods
    arr = impedance_arrangement(diagonal(jnp.array([1.0 / m])), k)
    O = Phiphase(arr)

    q0 = jnp.array([s_star + A])
    p0 = jnp.array([0.0])
    outs, states = run_tracking(O, (q0, p0), [s_star] * T)

    x = np.array([float(o[0]) for o in outs]) - s_star  # presented positions
    q = np.array([float(st[0][0]) for st in states]) - s_star  # true positions

    # -- period from zero crossings of the reported (presented) position --
    idx = np.nonzero(x[:-1] * x[1:] < 0)[0]
    assert len(idx) >= 60      # it genuinely oscillates
    t_cross = idx + x[idx] / (x[idx] - x[idx + 1])  # linear interpolation
    period = 2.0 * (t_cross[-1] - t_cross[0]) / (len(t_cross) - 1)

    theta_exact = np.arccos(1.0 - k / (2.0 * m))
    assert abs(period - 2.0 * np.pi / theta_exact) / period < 0.005
    assert abs(period - 2.0 * np.pi / np.sqrt(k / m)) / period < 0.01

    # -- no convergence, no divergence: amplitude neither decays nor grows --
    first, last = np.max(np.abs(q[: T // 4])), np.max(np.abs(q[-T // 4:]))
    assert 0.85 * A < first < 1.15 * A
    assert 0.85 * A < last < 1.15 * A
    assert 0.9 < last / first < 1.1


# ---------------------------------------------------------------------------
# 4. Two-sided accounting: equal-and-opposite covectors on the setpoint port.
# ---------------------------------------------------------------------------


def test_setpoint_port_receives_equal_and_opposite_covector():
    """The composite EMITS, in its outer position, the covector field
    ``omega_N(s) = dU_c/ds = -k(q_eval - s)`` on the setpoint port
    (eqn.omegaprime) -- equal and opposite to the parameter covector
    ``xi_Q = dU_c/dq = k(q_eval - s)`` the integrator consumes, because both
    are gradients of the one shared potential (Newton's third law of the
    virtual spring). ``q_eval`` is the integrator's presented position: ``q``
    for conf, ``q~ = q + p/m`` for phase.

    Our runs (tests 2-3) DISCARD this covector: physically, the setpoint
    holder is an infinitely stiff agent whose work against the spring is
    untracked by the closed loop.
    """
    # --- Phiconf: xi_Q recovered from the state update q' = q - dt * xi_Q ---
    dt, k = 0.05, 2.0
    q0, s = 1.3, 0.4
    O = Phiconf(impedance_arrangement(euclidean(1, dt), k))
    out_pos, _out_dir, new_q = _one_step(O, jnp.array([q0]), s)
    omega_N = out_pos[1]                       # the emitted covector FIELD
    omega_at_s = float(omega_N(jnp.array([s]))[0])
    xi_Q = float((jnp.array([q0]) - new_q)[0]) / dt
    np.testing.assert_allclose(xi_Q, k * (q0 - s), atol=1e-10)
    np.testing.assert_allclose(omega_at_s, -k * (q0 - s), atol=1e-10)
    np.testing.assert_allclose(omega_at_s + xi_Q, 0.0, atol=1e-10)

    # --- Phiphase: xi_Q recovered from the momentum update p' = p - xi_Q ---
    m, k = 0.8, 0.6
    q0, p0, s = -0.9, 0.35, 0.25
    O = Phiphase(impedance_arrangement(diagonal(jnp.array([1.0 / m])), k))
    out_pos, _out_dir, (new_q, new_p) = _one_step(O, (jnp.array([q0]), jnp.array([p0])), s)
    q_tilde = q0 + p0 / m                      # forces evaluated at q~
    omega_at_s = float(out_pos[1](jnp.array([s]))[0])
    xi_Q = float((jnp.array([p0]) - new_p)[0])
    np.testing.assert_allclose(xi_Q, k * (q_tilde - s), atol=1e-10)
    np.testing.assert_allclose(omega_at_s, -k * (q_tilde - s), atol=1e-10)
    np.testing.assert_allclose(omega_at_s + xi_Q, 0.0, atol=1e-10)
