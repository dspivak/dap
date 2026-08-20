"""Kuramoto = Phiconf of an edge-box arrangement; swing equation = Phiphase.

Audit of the claim (see ``dap/kuramoto.py``): the Kuramoto model

    theta_i' = omega_i + K * sum_{j ~ i} sin(theta_j - theta_i)

is the configuration dynamics of the closed arrangement with total potential

    U(theta) = -K * sum_{(i,j) in edges} cos(theta_i - theta_j)
               - sum_i omega_i * theta_i,

built compositionally (oscillator boxes + stateless edge coupling boxes +
routing wire, composed with ``compose_seq``/``tensor_arrangements``), and the
swing equation is ``Phiphase`` of the same arrangement data with sharp
``xi/m``. Conventions, as in test_graph_laplacian / test_heat_equation: the
framework's step is unit-time and the time step lives in the sharp --
``sharp_scale = dt`` for Phiconf (forward Euler with step ``dt``),
``sharp_scale = 1/m`` for Phiphase (semi-implicit Euler, unit step, inertia
``m``; resolution comes from taking ``m`` large, exactly as the graph-wave
test takes ``kappa/m`` small).

Checks:

0. the total potential and the routing bijection EMERGE from the composition;
1. one-step exactness of Phiconf against the hand-derived Euler-Kuramoto
   update (pins every sign);
2. two oscillators above threshold (K > |omega_0 - omega_1| / 2): phase
   locking at delta* = arcsin(dw / 2K), both frequencies -> mean(omega);
3. same below threshold: the phase difference drifts without bound;
4. Phiphase, omega = 0: the phase difference oscillates around 0 without
   decay (amplitude preserved, energy bounded), rather than converging.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.functors import Phiconf, Phiphase
from dap.interpretation import trivial_omega
from dap.kuramoto import kuramoto_arrangement, kuramoto_wire

_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def _stepper(O):
    """One closed-system step of the pc-morphism ``O`` as a jitted state map.

    This calls the framework's own ``run_one`` (nothing is re-derived); jit
    only compiles that very computation so the long runs are affordable.
    """

    def f(s):
        _, _, s2 = O.with_state(s).run_one(_IN_POS, _TRIV)
        return s2

    return jax.jit(f)


# A small irregular graph: path 0-1-2-3 plus chord 0-2 (degrees 2, 2, 3, 1).
_N = 4
_EDGES = [(0, 1), (1, 2), (2, 3), (0, 2)]


def _rhs(theta, omegas, edges, K):
    """Oracle: the Kuramoto right-hand side omega_i + K sum_{j~i} sin(theta_j - theta_i)."""
    out = np.array(omegas, dtype=float)
    for (i, j) in edges:
        out[i] += K * np.sin(theta[j] - theta[i])
        out[j] += K * np.sin(theta[i] - theta[j])
    return out


# ---------------------------------------------------------------------------
# 0. The composite is what the claim says it is.
# ---------------------------------------------------------------------------


def test_total_potential_emerges_from_composition():
    """U(theta) = -sum_i omega_i theta_i - K sum_e cos(theta_i - theta_j), from compose_seq."""
    K = 0.7
    omegas = [0.9, -0.4, 0.2, 1.1]
    arr = kuramoto_arrangement(omegas, _EDGES, K, sharp_scale=0.05)
    assert (arr.out_dim_M, arr.in_dim_M, arr.out_dim_N, arr.in_dim_N) == (0, 0, 0, 0)
    assert arr.Q.dim == _N
    rng = np.random.default_rng(0)
    for _ in range(5):
        theta = rng.standard_normal(_N)
        got = float(arr.U(jnp.asarray(theta), jnp.zeros(0), jnp.zeros(0)))
        want = -float(np.dot(omegas, theta)) - K * sum(
            np.cos(theta[i] - theta[j]) for (i, j) in _EDGES
        )
        np.testing.assert_allclose(got, want, atol=1e-12)


def test_wire_routing_is_a_bijection():
    """The wire routes the 2E broadcast out-ports bijectively onto the 2E edge in-ports."""
    wire = kuramoto_wire(_N, _EDGES)
    E2 = 2 * len(_EDGES)
    assert (wire.out_dim_M, wire.in_dim_M, wire.out_dim_N, wire.in_dim_N) == (E2, E2, 0, 0)
    m_out = jnp.arange(E2, dtype=float)
    routed = np.asarray(wire.in_f(jnp.zeros(0), m_out, jnp.zeros(0)))
    assert sorted(routed.tolist()) == list(range(E2))


# ---------------------------------------------------------------------------
# 1. One-step exactness of Phiconf (pins the signs).
# ---------------------------------------------------------------------------


def test_phiconf_one_step_is_euler_kuramoto():
    """One Phiconf step == theta_i + dt * (omega_i + K sum_{j~i} sin(theta_j - theta_i))."""
    K, dt = 0.7, 0.07
    omegas = [0.9, -0.4, 0.2, 1.1]
    O = Phiconf(kuramoto_arrangement(omegas, _EDGES, K, sharp_scale=dt))
    rng = np.random.default_rng(1)
    for _ in range(5):
        theta = rng.standard_normal(_N)
        _, _, new = O.with_state(jnp.asarray(theta)).run_one(_IN_POS, _TRIV)
        want = theta + dt * _rhs(theta, omegas, _EDGES, K)
        np.testing.assert_allclose(np.asarray(new), want, rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------
# 2 & 3. Two oscillators: locking above threshold, drift below.
# ---------------------------------------------------------------------------

_TWO_OMEGAS = (0.8, -0.2)  # dw = 1.0, mean = 0.3; threshold K_c = dw/2 = 0.5


def _run_two(K, dt, steps, theta0=(0.1, 0.0)):
    O = Phiconf(kuramoto_arrangement(list(_TWO_OMEGAS), [(0, 1)], K, sharp_scale=dt))
    step = _stepper(O)
    s = jnp.asarray(theta0)
    traj = np.empty((steps + 1, 2))
    traj[0] = np.asarray(s)
    for t in range(steps):
        s = step(s)
        traj[t + 1] = np.asarray(s)
    return traj


def test_two_oscillators_lock_above_threshold():
    """K = 0.8 > 0.5: delta -> arcsin(dw/2K), both frequencies -> mean(omega)."""
    K, dt, steps = 0.8, 0.02, 5000
    traj = _run_two(K, dt, steps)
    delta = traj[:, 0] - traj[:, 1]
    dw = _TWO_OMEGAS[0] - _TWO_OMEGAS[1]

    # The locked difference: delta* = arcsin(dw / 2K) is a fixed point of the
    # discrete map too (the per-step increment of delta is dt*(dw - 2K sin delta)),
    # so convergence is to it exactly, not merely O(dt)-near it.
    delta_star = np.arcsin(dw / (2 * K))
    np.testing.assert_allclose(delta[-1], delta_star, rtol=0, atol=1e-10)
    # ...and it has genuinely settled: constant over the last 500 steps.
    assert np.max(np.abs(delta[-500:] - delta[-1])) < 1e-10

    # Both frequencies converge to the mean frequency (0.3): measured over the
    # last 1000 steps, each oscillator advances at mean(omega) per unit time.
    window = 1000
    freqs = (traj[-1] - traj[-1 - window]) / (window * dt)
    np.testing.assert_allclose(freqs, np.mean(_TWO_OMEGAS), rtol=0, atol=1e-8)


def test_two_oscillators_drift_below_threshold():
    """K = 0.3 < 0.5: the phase difference never settles -- it drifts without bound."""
    K, dt, steps = 0.3, 0.02, 3000
    traj = _run_two(K, dt, steps)
    delta = traj[:, 0] - traj[:, 1]

    # d(delta)/dt = dw - 2K sin(delta) >= 1.0 - 0.6 = 0.4 > 0: monotone drift.
    assert np.all(np.diff(delta) > 0)
    # Far more than a full turn -- delta(T) - delta(0) >= 0.4 * T = 24 > 4 pi.
    assert delta[-1] - delta[0] > 4 * np.pi
    # No locking: the two measured frequencies stay distinct (by ~sqrt(dw^2-4K^2)).
    window = 1000
    freqs = (traj[-1] - traj[-1 - window]) / (window * dt)
    assert freqs[0] - freqs[1] > 0.4


# ---------------------------------------------------------------------------
# 4. Phiphase on the same arrangement data: swing oscillation, no decay.
# ---------------------------------------------------------------------------


def test_phiphase_oscillates_without_decay():
    """omega = 0, sharp = xi/m: delta oscillates around 0, amplitude and energy preserved."""
    K, m, steps = 1.0, 400.0, 2500
    arr = kuramoto_arrangement([0.0, 0.0], [(0, 1)], K, sharp_scale=1.0 / m)
    O = Phiphase(arr)
    step = _stepper(O)

    theta0 = jnp.array([0.5, -0.5])  # delta_0 = 1 rad, at rest
    s = (theta0, jnp.zeros(2))
    delta = np.empty(steps + 1)
    energy = np.empty(steps + 1)

    def _energy(state):
        q, xi = state
        return float(jnp.sum(xi**2) / (2 * m) + arr.U(q, jnp.zeros(0), jnp.zeros(0)))

    delta[0], energy[0] = float(theta0[0] - theta0[1]), _energy(s)
    for t in range(steps):
        s = step(s)
        delta[t + 1] = float(s[0][0] - s[0][1])
        energy[t + 1] = _energy(s)

    # Pendulum in delta: m ddot(delta) = -2K sin(delta), linear frequency
    # sqrt(2K/m) ~ 0.0707/step, period ~ 95 steps at amplitude 1 -> ~26 periods.
    # (a) It oscillates: many sign changes, symmetric about 0.
    crossings = int(np.sum(np.diff(np.sign(delta)) != 0))
    assert crossings > 20
    assert abs(np.mean(delta)) < 0.05

    # (b) No decay: the peak amplitude over the last 500 steps matches the
    # first 500 (>= 5 periods each) to a few percent -- and in particular has
    # not converged toward the minimum delta = 0 (contrast Phiconf, which
    # would settle there).
    a_first = np.max(np.abs(delta[:500]))
    a_last = np.max(np.abs(delta[-500:]))
    assert 0.95 < a_last / a_first < 1.05
    assert a_last > 0.9  # still swinging out to ~ the initial 1 rad

    # (c) Energy-conserving-ish: semi-implicit Euler keeps the energy in a
    # bounded band around E_0 (relative to the well depth E_0 - U_min), with
    # no systematic trend.
    U_min = -K  # U at delta = 0 (the omega terms vanish)
    band = np.max(np.abs(energy - energy[0])) / (energy[0] - U_min)
    assert band < 0.05
