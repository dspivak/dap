"""The heat equation via Phiconf -- the same chain as the wave equation, other semantics.

One diagram, two regimes (cf. the abstract): applied to a chain of harmonic
particles, Phiphase gives the conservative second-order wave equation
(test_wave_equation.py), while Phiconf gives the dissipative first-order discrete
heat equation. With particle sharp p |-> p/m and harmonic potential, the
configuration update q -> q - sharpR_q(xi_Q) is forward Euler for

    m (q_i(t+1) - q_i(t)) = kappa (q_{i-1} - 2 q_i + q_{i+1})(t),

the graph Laplacian of the chain. Unlike the wave/Phiphase case, this is a
genuinely stable, runnable time-stepper for kappa/m small: it dissipates to the
pinned equilibrium q = 0 (contrast rmk.euler_energy).
"""

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phiconf
from dap.rvect import diagonal
from dap.wiring import compose_chain


def _harmonic_particle(m: float, kappa: float) -> SmoothArrangement:
    Q = diagonal(jnp.array([1.0 / m]))
    return SmoothArrangement(
        Q=Q, out_dim_M=0, in_dim_M=0, out_dim_N=1, in_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2,
        label="Part",
    )


_IN_POS = (jnp.zeros(0), (jnp.zeros((0, 0)), jnp.zeros(0)))


def _step_pinned(O, q, kappa):
    """One pinned-end Phiconf step: q_prev = 0, xi_N = kappa * q_K (so q_{K+1} = 0)."""
    K = q.shape[0]
    q_prev = jnp.array([0.0])
    xi_N = jnp.array([kappa * float(q[K - 1])])
    _, _, new_q = O.with_state(q).run_one(_IN_POS, lambda _o: (xi_N, q_prev))
    return new_q


def test_discrete_heat_equation_identity():
    """m (q(t+1) - q(t)) = kappa * Laplacian q(t) at every step (exact, any kappa, m)."""
    K, T = 5, 10
    m, kappa = 1.5, 0.9
    O = Phiconf(compose_chain([_harmonic_particle(m, kappa)] * K))

    rng = np.random.default_rng(3)
    q = jnp.asarray(rng.standard_normal(K))
    traj = [q]
    for _ in range(T):
        q = _step_pinned(O, q, kappa)
        traj.append(q)

    arr = np.stack([np.asarray(x) for x in traj])
    for t in range(T):
        dq = arr[t + 1] - arr[t]
        aug = np.concatenate([[0.0], arr[t], [0.0]])
        laplacian = aug[:-2] - 2.0 * aug[1:-1] + aug[2:]
        np.testing.assert_allclose(m * dq - kappa * laplacian, np.zeros(K), atol=1e-10)


def test_heat_eigenmode_decays_geometrically():
    """A Dirichlet mode decays by the exact factor 1 - (kappa/m) 4 sin^2(n pi / 2(K+1))."""
    K = 7
    m, kappa = 1.0, 0.15  # kappa/m small => stable
    O = Phiconf(compose_chain([_harmonic_particle(m, kappa)] * K))

    js = np.arange(1, K + 1)
    for n in range(1, K + 1):
        q0 = jnp.asarray(np.sin(js * n * np.pi / (K + 1)))
        q1 = _step_pinned(O, q0, kappa)
        factor = 1.0 - (kappa / m) * 4.0 * np.sin(n * np.pi / (2.0 * (K + 1))) ** 2
        np.testing.assert_allclose(np.asarray(q1), factor * np.asarray(q0), atol=1e-10)


def test_heat_dissipates_to_equilibrium():
    """For real: in the stable regime the peak displacement decays monotonically to ~0."""
    K, T = 9, 500
    m, kappa = 1.0, 0.2  # sub-stochastic update => max-norm non-increasing
    O = Phiconf(compose_chain([_harmonic_particle(m, kappa)] * K))

    rng = np.random.default_rng(4)
    q = jnp.asarray(rng.standard_normal(K))
    peaks = [float(np.max(np.abs(np.asarray(q))))]
    for _ in range(T):
        q = _step_pinned(O, q, kappa)
        peaks.append(float(np.max(np.abs(np.asarray(q)))))

    assert peaks[-1] < 1e-3 * peaks[0]                                  # dissipated
    assert all(peaks[i + 1] <= peaks[i] + 1e-12 for i in range(T))      # monotone decay
