"""End-to-end test: the chain-of-K-particles Klein-Gordon equation via Phiphase.

The Klein-Gordon chain extends the harmonic chain (sec.wave_equation) by adding
an on-site mass term ``(m_0^2/2) q^2`` to each particle's potential:

    U(q, n_in) = (kappa/2)(q - n_in)^2 + (m_0^2/2) q^2.

The discrete recurrence is

    m (q_i(t+2) - 2 q_i(t+1) + q_i(t)) = kappa (q_{i-1} - 2 q_i + q_{i+1})(t) - m_0^2 q_i(t),

and for a pinned chain of length K the Dirichlet eigenmodes are
``q_j^{(n)} = sin(j n pi / (K+1))`` with
``omega_n^2 = m_0^2/m + (kappa/m) 4 sin^2(n pi / (2(K+1)))``.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest

from dap.interpretation import trivial_omega
from dap.arrangement import SmoothArrangement
from dap.functors import Phiphase
from dap.rvect import diagonal
from dap.wiring import compose_chain


def _kg_particle(m: float, kappa: float, m0: float) -> SmoothArrangement:
    Q = diagonal(jnp.array([1.0 / m]))
    return SmoothArrangement(
        Q=Q, out_dim_M=0, in_dim_M=0, out_dim_N=1, in_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2 + 0.5 * (m0 * m0) * q[0] * q[0],
        label="KGPart",
    )


def _kg_composite_final(q, p, q_prev_extern, xi_N_extern, m, kappa, m0):
    """Reference for the KG analogue of eqn.potlens_composite_final (on-site -m_0^2 q_i)."""
    K = q.shape[0]
    qt = q + p / m  # presented positions q~ (forces are evaluated here)
    q_new = qt
    p_new = []
    for i in range(K):
        if i < K - 1:
            q_prev = q_prev_extern if i == 0 else qt[i - 1]
            q_next = qt[i + 1]
            p_i_new = p[i] + kappa * (q_prev + q_next - 2.0 * qt[i]) - (m0 * m0) * qt[i]
        else:
            q_prev = qt[i - 1] if K > 1 else q_prev_extern
            p_i_new = p[i] + kappa * (q_prev - qt[i]) - xi_N_extern - (m0 * m0) * qt[i]
        p_new.append(p_i_new)
    return q_new, jnp.stack(p_new)


_IN_POS = (jnp.zeros(0), trivial_omega(0))


def _step_pinned(O, state, kappa, m):
    """One pinned-end step: q_prev = 0, xi_N = kappa * q~_K (presented), q~_{K+1} = 0."""
    q, p = state
    K = q.shape[0]
    q_prev = jnp.array([0.0])
    xi_N = jnp.array([kappa * float(q[K - 1] + p[K - 1] / m)])
    _, _, new_state = O.with_state(state).run_one(_IN_POS, lambda _o: (xi_N, q_prev))
    return new_state


def test_kg_state_update():
    K = 5
    m, kappa, m0 = 1.7, 2.3, 0.4
    composite = compose_chain([_kg_particle(m, kappa, m0)] * K)
    O = Phiphase(composite)

    rng = np.random.default_rng(20260508)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(K))
        p = jnp.asarray(rng.standard_normal(K))
        q_prev = float(rng.standard_normal())
        xi_N = float(rng.standard_normal())

        in_dir = (jnp.array([xi_N]), jnp.array([q_prev]))
        _, _, (new_q, new_p) = O.with_state((q, p)).run_one(_IN_POS, lambda _o: in_dir)

        ref_q, ref_p = _kg_composite_final(q, p, q_prev, xi_N, m, kappa, m0)
        np.testing.assert_allclose(np.asarray(new_q), np.asarray(ref_q), atol=1e-10)
        np.testing.assert_allclose(np.asarray(new_p), np.asarray(ref_p), atol=1e-10)


def test_kg_recurrence():
    K, T = 5, 10
    m, kappa, m0 = 1.5, 0.9, 0.3
    composite = compose_chain([_kg_particle(m, kappa, m0)] * K)
    O = Phiphase(composite)

    rng = np.random.default_rng(20260508)
    q0 = jnp.asarray(rng.standard_normal(K))
    state = (q0, jnp.zeros(K))
    q_traj = [q0]
    for _ in range(T + 2):
        state = _step_pinned(O, state, kappa, m)
        q_traj.append(state[0])

    q_arr = np.stack([np.asarray(q) for q in q_traj], axis=0)
    for t in range(T):
        ddot_q = q_arr[t + 2] - 2.0 * q_arr[t + 1] + q_arr[t]
        q_aug = np.concatenate([[0.0], q_arr[t + 1], [0.0]])
        laplacian = q_aug[:-2] - 2.0 * q_aug[1:-1] + q_aug[2:]
        residual = m * ddot_q - kappa * laplacian + (m0 * m0) * q_arr[t + 1]
        np.testing.assert_allclose(residual, np.zeros(K), atol=1e-10)


def test_kg_dispersion():
    """Initialize in Dirichlet eigenmode n, evolve two steps, check ddot q = -omega_n^2 q."""
    K = 7
    m, kappa, m0 = 1.3, 0.8, 0.5
    composite = compose_chain([_kg_particle(m, kappa, m0)] * K)
    O = Phiphase(composite)

    js = np.arange(1, K + 1)
    for n in range(1, K + 1):
        q0 = jnp.asarray(np.sin(js * n * np.pi / (K + 1)))
        state = (q0, jnp.zeros(K))
        q_traj = [q0]
        for _ in range(2):
            state = _step_pinned(O, state, kappa, m)
            q_traj.append(state[0])
        q_arr = np.stack([np.asarray(q) for q in q_traj], axis=0)
        ddot_q = q_arr[2] - 2.0 * q_arr[1] + q_arr[0]
        omega2 = (m0 * m0) / m + (kappa / m) * 4.0 * np.sin(n * np.pi / (2.0 * (K + 1))) ** 2
        np.testing.assert_allclose(ddot_q, -omega2 * np.asarray(q0), atol=1e-10)


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"),
                    reason="long-horizon stability diagnostic; set RUN_SLOW=1 to enable")
def test_kg_long_horizon_bounded():
    """The phase integrator is symplectic (presented-position readout), so over a
    long horizon the amplitude stays bounded rather than diverging."""
    K, T = 5, 2000
    m, kappa, m0 = 1.5, 0.9, 0.3
    composite = compose_chain([_kg_particle(m, kappa, m0)] * K)
    O = Phiphase(composite)

    rng = np.random.default_rng(20260508)
    q0 = jnp.asarray(1e-3 * rng.standard_normal(K))
    peak0 = float(np.max(np.abs(np.asarray(q0))))
    state = (q0, jnp.zeros(K))
    peak = peak0
    for _ in range(T):
        state = _step_pinned(O, state, kappa, m)
        peak = max(peak, float(np.max(np.abs(np.asarray(state[0])))))
    assert peak < 50.0 * peak0
