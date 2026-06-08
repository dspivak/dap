"""Golden end-to-end test: the chain-of-K-particles wave equation via Phiphase.

References sec.wave_equation of dynamic-algebra-potentials.tex:

* particle + wiring: sec.spring_first_pass (eqn.srw_morphism, eqn.sharp_chain).
* state-update equation: eqn.potlens_composite_final.
* discrete wave equation with pinned ends: eqn.discrete_wave, eqn.wave_pin,
  eqn.wave_bc, eqn.state_update_explicit.
"""

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phiphase
from dap.rvect import diagonal
from dap.wiring import compose_chain


def _harmonic_particle(m: float, kappa: float) -> SmoothArrangement:
    """Single-particle arrangement of sec.spring_first_pass:

    Q = R with sharp(p) = p/m, out_f = id, in_f vacuous, U(q, y) = (kappa/2)(q - y)^2.
    """
    Q = diagonal(jnp.array([1.0 / m]))
    return SmoothArrangement(
        Q=Q, out_dim_M=0, in_dim_M=0, out_dim_N=1, in_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2,
        label="Part",
    )


def _composite_final(q, p, q_prev_extern, xi_N_extern, m, kappa):
    """Reference implementation of eqn.potlens_composite_final."""
    K = q.shape[0]
    q_new = q + p / m
    p_new = []
    for i in range(K):
        if i < K - 1:
            q_prev = q_prev_extern if i == 0 else q[i - 1]
            q_next = q[i + 1]
            p_i_new = p[i] + kappa * (q_prev + q_next - 2.0 * q[i])
        else:
            q_prev = q[i - 1] if K > 1 else q_prev_extern
            p_i_new = p[i] + kappa * (q_prev - q[i]) - xi_N_extern
        p_new.append(p_i_new)
    return q_new, jnp.stack(p_new)


_IN_POS = (jnp.zeros(0), (jnp.zeros((0, 0)), jnp.zeros(0)))


def test_state_update_matches_composite_final():
    """Apply Phiphase to a 5-particle chain; check eqn.potlens_composite_final."""
    K = 5
    m, kappa = 1.7, 2.3
    composite = compose_chain([_harmonic_particle(m, kappa)] * K)
    O = Phiphase(composite)

    rng = np.random.default_rng(20260507)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(K))
        p = jnp.asarray(rng.standard_normal(K))
        q_prev = float(rng.standard_normal())
        xi_N = float(rng.standard_normal())

        in_dir = (jnp.array([xi_N]), jnp.array([q_prev]))
        _, _, (new_q, new_p) = O.with_state((q, p)).run_one(_IN_POS, lambda _o: in_dir)

        ref_q, ref_p = _composite_final(q, p, q_prev, xi_N, m, kappa)
        np.testing.assert_allclose(np.asarray(new_q), np.asarray(ref_q), atol=1e-10)
        np.testing.assert_allclose(np.asarray(new_p), np.asarray(ref_p), atol=1e-10)


def test_omega_N_is_harmonic_spring_field():
    """eqn.omegaprime: omega_N(q_0) = -kappa(q_1 - q_0), i.e. A_N = kappa, b_N = -kappa q_1."""
    K = 5
    m, kappa = 1.0, 0.7
    composite = compose_chain([_harmonic_particle(m, kappa)] * K)
    O = Phiphase(composite)

    q = jnp.array([0.4, -0.2, 1.1, 0.0, 2.3])
    p = jnp.zeros(K)
    _, fiber = O.step((q, p))
    out_pos, _ = fiber(_IN_POS)
    out_n, (A_N, b_N) = out_pos
    np.testing.assert_allclose(np.asarray(out_n), np.asarray(q[K - 1:K]), atol=1e-10)
    np.testing.assert_allclose(np.asarray(A_N), np.array([[kappa]]), atol=1e-10)
    np.testing.assert_allclose(np.asarray(b_N), np.array([-kappa * float(q[0])]), atol=1e-10)


def test_discrete_wave_equation_with_pinned_ends():
    """eqn.discrete_wave: m * ddot q_i = kappa (q_{i-1} - 2 q_i + q_{i+1}) at every step.

    Boundary conditions eqn.wave_bc with L = 0: q_0 := 0, xi_N := kappa * q_K.
    The recurrence is an exact algebraic identity at every step (cf. rmk.euler_energy),
    so a short horizon keeps residuals near float64 precision.
    """
    K = 5
    T = 10
    m, kappa = 1.5, 0.9
    composite = compose_chain([_harmonic_particle(m, kappa)] * K)
    O = Phiphase(composite)

    rng = np.random.default_rng(42)
    q0 = jnp.asarray(rng.standard_normal(K))
    p0 = jnp.zeros(K)

    q_traj = [q0]
    state = (q0, p0)
    for _ in range(T + 2):
        q, p = state
        q_prev = jnp.array([0.0])
        xi_N = jnp.array([kappa * float(q[K - 1])])
        _, _, state = O.with_state(state).run_one(_IN_POS, lambda _o: (xi_N, q_prev))
        q_traj.append(state[0])

    q_arr = np.stack([np.asarray(q) for q in q_traj], axis=0)
    for t in range(T):
        ddot_q = q_arr[t + 2] - 2.0 * q_arr[t + 1] + q_arr[t]
        q_aug = np.concatenate([[0.0], q_arr[t], [0.0]])
        laplacian = q_aug[:-2] - 2.0 * q_aug[1:-1] + q_aug[2:]
        residual = m * ddot_q - kappa * laplacian
        np.testing.assert_allclose(residual, np.zeros(K), atol=1e-10)
