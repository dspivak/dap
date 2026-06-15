"""The dynamics functors Phiconf / Phiphase and the cotangent functor (cor.functor)."""

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phiconf, Phiphase, cot_map
from dap.rvect import diagonal


def _harmonic(m, kappa):
    """Single harmonic particle: Q = R with sharp p |-> p/m, out_f = id, U = (k/2)(q-y)^2."""
    Q = diagonal(jnp.array([1.0 / m]))
    return SmoothArrangement(
        Q=Q, out_dim_M=0, in_dim_M=0, out_dim_N=1, in_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2,
        label="Part",
    )


_IN_POS = (jnp.zeros(0), (jnp.zeros((0, 0)), jnp.zeros(0)))


def test_cot_map_pullback():
    """def.cot: cot(f).pos = f, cot(f).dir at x = (T_x f)^T (via jax.vjp)."""
    f = lambda x: jnp.array([x[0] + 2.0 * x[1]])
    fp = cot_map(f, 2, 1)
    x = jnp.array([3.0, 4.0])
    np.testing.assert_allclose(fp.on_position(x), jnp.array([11.0]), atol=1e-10)
    np.testing.assert_allclose(fp.on_direction(x, jnp.array([7.0])),
                               jnp.array([7.0, 14.0]), atol=1e-10)


def test_phase_single_particle_update():
    """eqn.phase_update: (q,p) -> (q~, p - xi_Q), q~ = q + p/m, xi_Q = xi_N + kappa(q~ - q_prev)."""
    m, kappa = 2.0, 3.0
    O = Phiphase(_harmonic(m, kappa))
    q0, p0 = jnp.array([0.7]), jnp.array([1.1])
    xi_N, q_prev = jnp.array([0.4]), jnp.array([0.55])
    qt = q0 + p0 / m  # presented position; forces are evaluated here
    _, _, (new_q, new_p) = O.with_state((q0, p0)).run_one(_IN_POS, lambda _o: (xi_N, q_prev))
    np.testing.assert_allclose(new_q, qt, atol=1e-10)
    np.testing.assert_allclose(new_p, p0 - xi_N + kappa * (q_prev - qt), atol=1e-10)


def test_conf_single_particle_descent():
    """eqn.state_update_gradient: q -> q - sharpR_q(xi_Q); here sharpR = 1/m."""
    m, kappa = 2.0, 3.0
    O = Phiconf(_harmonic(m, kappa))
    q0, xi_N, q_prev = jnp.array([0.7]), jnp.array([0.4]), jnp.array([0.55])
    _, _, new_q = O.with_state(q0).run_one(_IN_POS, lambda _o: (xi_N, q_prev))
    xi_Q = xi_N + kappa * (q0 - q_prev)
    np.testing.assert_allclose(new_q, q0 - xi_Q / m, atol=1e-10)


def test_conf_phase_share_readout():
    """Phiconf and Phiphase share the readout *formulas*; Phiphase just evaluates
    them at the presented position q~ = q + sharp(p) (sec.phase_coalgebra)."""
    m, kappa = 1.3, 0.8
    P = _harmonic(m, kappa)
    q = jnp.array([0.3])
    p = jnp.array([0.5])
    qt = q + p / m  # presented position
    in_dir = (jnp.array([0.1]), jnp.array([0.05]))
    pc, dc, _ = Phiconf(P).with_state(qt).run_one(_IN_POS, lambda _o: in_dir)
    pp, dp, _ = Phiphase(P).with_state((q, p)).run_one(_IN_POS, lambda _o: in_dir)
    np.testing.assert_allclose(pc[0], pp[0], atol=1e-10)        # out_n
    np.testing.assert_allclose(pc[1][0], pp[1][0], atol=1e-10)  # A_N of omega_N
    np.testing.assert_allclose(pc[1][1], pp[1][1], atol=1e-10)  # b_N of omega_N
    np.testing.assert_allclose(dc[0], dp[0], atol=1e-10)        # returned xi_M
