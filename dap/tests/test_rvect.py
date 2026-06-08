"""Tests for ``ReactiveVectorSpace`` (def.rvect)."""

import jax
import jax.numpy as jnp
import numpy as np

from dap.rvect import (
    constant,
    diagonal,
    euclidean,
    inverse_hessian,
    trivial,
)


def test_constant_sharp_flat_inverse():
    Q = diagonal(jnp.array([0.5, 2.0, 1.0]))
    q = jnp.zeros(3)
    np.testing.assert_allclose(Q.sharp_at(q) @ Q.flat_at(q), jnp.eye(3), atol=1e-10)


def test_constant_sharp_ignores_position():
    Q = diagonal(jnp.array([3.0, 0.5]))
    np.testing.assert_allclose(Q.sharp_at(jnp.zeros(2)), Q.sharp_at(jnp.array([9.0, -4.0])))


def test_euclidean_learning_rate():
    """eqn.learning_sharp: sharpR_q = eta * I."""
    Q = euclidean(4, eta=0.1)
    np.testing.assert_allclose(Q.sharp_at(jnp.zeros(4)), 0.1 * jnp.eye(4), atol=1e-10)


def test_direct_sum_block_diagonal():
    Q = diagonal(jnp.array([2.0]))
    W = diagonal(jnp.array([5.0, 7.0]))
    QW = Q.direct_sum(W)
    assert QW.dim == 3
    np.testing.assert_allclose(
        QW.sharp_at(jnp.zeros(3)), jnp.diag(jnp.array([2.0, 5.0, 7.0])), atol=1e-10
    )


def test_inverse_hessian_is_position_dependent():
    """sec.newton_warmup: sharpR^U_q = (T_q dU)^{-1}, varying with q."""
    U = lambda q: jnp.exp(q[0]) - q[0]  # Hessian e^{q}, inverse e^{-q}
    Q = inverse_hessian(U, 1)
    for q0 in (-1.0, 0.0, 2.0):
        np.testing.assert_allclose(
            Q.sharp_at(jnp.array([q0])), jnp.array([[np.exp(-q0)]]), atol=1e-10
        )
    # genuinely non-constant
    assert not np.allclose(Q.sharp_at(jnp.array([0.0])), Q.sharp_at(jnp.array([2.0])))


def test_trivial():
    Q0 = trivial()
    assert Q0.dim == 0
    assert Q0.sharp_at(jnp.zeros(0)).shape == (0, 0)
