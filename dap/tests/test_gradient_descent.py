"""Gradient descent and backpropagation via Phiconf (sec.dl_warmup).

Three checks:
* the lens backward pass equals reverse-mode autodiff (backprop = chain rule);
* one Phiconf step on a Euclidean parameter space is vanilla gradient descent;
* the training loop actually fits a realizable regression (it runs, for real).
"""

import jax
import jax.numpy as jnp
import numpy as np

from dap.interpretation import trivial_omega
from dap.functors import Phiconf
from dap.learning import (
    forward_backward,
    parameterized_map,
    squared_error,
    squared_error_grad,
    train,
)
from dap.rvect import euclidean


def _two_layer(h, m, n):
    """A tanh two-layer net F(q, x) = W2 tanh(W1 x + b1) + b2, params flattened into q."""
    shapes = [(h, m), (h,), (n, h), (n,)]
    sizes = [int(np.prod(s)) for s in shapes]
    dim = sum(sizes)

    def unpack(q):
        out, i = [], 0
        for s, k in zip(shapes, sizes):
            out.append(q[i:i + k].reshape(s))
            i += k
        return out

    def F(q, x):
        W1, b1, W2, b2 = unpack(q)
        return W2 @ jnp.tanh(W1 @ x + b1) + b2

    return F, dim


def test_backprop_matches_autodiff():
    """eqn.dl_pullback: (xi_Q, xi_M) = (T F)^T xi equals jax.grad of the loss."""
    F, dim = _two_layer(4, 3, 2)
    arr = parameterized_map(F, euclidean(dim), in_dim=3, out_dim=2)

    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal(dim))
    x = jnp.asarray(rng.standard_normal(3))
    lam = jnp.asarray(rng.standard_normal(2))

    y0 = F(q, x)
    y, xi_Q, xi_M = forward_backward(arr, q, x, squared_error_grad(y0, lam))

    np.testing.assert_allclose(np.asarray(y), np.asarray(y0), atol=1e-12)
    grad_q = jax.grad(lambda qq: squared_error(F(qq, x), lam))(q)
    grad_x = jax.grad(lambda xx: squared_error(F(q, xx), lam))(x)
    np.testing.assert_allclose(np.asarray(xi_Q), np.asarray(grad_q), atol=1e-9)
    np.testing.assert_allclose(np.asarray(xi_M), np.asarray(grad_x), atol=1e-9)


def test_phiconf_step_is_gradient_descent():
    """eqn.dl_gradient_update with eqn.learning_sharp: new q = q - eta * grad(loss)."""
    F, dim = _two_layer(4, 3, 2)
    eta = 0.05
    arr = parameterized_map(F, euclidean(dim, eta), in_dim=3, out_dim=2)
    O = Phiconf(arr)

    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal(dim))
    x = jnp.asarray(rng.standard_normal(3))
    lam = jnp.asarray(rng.standard_normal(2))

    in_pos = (x, trivial_omega(0))
    _, _, new_q = O.with_state(q).run_one(
        in_pos, lambda op: (squared_error_grad(op[0], lam), jnp.zeros(0))
    )
    grad_q = jax.grad(lambda qq: squared_error(F(qq, x), lam))(q)
    np.testing.assert_allclose(np.asarray(new_q), np.asarray(q - eta * grad_q), atol=1e-9)


def test_training_fits_realizable_regression():
    """The training loop converges: a linear model fits realizable data to ~0 loss."""
    m, n = 3, 2
    dim = n * m + n

    def F(q, x):
        W = q[: n * m].reshape(n, m)
        b = q[n * m:]
        return W @ x + b

    rng = np.random.default_rng(2)
    W_true = jnp.asarray(rng.standard_normal((n, m)))
    b_true = jnp.asarray(rng.standard_normal(n))
    data = []
    for _ in range(20):
        x = jnp.asarray(rng.standard_normal(m))
        data.append((x, W_true @ x + b_true))

    arr = parameterized_map(F, euclidean(dim, eta=0.05), in_dim=m, out_dim=n)
    q, history = train(arr, jnp.zeros(dim), data, steps=5000)

    full_batch_loss = float(np.mean([float(squared_error(F(q, x), lam)) for x, lam in data]))
    assert full_batch_loss < 1e-4
    assert np.mean(history[-50:]) < np.mean(history[:50])
