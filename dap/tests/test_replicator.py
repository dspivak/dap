"""Replicator dynamics as Phiconf of the Shahshahani arrangement (dap/replicator.py).

Four checks:
* one Phiconf step from a simplex point IS the forward-Euler replicator step
  ``x + dt * (x_i ((A x)_i - x^T A x))_i``, to machine precision (keystone);
* the Shahshahani sharp is symmetric with the constants in its kernel exactly
  on the simplex, and ``sum(x) = 1`` is preserved to rounding level (measured
  max drift ~3e-16 over 5000 steps, non-accumulating -- the drift is
  self-correcting, see the module docstring);
* convergence to the ESS: a diagonal coordination game reaches the pure
  strategy of the largest payoff from the barycenter, and (column-symmetrized)
  Hawk-Dove reaches the mixed ESS ``x_Hawk = v/c`` from interior starts.

Non-symmetric games (rock-paper-scissors) are OUT OF SCOPE by construction --
they are not potential games -- and the arrangement constructor rejects them;
we test the rejection, not the dynamics.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from dap.replicator import (
    replicator_arrangement,
    replicator_dynamics,
    replicator_step,
    shahshahani,
)


def _random_symmetric(rng, n):
    B = rng.standard_normal((n, n))
    return jnp.asarray((B + B.T) / 2.0)


def _random_simplex(rng, n):
    return jnp.asarray(rng.dirichlet(np.ones(n)))


def test_one_step_is_euler_replicator():
    """Keystone: Phiconf step = x + dt * (x_i((Ax)_i - x^T A x))_i, machine precision."""
    rng = np.random.default_rng(0)
    for n in (2, 3, 5):
        A = _random_symmetric(rng, n)
        for dt in (1.0, 0.05):
            O = replicator_dynamics(A, dt)
            for _ in range(3):
                x = _random_simplex(rng, n)
                new_x = replicator_step(O, x)
                Ax = A @ x
                exact = x + dt * (x * (Ax - jnp.dot(x, Ax)))
                np.testing.assert_allclose(
                    np.asarray(new_x), np.asarray(exact), rtol=0.0, atol=1e-14
                )


def test_sharp_kernel_and_simplex_invariance():
    """sharpR_x is symmetric; ker contains constants iff sum(x)=1; sum drifts only at rounding."""
    rng = np.random.default_rng(1)
    n, dt = 3, 0.05
    Q = shahshahani(n, dt)

    # Symmetric, and constants in the kernel exactly on the simplex:
    # sharpR_x(1) = dt * x * (1 - sum x).
    x_on = _random_simplex(rng, n)
    S = Q.sharp_at(x_on)
    np.testing.assert_allclose(np.asarray(S), np.asarray(S.T), atol=0.0)
    np.testing.assert_allclose(np.asarray(S @ jnp.ones(n)), np.zeros(n), atol=1e-15)
    x_off = 1.7 * x_on  # sum = 1.7: constants NOT in the kernel off the simplex
    S_off = Q.sharp_at(x_off)
    np.testing.assert_allclose(
        np.asarray(S_off @ jnp.ones(n)), np.asarray(dt * x_off * (1.0 - 1.7)), atol=1e-14
    )
    assert float(jnp.max(jnp.abs(S_off @ jnp.ones(n)))) > 1e-3

    # Invariance of sum(x) = 1 along the Euler orbit. Not exact in floats: the
    # per-step increment is dt * (x^T A x) * (1 - sum x) plus rounding, so the
    # drift stays at rounding level and does not accumulate (measured max
    # ~3e-16 over 5000 steps); we assert the rounding-level bound, not exactness.
    A = _random_symmetric(rng, n) + 2.0 * jnp.eye(n) + 1.0  # positive on the orthant
    O = replicator_dynamics(A, dt)
    x = _random_simplex(rng, n)
    max_drift = 0.0
    for _ in range(2000):
        x = replicator_step(O, x)
        max_drift = max(max_drift, abs(float(jnp.sum(x)) - 1.0))
    assert max_drift < 1e-13


def test_coordination_game_converges_to_dominant_pure_strategy():
    """A = diag(3,2,1): from the barycenter the largest-payoff pure strategy wins."""
    A = jnp.diag(jnp.array([3.0, 2.0, 1.0]))
    O = replicator_dynamics(A, dt=0.1)
    x = jnp.ones(3) / 3.0
    fitness = [float(jnp.dot(x, A @ x))]
    for _ in range(2000):
        x = replicator_step(O, x)
        fitness.append(float(jnp.dot(x, A @ x)))
    np.testing.assert_allclose(np.asarray(x), np.array([1.0, 0.0, 0.0]), atol=1e-8)
    # descent of U = ascent of mean fitness, monotone in this run
    assert all(f2 >= f1 - 1e-12 for f1, f2 in zip(fitness, fitness[1:]))


def test_hawk_dove_converges_to_mixed_ess():
    """Column-symmetrized Hawk-Dove (v=2, c=3) reaches the mixed ESS x_H = v/c = 2/3.

    The textbook matrix [[(v-c)/2, v], [0, v/2]] is not symmetric, but replicator
    dynamics is invariant under column shifts A -> A + 1 c^T; the shift c = (v, 0)
    gives the symmetric representative [[(v-c)/2 + v, v], [v, v/2]] with the same
    dynamics, and THAT is the partnership game fed to the construction.
    """
    v, c = 2.0, 3.0
    A = jnp.array([[(v - c) / 2.0 + v, v], [v, v / 2.0]])
    O = replicator_dynamics(A, dt=0.1)
    for x0 in (jnp.array([0.15, 0.85]), jnp.array([0.9, 0.1])):
        x = x0
        for _ in range(3000):
            x = replicator_step(O, x)
        np.testing.assert_allclose(
            np.asarray(x), np.array([v / c, 1.0 - v / c]), atol=1e-10
        )


def test_rock_paper_scissors_is_rejected():
    """RPS is not a partnership game; the constructor refuses it (scope limit)."""
    rps = jnp.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    with pytest.raises(ValueError):
        replicator_arrangement(rps, dt=0.1)
