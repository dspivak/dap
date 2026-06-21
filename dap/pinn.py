"""A physics-informed network: deep-Ritz for 1D Poisson, via ``Phiconf``.

EXTENSION (beyond the paper). This reuses the paper's configuration functor
``Phiconf`` and the learner of sec.dl_warmup unchanged, but the deep-Ritz Dirichlet
energy, the coordinate MLP ansatz, and the Poisson discretization are standard
numerics, not part of the paper's formal development. It is a toy demonstration
that the primitives compose -- not an implementation of a paper result. See the
"Extensions" section of the README.

A field network ``u_theta : [0,1] -> R`` is trained to solve the boundary-value
problem ``-u'' = f`` on ``[0,1]`` with ``u(0) = u(1) = 0`` by descending the
discrete *Dirichlet energy* (the variational / deep-Ritz form)

    E(u) = 1/2 u . L u - b . u,     L = tridiag(-1, 2, -1),  b_i = h^2 f(x_i),

whose unique minimizer is the discrete Poisson solution ``L u = b`` (the
second-difference of ``-u''=f``). The network outputs the field on a grid; under
the configuration functor ``Phiconf`` (sec.dl_warmup) its weights descend ``E``,
the gradient being ``dE/du = L u - b`` pulled back through the net. Hard Dirichlet
boundary conditions are baked in by the ansatz ``u_theta(x) = x(1-x) g_theta(x)``.

This is the *energy* (deep-Ritz) reading of a physics-informed network: the loss
is the field's own Dirichlet energy, assembled as the paper's harmonic bonds are
(rmk.symmetric_bonds). It is elliptic only -- the wave equation's action is
indefinite and cannot be descended (see the conclusion).
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .learning import parameterized_map, train
from .rvect import euclidean


# ---------------------------------------------------------------------------
# The discrete 1D Poisson problem.
# ---------------------------------------------------------------------------


def poisson_1d(V: int, f: Callable[[Array], Array]) -> Tuple[Array, Array, Array, Array]:
    """Discrete ``-u'' = f`` on ``[0,1]``, Dirichlet BC, ``V`` interior points.

    Returns ``(x, L, b, u_star)``: the interior grid ``x_i = i h`` (``h=1/(V+1)``),
    the second-difference matrix ``L = tridiag(-1, 2, -1)``, the right-hand side
    ``b_i = h^2 f(x_i)``, and the exact discrete solution ``u_star = L^{-1} b``.
    """

    h = 1.0 / (V + 1)
    x = jnp.linspace(h, 1.0 - h, V)
    main = 2.0 * jnp.ones(V)
    off = -1.0 * jnp.ones(V - 1)
    L = jnp.diag(main) + jnp.diag(off, 1) + jnp.diag(off, -1)
    b = (h ** 2) * jax.vmap(f)(x)
    u_star = jnp.linalg.solve(L, b)
    return x, L, b, u_star


# ---------------------------------------------------------------------------
# The field network: a coordinate MLP with hard Dirichlet BC, evaluated on the grid.
# ---------------------------------------------------------------------------


def coordinate_field(hidden: int, x: Array) -> Tuple[Callable[[Array, Array], Array], int]:
    """A coordinate MLP ``u_theta(x) = x(1-x) g_theta(x)`` (hard Dirichlet BC),
    evaluated at the grid ``x`` to give a field in ``R^V``.

    Returns ``(F, n_params)`` with ``F(theta, _) : R^0 -> R^V`` -- a *single*
    network (weights shared across grid points), so the field is a smooth ansatz,
    not ``V`` free values.
    """

    shapes = [(hidden, 1), (hidden,), (1, hidden), (1,)]
    sizes = [int(np.prod(s)) for s in shapes]
    n = sum(sizes)

    def unpack(theta):
        out, i = [], 0
        for s, k in zip(shapes, sizes):
            out.append(theta[i : i + k].reshape(s))
            i += k
        return out

    def g(theta, xi):  # scalar coordinate -> scalar
        W1, b1, W2, b2 = unpack(theta)
        return (W2 @ jnp.tanh(W1 @ jnp.array([xi]) + b1) + b2)[0]

    def F(theta, _x0):
        return jax.vmap(lambda xi: xi * (1.0 - xi) * g(theta, xi))(x)

    return F, n


# ---------------------------------------------------------------------------
# Solve, by descending the Dirichlet energy under Phiconf.
# ---------------------------------------------------------------------------


def solve_deep_ritz(
    V: int = 31,
    f: Callable[[Array], Array] = lambda x: (jnp.pi ** 2) * jnp.sin(jnp.pi * x),
    *,
    hidden: int = 32,
    eta: float = 0.3,
    steps: int = 4000,
    seed: int = 0,
):
    """Train the field network to minimize ``E`` (the deep-Ritz solve).

    Returns ``(u, u_star, history, x)``: the learned field on the grid, the exact
    discrete Poisson solution, the per-step energy, and the grid.
    """

    x, L, b, u_star = poisson_1d(V, f)
    F, n = coordinate_field(hidden, x)

    def grad_energy(u, _lam):  # dE/du = L u - b
        return L @ u - b

    def energy(u, _lam):  # E(u) = 1/2 u.L.u - b.u
        return 0.5 * u @ (L @ u) - b @ u

    rng = np.random.default_rng(seed)
    init = 0.1 * jnp.asarray(rng.standard_normal(n))
    arr = parameterized_map(F, euclidean(n, eta), in_dim=0, out_dim=V)
    params, history = train(
        arr,
        init,
        [(jnp.zeros(0), jnp.zeros(V))],  # no input, no label; the loss is the energy
        grad_loss=grad_energy,
        loss=energy,
        steps=steps,
    )
    u = F(params, jnp.zeros(0))
    return u, u_star, history, x
