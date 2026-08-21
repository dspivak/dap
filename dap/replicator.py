"""Replicator dynamics as ``Phiconf`` of a Shahshahani sarr-scalar.

The replicator equation for a *symmetric* payoff matrix ``A`` (a partnership
game),

    x_i' = x_i ((A x)_i - x^T A x),

is the configuration dynamics of the closed arrangement (an sarr-scalar
``I -> I``, cf. test_newton.py) whose parameter reactive vector space is
``R^n`` with the *state-dependent* Shahshahani sharp

    sharpR_x(xi) = dt * (diag(x) xi - (x . xi) x),      i.e. the matrix
    sharpR_x    = dt * (diag(x) - x x^T),

and whose potential is ``U(x) = -(1/2) x^T A x``.

Sign derivation against the code's conventions (no free choices left):
the configuration integrator steps ``x -> x - sharpR_x(xi_Q)``
(``integrator.configuration_integrator``, eqn.conf_integrator), and for a
closed scalar the interpretation's parameter covector is ``xi_Q = dU|_x``
(eqn.bigtheta, the other two pullbacks vanish). With ``A`` symmetric,
``dU|_x = -A x``, so one step is

    x -> x + dt * (diag(x) A x - (x^T A x) x)
       = x + dt * (x_i ((A x)_i - x^T A x))_i,

the forward-Euler replicator step. The minus sign in ``U`` makes the
integrator's descent *ascend* the mean fitness ``(1/2) x^T A x`` -- the
Shahshahani sharp is positive semidefinite on the positive orthant
(``xi^T sharpR_x xi = dt * (E_x[xi^2] - E_x[xi]^2) >= 0`` for ``x`` a
probability vector, by Jensen), so ``U`` decreases and fitness increases
along the flow (Fisher's fundamental theorem, in this discretization only
approximately per step).

The time step ``dt`` is carried by the sharp (a positive scalar multiple of a
sharp is a sharp), exactly as ``rvect.euclidean(dim, eta)`` carries the
learning rate; the paper's configuration integrator itself has no ``dt``.

Degeneracy and the simplex: ``sharpR_x`` is symmetric and degenerate. Its
kernel contains the constant covectors *exactly on the simplex*:

    sharpR_x(c * 1) = dt * c * x * (1 - sum(x)),

which vanishes iff ``sum(x) = 1`` (or ``c = 0``, or ``x = 0``). Dually,
``1^T sharpR_x = dt * (1 - sum(x)) x^T``, so the increment of ``sum(x)``
in one step is ``dt * (x^T A x) * (1 - sum(x))``: the simplex ``sum = 1``
is invariant (exactly in real arithmetic; in floats the per-step drift is
at rounding level and is self-correcting, since ``x^T A x > 0`` makes
``sum = 1`` attracting for positive-definite-on-the-orthant payoffs).
Positivity of the coordinates is preserved by the flow but only
approximately by Euler steps (a coordinate can overshoot 0 for large
``dt``); we keep ``dt`` small and start in the interior.

SCOPE LIMIT: this construction covers partnership games only. For a
non-symmetric payoff matrix -- e.g. rock-paper-scissors -- the replicator
vector field is NOT a Shahshahani gradient of any potential (RPS orbits
cycle around the interior equilibrium and ascend nothing), so it does not
arise as ``Phiconf`` of this arrangement; ``dU|_x = -(1/2)(A + A^T) x``
would symmetrize the game, changing the dynamics. Hawk-Dove is covered
despite its textbook matrix being non-symmetric, because replicator
dynamics is invariant under column shifts ``A -> A + 1 c^T`` and every
2x2 game is column-equivalent to a symmetric one (see the tests).

LOG COORDINATES (cf. reaction_network, encoding 5): the same flow can be run
in the chart ``y = log x``, pushing the Shahshahani sharp forward along the
diffeomorphism (``sharp' = J sharp J^T``, ``J = diag(1/x)``):

    sharpR^log_y = dt * (diag(1/x) - 1 1^T),      x = e^y,

symmetric, and PSD on the simplex (Cauchy--Schwarz), with the SAME potential
``U = -(1/2) x^T A x`` read through the chart. One step is then

    y_i -> y_i + dt * ((A x)_i - x^T A x),
    x_i -> x_i * exp(dt * ((A x)_i - x^T A x)),

the forward-Euler replicator step in log coordinates (the multiplicative-
weights update). Fractions are POSITIVE BY CONSTRUCTION at any ``dt`` --
where the standard tick can overshoot a coordinate below 0 -- but the sum
``sum(x) = 1``, exactly invariant for the standard tick, now drifts by
``O(dt^2)`` per step: the same positivity-for-conservation trade as the
Petri encodings 4 vs 5. No doubled state is needed here, unlike encoding 5,
because the replicator field IS a sharp-gradient (potential game).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .functors import Phiconf
from .interpretation import trivial_omega
from .pc import PCMorphism
from .rvect import ReactiveVectorSpace


def shahshahani(dim: int, dt: float = 1.0) -> ReactiveVectorSpace:
    """The Shahshahani reactive vector space ``(R^dim, dt * (diag(x) - x x^T))``.

    A state-dependent, symmetric, degenerate sharp (see the module docstring
    for its kernel); ``dt`` is the Euler step size, carried by the sharp as
    ``euclidean`` carries the learning rate.
    """

    def sharp_fn(x: Array) -> Array:
        return dt * (jnp.diag(x) - jnp.outer(x, x))

    return ReactiveVectorSpace(dim=dim, sharp_fn=sharp_fn)


def replicator_arrangement(A: Array, dt: float = 1.0) -> SmoothArrangement:
    """The sarr-scalar ``((R^n, sharpR^Shah), !, !, -(1/2) x^T A x) : I -> I``.

    ``A`` must be symmetric (a partnership game); this is asserted, since for
    non-symmetric ``A`` the construction would silently symmetrize the game
    (module docstring, scope limit).
    """
    A = jnp.asarray(A, dtype=float)
    n = int(A.shape[0])
    if A.shape != (n, n) or not bool(jnp.allclose(A, A.T)):
        raise ValueError("replicator_arrangement: payoff matrix must be square and symmetric")

    return SmoothArrangement(
        Q=shahshahani(n, dt),
        out_dim_M=0, in_dim_M=0, out_dim_N=0, in_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: -0.5 * jnp.dot(q, A @ q),
        label="replicator",
    )


def replicator_dynamics(A: Array, dt: float = 1.0) -> PCMorphism:
    """``Phiconf`` of the replicator arrangement: the closed pc-morphism."""
    return Phiconf(replicator_arrangement(A, dt))


_IN_POS = (jnp.zeros(0), trivial_omega(0))
_IN_DIR = (jnp.zeros(0), jnp.zeros(0))


def replicator_step(O: PCMorphism, x: Array) -> Array:
    """One Phiconf Euler step of the closed system from state ``x``."""
    _, _, new_x = O.with_state(x).run_one(_IN_POS, lambda _o: _IN_DIR)
    return new_x


def log_shahshahani(dim: int, dt: float = 1.0) -> ReactiveVectorSpace:
    """The pushforward of ``shahshahani`` along ``y = log x`` (module docstring):
    ``sharpR^log_y = dt * (diag(1/x) - 1 1^T)``, ``x = e^y``. Symmetric,
    state-dependent, PSD on the simplex."""

    def sharp_fn(y: Array) -> Array:
        x = jnp.exp(y)
        return dt * (jnp.diag(1.0 / x) - jnp.ones((dim, dim)))

    return ReactiveVectorSpace(dim=dim, sharp_fn=sharp_fn)


def log_replicator_arrangement(A: Array, dt: float = 1.0) -> SmoothArrangement:
    """The sarr-scalar ``((R^n, sharpR^log), !, !, -(1/2) e^y^T A e^y) : I -> I``:
    the replicator arrangement read through the chart ``y = log x``."""
    A = jnp.asarray(A, dtype=float)
    n = int(A.shape[0])
    if A.shape != (n, n) or not bool(jnp.allclose(A, A.T)):
        raise ValueError("log_replicator_arrangement: payoff matrix must be square and symmetric")

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        x = jnp.exp(q)
        return -0.5 * jnp.dot(x, A @ x)

    return SmoothArrangement(
        Q=log_shahshahani(n, dt),
        out_dim_M=0, in_dim_M=0, out_dim_N=0, in_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=U,
        label="log_replicator",
    )


def log_replicator_dynamics(A: Array, dt: float = 1.0) -> PCMorphism:
    """``Phiconf`` of the log-coordinate replicator arrangement. The state is
    ``y = log x``; step it with ``replicator_step`` and read fractions off as
    ``exp(y)``, which are positive by construction."""
    return Phiconf(log_replicator_arrangement(A, dt))
