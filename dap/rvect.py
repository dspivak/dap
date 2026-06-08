"""Reactive vector spaces (def.rvect).

A *reactive vector space* is a finite-dimensional real vector space ``Q``
together with a smooth *sharp* map

    sharpR_Q : Q -> vect(Q^*, Q),

i.e. a (possibly position-dependent) bundle map from covectors to vectors;
we write ``sharpR_q := sharpR_Q(q) : Q^* -> Q`` for the value at ``q in Q``
(def.rvect). The reaction is *constant* if ``sharpR_q = sharpR_0`` for all
``q``, and *nondegenerate* if every ``sharpR_q`` is an isomorphism.

This generalizes the old ``PairedVectorSpace`` (whose sharp was a single
constant isomorphism): there the sharp was one matrix; here it is a function
of position. The constant case (``constant``/``diagonal``/``euclidean``)
recovers the physics examples; the position-dependent case is what Newton's
method needs (``inverse_hessian``; sec.newton_warmup).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


SharpFn = Callable[[Array], Array]  # q |-> (dim, dim) matrix for sharpR_q : Q^* -> Q


@dataclass(frozen=True)
class ReactiveVectorSpace:
    """A reactive vector space ``(Q, sharpR_Q)`` per def.rvect.

    ``sharp_fn(q)`` returns the ``(dim, dim)`` matrix of ``sharpR_q : Q^* -> Q``.
    """

    dim: int
    sharp_fn: SharpFn

    # ---- the reaction ----

    def sharp_at(self, q: Array) -> Array:
        """The matrix ``sharpR_q : Q^* -> Q`` at the basepoint ``q``."""
        return self.sharp_fn(q)

    def apply_sharp(self, q: Array, xi: Array) -> Array:
        """Apply ``sharpR_q`` to a covector ``xi in Q^*``."""
        return self.sharp_fn(q) @ xi

    # ---- the nondegenerate (paired) case ----

    def flat_at(self, q: Array) -> Array:
        """The flat map ``flat_q := sharpR_q^{-1} : Q -> Q^*`` (nondegenerate case)."""
        return jnp.linalg.inv(self.sharp_fn(q))

    def pairing(self, q: Array, v: Array, vp: Array) -> Array:
        """The pairing at ``q``: ``<v, vp>_q := flat_q(v) . vp`` (def.rvect)."""
        return jnp.dot(self.flat_at(q) @ v, vp)

    # ---- monoidal structure (prop.rvect_monoidal) ----

    def direct_sum(self, other: "ReactiveVectorSpace") -> "ReactiveVectorSpace":
        """Direct-sum reactive vector space with block-diagonal sharp.

        Per eqn.directsum_sharp, ``sharpR_{(q,w)} = sharpR_q (+) sharpR_w``.
        """
        n, m = self.dim, other.dim

        def sharp_fn(q: Array) -> Array:
            q1, q2 = q[:n], q[n:]
            S = jnp.zeros((n + m, n + m))
            S = S.at[:n, :n].set(self.sharp_fn(q1))
            S = S.at[n:, n:].set(other.sharp_fn(q2))
            return S

        return ReactiveVectorSpace(dim=n + m, sharp_fn=sharp_fn)


# ---------------------------------------------------------------------------
# Constructors.
# ---------------------------------------------------------------------------


def constant(matrix: Array) -> ReactiveVectorSpace:
    """A reactive vector space with *constant* sharp given by ``matrix`` (def.rvect)."""
    matrix = jnp.asarray(matrix)
    return ReactiveVectorSpace(dim=int(matrix.shape[0]), sharp_fn=lambda q: matrix)


def diagonal(diag: Array) -> ReactiveVectorSpace:
    """Constant sharp that is diagonal with the given entries (e.g. inverse masses)."""
    diag = jnp.asarray(diag)
    return constant(jnp.diag(diag))


def euclidean(dim: int, eta: float = 1.0) -> ReactiveVectorSpace:
    """The Euclidean sharp ``eta * I`` (ex.euclidean_sharp, eqn.learning_sharp).

    With ``eta = eta_LR`` this is the constant learning-rate metric whose
    configuration dynamics is vanilla gradient descent (sec.dl_warmup).
    """
    return constant(eta * jnp.eye(dim))


def inverse_hessian(U: Callable[[Array], Array], dim: int) -> ReactiveVectorSpace:
    """The Newton reaction ``sharpR^U_q = (T_q dU)^{-1}`` (sec.newton_warmup).

    Here ``U : R^dim -> R`` and ``T_q(dU)`` is its Hessian at ``q``; the
    standing hypothesis is that it is invertible at every ``q``
    (cf. eqn.newton). This is the prototypical *position-dependent* sharp.
    """

    hess = jax.hessian(U)
    return ReactiveVectorSpace(dim=dim, sharp_fn=lambda q: jnp.linalg.inv(hess(q)))


def trivial() -> ReactiveVectorSpace:
    """The monoidal unit of rvect: the zero reactive vector space ``R^0``."""
    return ReactiveVectorSpace(dim=0, sharp_fn=lambda q: jnp.zeros((0, 0)))


SmoothMap = Callable[..., Array]
