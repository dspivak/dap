"""Wiring constructors in ``sarr`` (sec.wd_operads, sec.wave_equation).

Provides:

* ``finset_chain_wire(K)`` -- the lens ``phi_K`` of ex.lens_finsetop /
  sec.spring_first_pass, as the underlying pair of finite-set functions.
* ``chain_wire(K)``        -- its image under ``R^-`` (lem.lens_pow),
  realized as a smooth (static) arrangement: a ``K``-ary wiring morphism
  ``box^K -> box`` with trivial parameter ``R^0`` and ``U = 0``.
* ``compose_chain(parts)`` -- the ``K``-ary composite
  ``chain_wire(K)(Part_1, ..., Part_K)`` directly in ``sarr``, the 0-ary
  morphism of sec.spring_first_pass (eqn.sharp_chain and surrounding prose).
* ``parallel_arrangements(p1, p2)`` -- the monoidal product in ``sarr``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .rvect import ReactiveVectorSpace, trivial


# ---------------------------------------------------------------------------
# The finset-op lens phi_K (ex.lens_finsetop, sec.spring_first_pass).
# ---------------------------------------------------------------------------


def finset_chain_wire(K: int) -> Tuple[List[int], List[int]]:
    """Return ``(out_f, in_f)`` for ``phi_K : <K|K> -> <1|1>``.

    Per sec.spring_first_pass, with the inner finite set ``{1,...,K}`` and the
    outer box ports labeled by ``{K}`` (output, right) and ``{0}`` (input, left):

        out_f : {K} -> {1,...,K}        the inclusion, out_f(0) = K-1 (0-indexed),
        in_f  : {1,...,K} -> {1,...,K}+{0}, n |-> n-1 (feed each from the previous wire).
    """
    out_f = [K - 1]               # singleton; selects index of last small box
    in_f = list(range(K))         # picks the first K of K+1 wires
    return out_f, in_f


# ---------------------------------------------------------------------------
# chain_wire(K) as a SmoothArrangement, the image under R^- (lem.lens_pow).
# ---------------------------------------------------------------------------


def chain_wire(K: int) -> SmoothArrangement:
    """Image of ``phi_K`` under ``R^-`` (lem.lens_pow), as a smooth (static) arrangement.

    Source ``M = box^K = <R^K | R^K>``, target ``N = box = <R | R>``, trivial
    parameter ``R^0``, no potential. On coordinates (sec.spring_first_pass):

        out_f(q_1, ..., q_K)      = q_K
        in_f((q_1, ..., q_K), q_0) = (q_0, q_1, ..., q_{K-1}).
    """

    if K < 1:
        raise ValueError("K must be >= 1")

    Q = trivial()

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return m_out[K - 1:K]  # shape (1,); the last coordinate q_K

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        q_0 = n_in[0]
        return jnp.concatenate([jnp.array([q_0]), m_out[: K - 1]])

    def U(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.array(0.0)

    return SmoothArrangement(
        Q=Q,
        out_dim_M=K,
        in_dim_M=K,
        out_dim_N=1,
        in_dim_N=1,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label=f"chain_wire({K})",
    )


# ---------------------------------------------------------------------------
# Composition in sarr for the chain-of-particles example.
#
# Each Part_i : <R^0|R^0> -> <R|R> has parameter Q_i and data
#   out_f_i : Q_i -> R      (here identity),
#   in_f_i  : Q_i x R -> R^0 (vacuous),
#   U_i     : Q_i x R -> R   (the per-particle potential).
# Composing chain_wire(K)(Part_1, ..., Part_K) in sarr gives a 0-ary morphism
# with parameter Q_tot = Q_1 (+) ... (+) Q_K (sec.spring_first_pass).
# ---------------------------------------------------------------------------


def compose_chain(parts: Sequence[SmoothArrangement]) -> SmoothArrangement:
    """Compose ``chain_wire(K)(Part_1, ..., Part_K)`` directly in ``sarr``.

    Specialized to the wave-equation chain: each ``Part_i`` must have source
    ``<R^0|R^0>`` and target ``<R|R>``. The composite has

        Q_tot     = Q_1 (+) ... (+) Q_K
        out_f_tot = q_K
        in_f_tot  vacuous (target is <R^0|R^0>)
        U_tot     = sum_i U_i(q_i, q_{i-1})   (with q_0 the external input).
    """

    K = len(parts)
    if K < 1:
        raise ValueError("Need at least one particle in chain")

    for i, P in enumerate(parts):
        if not (P.out_dim_M == 0 and P.in_dim_M == 0):
            raise ValueError(f"Part {i}: expected source <R^0|R^0>")
        if not (P.out_dim_N == 1 and P.in_dim_N == 1):
            raise ValueError(f"Part {i}: expected target <R|R>")

    # Combined reactive vector space Q_tot = Q_1 (+) ... (+) Q_K.
    Q_tot = parts[0].Q
    for P in parts[1:]:
        Q_tot = Q_tot.direct_sum(P.Q)

    dims = [P.Q.dim for P in parts]
    offsets = [0]
    for d in dims:
        offsets.append(offsets[-1] + d)

    def split_q(q_tot: Array):
        return [q_tot[offsets[i]: offsets[i + 1]] for i in range(K)]

    def out_f(q_tot: Array, m_out: Array) -> Array:
        q_chunks = split_q(q_tot)
        q_K = parts[K - 1].out_f(q_chunks[K - 1], jnp.zeros(0))
        return q_K  # shape (1,)

    def in_f(q_tot: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)  # target in_dim_M = 0

    def U(q_tot: Array, m_out: Array, n_in: Array) -> Array:
        q_chunks = split_q(q_tot)
        q_outs = [parts[i].out_f(q_chunks[i], jnp.zeros(0)) for i in range(K)]
        # Each Part_i sees in_n = q_{i-1}; q_0 := n_in (external input).
        total = jnp.array(0.0)
        for i in range(K):
            neighbor = n_in if i == 0 else q_outs[i - 1]
            total = total + parts[i].U(q_chunks[i], jnp.zeros(0), neighbor)
        return total

    return SmoothArrangement(
        Q=Q_tot,
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=1,
        in_dim_N=1,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label=f"chain_wire({K})(parts)",
    )


# ---------------------------------------------------------------------------
# Monoidal (parallel) composition in sarr (sec.para_general).
# ---------------------------------------------------------------------------


def parallel_arrangements(
    p1: SmoothArrangement, p2: SmoothArrangement
) -> SmoothArrangement:
    """Monoidal product in ``sarr``: side-by-side independent boxes."""

    Q = p1.Q.direct_sum(p2.Q)
    n1 = p1.Q.dim

    def split(q: Array) -> Tuple[Array, Array]:
        return q[:n1], q[n1:]

    def out_f(q: Array, m_out: Array) -> Array:
        q1, q2 = split(q)
        m1, m2 = m_out[: p1.out_dim_M], m_out[p1.out_dim_M:]
        return jnp.concatenate([p1.out_f(q1, m1), p2.out_f(q2, m2)])

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        q1, q2 = split(q)
        m1, m2 = m_out[: p1.out_dim_M], m_out[p1.out_dim_M:]
        n1_in, n2_in = n_in[: p1.in_dim_N], n_in[p1.in_dim_N:]
        a = p1.in_f(q1, m1, n1_in)
        b = p2.in_f(q2, m2, n2_in)
        return jnp.concatenate([a, b])

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        q1, q2 = split(q)
        m1, m2 = m_out[: p1.out_dim_M], m_out[p1.out_dim_M:]
        n1_in, n2_in = n_in[: p1.in_dim_N], n_in[p1.in_dim_N:]
        return p1.U(q1, m1, n1_in) + p2.U(q2, m2, n2_in)

    return SmoothArrangement(
        Q=Q,
        out_dim_M=p1.out_dim_M + p2.out_dim_M,
        in_dim_M=p1.in_dim_M + p2.in_dim_M,
        out_dim_N=p1.out_dim_N + p2.out_dim_N,
        in_dim_N=p1.in_dim_N + p2.in_dim_N,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label=f"({p1.label} || {p2.label})",
    )
