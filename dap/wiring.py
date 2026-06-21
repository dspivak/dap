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
* ``graph_wire(V, edges)`` / ``harmonic_vertex(...)`` / ``compose_graph(...)``
  -- the prism wiring ``R^{varphi_G}``, the vertex particle ``Part_v``, and their
  composite ``wire_G`` of sec.graph_laplacian (eqn.prism_f, eqn.graph_wire).
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .rvect import ReactiveVectorSpace, diagonal, trivial


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


def compose_seq(f: SmoothArrangement, g: SmoothArrangement) -> SmoothArrangement:
    """Sequential composition ``g . f`` in ``sarr`` (rem.huge_wiring_diagram).

    For ``f : L -> M`` and ``g : M -> N`` (so ``f.out_dim_N == g.out_dim_M`` and
    ``f.in_dim_N == g.in_dim_M``), the composite ``g . f : L -> N`` is the
    coKleisli/lens composition of def.lens carried into ``Lcokl{mfd}{R}``: forward
    and backward components inherited from the lens composite, and the potential
    *added* via the writer monad's multiplication ``(+)`` (lem.monoid_to_monad).
    Its parameter is ``Q_f (+) Q_g`` (lem.para_monoidal). On coordinates, with
    ``m = out_f(q_f, l)`` the intermediate forward value:

        out      : n   = out_g(q_g, m)
        in       : l_in = in_f(q_f, l, in_g(q_g, m, n_in))
        U        : U_f(q_f, l, in_g(q_g, m, n_in)) + U_g(q_g, m, n_in).
    """

    if f.out_dim_N != g.out_dim_M or f.in_dim_N != g.in_dim_M:
        raise ValueError(
            f"compose_seq: codomain of f <{f.in_dim_N}|{f.out_dim_N}> "
            f"!= domain of g <{g.in_dim_M}|{g.out_dim_M}>"
        )

    Q = f.Q.direct_sum(g.Q)
    nf = f.Q.dim

    def split(q: Array) -> Tuple[Array, Array]:
        return q[:nf], q[nf:]

    def out_f(q: Array, l_out: Array) -> Array:
        q_f, q_g = split(q)
        m = f.out_f(q_f, l_out)
        return g.out_f(q_g, m)

    def in_f(q: Array, l_out: Array, n_in: Array) -> Array:
        q_f, q_g = split(q)
        m = f.out_f(q_f, l_out)
        m_in = g.in_f(q_g, m, n_in)
        return f.in_f(q_f, l_out, m_in)

    def U(q: Array, l_out: Array, n_in: Array) -> Array:
        q_f, q_g = split(q)
        m = f.out_f(q_f, l_out)
        m_in = g.in_f(q_g, m, n_in)
        return f.U(q_f, l_out, m_in) + g.U(q_g, m, n_in)

    return SmoothArrangement(
        Q=Q,
        out_dim_M=f.out_dim_M,
        in_dim_M=f.in_dim_M,
        out_dim_N=g.out_dim_N,
        in_dim_N=g.in_dim_N,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label=f"({g.label} . {f.label})",
    )


def tensor_arrangements(parts: Sequence[SmoothArrangement]) -> SmoothArrangement:
    """The ``K``-fold monoidal product ``Part_1 (x) ... (x) Part_K`` in ``sarr``
    (lem.para_monoidal), by folding ``parallel_arrangements``."""

    if len(parts) < 1:
        raise ValueError("tensor_arrangements: need at least one factor")
    acc = parts[0]
    for P in parts[1:]:
        acc = parallel_arrangements(acc, P)
    return acc


def compose_chain(parts: Sequence[SmoothArrangement]) -> SmoothArrangement:
    """The operad composite ``chain_wire(K)(Part_1, ..., Part_K)`` of
    sec.spring_first_pass, built by genuine composition in ``sarr``:

        chain_wire(K)(Parts) = compose_seq( tensor(Parts), chain_wire(K) ),

    i.e. tensor the ``K`` particles (``box^0 -> box^K``) and post-compose with the
    wiring (``box^K -> box``). No hand-specialization -- the closed quadratic
    potential ``sum_i (kappa/2)(q_i - q_{i-1})^2`` emerges from ``compose_seq``.
    Each ``Part_i`` must have source ``<R^0|R^0>`` and target ``<R|R>``.
    """

    K = len(parts)
    if K < 1:
        raise ValueError("Need at least one particle in chain")
    for i, P in enumerate(parts):
        if not (P.out_dim_M == 0 and P.in_dim_M == 0):
            raise ValueError(f"Part {i}: expected source <R^0|R^0>")
        if not (P.out_dim_N == 1 and P.in_dim_N == 1):
            raise ValueError(f"Part {i}: expected target <R|R>")

    return compose_seq(tensor_arrangements(list(parts)), chain_wire(K))


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


# ---------------------------------------------------------------------------
# Graph wiring: the prism varphi_G and the vertex particle (sec.graph_laplacian).
# ---------------------------------------------------------------------------


def _graph_ports(
    num_vertices: int, edges: Sequence[Tuple[int, int]]
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """Out-edge / in-edge index lists per vertex, and the prism permutation.

    For ``edges[e] = (src, tgt)`` and each ``v``, ``out(v) = src^{-1}(v)`` and
    ``in(v) = tgt^{-1}(v)`` (sec.graph_laplacian). Concatenating over ``v`` in order
    gives two orderings of ``E``, ``OUT = sum_v out(v)`` and ``IN = sum_v in(v)``
    (each a permutation of ``E``). The prism input map ``inpt f`` of eqn.prism_f --
    ``sum_v in(v) -> sum_v out(v)`` via ``tgt`` then ``src`` -- is the permutation
    ``perm`` with ``OUT[perm[j]] == IN[j]`` (same edge at in-port ``j``).
    """
    out_lists = [[e for e, (s, t) in enumerate(edges) if s == v] for v in range(num_vertices)]
    in_lists = [[e for e, (s, t) in enumerate(edges) if t == v] for v in range(num_vertices)]
    OUT = [e for v in range(num_vertices) for e in out_lists[v]]
    IN = [e for v in range(num_vertices) for e in in_lists[v]]
    pos_in_out = {e: k for k, e in enumerate(OUT)}
    perm = [pos_in_out[e] for e in IN]  # in_f(m_out)[j] = m_out[perm[j]]
    return out_lists, in_lists, perm


def graph_wire(num_vertices: int, edges: Sequence[Tuple[int, int]]) -> SmoothArrangement:
    """The prism wiring ``R^{varphi_G}`` of sec.graph_laplacian as a static arrangement.

    Image under ``R^-`` (lem.lens_pow) of the prism ``varphi_G : (x)_v <in(v)/out(v)>
    -> <0/0>`` (eqn.prism_f). Source ``(x)_v <R^{in(v)}|R^{out(v)}> = <R^E|R^E>``,
    target the unit ``<R^0|R^0>``, trivial parameter ``R^0``, ``U' = 0``. The input
    map ``prod_v R^{out(v)} -> prod_v R^{in(v)}`` is the identity on ``R^E`` shuttled
    through ``src``/``tgt``, i.e. ``in_f(m_out) = m_out[perm]`` (see ``_graph_ports``).
    """
    E = len(edges)
    _, _, perm = _graph_ports(num_vertices, edges)
    perm = jnp.asarray(perm, dtype=int)

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return jnp.zeros(0)  # bang : 0 -> sum_v out(v)

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return m_out[perm]  # route each in-edge from its source's out-edge

    def U(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.array(0.0)

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=E,
        in_dim_M=E,
        out_dim_N=0,
        in_dim_N=0,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label=f"graph_wire(V={num_vertices}, E={E})",
    )


def harmonic_vertex(d_in: int, d_out: int, m: float, kappas: Sequence[float]) -> SmoothArrangement:
    """The harmonic particle ``Part_v : <1/1> -> <R^{in(v)}|R^{out(v)}>`` at a vertex.

    Parameter ``Q_v = R`` with constant ``sharp(xi) = xi/m``; output map the diagonal
    ``q |-> (q)_{e in out(v)}`` broadcasting position to the ``d_out`` out-edges; input
    map vacuous; potential ``U_v(q, (q'_e)) = sum_e (kappa_e/2)(q - q'_e)^2`` over the
    ``d_in`` in-edges (sec.graph_laplacian). The chain particle of sec.spring_first_pass
    is the case ``d_in = d_out = 1``.
    """
    kappas = jnp.asarray(kappas, float)

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.full((d_out,), q[0])  # diagonal: broadcast position to out-edges

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)  # bang

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return 0.5 * jnp.sum(kappas * (q[0] - n_in) ** 2)

    return SmoothArrangement(
        Q=diagonal(jnp.array([1.0 / m])),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=d_out,
        in_dim_N=d_in,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label="Part_v",
    )


def compose_graph(
    num_vertices: int, edges: Sequence[Tuple[int, int]], m: float, kappa: float
) -> SmoothArrangement:
    """``wire_G = R^{varphi_G}((Part_v)_v)`` of eqn.graph_wire, by genuine composition.

        wire_G = compose_seq( tensor(Part_v for v), graph_wire(G) ) : <1/1> -> <1/1>.

    A closed 0-ary arrangement with parameter ``R^V``, ``sharp = (xi_v/m)_v``, and
    composite potential ``sum_e (kappa/2)(q_tgt(e) - q_src(e))^2`` (eqn.graph_potential)
    that *emerges* from the prism wiring composed with the vertex particles -- it is
    not written by hand. Uniform mass ``m`` and spring constant ``kappa``.
    """
    out_lists, in_lists, _ = _graph_ports(num_vertices, edges)
    vertices = [
        harmonic_vertex(len(in_lists[v]), len(out_lists[v]), m, [kappa] * len(in_lists[v]))
        for v in range(num_vertices)
    ]
    wired = compose_seq(tensor_arrangements(vertices), graph_wire(num_vertices, edges))
    return replace(wired, label=f"graph(V={num_vertices}, E={len(edges)})")
