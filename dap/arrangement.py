"""Smooth adaptive arrangements: morphisms in ``sarr`` (def.potlens).

A morphism ``f : Q . (M_1 (x) ... (x) M_K) -> N`` in ``sarr`` is a tuple
``(rv Q, (in_f / out_f), U)`` with ``rv Q = (Q, sharpR_Q) : rvect`` and smooth
maps (eqn.srw_morphism; the ``K = 1`` case is eqn.para_potential_lens_maps)

    out_f : Q x out_M          -> out_N
    in_f  : Q x out_M x in_N   -> in_M
    U     : Q x out_M x in_N   -> R

In v1 the manifolds are all ``R^d``, recorded by their dimensions. This is the
object the dynamics functors ``Phiconf, Phiphase`` (cor.functor) act on; it was
called ``PotLensMap`` in the previous version (with ``V`` for the parameter).
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from .rvect import ReactiveVectorSpace, SmoothMap


@dataclass(frozen=True)
class SmoothArrangement:
    """A smooth adaptive arrangement (def.potlens, eqn.srw_morphism).

    The tensored domain ``M = M_1 (x) ... (x) M_K`` has output dimension
    ``out_dim_M`` and input dimension ``in_dim_M``; the codomain ``N`` has
    ``out_dim_N`` and ``in_dim_N``. The parameter is the reactive vector
    space ``Q``.
    """

    Q: ReactiveVectorSpace
    out_dim_M: int
    in_dim_M: int
    out_dim_N: int
    in_dim_N: int
    out_f: SmoothMap  # (q, m_out)              -> n_out
    in_f: SmoothMap   # (q, m_out, n_in)        -> m_in
    U: SmoothMap      # (q, m_out, n_in)        -> R
    label: str = ""
