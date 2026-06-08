"""Named polynomial constructors (VOCAB design decision (1)).

Polynomials in our subcategory of ``poly`` are built from four named
constructors:

* ``Yon``                 -- the unit ``y`` (sec.poly).
* ``Cot(dim)``            -- ``cot(R^dim)`` (def.cot).
* ``DirichletProduct``    -- the Dirichlet product ``p1 (x) ... (x) pk``.
* ``PolyMap``             -- a polynomial map between any of the above.

We do not provide a generic ``Polynomial`` class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple, Union

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class Yon:
    """The unit polynomial ``y`` of the Dirichlet product (sec.poly)."""

    @property
    def position_shape(self) -> Tuple[int, ...]:
        return ()  # a single position '*'

    @property
    def direction_shape(self) -> Tuple[int, ...]:
        return ()  # a single direction '*'


@dataclass(frozen=True)
class Cot:
    """The polynomial ``cot(R^dim)`` (def.cot).

    Positions are points of ``R^dim`` (the manifold), directions at a
    position are cotangent vectors in ``(R^dim)^*`` ~= ``R^dim``.
    """

    dim: int

    @property
    def position_shape(self) -> Tuple[int, ...]:
        return (self.dim,)

    @property
    def direction_shape(self) -> Tuple[int, ...]:
        return (self.dim,)


Poly = Union[Yon, Cot, "DirichletProduct"]


@dataclass(frozen=True)
class DirichletProduct:
    """``p1 (x) ... (x) pk``, the Dirichlet product (sec.poly).

    A position is a tuple of positions, a direction at it is a tuple of
    directions.
    """

    factors: Tuple[Poly, ...]

    def __init__(self, *factors: Poly):
        object.__setattr__(self, "factors", tuple(factors))


# A position or direction is a JAX array (scalar/vector) or a Python
# tuple thereof, recursively, mirroring the tree structure of a
# DirichletProduct. We use jax pytrees so it composes with jit/vjp.
PolyValue = Union[Array, Tuple["PolyValue", ...]]


@dataclass(frozen=True)
class PolyMap:
    """A polynomial map ``src -> tgt`` (sec.poly).

    A morphism in poly between any two of ``Yon``, ``Cot``, and
    ``DirichletProduct``. Specified by:

    * ``position_action``:   ``src.position -> tgt.position``.
    * ``direction_action``:  ``(src.position, tgt.direction) -> src.direction``.

    The direction action is the contravariant fiber map from cotangent
    fibers at the target to cotangent fibers at the source.
    """

    src: Poly
    tgt: Poly
    position_action: Callable[[PolyValue], PolyValue]
    direction_action: Callable[[PolyValue, PolyValue], PolyValue]
    label: str = ""

    def on_position(self, i: PolyValue) -> PolyValue:
        return self.position_action(i)

    def on_direction(self, i: PolyValue, d_tgt: PolyValue) -> PolyValue:
        return self.direction_action(i, d_tgt)


def identity_poly_map(p: Poly) -> PolyMap:
    """The identity ``p -> p`` in poly."""
    return PolyMap(
        src=p,
        tgt=p,
        position_action=lambda i: i,
        direction_action=lambda i, d: d,
        label="id",
    )
