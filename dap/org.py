"""``OrgMorphism`` (sec.org).

A Moore-style coalgebraic representation of an ``[p, q]``-coalgebra
(def.pq_coalg). Following VOCAB design (2)-(3), we never materialize
the internal hom ``[p, q]``; instead an ``OrgMorphism`` carries:

* ``state``: the current state ``s in S``.
* ``step``:  ``s -> (PolyMap p -> q,  in_dir -> (out_dir, new_state))``.

The polynomial-map component is the action ``act^beta(s)`` of
def.pq_coalg; the closure is the update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

from .polynomial import PolyMap, PolyValue


# Step signature, def.pq_coalg, in fully-curried Moore form (sec.dynamics_functor):
#     state -> (act: PolyMap p -> q,
#               in_pos -> (out_pos,
#                          in_dir -> (out_dir, new_state)))
# The PolyMap component IS act^beta(state) (def.pq_coalg); it
# duplicates the position/direction logic that the curried closure
# also exposes, but having it as a first-class poly map makes
# composition with static poly maps in ``then_static`` clean.
StepFn = Callable[
    [Any],
    Tuple[PolyMap,
          Callable[[PolyValue],
                   Tuple[PolyValue,
                         Callable[[PolyValue], Tuple[PolyValue, Any]]]]],
]


@dataclass(frozen=True)
class OrgMorphism:
    """An object of the org-bicategory's homset ``org(p, q)``.

    Implements the unpacked coalgebra of def.pq_coalg in Moore form.
    Composition is provided as a method rather than via a category
    class (VOCAB design (3)).
    """

    src_poly: Any  # the polynomial p
    tgt_poly: Any  # the polynomial q
    state: Any
    step: StepFn

    def run_one(self, in_pos: PolyValue, in_dir_from: Callable[[PolyValue], PolyValue]):
        """Apply the coalgebra at ``state`` to one external position.

        ``in_dir_from`` is provided by the surrounding context: given the
        out-position emitted by ``self``, it returns the in-direction.
        Returns ``(out_pos, out_dir, new_state)``.
        """
        act, fiber = self.step(self.state)
        out_pos, fiber_at_pos = fiber(in_pos)
        in_dir = in_dir_from(out_pos)
        out_dir, new_state = fiber_at_pos(in_dir)
        return out_pos, out_dir, new_state

    def with_state(self, new_state: Any) -> "OrgMorphism":
        """Return a copy of this OrgMorphism with the state replaced."""
        return OrgMorphism(
            src_poly=self.src_poly,
            tgt_poly=self.tgt_poly,
            state=new_state,
            step=self.step,
        )

    # ---- composition (sec.org) ----

    def then_static(self, outer: PolyMap) -> "OrgMorphism":
        """Post-compose with a static polynomial map ``q -> r`` (sec.org).

        The result has the same state set as ``self``; the action on
        state ``s`` is ``outer o act^beta(s)``.
        """
        from .polynomial import PolyMap as _PM

        def new_step(s):
            inner_act, inner_fiber = self.step(s)

            composed_act = _PM(
                src=outer.src,
                tgt=inner_act.tgt,
                position_action=lambda i: inner_act.on_position(outer.on_position(i)),
                direction_action=lambda i, d: outer.on_direction(
                    i, inner_act.on_direction(outer.on_position(i), d)
                ),
                label=f"{outer.label};{inner_act.label}",
            )

            def fiber(i_outer):
                # outer.on_position(i_outer) is the inner-src position.
                inner_pos = outer.on_position(i_outer)
                inner_out_pos, inner_at_pos = inner_fiber(inner_pos)
                # outer's position-action turns inner_out_pos into an outer-tgt position?
                # No: ``outer`` here is OUTER, going from outer.src to outer.tgt = inner.src.
                # So we composed inner-after-outer; the new target = inner.tgt.
                out_pos = inner_out_pos

                def at_pos(d_outer_tgt):
                    # d_outer_tgt is in inner.tgt direction-fiber at out_pos.
                    # Forward through inner_at_pos which expects an inner-tgt direction.
                    out_dir_inner, new_state = inner_at_pos(d_outer_tgt)
                    # Push inner-src direction back through outer to get outer-src direction.
                    out_dir = outer.on_direction(i_outer, out_dir_inner)
                    return out_dir, new_state

                return out_pos, at_pos

            return composed_act, fiber

        return OrgMorphism(
            src_poly=outer.src,
            tgt_poly=self.tgt_poly,
            state=self.state,
            step=new_step,
        )

    def parallel(self, other: "OrgMorphism") -> "OrgMorphism":
        """Monoidal parallel composition (sec.org), state-spaces multiply."""

        from .polynomial import PolyMap as _PM, DirichletProduct

        def new_step(s):
            s1, s2 = s
            act1, fiber1 = self.step(s1)
            act2, fiber2 = other.step(s2)

            act = _PM(
                src=DirichletProduct(act1.src, act2.src),
                tgt=DirichletProduct(act1.tgt, act2.tgt),
                position_action=lambda i: (act1.on_position(i[0]),
                                           act2.on_position(i[1])),
                direction_action=lambda i, d: (
                    act1.on_direction(i[0], d[0]),
                    act2.on_direction(i[1], d[1]),
                ),
            )

            def fiber(i):
                op1, at1 = fiber1(i[0])
                op2, at2 = fiber2(i[1])

                def at_pos(d):
                    od1, ns1 = at1(d[0])
                    od2, ns2 = at2(d[1])
                    return (od1, od2), (ns1, ns2)

                return (op1, op2), at_pos

            return act, fiber

        return OrgMorphism(
            src_poly=DirichletProduct(self.src_poly, other.src_poly),
            tgt_poly=DirichletProduct(self.tgt_poly, other.tgt_poly),
            state=(self.state, other.state),
            step=new_step,
        )
