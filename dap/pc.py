"""``PCMorphism`` (sec.pc).

A Moore-style coalgebraic representation of an ``[p, q]``-coalgebra
(def.pq_coalg). Following VOCAB design (2)-(3), we never materialize
the internal hom ``[p, q]``; instead an ``PCMorphism`` carries:

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
# composition with static poly maps (``pre_static``/``post_static``) clean.
StepFn = Callable[
    [Any],
    Tuple[PolyMap,
          Callable[[PolyValue],
                   Tuple[PolyValue,
                         Callable[[PolyValue], Tuple[PolyValue, Any]]]]],
]


@dataclass(frozen=True)
class PCMorphism:
    """An object of the pc-bicategory's homset ``pc(p, q)``.

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

    def with_state(self, new_state: Any) -> "PCMorphism":
        """Return a copy of this PCMorphism with the state replaced."""
        return PCMorphism(
            src_poly=self.src_poly,
            tgt_poly=self.tgt_poly,
            state=new_state,
            step=self.step,
        )

    # ---- composition (sec.pc) ----

    def pre_static(self, before: PolyMap) -> "PCMorphism":
        """Pre-compose a static polynomial map ``before : o -> p``, giving ``o -> q``.

        The result has the same state set as ``self``; the action on state ``s``
        is ``act^beta(s) o before``. (``self`` runs *after* the static map: an
        ``o``-position is carried to a ``p``-position by ``before`` and then read
        by ``self``, and the ``p``-direction that comes back is pushed on to an
        ``o``-direction by ``before``.)
        """
        from .polynomial import PolyMap as _PM

        if before.tgt != self.src_poly:
            raise ValueError(
                f"pre_static: target {before.tgt} of the static map "
                f"!= source {self.src_poly} of this morphism"
            )

        def new_step(s):
            inner_act, inner_fiber = self.step(s)

            composed_act = _PM(
                src=before.src,
                tgt=inner_act.tgt,
                position_action=lambda i: inner_act.on_position(before.on_position(i)),
                direction_action=lambda i, d: before.on_direction(
                    i, inner_act.on_direction(before.on_position(i), d)
                ),
                label=f"{before.label};{inner_act.label}",
            )

            def fiber(i_before):             # i_before: o-position
                inner_pos = before.on_position(i_before)          # p-position
                out_pos, inner_at_pos = inner_fiber(inner_pos)    # q-position

                def at_pos(d_q):             # d_q: q-direction at out_pos
                    d_p, new_state = inner_at_pos(d_q)            # p-direction
                    d_o = before.on_direction(i_before, d_p)      # o-direction
                    return d_o, new_state

                return out_pos, at_pos

            return composed_act, fiber

        return PCMorphism(
            src_poly=before.src,
            tgt_poly=self.tgt_poly,
            state=self.state,
            step=new_step,
        )

    def post_static(self, after: PolyMap) -> "PCMorphism":
        """Post-compose a static polynomial map ``after : q -> r``, giving ``p -> r``.

        The result has the same state set as ``self``; the action on state ``s``
        is ``after o act^beta(s)``. This is ``then`` in the special case where the
        second factor carries a single state, i.e. is a polynomial map (sec.pc);
        the paper's second pass post-composes exactly such a map, the static
        ``Phi(wire_K)`` (sec.spring_second_pass).
        """
        from .polynomial import PolyMap as _PM

        if self.tgt_poly != after.src:
            raise ValueError(
                f"post_static: target {self.tgt_poly} of this morphism "
                f"!= source {after.src} of the static map"
            )

        def new_step(s):
            inner_act, inner_fiber = self.step(s)

            composed_act = _PM(
                src=inner_act.src,
                tgt=after.tgt,
                position_action=lambda i: after.on_position(inner_act.on_position(i)),
                direction_action=lambda i, d_r: inner_act.on_direction(
                    i, after.on_direction(inner_act.on_position(i), d_r)
                ),
                label=f"{inner_act.label};{after.label}",
            )

            def fiber(i):                    # i: p-position
                j, inner_at_pos = inner_fiber(i)                  # q-position
                k = after.on_position(j)                          # r-position

                def at_pos(d_r):             # d_r: r-direction at k
                    d_q = after.on_direction(j, d_r)              # q-direction at j
                    return inner_at_pos(d_q)                      # (p-direction, new state)

                return k, at_pos

            return composed_act, fiber

        return PCMorphism(
            src_poly=self.src_poly,
            tgt_poly=after.tgt,
            state=self.state,
            step=new_step,
        )

    def then(self, other: "PCMorphism") -> "PCMorphism":
        """Sequential composition in pc: ``self : p -> q`` then ``other : q -> r``,
        giving ``p -> r`` (sec.pc). State spaces multiply.

        Forward, a ``p``-position runs through ``self`` then ``other`` to an
        ``r``-position; backward, an ``r``-direction runs through ``other`` then
        ``self`` to a ``p``-direction, updating both states. This is the general
        ``pc`` composition on which the functoriality of ``Phi`` rests (the
        second-pass audit, sec.spring_second_pass); ``post_static`` is the special
        case where ``other`` carries a single state. Requires ``self.tgt_poly``
        and ``other.src_poly`` to agree.
        """

        from .polynomial import PolyMap as _PM

        if self.tgt_poly != other.src_poly:
            raise ValueError(
                f"then: target {self.tgt_poly} of the first factor "
                f"!= source {other.src_poly} of the second"
            )

        def new_step(s):
            s_self, s_other = s
            act_self, fiber_self = self.step(s_self)
            act_other, fiber_other = other.step(s_other)

            composed_act = _PM(
                src=act_self.src,
                tgt=act_other.tgt,
                position_action=lambda i: act_other.on_position(act_self.on_position(i)),
                direction_action=lambda i, d_r: act_self.on_direction(
                    i, act_other.on_direction(act_self.on_position(i), d_r)
                ),
                label=f"{act_self.label};{act_other.label}",
            )

            def fiber(i):                       # i: p-position
                j, at_self = fiber_self(i)      # j: q-position
                k, at_other = fiber_other(j)    # k: r-position

                def at_pos(d_r):                # d_r: r-direction at k
                    d_q, ns_other = at_other(d_r)   # d_q: q-direction at j
                    d_p, ns_self = at_self(d_q)     # d_p: p-direction at i
                    return d_p, (ns_self, ns_other)

                return k, at_pos

            return composed_act, fiber

        return PCMorphism(
            src_poly=self.src_poly,
            tgt_poly=other.tgt_poly,
            state=(self.state, other.state),
            step=new_step,
        )

    def parallel(self, other: "PCMorphism") -> "PCMorphism":
        """Monoidal parallel composition (sec.pc), state-spaces multiply."""

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

        return PCMorphism(
            src_poly=DirichletProduct(self.src_poly, other.src_poly),
            tgt_poly=DirichletProduct(self.tgt_poly, other.tgt_poly),
            state=(self.state, other.state),
            step=new_step,
        )
