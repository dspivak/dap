"""``org^(2)``: general two-stage coalgebras ``[p,q]^{∘2}``-Coalg (rmk.org_N).

A morphism in ``org^(2)`` from ``p`` to ``q`` is a coalgebra
``β : S → [p,q]([p,q](S))`` for the substitution ``[p,q]^{∘2} = [p,q] ◁ [p,q]``.
In Moore form it is **two interaction rounds** per macro-tick. The key is the
substitution: round 1 emits a ``q``-position, receives a ``q``-direction, returns
a ``p``-direction, and lands *not* in a new state ``S`` but in an **inner 1-stage
``[p,q]``-coalgebra** — an element of ``[p,q](S)``, i.e. an ``org.OrgMorphism``.
Round 2 runs that inner coalgebra and lands in ``S``.

This module is the **general datatype**, independent of any integrator: ``step``
may be *any* such two-round behavior. A two-stage integrator (leapfrog,
``leapfrog.py``) is one instance built on top of it. Composition ``parallel`` and
``then_static`` mirror ``org.OrgMorphism``, delegating the inner round to the
1-stage versions.

Caveat: this provides the datatype, its execution, and these two composites
(tested). The claim that ``sarr → org^(2)`` is a lax monoidal *functor*
(the K=2 case of rmk.org_N) is conjectural and is **not** proved here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .org import OrgMorphism
from .polynomial import DirichletProduct, PolyMap


@dataclass(frozen=True)
class OrgMorphism2:
    """A ``[p,q]^{∘2}``-coalgebra: a morphism in ``org^(2)``, in two-round Moore form.

    ``step(state)`` returns ``(act, fiber)`` with

        act    : PolyMap p → q                    -- round-1 action a^β(state)
        fiber  : in_pos → (out_pos, at_pos)
        at_pos : in_dir → (out_dir, inner)        -- inner : org.OrgMorphism (round 2)

    where ``inner`` is the 1-stage coalgebra that round 2 runs.
    """

    src_poly: Any
    tgt_poly: Any
    state: Any
    step: Callable

    def with_state(self, state: Any) -> "OrgMorphism2":
        return OrgMorphism2(self.src_poly, self.tgt_poly, state, self.step)

    # ---- execution ----

    def run_two(self, in_pos1, in_dir_from1, in_pos2, in_dir_from2):
        """General two-round execution.

        Returns ``(out_pos1, out_dir1, out_pos2, out_dir2, new_state)``. The two
        environments may differ (round 2 sees the inner coalgebra produced by
        round 1).
        """
        act, fiber = self.step(self.state)
        out_pos1, at_pos1 = fiber(in_pos1)
        out_dir1, inner = at_pos1(in_dir_from1(out_pos1))
        out_pos2, out_dir2, new_state = inner.run_one(in_pos2, in_dir_from2)
        return out_pos1, out_dir1, out_pos2, out_dir2, new_state

    def run_one(self, in_pos, in_dir_from):
        """Closed-system convenience: the same environment at both rounds.

        Returns ``(out_pos1, out_pos2, new_state)``.
        """
        op1, _, op2, _, new_state = self.run_two(in_pos, in_dir_from, in_pos, in_dir_from)
        return op1, op2, new_state

    # ---- composition (mirrors org.OrgMorphism, lifted to two stages) ----

    def then_static(self, outer: PolyMap) -> "OrgMorphism2":
        """Compose round-1 and round-2 outputs through a static poly map ``outer``."""

        def new_step(s):
            inner_act, inner_fiber = self.step(s)
            composed_act = PolyMap(
                src=outer.src,
                tgt=inner_act.tgt,
                position_action=lambda i: inner_act.on_position(outer.on_position(i)),
                direction_action=lambda i, d: outer.on_direction(
                    i, inner_act.on_direction(outer.on_position(i), d)
                ),
                label=f"{outer.label};{inner_act.label}",
            )

            def fiber(i_outer):
                inner_pos = outer.on_position(i_outer)
                inner_out_pos, inner_at_pos = inner_fiber(inner_pos)

                def at_pos(d_outer_tgt):
                    out_dir_inner, inner_round2 = inner_at_pos(d_outer_tgt)
                    out_dir = outer.on_direction(i_outer, out_dir_inner)
                    return out_dir, inner_round2.then_static(outer)

                return inner_out_pos, at_pos

            return composed_act, fiber

        return OrgMorphism2(outer.src, self.tgt_poly, self.state, new_step)

    def parallel(self, other: "OrgMorphism2") -> "OrgMorphism2":
        """Monoidal product: two ``org^(2)`` morphisms run side by side."""

        def new_step(s):
            s1, s2 = s
            act1, fiber1 = self.step(s1)
            act2, fiber2 = other.step(s2)

            act = PolyMap(
                src=DirichletProduct(act1.src, act2.src),
                tgt=DirichletProduct(act1.tgt, act2.tgt),
                position_action=lambda i: (act1.on_position(i[0]), act2.on_position(i[1])),
                direction_action=lambda i, d: (
                    act1.on_direction(i[0], d[0]),
                    act2.on_direction(i[1], d[1]),
                ),
            )

            def fiber(i):
                op1, at1 = fiber1(i[0])
                op2, at2 = fiber2(i[1])

                def at_pos(d):
                    od1, inner1 = at1(d[0])
                    od2, inner2 = at2(d[1])
                    return (od1, od2), inner1.parallel(inner2)  # round 2 in parallel

                return (op1, op2), at_pos

            return act, fiber

        return OrgMorphism2(
            DirichletProduct(self.src_poly, other.src_poly),
            DirichletProduct(self.tgt_poly, other.tgt_poly),
            (self.state, other.state),
            new_step,
        )
