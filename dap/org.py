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

from .polynomial import DirichletProduct, PolyMap, PolyValue


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

    def then(self, other: "OrgMorphism") -> "OrgMorphism":
        """Sequential composition in pc: ``self : p -> q`` then ``other : q -> r``,
        giving ``p -> r`` (sec.org). State spaces multiply.

        Forward, a ``p``-position runs through ``self`` then ``other`` to an
        ``r``-position; backward, an ``r``-direction runs through ``other`` then
        ``self`` to a ``p``-direction, updating both states. This is the general
        ``pc`` composition on which the functoriality of ``Phi`` rests (the
        second-pass audit, sec.spring_second_pass); ``then_static`` is the special
        case where ``other`` carries a single state. Requires ``self.tgt_poly``
        and ``other.src_poly`` to agree.
        """

        from .polynomial import PolyMap as _PM

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

        return OrgMorphism(
            src_poly=self.src_poly,
            tgt_poly=other.tgt_poly,
            state=(self.state, other.state),
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


# ---- structure morphisms: the identity and the symmetry (sec.org) ----
#
# ``then``/``parallel`` are the composition and the monoidal product; to state the
# *laws* that make ``org(p, q)`` a symmetric monoidal category one also needs the
# unit for ``then`` (the identity coalgebra) and the braiding for ``parallel`` (the
# swap). Both are **stateless** structure maps: they carry no dynamics, only rewire
# positions forward and directions backward, so their state is the trivial ``None``
# (which ``jax`` pytrees flatten to no leaves -- exactly the unit ``I`` of the state
# monoid ``S1 x S2``, so ``identity.then(a)`` recovers ``a``'s state on the nose up to
# the canonical ``I x S ~= S``). These are the coherence isomorphisms of the monoidal
# category made concrete; the laws they satisfy are checked in ``test_monoidal_laws``.


def identity(poly) -> OrgMorphism:
    """The identity morphism ``id_p : p -> p`` in ``org`` -- the unit for ``then``.

    Emits the incoming position unchanged, returns the incoming direction unchanged,
    and never mutates its (trivial) state. It is the coalgebra whose action is
    ``identity_poly_map(p)`` at every tick. ``identity(p).then(a) == a`` and
    ``a.then(identity(r)) == a`` in observable behavior, with the state wrapped by the
    unit iso ``None x S ~= S`` (``test_monoidal_laws``).
    """
    from .polynomial import identity_poly_map

    def step(s):
        act = identity_poly_map(poly)

        def fiber(in_pos):
            def at_pos(in_dir):
                return in_dir, s  # trivial state, unchanged

            return in_pos, at_pos

        return act, fiber

    return OrgMorphism(src_poly=poly, tgt_poly=poly, state=None, step=step)


def braiding(p, q) -> OrgMorphism:
    """The symmetry ``sigma_{p,q} : p (x) q -> q (x) p`` in ``org`` (sec.org).

    The stateless coalgebra that swaps the two factors: forward it sends a position
    ``(i_p, i_q)`` to ``(i_q, i_p)``; backward it sends a ``q (x) p``-direction
    ``(d_q, d_p)`` to the ``p (x) q``-direction ``(d_p, d_q)``. It is an involution up
    to the swap (``braiding(p, q).then(braiding(q, p)) == identity(p (x) q)``) and is
    natural in both arguments, making ``parallel`` a *symmetric* monoidal product
    (``test_monoidal_laws``).
    """
    src = DirichletProduct(p, q)
    tgt = DirichletProduct(q, p)
    swap = PolyMap(
        src=src,
        tgt=tgt,
        position_action=lambda i: (i[1], i[0]),
        direction_action=lambda i, d: (d[1], d[0]),
        label="braiding",
    )

    def step(s):
        def fiber(in_pos):
            out_pos = (in_pos[1], in_pos[0])

            def at_pos(in_dir):  # in_dir: a (q (x) p)-direction (d_q, d_p)
                return (in_dir[1], in_dir[0]), s

            return out_pos, at_pos

        return swap, fiber

    return OrgMorphism(src_poly=src, tgt_poly=tgt, state=None, step=step)
