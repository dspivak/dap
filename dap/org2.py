"""``org^(2)``: general two-stage coalgebras ``[p,q]^{∘2}``-Coalg (rmk.multistage).

A morphism in ``org^(2)`` from ``p`` to ``q`` is a coalgebra
``β : S → [p,q]([p,q](S))`` for the substitution ``[p,q]^{∘2} = [p,q] ◁ [p,q]``.
In Moore form it is **two interaction rounds** per macro-tick. The key is the
substitution: round 1 emits a ``q``-position, receives a ``q``-direction, returns
a ``p``-direction, and lands *not* in a new state ``S`` but in an **inner 1-stage
``[p,q]``-coalgebra** — an element of ``[p,q](S)``, i.e. an ``org.OrgMorphism``.
Round 2 runs that inner coalgebra and lands in ``S``.

This module is the **general datatype**, independent of any integrator: ``step``
may be *any* such two-round behavior. A two-stage integrator (leapfrog,
``leapfrog.py``) is one instance built on top of it. Composition ``parallel``,
``then_static`` and the sequential ``then`` mirror ``org.OrgMorphism``, delegating the
inner round to the 1-stage versions.

Caveat: this provides the datatype, its execution, and these composites (tested).
The claim that ``sarr → org^(2)`` is a lax monoidal *functor* (the K=2 case of
rmk.multistage) is conjectural and is **not** proved here -- but ``then`` (the general
``pc`` composite) now lets the second-pass functoriality audit of sec.spring_second_pass
be run for ``Phileap`` (``test_multistage_functoriality.py``), as ``org.OrgMorphism.then``
does at K=1: it passes (both passes agree exactly), evidence for the K=2 case, still not
a proof.
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

    def then(self, other: "OrgMorphism2") -> "OrgMorphism2":
        """Sequential composition in pc, lifted to two stages: ``self : p -> q`` then
        ``other : q -> r``, giving ``p -> r`` (sec.org). State spaces multiply.

        The two-stage analogue of ``org.OrgMorphism.then``. Each round threads a
        ``p``-position forward through ``self`` then ``other`` to an ``r``-position and
        an ``r``-direction backward through ``other`` then ``self`` to a ``p``-direction,
        updating both states; **round 2 delegates to the 1-stage ``OrgMorphism.then``**
        on the inner coalgebras (exactly as ``then_static``/``parallel`` delegate their
        second round). This is the general ``pc`` composite on which the functoriality of
        a two-stage ``Phi`` (``leapfrog.Phileap``) rests -- the K=2 case of the audit that
        ``org.OrgMorphism.then`` supports at K=1 (sec.spring_second_pass). Requires
        ``self.tgt_poly`` and ``other.src_poly`` to agree.
        """

        def new_step(s):
            s_self, s_other = s
            act_self, fiber_self = self.step(s_self)
            act_other, fiber_other = other.step(s_other)

            composed_act = PolyMap(
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
                    d_q, inner_other = at_other(d_r)   # d_q: q-direction; inner_other: round 2
                    d_p, inner_self = at_self(d_q)     # d_p: p-direction; inner_self: round 2
                    return d_p, inner_self.then(inner_other)  # round 2: 1-stage org then

                return k, at_pos

            return composed_act, fiber

        return OrgMorphism2(self.src_poly, other.tgt_poly, (self.state, other.state), new_step)

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


# ---- structure morphisms: identity and symmetry, lifted to two stages ----
#
# The ``org^(2)`` analogues of ``org.identity`` / ``org.braiding``. Each is a two-round
# stateless coalgebra whose **round 2 is the 1-stage structure map** on the inner
# coalgebra -- exactly the delegation ``then``/``parallel`` already use. They are the
# unit for ``OrgMorphism2.then`` and the braiding for ``OrgMorphism2.parallel``, so
# ``org^(2)(p, q)`` is a symmetric monoidal category (``test_monoidal_laws``).


def identity(poly) -> OrgMorphism2:
    """The identity ``id_p : p -> p`` in ``org^(2)`` -- the unit for ``then`` (K = 2)."""
    from .org import identity as _identity_org
    from .polynomial import identity_poly_map

    def step(s):
        act = identity_poly_map(poly)

        def fiber(in_pos):
            def at_pos(in_dir):
                return in_dir, _identity_org(poly)  # round 2: the 1-stage identity

            return in_pos, at_pos

        return act, fiber

    return OrgMorphism2(poly, poly, None, step)


def braiding(p, q) -> OrgMorphism2:
    """The symmetry ``sigma_{p,q} : p (x) q -> q (x) p`` in ``org^(2)`` (K = 2)."""
    from .org import braiding as _braiding_org

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

            def at_pos(in_dir):
                return (in_dir[1], in_dir[0]), _braiding_org(p, q)  # round 2: 1-stage swap

            return out_pos, at_pos

        return swap, fiber

    return OrgMorphism2(src, tgt, None, step)


def org2_from_integrator(arr, intg2) -> OrgMorphism2:
    """Turn a two-stage integrator into an ``org^(2)`` morphism (the K=2 analog of
    ``functors.Phi`` / ``prop.integrator_to_org``).

    Round 1 runs the interpretation ``Phi'`` at ``read1(state)`` and reads the
    parameter covector ``xi_Q1``; ``advance`` produces the intermediate. Round 2 is
    a genuine **1-stage** ``org.OrgMorphism`` built by ``functors.Phi`` from a
    1-stage integrator derived from ``read2``/``finish`` — so ``org^(2)`` is
    assembled the same way ``org`` is, just from a two-stage integrator.
    """
    from .functors import Phi, phi_box_poly
    from .integrator import Integrator
    from .interpretation import smooth_interpretation

    Q = arr.Q
    interp = smooth_interpretation(arr)
    src_p = phi_box_poly(arr.out_dim_M, arr.in_dim_M)
    tgt_p = phi_box_poly(arr.out_dim_N, arr.in_dim_N)

    def step(state):
        q1 = intg2.read1(Q, state)
        position_action, direction_action = interp(q1)

        def act_positions(in_pos):
            out_m, omega_M = in_pos
            return position_action(out_m, omega_M)

        def act_directions(in_pos, in_dir):
            out_m, omega_M = in_pos
            xi_N, in_n = in_dir
            _, xi_M, in_m = direction_action(out_m, omega_M, xi_N, in_n)
            return (xi_M, in_m)

        act = PolyMap(src_p, tgt_p, act_positions, act_directions, label="stage1")

        def fiber(in_pos):
            out_m, omega_M = in_pos
            out_pos = position_action(out_m, omega_M)  # emit at read1(state)

            def at_pos(in_dir):
                xi_N, in_n = in_dir
                xi_Q1, xi_M, in_m = direction_action(out_m, omega_M, xi_N, in_n)
                mid = intg2.advance(Q, state, xi_Q1)
                # round 2: a 1-stage org-morphism from a 1-stage integrator (read2/finish)
                round2 = Integrator(
                    init=lambda Q_, m=mid: m,
                    position=lambda Q_, s: intg2.read2(Q_, s),
                    step=lambda Q_, s, xi_Q2, st=state: intg2.finish(Q_, st, s, xi_Q2),
                    label="stage2",
                )
                inner = Phi(arr, round2).with_state(mid)
                return (xi_M, in_m), inner

            return out_pos, at_pos

        return act, fiber

    return OrgMorphism2(src_p, tgt_p, intg2.init(Q), step)
