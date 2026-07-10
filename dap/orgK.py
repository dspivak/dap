"""``org^(K)``: general K-stage coalgebras ``[p,q]^{∘K}``-Coalg (rmk.multistage).

A morphism in ``org^(K)`` from ``p`` to ``q`` is a coalgebra
``β : S → [p,q]^{∘K}(S)`` for the K-fold substitution
``[p,q]^{∘K} = [p,q] ◁ … ◁ [p,q]``. In Moore form it is **K interaction rounds**
per macro-tick. Round 1 emits a ``q``-position, receives a ``q``-direction, returns
a ``p``-direction, and lands *not* in a new state ``S`` but in an **inner
``[p,q]^{∘(K-1)}``-coalgebra** -- an element of ``[p,q]^{∘(K-1)}(S)``; rounds
``2…K`` run that inner coalgebra. This is exactly ``org2.OrgMorphism2`` one
dimension up, built recursively rather than by copy-paste: the round-``i`` result
is the ``(K-i)``-round coalgebra ``org^(K-i)``, and the base case ``K = 1`` is the
single-stage ``org.OrgMorphism`` (its last round lands directly in a new state).

This module is the **general datatype**, independent of any integrator: ``step``
may be *any* such K-round behavior. A K-stage integrator (RK4, ``rk4.py``) is one
instance built on top of it via ``orgK_from_integrator``. Composition ``parallel``,
``then_static`` and the sequential ``then`` mirror ``org.OrgMorphism`` /
``org2.OrgMorphism2``, recursing on the remaining rounds.

Caveat: this provides the datatype, its execution, and these composites (tested).
The claim that ``sarr → org^(K)`` is a lax monoidal *functor* (the general case of
rmk.multistage) is conjectural and is **not** proved here -- exactly as for ``org^(2)``.
But ``then`` (the general ``pc`` composite, on which the functoriality of ``Phi`` rests)
now lets the second-pass audit of sec.spring_second_pass be run for the K=4 functor
``Phirk4`` (``test_multistage_functoriality.py``), as ``org.OrgMorphism.then`` does at
K=1: it passes (both passes agree exactly) -- computational evidence for the ``org^(K)``
case, still not a proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

from .polynomial import DirichletProduct, PolyMap


@dataclass(frozen=True)
class OrgMorphismK:
    """A ``[p,q]^{∘K}``-coalgebra: a morphism in ``org^(K)``, in K-round Moore form.

    ``step(state)`` returns ``(act, fiber)`` with

        act    : PolyMap p → q                    -- round-1 action a^β(state)
        fiber  : in_pos → (out_pos, at_pos)
        at_pos : in_dir → (out_dir, rest)

    where ``rest`` is the **(K-1)-round inner coalgebra** ``OrgMorphismK`` (its state
    baked in) when ``K > 1``, and the **new macro-state** when ``K = 1``. The field
    ``K`` records how many rounds remain; ``run`` reads it to know when ``rest`` is a
    coalgebra to recurse into versus a final state.
    """

    src_poly: Any
    tgt_poly: Any
    K: int
    state: Any
    step: Callable

    def with_state(self, state: Any) -> "OrgMorphismK":
        return OrgMorphismK(self.src_poly, self.tgt_poly, self.K, state, self.step)

    # ---- execution ----

    def run(self, in_poss: List, in_dir_froms: List):
        """General K-round execution.

        ``in_poss`` and ``in_dir_froms`` are length-``K`` lists, one per round (the
        rounds may see different environments). Returns ``(out_poss, out_dirs,
        new_state)`` with ``out_poss``/``out_dirs`` length-``K`` lists.
        """
        coalg = self
        out_poss, out_dirs = [], []
        for i in range(self.K):
            _act, fiber = coalg.step(coalg.state)
            out_pos, at_pos = fiber(in_poss[i])
            out_dir, rest = at_pos(in_dir_froms[i](out_pos))
            out_poss.append(out_pos)
            out_dirs.append(out_dir)
            coalg = rest  # an inner OrgMorphismK for i < K-1; the new state for i = K-1
        return out_poss, out_dirs, coalg  # after the loop, coalg is the new macro-state

    def run_one(self, in_pos, in_dir_from):
        """Closed-system convenience: the same environment at every round.

        Returns ``(out_poss, new_state)`` with ``out_poss`` a length-``K`` list.
        """
        out_poss, _out_dirs, new_state = self.run(
            [in_pos] * self.K, [in_dir_from] * self.K
        )
        return out_poss, new_state

    # ---- composition (mirrors org.OrgMorphism / org2.OrgMorphism2, lifted to K rounds) ----

    def then_static(self, outer: PolyMap) -> "OrgMorphismK":
        """Compose every round's output through a static poly map ``outer`` (sec.org)."""

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
                    out_dir_inner, rest = inner_at_pos(d_outer_tgt)
                    out_dir = outer.on_direction(i_outer, out_dir_inner)
                    # recurse into the remaining rounds; pass the state through at K = 1
                    new_rest = rest.then_static(outer) if self.K > 1 else rest
                    return out_dir, new_rest

                return inner_out_pos, at_pos

            return composed_act, fiber

        return OrgMorphismK(outer.src, self.tgt_poly, self.K, self.state, new_step)

    def then(self, other: "OrgMorphismK") -> "OrgMorphismK":
        """Sequential composition in pc, lifted to K stages: ``self : p -> q`` then
        ``other : q -> r``, giving ``p -> r`` (sec.org). State spaces multiply.

        The K-round analogue of ``org.OrgMorphism.then`` (and the ``org^(2)`` version in
        ``org2.OrgMorphism2.then``). Each of the K rounds threads a ``p``-position forward
        through ``self`` then ``other`` and an ``r``-direction backward through ``other``
        then ``self``, updating both states; the **remaining rounds recurse** --
        ``rest_self.then(rest_other)`` when ``K > 1`` (both ``rest`` are ``(K-1)``-round
        inner coalgebras), tupling the two final macro-states when ``K = 1`` (exactly as
        ``then_static``/``parallel`` recurse). Both stage counts must agree.

        This is the general ``pc`` composite on which the functoriality of a K-stage
        ``Phi`` (``functors.Phirk4`` at K=4) rests: the multi-stage case of the
        second-pass audit (sec.spring_second_pass) that ``org.OrgMorphism.then`` supports
        at K=1. Whether ``sarr -> org^(K)`` *is* a functor (rmk.multistage) is still open;
        this method is what lets that be tested round-by-round, not a proof.
        """
        if self.K != other.K:
            raise ValueError(f"then needs equal stage counts, got {self.K} and {other.K}")

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
                    d_q, rest_other = at_other(d_r)   # d_q: q-direction; rest: inner or state
                    d_p, rest_self = at_self(d_q)     # d_p: p-direction; rest: inner or state
                    if self.K > 1:
                        new_rest = rest_self.then(rest_other)  # remaining rounds composed
                    else:
                        new_rest = (rest_self, rest_other)     # both are final macro-states
                    return d_p, new_rest

                return k, at_pos

            return composed_act, fiber

        return OrgMorphismK(
            self.src_poly, other.tgt_poly, self.K, (self.state, other.state), new_step
        )

    def parallel(self, other: "OrgMorphismK") -> "OrgMorphismK":
        """Monoidal product: two ``org^(K)`` morphisms run side by side (sec.org)."""
        if self.K != other.K:
            raise ValueError(f"parallel needs equal stage counts, got {self.K} and {other.K}")

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
                    od1, rest1 = at1(d[0])
                    od2, rest2 = at2(d[1])
                    if self.K > 1:
                        new_rest = rest1.parallel(rest2)  # remaining rounds in parallel
                    else:
                        new_rest = (rest1, rest2)  # both are final states
                    return (od1, od2), new_rest

                return (op1, op2), at_pos

            return act, fiber

        return OrgMorphismK(
            DirichletProduct(self.src_poly, other.src_poly),
            DirichletProduct(self.tgt_poly, other.tgt_poly),
            self.K,
            (self.state, other.state),
            new_step,
        )


# ---- structure morphisms: identity and symmetry, general K rounds ----
#
# The K-round analogues of ``org.identity`` / ``org.braiding`` (and the ``org^(2)``
# versions), built by the same recursion as ``then``/``parallel``: each round is the
# structure map, and ``rest`` is the ``(K-1)``-round structure map when ``K > 1`` or
# the (trivial) new macro-state when ``K = 1``. They are the unit for ``then`` and the
# braiding for ``parallel`` at every K, so ``org^(K)`` is a symmetric monoidal category
# (``test_monoidal_laws``). Both are stateless (``None``): pure rewiring, no dynamics.


def identity(poly, K: int) -> OrgMorphismK:
    """The identity ``id_p : p -> p`` in ``org^(K)`` -- the unit for ``then`` at K rounds."""
    from .polynomial import identity_poly_map

    def make(rounds_left: int) -> OrgMorphismK:
        def step(s):
            act = identity_poly_map(poly)

            def fiber(in_pos):
                def at_pos(in_dir):
                    rest = s if rounds_left == 1 else make(rounds_left - 1).with_state(s)
                    return in_dir, rest

                return in_pos, at_pos

            return act, fiber

        return OrgMorphismK(poly, poly, rounds_left, None, step)

    return make(K)


def braiding(p, q, K: int) -> OrgMorphismK:
    """The symmetry ``sigma_{p,q} : p (x) q -> q (x) p`` in ``org^(K)`` (K rounds)."""
    src = DirichletProduct(p, q)
    tgt = DirichletProduct(q, p)
    swap = PolyMap(
        src=src,
        tgt=tgt,
        position_action=lambda i: (i[1], i[0]),
        direction_action=lambda i, d: (d[1], d[0]),
        label="braiding",
    )

    def make(rounds_left: int) -> OrgMorphismK:
        def step(s):
            def fiber(in_pos):
                out_pos = (in_pos[1], in_pos[0])

                def at_pos(in_dir):
                    rest = s if rounds_left == 1 else make(rounds_left - 1).with_state(s)
                    return (in_dir[1], in_dir[0]), rest

                return out_pos, at_pos

            return swap, fiber

        return OrgMorphismK(src, tgt, rounds_left, None, step)

    return make(K)


def orgK_from_integrator(arr, intg) -> OrgMorphismK:
    """Turn a K-stage integrator into an ``org^(K)`` morphism (the K-fold analog of
    ``functors.Phi`` / ``org2.org2_from_integrator``).

    Round ``i`` runs the interpretation ``Phi'`` at ``reads[i]`` and reads the
    parameter covector ``xi_Q``; ``advances[i]`` produces the next intermediate. The
    rounds are assembled recursively: round 1 lands in the ``(K-1)``-round inner
    ``org^(K-1)`` coalgebra (built by the same recursion at the next stage), and the
    final round (``K = 1``) lands in a new macro-state -- so ``org^(K)`` is built the
    same way ``org`` is, just from a K-stage integrator.
    """
    from .functors import phi_box_poly
    from .interpretation import smooth_interpretation

    Q = arr.Q
    interp = smooth_interpretation(arr)
    src_p = phi_box_poly(arr.out_dim_M, arr.in_dim_M)
    tgt_p = phi_box_poly(arr.out_dim_N, arr.in_dim_N)
    K = intg.K
    reads, advances = intg.reads, intg.advances

    def make(level: int, mid: Any) -> OrgMorphismK:
        rounds_left = K - level

        def step(state):
            q = reads[level](Q, state)
            position_action, direction_action = interp(q)

            def act_positions(in_pos):
                out_m, omega_M = in_pos
                return position_action(out_m, omega_M)

            def act_directions(in_pos, in_dir):
                out_m, omega_M = in_pos
                xi_N, in_n = in_dir
                _, xi_M, in_m = direction_action(out_m, omega_M, xi_N, in_n)
                return (xi_M, in_m)

            act = PolyMap(src_p, tgt_p, act_positions, act_directions, label=f"stage{level + 1}")

            def fiber(in_pos):
                out_m, omega_M = in_pos
                out_pos = position_action(out_m, omega_M)  # emit at reads[level](state)

                def at_pos(in_dir):
                    xi_N, in_n = in_dir
                    xi_Q, xi_M, in_m = direction_action(out_m, omega_M, xi_N, in_n)
                    nxt = advances[level](Q, state, xi_Q)
                    if rounds_left == 1:
                        rest = nxt  # finish: a new macro-state
                    else:
                        rest = make(level + 1, nxt)  # the (K-level-1)-round inner coalgebra
                    return (xi_M, in_m), rest

                return out_pos, at_pos

            return act, fiber

        return OrgMorphismK(src_p, tgt_p, rounds_left, mid, step)

    return make(0, intg.init(Q))
