"""``org^(K)`` is a symmetric monoidal category (sec.org, rmk.multistage).

PR #1 (``test_composition`` / ``test_multistage_functoriality``) gave the ``;``-half of
functoriality: ``Phi`` preserves *sequential* composition, using ``then``. But a functor
needs a *category* to land in, and a *monoidal* functor a *monoidal* category. This
module supplies the missing target structure: it checks that ``then`` (composition),
``parallel`` (monoidal product), ``identity`` (unit) and ``braiding`` (symmetry) satisfy
the laws of a **symmetric monoidal category**, at every stage count the repo ships --
``K = 1`` (``org``), ``K = 2`` (``org^(2)``), ``K = 4`` (``org^(4)``):

  * unit laws          ``identity.then(a) == a == a.then(identity)``
  * associativity      ``(a.then(b)).then(c) == a.then(b.then(c))``
  * interchange        ``(a ∥ b).then(c ∥ d) == (a.then(c)) ∥ (b.then(d))``
  * symmetry           ``braiding.then(braiding) == identity``  (involution)
                       ``(a ∥ b).then(braiding) == braiding.then(b ∥ a)``  (naturality)

The subjects are small *stateful, observable* toy coalgebras over ``cot(R)`` -- each
emits a position, returns a direction, and updates a real state per round -- so the
laws are exercised on genuine dynamics, not on identities. **Evidence, not proof**, in
the repo's idiom, but the evidence is sharp and its form is the honest categorical
statement: every law holds

  * **on the nose in observable behavior** -- same emitted positions and returned
    directions at every one of the K rounds, for several consecutive macro-ticks, to
    machine precision (checked here by driving both sides forward and comparing); and
  * **up to the canonical state bijection** -- the two sides differ only by a retupling
    of the multiplied state spaces ``S1 x S2`` (the associator / interchanger / unitor),
    which is exactly the coherence isomorphism of the monoidal category. We witness this
    by checking the two final states carry the *same multiset of leaves* (reassociation
    permutes leaves, it does not change them), and -- more strongly -- by the multi-tick
    behavioural agreement, which would break on the very next tick if the states did not
    correspond under that bijection.

Together with the ``⊗``-preservation of ``test_monoidal_functoriality`` and the
``;``-preservation of PR #1, this is the sharpest statement yet of the open
``rmk.multistage`` question: ``Phi : sarr -> org^(K)`` is a *symmetric monoidal functor*
(evidenced), for K = 1, 2, 4.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from dap import org, org2, orgK
from dap.polynomial import Cot, DirichletProduct, PolyMap

C1 = Cot(1)
PROD = DirichletProduct(C1, C1)


# ---------------------------------------------------------------------------
# Toy coalgebras: observable and stateful, over cot(R). Each round emits
# ``scale * i + state``, returns ``scale * d`` backward, and integrates the
# incoming direction into the state (``state + gain * d``) -- so ``then`` and
# ``parallel`` have real state to thread and real dynamics to preserve.
# ---------------------------------------------------------------------------


def _toy_org(scale, gain, s0):
    def step(state):
        act = PolyMap(C1, C1, lambda i: scale * i + state, lambda i, d: scale * d, label="toy")

        def fiber(i):
            out = scale * i + state

            def at_pos(d):
                return scale * d, state + gain * d

            return out, at_pos

        return act, fiber

    return org.OrgMorphism(C1, C1, s0, step)


def _toy_org2(scale, gain, s0):
    def step(state):
        act = PolyMap(C1, C1, lambda i: scale * i + state, lambda i, d: scale * d, label="toy")

        def fiber(i):
            out = scale * i + state

            def at_pos(d):
                # round 2 is a 1-stage toy carrying the updated state
                return scale * d, _toy_org(scale, gain, state + gain * d)

            return out, at_pos

        return act, fiber

    return org2.OrgMorphism2(C1, C1, s0, step)


def _toy_orgK(scale, gain, s0, K):
    def make(rounds_left, s):
        def step(state):
            act = PolyMap(C1, C1, lambda i: scale * i + state, lambda i, d: scale * d, label="toy")

            def fiber(i):
                out = scale * i + state

                def at_pos(d):
                    nxt = state + gain * d
                    rest = nxt if rounds_left == 1 else make(rounds_left - 1, nxt)
                    return scale * d, rest

                return out, at_pos

            return act, fiber

        return orgK.OrgMorphismK(C1, C1, rounds_left, s, step)

    return make(K, s0)


# One "kind" per stage count: how to build a toy, the identity, the braiding, and
# the identity on the product (for the braiding-involution law), in that datatype.
def _kind_org():
    return dict(
        mk=_toy_org,
        ident=lambda: org.identity(C1),
        braid=lambda: org.braiding(C1, C1),
        id_prod=lambda: org.identity(PROD),
    )


def _kind_org2():
    return dict(
        mk=_toy_org2,
        ident=lambda: org2.identity(C1),
        braid=lambda: org2.braiding(C1, C1),
        id_prod=lambda: org2.identity(PROD),
    )


def _kind_orgK(K):
    return dict(
        mk=lambda scale, gain, s0: _toy_orgK(scale, gain, s0, K),
        ident=lambda: orgK.identity(C1, K),
        braid=lambda: orgK.braiding(C1, C1, K),
        id_prod=lambda: orgK.identity(PROD, K),
    )


KINDS = [("org", _kind_org()), ("org2", _kind_org2()), ("orgK4", _kind_orgK(4))]


# ---------------------------------------------------------------------------
# Driving + comparison. ``_run_macro`` runs one macro-tick of any of the three
# datatypes under a fixed environment; ``_assert_same`` drives both morphisms for
# several ticks and asserts (i) identical observable behavior each round and
# (ii) same final-state leaf multiset (agreement up to the coherence iso).
# ---------------------------------------------------------------------------


def _run_macro(O, in_pos, env):
    if hasattr(O, "K"):  # OrgMorphismK (K interaction rounds)
        return O.run([in_pos] * O.K, [env] * O.K)
    if hasattr(O, "run_two"):  # OrgMorphism2 (two rounds)
        op1, od1, op2, od2, ns = O.run_two(in_pos, env, in_pos, env)
        return [op1, op2], [od1, od2], ns
    op, od, ns = O.run_one(in_pos, env)  # OrgMorphism (one round)
    return [op], [od], ns


def _probe(poly, rng):
    if isinstance(poly, Cot):
        return jnp.asarray(rng.standard_normal(poly.dim))
    if isinstance(poly, DirichletProduct):
        return tuple(_probe(f, rng) for f in poly.factors)
    return jnp.zeros(0)  # Yon


def _env(out_pos):
    # A direction has the same tree/shape as a cot-position; make it depend on the
    # emitted position so the drive is non-trivial (and identical for both sides).
    return jax.tree_util.tree_map(lambda x: 0.4 + 0.15 * x, out_pos)


def _scalar_leaves(tree):
    return sorted(float(v) for leaf in jax.tree_util.tree_leaves(tree) for v in np.ravel(leaf))


def _assert_same(O1, O2, T=6, seed=0):
    """Assert two ``org^(K)`` morphisms are equal as arrows: identical observable
    behavior for ``T`` macro-ticks, and final states equal up to the canonical
    reassociation (same leaf multiset)."""
    rng = np.random.default_rng(seed)
    in_pos = _probe(O1.src_poly, rng)  # both sides share src_poly, hence in_pos

    st1, st2 = O1.state, O2.state
    for _ in range(T):
        ops1, ods1, st1 = _run_macro(O1.with_state(st1), in_pos, _env)
        ops2, ods2, st2 = _run_macro(O2.with_state(st2), in_pos, _env)
        assert len(ops1) == len(ops2)  # same number of rounds
        for a, b in zip(jax.tree_util.tree_leaves((ops1, ods1)),
                        jax.tree_util.tree_leaves((ops2, ods2))):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-10)

    l1, l2 = _scalar_leaves(st1), _scalar_leaves(st2)
    assert len(l1) == len(l2)
    np.testing.assert_allclose(l1, l2, atol=1e-10)


# ---------------------------------------------------------------------------
# The laws.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,kind", KINDS)
def test_left_unit(name, kind):
    """``identity.then(a) == a``: the identity coalgebra is a left unit for ``then``."""
    a = kind["mk"](1.3, 0.5, jnp.array([0.2]))
    _assert_same(kind["ident"]().then(a), a)


@pytest.mark.parametrize("name,kind", KINDS)
def test_right_unit(name, kind):
    """``a.then(identity) == a``: the identity coalgebra is a right unit for ``then``."""
    a = kind["mk"](1.3, 0.5, jnp.array([0.2]))
    _assert_same(a.then(kind["ident"]()), a)


@pytest.mark.parametrize("name,kind", KINDS)
def test_associativity(name, kind):
    """``(a.then(b)).then(c) == a.then(b.then(c))``: ``then`` is associative."""
    a = kind["mk"](1.1, 0.3, jnp.array([0.2]))
    b = kind["mk"](0.9, -0.4, jnp.array([-0.1]))
    c = kind["mk"](1.4, 0.2, jnp.array([0.05]))
    _assert_same((a.then(b)).then(c), a.then(b.then(c)))


@pytest.mark.parametrize("name,kind", KINDS)
def test_interchange(name, kind):
    """``(a ∥ b).then(c ∥ d) == (a.then(c)) ∥ (b.then(d))``.

    The interchange law: ``parallel`` is a bifunctor with respect to ``then`` -- i.e.
    ``org^(K)`` is a genuinely *monoidal* category, not just a category with a product.
    """
    a = kind["mk"](1.1, 0.3, jnp.array([0.2]))
    b = kind["mk"](0.8, -0.2, jnp.array([-0.3]))
    c = kind["mk"](1.3, 0.15, jnp.array([0.1]))
    d = kind["mk"](0.95, 0.4, jnp.array([0.25]))
    lhs = (a.parallel(b)).then(c.parallel(d))
    rhs = (a.then(c)).parallel(b.then(d))
    _assert_same(lhs, rhs)


@pytest.mark.parametrize("name,kind", KINDS)
def test_braiding_involution(name, kind):
    """``braiding.then(braiding) == identity``: the symmetry is self-inverse."""
    _assert_same(kind["braid"]().then(kind["braid"]()), kind["id_prod"]())


@pytest.mark.parametrize("name,kind", KINDS)
def test_braiding_naturality(name, kind):
    """``(a ∥ b).then(braiding) == braiding.then(b ∥ a)``: the braiding is natural.

    With involution, this makes ``parallel`` a *symmetric* monoidal product.
    """
    a = kind["mk"](1.2, 0.3, jnp.array([0.15]))
    b = kind["mk"](0.85, -0.25, jnp.array([-0.2]))
    lhs = (a.parallel(b)).then(kind["braid"]())
    rhs = kind["braid"]().then(b.parallel(a))
    _assert_same(lhs, rhs)
