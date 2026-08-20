"""Functoriality of the dynamics functors (sec.spring_second_pass).

Operad-functoriality has two legs, and they are tested separately here.

* The **∘-leg** (``test_*_functoriality_wave_chain``). The wave/heat chain
  ``wire_K(Part,...,Part)`` is the genuine composite
  ``compose_seq(tensor(Parts), chain_wire(K))`` in ``sarr``, so

    ``Phi(compose_seq(tensor(Parts), chain_wire))``  (compose in sarr, then Phi)
    ``Phi(tensor(Parts)).then(Phi(chain_wire))``     (Phi each, then compose in pc)

  must agree. This exercises ``PCMorphism.then`` on *non-identity, stateful*
  coalgebras (the parts under ``Phiphase`` carry the phase state ``T*R^K``).

* The **⊗-leg** (``test_laxator_coherence_*``). ``Phi`` is only *lax* monoidal on
  objects: ``Phi(A (x) B)`` carries one covector field on the product interface,
  while ``Phi(A) (x) Phi(B)`` carries one field per factor, and only the direct
  sum ``phi_laxator`` goes between them (functors.py). The coherence square

    ``(Phi(A) (x) Phi(B)).post_static(lax_N)  ==  Phi(A (x) B).pre_static(lax_M)``

  is checked on boxes whose interfaces are non-trivial on *both* sides -- where
  the flat/nested distinction is not vacuous -- with nonlinear ``out_f``,
  ``in_f``, ``U`` and a nonlinear incoming covector field.

* The paper's **second pass** (``test_second_pass_*``) uses both legs at once:
  combine the ``K`` particle coalgebras by the lax monoidal structure of ``pc``,
  then post-compose the static ``Phi(wire_K)`` (sec.spring_second_pass), and
  compare with the first pass ``Phi(compose_chain(Parts))``.
"""

import jax
jax.config.update("jax_enable_x64", True)

from functools import reduce

import jax.numpy as jnp
import numpy as np

from dap.interpretation import trivial_omega
from dap.arrangement import SmoothArrangement
from dap.functors import Phiconf, Phiphase, phi_laxator
from dap.polynomial import nest_left, unnest_left
from dap.rvect import diagonal
from dap.wiring import (
    chain_wire,
    compose_chain,
    parallel_arrangements,
    tensor_arrangements,
)

_IN_POS = (jnp.zeros(0), trivial_omega(0))


def _harmonic_particle(m, kappa):
    return SmoothArrangement(
        diagonal(jnp.array([1.0 / m])), 0, 0, 1, 1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2,
    )


def _step(O, state, in_dir, in_pos=_IN_POS):
    return O.with_state(state).run_one(in_pos, lambda _o: in_dir)


def _flatten_state(nested, K, phase):
    """The ``K``-fold tensor state ``((s_1, s_2), s_3)`` as the direct-sum state.

    Tensoring in ``sarr`` direct-sums the parameter spaces, so ``Phi`` of the
    tensor stores one array over ``Q_1 (+) ... (+) Q_K``; tensoring in ``pc``
    multiplies state sets, so it stores a nested tuple. This is the comparison.
    """
    parts = unnest_left(nested, K)
    if phase:
        return (jnp.concatenate([q for q, _ in parts]),
                jnp.concatenate([xi for _, xi in parts]))
    return jnp.concatenate(parts)


# ---------------------------------------------------------------------------
# The compose-leg: composition in sarr vs composition in pc.
# ---------------------------------------------------------------------------


def _assert_functorial(Phi, K, m, kappa, phase):
    parts = [_harmonic_particle(m, kappa) for _ in range(K)]
    first = Phi(compose_chain(parts))                            # compose in sarr, then Phi
    second = Phi(tensor_arrangements(parts)).then(Phi(chain_wire(K)))  # Phi each, compose in pc

    wire_state = second.state[1]  # second.state = (tensor-state, wire-state)
    rng = np.random.default_rng(7)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(K))
        if phase:
            p = jnp.asarray(rng.standard_normal(K))
            s1, s2 = (q, p), ((q, p), wire_state)
        else:
            s1, s2 = q, (q, wire_state)
        in_dir = (jnp.asarray(rng.standard_normal(1)), jnp.asarray(rng.standard_normal(1)))

        (on1, om1), d1, ns1 = _step(first, s1, in_dir)
        (on2, om2), d2, ns2 = _step(second, s2, in_dir)
        ns2 = ns2[0]  # second nests the tensor-state under the wire-state

        # same output position and covector field (probe the field at a point)
        np.testing.assert_allclose(np.asarray(on1), np.asarray(on2), atol=1e-10)
        z = jnp.array([0.37])
        np.testing.assert_allclose(np.asarray(om1(z)), np.asarray(om2(z)), atol=1e-10)
        # same returned direction at the (unit) source interface
        for a, b in zip(d1, d2):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-10)
        # same updated state
        if phase:
            np.testing.assert_allclose(np.asarray(ns1[0]), np.asarray(ns2[0]), atol=1e-10)
            np.testing.assert_allclose(np.asarray(ns1[1]), np.asarray(ns2[1]), atol=1e-10)
        else:
            np.testing.assert_allclose(np.asarray(ns1), np.asarray(ns2), atol=1e-10)


def test_phiphase_functoriality_wave_chain():
    _assert_functorial(Phiphase, K=5, m=1.7, kappa=2.3, phase=True)


def test_phiconf_functoriality_heat_chain():
    _assert_functorial(Phiconf, K=5, m=1.5, kappa=0.9, phase=False)


# ---------------------------------------------------------------------------
# The tensor-leg: the laxator's coherence square, on non-trivial interfaces.
# ---------------------------------------------------------------------------


def _generic_box(dim_Q, d_out_M, d_in_M, d_out_N, d_in_N, seed):
    """A box with nonlinear ``out_f``, ``in_f``, ``U`` and non-trivial interfaces
    on both sides -- so every summand of eqn.covector_triple is exercised."""
    rng = np.random.default_rng(seed)
    A = jnp.asarray(rng.standard_normal((d_out_N, dim_Q + d_out_M)))
    B = jnp.asarray(rng.standard_normal((d_in_M, dim_Q + d_out_M + d_in_N)))
    c = jnp.asarray(rng.standard_normal(dim_Q + d_out_M + d_in_N))

    def out_f(q, m_out):
        return jnp.tanh(A @ jnp.concatenate([q, m_out]))

    def in_f(q, m_out, n_in):
        return jnp.sin(B @ jnp.concatenate([q, m_out, n_in]))

    def U(q, m_out, n_in):
        z = jnp.concatenate([q, m_out, n_in])
        return jnp.sum(jnp.cos(c * z)) + 0.5 * jnp.sum(z ** 2)

    return SmoothArrangement(
        diagonal(jnp.asarray(0.5 + rng.random(dim_Q))),
        d_out_M, d_in_M, d_out_N, d_in_N,
        out_f=out_f, in_f=in_f, U=U, label=f"box{seed}",
    )


def _assert_laxator_coherence(Phi, phase):
    A = _generic_box(2, 2, 3, 2, 2, seed=1)
    B = _generic_box(3, 1, 2, 3, 1, seed=2)
    boxes = [A, B]

    lax_M = phi_laxator([(P.out_dim_M, P.in_dim_M) for P in boxes], label="lax_M")
    lax_N = phi_laxator([(P.out_dim_N, P.in_dim_N) for P in boxes], label="lax_N")

    lhs = Phi(A).parallel(Phi(B)).post_static(lax_N)          # tensor in pc, then compare
    rhs = Phi(parallel_arrangements(A, B)).pre_static(lax_M)  # tensor in sarr, then Phi
    assert lhs.src_poly == rhs.src_poly and lhs.tgt_poly == rhs.tgt_poly

    d_out_M, d_in_M = A.out_dim_M + B.out_dim_M, A.in_dim_M + B.in_dim_M
    d_out_N, d_in_N = A.out_dim_N + B.out_dim_N, A.in_dim_N + B.in_dim_N
    rng = np.random.default_rng(11)
    for _ in range(5):
        # a nested source position: one output value and one covector field per factor
        def field(seed, d):
            w = jnp.asarray(rng.standard_normal(d))
            return lambda z: jnp.cos(3.0 * z) + w * z ** 2
        in_pos = ((jnp.asarray(rng.standard_normal(A.out_dim_M)), field(0, A.in_dim_M)),
                  (jnp.asarray(rng.standard_normal(B.out_dim_M)), field(1, B.in_dim_M)))
        in_dir = (jnp.asarray(rng.standard_normal(d_out_N)),
                  jnp.asarray(rng.standard_normal(d_in_N)))

        qA = jnp.asarray(rng.standard_normal(A.Q.dim))
        qB = jnp.asarray(rng.standard_normal(B.Q.dim))
        if phase:
            pA = jnp.asarray(rng.standard_normal(A.Q.dim))
            pB = jnp.asarray(rng.standard_normal(B.Q.dim))
            s_lhs = ((qA, pA), (qB, pB))
        else:
            s_lhs = (qA, qB)
        s_rhs = _flatten_state(s_lhs, 2, phase)

        (on1, om1), d1, ns1 = _step(lhs, s_lhs, in_dir, in_pos)
        (on2, om2), d2, ns2 = _step(rhs, s_rhs, in_dir, in_pos)

        # same output position, and the same (flat) covector field on the product
        np.testing.assert_allclose(np.asarray(on1), np.asarray(on2), atol=1e-10)
        z = jnp.asarray(rng.standard_normal(d_in_N))
        np.testing.assert_allclose(np.asarray(om1(z)), np.asarray(om2(z)), atol=1e-10)
        # same returned directions, factor by factor
        for (xi_a, m_a), (xi_b, m_b) in zip(unnest_left(d1, 2), unnest_left(d2, 2)):
            np.testing.assert_allclose(np.asarray(xi_a), np.asarray(xi_b), atol=1e-10)
            np.testing.assert_allclose(np.asarray(m_a), np.asarray(m_b), atol=1e-10)
        # same updated state, under the direct-sum comparison
        f1 = _flatten_state(ns1, 2, phase)
        for a, b in zip(f1 if phase else (f1,), ns2 if phase else (ns2,)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-10)


def test_laxator_coherence_phase():
    _assert_laxator_coherence(Phiphase, phase=True)


def test_laxator_coherence_conf():
    _assert_laxator_coherence(Phiconf, phase=False)


def test_laxator_is_not_strong():
    """The laxator is not invertible: a covector field on the product interface is
    not a tuple of fields on the factors (which is why ``Phi`` is only lax)."""
    lax = phi_laxator([(1, 2), (1, 2)])
    m = jnp.zeros(1)
    omega = lambda z: jnp.cos(z)
    flat, field = lax.on_position(((m, omega), (m, omega)))
    assert flat.shape == (2,)
    # the image consists of the direct sums: no cross-factor coupling survives
    z = jnp.array([0.3, 0.7, -0.2, 1.1])
    np.testing.assert_allclose(
        np.asarray(field(z)),
        np.asarray(jnp.concatenate([omega(z[:2]), omega(z[2:])])),
        atol=1e-12,
    )
    coupled = lambda z: jnp.stack([z[0] * z[3], z[1], z[2], z[3]])  # not a direct sum
    assert not np.allclose(np.asarray(coupled(z)), np.asarray(field(z)))


# ---------------------------------------------------------------------------
# The paper's second pass (sec.spring_second_pass), both legs at once.
# ---------------------------------------------------------------------------


def _second_pass(Phi, parts):
    """``Phi(wire_K)(Phi(Part),...,Phi(Part))`` exactly as sec.spring_second_pass
    builds it: combine the ``K`` coalgebras via the lax monoidal structure of
    ``pc``, then post-compose the static polynomial map ``Phi(wire_K)``."""
    K = len(parts)
    tensored = reduce(lambda a, b: a.parallel(b), [Phi(P) for P in parts])
    lax_N = phi_laxator([(P.out_dim_N, P.in_dim_N) for P in parts], label="lax_N")
    return tensored.post_static(lax_N).then(Phi(chain_wire(K)))


def _assert_second_pass(Phi, K, m, kappa, phase):
    parts = [_harmonic_particle(m, kappa) for _ in range(K)]
    first = Phi(compose_chain(parts))
    second = _second_pass(Phi, parts)

    # the source of the second pass is the K-fold tensor of the unit interface,
    # so its position is the K-fold nesting of the unit's single position
    in_pos_2 = nest_left([_IN_POS] * K)
    wire_state = second.state[1]
    rng = np.random.default_rng(19)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(K))
        if phase:
            p = jnp.asarray(rng.standard_normal(K))
            s1 = (q, p)
            nested = nest_left([(q[i:i + 1], p[i:i + 1]) for i in range(K)])
        else:
            s1 = q
            nested = nest_left([q[i:i + 1] for i in range(K)])
        s2 = (nested, wire_state)
        in_dir = (jnp.asarray(rng.standard_normal(1)), jnp.asarray(rng.standard_normal(1)))

        (on1, om1), _, ns1 = _step(first, s1, in_dir)
        (on2, om2), _, ns2 = _step(second, s2, in_dir, in_pos_2)
        ns2 = _flatten_state(ns2[0], K, phase)

        np.testing.assert_allclose(np.asarray(on1), np.asarray(on2), atol=1e-10)
        z = jnp.array([0.41])
        np.testing.assert_allclose(np.asarray(om1(z)), np.asarray(om2(z)), atol=1e-10)
        if phase:
            np.testing.assert_allclose(np.asarray(ns1[0]), np.asarray(ns2[0]), atol=1e-10)
            np.testing.assert_allclose(np.asarray(ns1[1]), np.asarray(ns2[1]), atol=1e-10)
        else:
            np.testing.assert_allclose(np.asarray(ns1), np.asarray(ns2), atol=1e-10)


def test_second_pass_wave_chain():
    _assert_second_pass(Phiphase, K=5, m=1.7, kappa=2.3, phase=True)


def test_second_pass_heat_chain():
    _assert_second_pass(Phiconf, K=5, m=1.5, kappa=0.9, phase=False)
