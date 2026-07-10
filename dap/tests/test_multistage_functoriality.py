"""Functoriality of the *multi-stage* dynamics functors (sec.spring_second_pass).

``test_composition.py`` runs the paper's second-pass audit -- ``Phi`` preserves
composition -- for the SINGLE-stage functors ``Phiconf``/``Phiphase``, using
``org.OrgMorphism.then``. This module runs the SAME audit for the MULTI-stage
functors: leapfrog ``Phileap`` (``org^(2)``) and RK4 ``Phirk4`` (``org^(4)``), using
the K-round sequential composites ``org2.OrgMorphism2.then`` / ``orgK.OrgMorphismK.then``:

  first pass:   ``Phi(compose_chain(Parts))``                       (compose in sarr, then Phi)
  second pass:  ``Phi(tensor(Parts)).then(Phi(chain_wire(K)))``     (Phi each, then compose in pc)

The wave/heat chain ``compose_chain(Parts) = compose_seq(tensor(Parts), chain_wire(K))``
is the genuine operad composite in ``sarr``; the audit checks, at every one of the K
interaction rounds, that the two passes emit the same output position and covector
field and return the same source direction, and end at the same macro-state.

Passing is the first computational evidence that ``sarr -> org^(K)`` preserves
*sequential* composition for ``K > 1`` -- the crux of the functoriality question
``rmk.multistage`` leaves open. The datatype plus ``parallel``/``then_static`` were
already there and tested; the general ``pc`` composite ``then``, *on which the
functoriality of ``Phi`` rests* (cf. ``test_composition.py``), was the missing piece
at ``K > 1``. It is **evidence, not a proof**: agreement holds round-by-round, from
arbitrary matched states, to machine precision (in fact exactly -- the two passes run
the same operations threaded differently).
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from dap.arrangement import SmoothArrangement
from dap.functors import Phirk4
from dap.integrator import IntegratorK
from dap.interpretation import trivial_omega
from dap.leapfrog import Phileap
from dap.orgK import orgK_from_integrator
from dap.rvect import diagonal, euclidean
from dap.wiring import chain_wire, compose_chain, tensor_arrangements

_IN_POS = (jnp.zeros(0), trivial_omega(0))


def _harmonic_particle(m, kappa):
    """A wave/heat chain particle ``<R^0|R^0> -> <R|R>`` (as in ``test_composition``)."""
    return SmoothArrangement(
        diagonal(jnp.array([1.0 / m])), 0, 0, 1, 1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2,
    )


def _run_rounds(O, env):
    """One macro-tick of a multi-stage coalgebra under a fixed environment ``env``.

    Returns ``(out_poss, out_dirs, new_state)`` with length-K lists, for either datatype
    -- ``OrgMorphismK`` (``.run``) or ``OrgMorphism2`` (``.run_two``).
    """
    if hasattr(O, "K"):  # OrgMorphismK (K interaction rounds)
        return O.run([_IN_POS] * O.K, [env] * O.K)
    op1, od1, op2, od2, ns = O.run_two(_IN_POS, env, _IN_POS, env)  # OrgMorphism2
    return [op1, op2], [od1, od2], ns


def _assert_functorial(PhiK, K, m, kappa, phase):
    parts = [_harmonic_particle(m, kappa) for _ in range(K)]
    first = PhiK(compose_chain(parts))                                   # compose in sarr, then Phi
    second = PhiK(tensor_arrangements(parts)).then(PhiK(chain_wire(K)))  # Phi each, compose in pc

    wire_state = second.state[1]  # second.state = (tensor-state, wire-state)
    rng = np.random.default_rng(7)
    z = jnp.array([0.37])         # a probe point for the emitted covector field
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(K))
        if phase:
            p = jnp.asarray(rng.standard_normal(K))
            s1, s2 = (q, p), ((q, p), wire_state)
        else:
            s1, s2 = q, (q, wire_state)
        in_dir = (jnp.asarray(rng.standard_normal(1)), jnp.asarray(rng.standard_normal(1)))
        env = lambda _op: in_dir  # noqa: E731 -- same environment at every round (closed drive)

        op1, od1, ns1 = _run_rounds(first.with_state(s1), env)
        op2, od2, ns2 = _run_rounds(second.with_state(s2), env)
        ns2 = ns2[0]  # second nests the tensor-state under the wire-state

        # Per round (rounds = the integrator's stage count, 2 for leapfrog / 4 for RK4;
        # distinct from the chain length K): same output position, same output covector
        # field, same returned source direction.
        assert len(op1) == len(op2)
        for r in range(len(op1)):
            (on1, om1), (on2, om2) = op1[r], op2[r]
            np.testing.assert_allclose(np.asarray(on1), np.asarray(on2), atol=1e-10)
            np.testing.assert_allclose(np.asarray(om1(z)), np.asarray(om2(z)), atol=1e-10)
            (xi_M1, _in_m1), (xi_M2, _in_m2) = od1[r], od2[r]
            np.testing.assert_allclose(np.asarray(xi_M1), np.asarray(xi_M2), atol=1e-10)

        # Same updated macro-state.
        if phase:
            np.testing.assert_allclose(np.asarray(ns1[0]), np.asarray(ns2[0]), atol=1e-10)
            np.testing.assert_allclose(np.asarray(ns1[1]), np.asarray(ns2[1]), atol=1e-10)
        else:
            np.testing.assert_allclose(np.asarray(ns1), np.asarray(ns2), atol=1e-10)


def test_phileap_functoriality_wave_chain():
    """org^(2): ``Phileap(compose_chain) == Phileap(tensor).then(Phileap(chain_wire))``.

    The K=2 (leapfrog) analogue of ``test_phiphase_functoriality_wave_chain``: the two
    force evaluations of velocity Verlet are the two ``org^(2)`` rounds, and both passes
    agree at each round -- so leapfrog dynamics preserves the wave-chain composition.
    """
    _assert_functorial(Phileap, K=5, m=1.7, kappa=2.3, phase=True)


def test_phirk4_functoriality_heat_chain():
    """org^(4): ``Phirk4(compose_chain) == Phirk4(tensor).then(Phirk4(chain_wire))``.

    The K=4 (RK4) analogue of ``test_phiconf_functoriality_heat_chain``: the four RK4
    stage forces are the four ``org^(4)`` rounds -- each evaluated at a *different*
    intermediate position -- and both passes agree stage-by-stage, so RK4 dynamics
    preserves the composition even though the stages read at shifted positions.
    """
    _assert_functorial(lambda arr: Phirk4(arr, h=0.1), K=5, m=1.5, kappa=0.9, phase=False)


def _closed_well():
    """A closed ``<R^0|R^0> -> <R^0|R^0>`` quadratic well (Euclidean sharp)."""
    return SmoothArrangement(
        euclidean(1, 1.0), 0, 0, 0, 0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * q[0] ** 2,
        label="well",
    )


def test_then_rejects_mismatched_stage_counts():
    """``then`` across different stage counts is a clear error -- like ``parallel``.

    Sequential composition threads the two coalgebras round-for-round, so it is only
    defined between equal stage counts (``org^(K)`` with ``org^(K)``). A ``K=4`` (RK4)
    morphism composed with a ``K=2`` one is rejected, mirroring the existing guard on
    ``OrgMorphismK.parallel``.
    """
    K4 = Phirk4(_closed_well())  # K = 4
    two_stage = IntegratorK(
        init=lambda Q: jnp.zeros(Q.dim),
        reads=(lambda Q, s: s, lambda Q, s: s),
        advances=(lambda Q, s, xi: s, lambda Q, s, xi: s),
        label="noop2",
    )
    K2 = orgK_from_integrator(_closed_well(), two_stage)  # K = 2, same source/target polys
    assert K4.K == 4 and K2.K == 2
    with pytest.raises(ValueError, match="stage counts"):
        K4.then(K2)
