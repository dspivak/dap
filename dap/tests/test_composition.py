"""Functoriality of the dynamics functors (sec.spring_second_pass).

The wave/heat chain ``wire_K(Part,...,Part)`` is the genuine operad composite
``compose_seq(tensor(Parts), chain_wire(K))`` in ``sarr``. We check the paper's
two passes agree -- i.e. ``Phi`` preserves composition:

  first pass:   ``Phi(compose_seq(tensor(Parts), chain_wire))``  (compose in sarr, then Phi)
  second pass:  ``Phi(tensor(Parts)).then(Phi(chain_wire))``     (Phi each, then compose in pc)

This exercises ``OrgMorphism.then`` on *non-identity, stateful* coalgebras (the
parts under ``Phiphase`` carry the phase state ``T*R^K``), establishing the general
``pc`` composition that the functoriality audit asserts -- not merely identity.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.interpretation import trivial_omega
from dap.arrangement import SmoothArrangement
from dap.functors import Phiconf, Phiphase
from dap.rvect import diagonal
from dap.wiring import chain_wire, compose_chain, tensor_arrangements

_IN_POS = (jnp.zeros(0), trivial_omega(0))


def _harmonic_particle(m, kappa):
    return SmoothArrangement(
        diagonal(jnp.array([1.0 / m])), 0, 0, 1, 1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2,
    )


def _step(O, state, in_dir):
    return O.with_state(state).run_one(_IN_POS, lambda _o: in_dir)


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
