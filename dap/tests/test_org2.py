"""General org^(2) datatype and its composition (org2.py).

`Phileap` is now an instance of the general two-stage coalgebra `OrgMorphism2`.
These tests exercise the *general* structure (not leapfrog-specific): that the
two-round execution composes correctly under `parallel` and `then_static`.
"""

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phiphase
from dap.integrator import Integrator2
from dap.leapfrog import Phileap
from dap.org2 import OrgMorphism2, org2_from_integrator
from dap.polynomial import identity_poly_map
from dap.rvect import diagonal


def _oscillator(omega2):
    """A closed (I -> I) harmonic oscillator: Q = R, mass 1, U(q) = (omega2/2) q^2."""
    return SmoothArrangement(
        diagonal(jnp.array([1.0])), 0, 0, 0, 0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * omega2 * q[0] ** 2,
        label="osc",
    )


_IN_POS0 = (jnp.zeros(0), (jnp.zeros((0, 0)), jnp.zeros(0)))  # closed position
_IN_DIR0 = (jnp.zeros(0), jnp.zeros(0))                        # closed direction


def test_phileap_is_a_general_org2_morphism():
    """Leapfrog is now an instance of the general datatype, not a bespoke class."""
    assert isinstance(Phileap(_oscillator(1.0)), OrgMorphism2)


def test_parallel_runs_two_systems_independently():
    """`A.parallel(B)` evolves the combined state exactly as A and B run separately."""
    o1, o2 = _oscillator(0.5), _oscillator(1.3)
    A = Phileap(o1).parallel(Phileap(o2))
    B1, B2 = Phileap(o1), Phileap(o2)

    s1 = (jnp.array([1.0]), jnp.array([0.0]))
    s2 = (jnp.array([0.5]), jnp.array([0.2]))
    A_state = (s1, s2)

    for _ in range(20):
        _, _, A_state = A.with_state(A_state).run_one(
            (_IN_POS0, _IN_POS0), lambda op: (_IN_DIR0, _IN_DIR0)
        )
        _, _, s1 = B1.with_state(s1).run_one(_IN_POS0, lambda op: _IN_DIR0)
        _, _, s2 = B2.with_state(s2).run_one(_IN_POS0, lambda op: _IN_DIR0)

    for got, want in ((A_state[0], s1), (A_state[1], s2)):
        np.testing.assert_allclose(np.asarray(got[0]), np.asarray(want[0]), atol=1e-10)
        np.testing.assert_allclose(np.asarray(got[1]), np.asarray(want[1]), atol=1e-10)


def test_then_static_identity_is_a_noop():
    """Post-composing both rounds with the identity poly map changes nothing."""
    o = _oscillator(0.7)
    base = Phileap(o)
    comp = base.then_static(identity_poly_map(base.tgt_poly))

    sb = sc = (jnp.array([1.0]), jnp.array([0.3]))
    for _ in range(15):
        _, _, sb = base.with_state(sb).run_one(_IN_POS0, lambda op: _IN_DIR0)
        _, _, sc = comp.with_state(sc).run_one(_IN_POS0, lambda op: _IN_DIR0)
    np.testing.assert_allclose(np.asarray(sc[0]), np.asarray(sb[0]), atol=1e-10)
    np.testing.assert_allclose(np.asarray(sc[1]), np.asarray(sb[1]), atol=1e-10)


def test_org2_from_integrator_is_general():
    """The builder works for ANY two-stage integrator, not just leapfrog: a
    'two phase-Euler steps' Integrator2 equals running Phiphase twice."""

    def advance(Q, s, xi_Q1):                       # one phase-Euler step
        q, xi = s
        return (q + Q.apply_sharp(q, xi), xi - xi_Q1)

    def finish(Q, s, mid, xi_Q2):                   # another phase-Euler step
        q, xi = mid
        return (q + Q.apply_sharp(q, xi), xi - xi_Q2)

    two_phase = Integrator2(
        init=lambda Q: (jnp.zeros(Q.dim), jnp.zeros(Q.dim)),
        read1=lambda Q, s: s[0] + Q.apply_sharp(s[0], s[1]), advance=advance,
        read2=lambda Q, mid: mid[0] + Q.apply_sharp(mid[0], mid[1]), finish=finish, label="2phase",
    )

    o = _oscillator(0.6)
    A = org2_from_integrator(o, two_phase)
    B = Phiphase(o)

    s = (jnp.array([1.0]), jnp.array([0.2]))
    _, _, a_state = A.with_state(s).run_one(_IN_POS0, lambda op: _IN_DIR0)
    _, _, b1 = B.with_state(s).run_one(_IN_POS0, lambda op: _IN_DIR0)
    _, _, b2 = B.with_state(b1).run_one(_IN_POS0, lambda op: _IN_DIR0)

    np.testing.assert_allclose(np.asarray(a_state[0]), np.asarray(b2[0]), atol=1e-10)
    np.testing.assert_allclose(np.asarray(a_state[1]), np.asarray(b2[1]), atol=1e-10)
