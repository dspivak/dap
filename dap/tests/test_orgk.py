"""General ``org^(K)`` datatype + RK4 as ``org^(4)`` (orgK.py, rk4.py; rmk.multistage).

``OrgMorphismK`` is the K-fold substitution ``[p,q]^{∘K}``, the general case of the
two-stage ``OrgMorphism2``. These tests exercise:

* the *general* structure -- ``parallel`` and ``then_static`` compose correctly over
  the K rounds (here K = 4, RK4 instances);
* **subsumption** -- ``K = 1`` reproduces the single-stage ``Phiconf`` (org), and
  ``K = 2`` reproduces leapfrog (``org^(2)``), so the general machinery contains both;
* **RK4 is really RK4** -- one macro-tick equals a hand-written RK4 step, and the
  global error of the staged ``Phirk4`` falls like ``h^4`` (halving ``h`` cuts the
  error by ~16x). The ``h^4`` rate is the falsifiable proof that the four ``org^(4)``
  rounds carry the right intermediate positions and Butcher weights.
"""

import jax
import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phiconf, Phirk4
from dap.integrator import IntegratorK, configuration_integrator
from dap.interpretation import trivial_omega
from dap.leapfrog import Phileap
from dap.orgK import OrgMorphismK, orgK_from_integrator
from dap.polynomial import identity_poly_map
from dap.rk4 import rk4_integrator
from dap.rvect import diagonal, euclidean

_IN_POS0 = (jnp.zeros(0), trivial_omega(0))  # closed position
_IN_DIR0 = (jnp.zeros(0), jnp.zeros(0))       # closed direction
_TRIV = lambda op: _IN_DIR0


def _well(A):
    """A closed (I -> I) quadratic well ``U(q) = 1/2 q.A.q`` with Euclidean sharp.

    The configuration/RK4 flow is ``qdot = -A q`` (sharpR = I), whose exact solution
    ``q(t) = e^{-At} q0`` lets us measure the integrator's order against a closed form.
    """
    A = jnp.asarray(A)
    return SmoothArrangement(
        euclidean(A.shape[0], 1.0), 0, 0, 0, 0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * q @ (A @ q),
        label="well",
    )


def _oscillator(omega2):
    """A closed harmonic oscillator: Q = R, mass 1, U(q) = (omega2/2) q^2 (for leapfrog)."""
    return SmoothArrangement(
        diagonal(jnp.array([1.0])), 0, 0, 0, 0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * omega2 * q[0] ** 2,
        label="osc",
    )


# ---------------------------------------------------------------------------
# Datatype basics.
# ---------------------------------------------------------------------------


def test_phirk4_is_a_general_orgK_morphism_with_K_4():
    """RK4 is an instance of the general datatype, with four interaction rounds."""
    O = Phirk4(_well(jnp.eye(2)))
    assert isinstance(O, OrgMorphismK)
    assert O.K == 4


def test_run_one_returns_K_out_positions():
    """A closed macro-tick emits one out-position per round (K = 4)."""
    O = Phirk4(_well(jnp.eye(2)), h=0.1)
    out_poss, _new_state = O.run_one(_IN_POS0, _TRIV)
    assert len(out_poss) == 4


# ---------------------------------------------------------------------------
# Subsumption: K = 1 is org, K = 2 is leapfrog.
# ---------------------------------------------------------------------------


def _as_K1(intg):
    """Re-express a single-stage ``Integrator`` as a one-round ``IntegratorK``."""
    return IntegratorK(
        init=intg.init,
        reads=(intg.position,),
        advances=(lambda Q, s, xi_Q: intg.step(Q, s, xi_Q),),
        label=f"{intg.label}-K1",
    )


def test_K1_reproduces_phiconf():
    """The K = 1 case of the general machinery IS the single-stage org (Phiconf)."""
    arr = _well(jnp.diag(jnp.array([1.0, 2.0, 4.0])))
    A = orgK_from_integrator(arr, _as_K1(configuration_integrator()))
    B = Phiconf(arr)

    sa = sb = jnp.array([1.0, -0.5, 0.3])
    for _ in range(25):
        _outs, sa = A.with_state(sa).run_one(_IN_POS0, _TRIV)
        _op, _od, sb = B.with_state(sb).run_one(_IN_POS0, _TRIV)
    np.testing.assert_allclose(np.asarray(sa), np.asarray(sb), atol=1e-12)


def _leapfrog_K():
    """Leapfrog (velocity Verlet) re-expressed as a two-round ``IntegratorK``."""

    def advance1(Q, s, xi_Q1):
        q, xi = s
        xi_half = xi - 0.5 * xi_Q1
        return (q + Q.apply_sharp(q, xi_half), xi_half)

    def finish(Q, mid, xi_Q2):
        q2, xi_half = mid
        return (q2, xi_half - 0.5 * xi_Q2)

    return IntegratorK(
        init=lambda Q: (jnp.zeros(Q.dim), jnp.zeros(Q.dim)),
        reads=(lambda Q, s: s[0], lambda Q, mid: mid[0]),
        advances=(advance1, finish),
        label="leapfrog-K2",
    )


def test_K2_reproduces_leapfrog():
    """The K = 2 case reproduces leapfrog exactly -- the same step as org^(2) Phileap."""
    osc = _oscillator(0.8)
    A = orgK_from_integrator(osc, _leapfrog_K())
    B = Phileap(osc)

    sa = sb = (jnp.array([1.0]), jnp.array([0.3]))
    for _ in range(30):
        _outs, sa = A.with_state(sa).run_one(_IN_POS0, _TRIV)
        _op1, _op2, sb = B.with_state(sb).run_one(_IN_POS0, _TRIV)
    np.testing.assert_allclose(np.asarray(sa[0]), np.asarray(sb[0]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(sa[1]), np.asarray(sb[1]), atol=1e-12)


# ---------------------------------------------------------------------------
# Composition (general structure, exercised on K = 4).
# ---------------------------------------------------------------------------


def test_parallel_runs_two_systems_independently():
    """`A.parallel(B)` evolves the combined state exactly as A and B run separately."""
    o1, o2 = _well(jnp.diag(jnp.array([1.0, 3.0]))), _well(jnp.diag(jnp.array([2.0])))
    A = Phirk4(o1, 0.1).parallel(Phirk4(o2, 0.1))
    B1, B2 = Phirk4(o1, 0.1), Phirk4(o2, 0.1)

    s1 = jnp.array([1.0, -0.5])
    s2 = jnp.array([0.7])
    A_state = (s1, s2)
    for _ in range(15):
        _outs, A_state = A.with_state(A_state).run_one(
            (_IN_POS0, _IN_POS0), lambda op: (_IN_DIR0, _IN_DIR0)
        )
        _o1, s1 = B1.with_state(s1).run_one(_IN_POS0, _TRIV)
        _o2, s2 = B2.with_state(s2).run_one(_IN_POS0, _TRIV)

    np.testing.assert_allclose(np.asarray(A_state[0]), np.asarray(s1), atol=1e-12)
    np.testing.assert_allclose(np.asarray(A_state[1]), np.asarray(s2), atol=1e-12)


def test_then_static_identity_is_a_noop():
    """Post-composing every round with the identity poly map changes nothing."""
    o = _well(jnp.diag(jnp.array([1.0, 2.0])))
    base = Phirk4(o, 0.1)
    comp = base.then_static(identity_poly_map(base.tgt_poly))

    sb = sc = jnp.array([1.0, -0.4])
    for _ in range(15):
        _ob, sb = base.with_state(sb).run_one(_IN_POS0, _TRIV)
        _oc, sc = comp.with_state(sc).run_one(_IN_POS0, _TRIV)
    np.testing.assert_allclose(np.asarray(sc), np.asarray(sb), atol=1e-12)


# ---------------------------------------------------------------------------
# RK4 really is RK4.
# ---------------------------------------------------------------------------


def test_one_step_matches_hand_written_rk4():
    """One ``Phirk4`` macro-tick equals a hand-rolled RK4 step of qdot = -A q."""
    A = np.array([[1.0, 0.2], [0.2, 3.0]])
    h = 0.1
    O = Phirk4(_well(jnp.asarray(A)), h)
    x0 = jnp.array([0.6, -0.9])
    _outs, x_new = O.with_state(x0).run_one(_IN_POS0, _TRIV)

    f = lambda x: -A @ x
    x = np.asarray(x0)
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    x_ref = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    np.testing.assert_allclose(np.asarray(x_new), x_ref, atol=1e-12)


def test_rk4_is_fourth_order():
    """Global error of the staged Phirk4 falls like h^4: halving h cuts it by ~16x.

    Integrated against the closed form e^{-AT} q0 over a fixed horizon T; the ratio of
    successive errors approaches 16 (and stays well above 8 = h^3), the signature of a
    genuine 4th-order method running through the four org^(4) rounds.
    """
    A = jnp.diag(jnp.array([1.0, 2.0, 5.0]))
    arr = _well(A)
    x0 = jnp.array([1.0, 1.0, 1.0])
    T = 1.0
    exact = np.asarray(jax.scipy.linalg.expm(-np.asarray(A) * T) @ np.asarray(x0))

    def err(N):
        h = T / N
        O = Phirk4(arr, h)
        s = x0
        for _ in range(N):
            _outs, s = O.with_state(s).run_one(_IN_POS0, _TRIV)
        return float(np.linalg.norm(np.asarray(s) - exact))

    e8, e16, e32 = err(8), err(16), err(32)
    assert e32 < 1e-6                  # the fine solution is accurate
    assert e8 / e16 > 12.0             # h^4 (=16), cleanly above h^3 (=8)
    assert e16 / e32 > 12.0
