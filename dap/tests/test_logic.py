"""Tests for ``logic`` -- CMOS from non-reciprocal MOSFETs (EXTENSION, not in the paper).

The claims under test:

1.  *Reciprocity is forced instantaneously.*  At a fixed parameter position the
    port currents are ``dU`` on the input interface, so the port Jacobian is
    ``d^2 U`` and is symmetric -- for every arrangement, transistor or not.
2.  *Non-reciprocity comes from the sharp.*  Once the state is adiabatically
    eliminated the same device has ``dI_G/de_D = 0`` exactly (the sharp's kernel)
    while ``dI_D/de_G`` is the transconductance.
3.  *Both survive composition.*  Sharps direct-sum and potentials add, and the
    devices have no output ports, so ``dU_tot/dq_j = dU_j/dq_j``; a device whose
    gate-source pair is pinned does not move when its drain swings.
4.  The composite is a NAND, and two of them cross-coupled are a bistable latch.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dap.functors import Phiconf, Phirk4
from dap.interpretation import trivial_omega
from dap.logic import (
    KERNEL_SHARP,
    chain_state,
    inverter,
    inverter_chain,
    inverter_state,
    latch_state,
    mosfet,
    nand_gate,
    nand_state,
    rail,
    run,
    sr_latch,
    stepper,
)
from dap.wiring import tensor_arrangements

_IN_POS = (jnp.zeros(0), trivial_omega(0))
HI, LO = 1.0, 0.0


def _settle_fet(fet, e, n=300):
    """Run one device's ``Phiconf`` to its fixed point at terminal voltages ``e``."""
    O, q = Phiconf(fet), jnp.zeros(2)
    for _ in range(n):
        _, fiber = O.with_state(q).step(q)
        _, at_pos = fiber(_IN_POS)
        _, q = at_pos((jnp.zeros(0), e))
    return q


def _fet_currents(fet, q, e):
    """The device's port covector ``omega_N(e)`` at parameter position ``q``."""
    _, fiber = Phiconf(fet).with_state(q).step(q)
    (_, omega_N), _ = fiber(_IN_POS)
    return omega_N(e)


# ---------------------------------------------------------------------------
# 1-2.  The device.
# ---------------------------------------------------------------------------


def test_sharp_is_degenerate_and_nonsymmetric():
    """def.rvect asks only for ``Q -> vect(Q^*, Q)``: no symmetry, no invertibility."""
    S = KERNEL_SHARP
    assert not np.allclose(S, S.T)
    assert np.linalg.matrix_rank(np.asarray(S)) == 1


def test_gate_state_tracks_vgs_independently_of_drain():
    """The sharp reads only the ``q1``-component, where ``vds`` does not appear."""
    fet = mosfet("n")
    qs = [_settle_fet(fet, jnp.array([0.8, vd, 0.0])) for vd in (0.2, 0.5, 0.9, 2.0)]
    for q in qs:
        np.testing.assert_allclose(q[0], 0.8, atol=1e-5)  # q0 -> vgs
        np.testing.assert_allclose(q[1], 0.0, atol=1e-12)  # q1 frozen
    for q in qs[1:]:
        np.testing.assert_allclose(q, qs[0], atol=1e-6)  # identical, not merely close


def test_device_currents_sum_to_zero():
    """``U`` sees only voltage differences, so ``dU/deG + dU/deD + dU/deS = 0``."""
    fet = mosfet("n")
    e = jnp.array([0.8, 0.5, 0.1])
    I = _fet_currents(fet, _settle_fet(fet, e), e)
    np.testing.assert_allclose(float(I.sum()), 0.0, atol=1e-9)


def test_instantaneous_port_jacobian_is_symmetric():
    """Claim 1: at fixed ``q`` the port Jacobian is ``d^2 U``, hence reciprocal."""
    fet = mosfet("n")
    e = jnp.array([0.8, 0.5, 0.0])
    q = _settle_fet(fet, e)
    J = jax.jacobian(lambda ee: _fet_currents(fet, q, ee))(e)
    np.testing.assert_allclose(J, J.T, atol=1e-6)


def test_steady_state_port_jacobian_is_not():
    """Claim 2: adiabatic elimination breaks it, and ``dI_G/de_D`` is exactly 0."""
    fet = mosfet("n")
    e = jnp.array([0.8, 0.5, 0.0])
    J = jax.jacobian(lambda ee: _fet_currents(fet, _settle_fet(fet, ee), ee))(e)
    assert abs(float(J[1, 0])) > 1.0            # transconductance dI_D/de_G
    np.testing.assert_allclose(J[0, 1], 0.0, atol=1e-9)   # dI_G/de_D, the kernel
    assert float(np.abs(J - J.T).max()) > 1.0


def test_rail_is_frozen():
    """An ideal source is a state with ``sharpR = 0``."""
    np.testing.assert_allclose(rail().Q.sharp_at(jnp.zeros(1)), jnp.zeros((1, 1)))


# ---------------------------------------------------------------------------
# 3-4.  The composites.
# ---------------------------------------------------------------------------


def test_nand_is_built_by_composition():
    """Seven boxes, no hand-written composite potential."""
    g = nand_gate()
    assert g.Q.dim == 4 * 2 + 3          # four devices (R^2), three reporters (R^1)
    assert (g.out_dim_M, g.in_dim_M) == (0, 0)
    assert (g.in_dim_N, g.out_dim_N) == (2, 1)


@pytest.mark.parametrize(
    "a,b,expected", [(LO, LO, HI), (LO, HI, HI), (HI, LO, HI), (HI, HI, LO)]
)
def test_nand_truth_table(a, b, expected):
    _, outs, currents = run(nand_gate(), nand_state(0.5), lambda t: (a, b), 600)
    np.testing.assert_allclose(float(outs[-1][0]), expected, atol=1e-3)
    assert float(jnp.abs(currents[-1]).max()) < 1e-4   # no DC input current


def test_input_current_is_purely_capacitive():
    """The residual reciprocal coupling is the gate capacitance ``kappa`` and nothing else."""
    g = nand_gate()
    st, _, _ = run(g, nand_state(1.0), lambda t: (LO, HI), 400)
    _, _, currents = run(g, st, lambda t: (HI, HI), 200)
    assert float(jnp.abs(currents[0][0])) > 0.1    # a transient on the transition
    assert float(jnp.abs(currents[-1][0])) < 1e-3  # gone once the gates settle


def test_drain_swing_does_not_move_a_pinned_gate_state():
    """Claim 3: N2's gate-source is pinned to (b, ground); its drain V_mid is not."""
    g, states = nand_gate(), []
    for load in (0.0, 0.5, 2.0):
        st, outs, _ = run(g, nand_state(0.5), lambda t: (HI, HI), 400, load=[load])
        states.append(st)
    assert abs(float(states[0][8]) - float(states[2][8])) > 0.3   # V_out really swung
    for st in states[1:]:
        np.testing.assert_allclose(st[6], states[0][6], atol=1e-9)  # N2 gate state
        np.testing.assert_allclose(st[0], states[0][0], atol=1e-9)  # P1 gate state
    np.testing.assert_allclose(jnp.abs(states[-1][1:8:2]).max(), 0.0, atol=1e-12)


def test_latch_holds_both_states():
    L = sr_latch()
    for q, qbar in ((0.9, 0.1), (0.1, 0.9)):
        _, outs, _ = run(L, latch_state(q, qbar), lambda t: (HI, HI), 1500)
        np.testing.assert_allclose(float(outs[-1][0]), round(q), atol=1e-3)
        np.testing.assert_allclose(float(outs[-1][1]), round(qbar), atol=1e-3)


def test_latch_set_reset_and_remember():
    L, st = sr_latch(), latch_state(0.5, 0.5)
    seen = []
    for sbar, rbar in [(LO, HI), (HI, HI), (HI, LO), (HI, HI), (LO, HI), (HI, HI)]:
        st, outs, _ = run(L, st, lambda t: (sbar, rbar), 500)
        seen.append(round(float(outs[-1][0])))
    assert seen == [1, 1, 0, 0, 1, 1]   # set, hold, reset, hold, set, hold


def test_latch_metastable_point_is_unstable():
    """The symmetric fixed point is exact; either sign of perturbation resolves it."""
    L = sr_latch()
    outs = {}
    for eps in (0.0, 1e-4, -1e-4):
        _, o, _ = run(L, latch_state(0.5 + eps, 0.5), lambda t: (HI, HI), 3000)
        outs[eps] = float(o[-1][0])
    np.testing.assert_allclose(outs[0.0], 0.5, atol=1e-3)
    np.testing.assert_allclose(outs[1e-4], 1.0, atol=1e-3)
    np.testing.assert_allclose(outs[-1e-4], 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 5.  What the tick budget actually scales with.
# ---------------------------------------------------------------------------


def _settle_ticks(arr, st, n_in, tol=0.01, maxt=4000):
    one, xi = stepper(arr), jnp.zeros(arr.out_dim_N)
    traj = []
    for _ in range(maxt):
        out, _, st = one(st, jnp.asarray(n_in, float), xi)
        traj.append(np.asarray(out))
    fin = traj[-1]
    return next(t for t in range(len(traj)) if np.abs(traj[t] - fin).max() < tol), fin


def test_ticks_scale_with_logic_depth_not_component_count():
    """Depth costs ticks; width does not.

    A chain of ``K`` inverters settles in time affine in ``K`` (a signal really
    does have to cross ``K`` stages), while ``K`` *independent* inverters settle
    in the same number of ticks no matter how many there are.  So a wide circuit
    is free and a deep one costs its depth -- the physical delay, not a
    numerical penalty.
    """
    depths = [1, 2, 4, 6]
    ticks = [_settle_ticks(inverter_chain(K), chain_state(K), [HI])[0] for K in depths]
    assert ticks == sorted(ticks)
    per_stage = [(ticks[i] - ticks[0]) / (depths[i] - depths[0]) for i in range(1, 4)]
    assert max(per_stage) - min(per_stage) < 3.0        # affine in depth
    assert ticks[-1] < 4 * ticks[0]                     # and cheaply so

    wide = [
        _settle_ticks(
            tensor_arrangements([inverter() for _ in range(K)]),
            jnp.concatenate([inverter_state(0.5) for _ in range(K)]),
            [HI] * K,
        )[0]
        for K in (1, 4, 16)
    ]
    assert len(set(wide)) == 1                          # width is free


def test_rk4_buys_1_4x_the_step_for_4x_the_work():
    """Explicit multistage does not pay for itself on a step-limited circuit.

    Fixing the circuit's *rate ratio* and sweeping a single timestep ``h``,
    ``Phirk4`` (``pc^(4)``, four gradients per macro-tick) is stable where Euler
    is not -- but only by the classical linear-stability factor ``2.785/2 = 1.39``,
    so its net throughput is about ``0.35``.  If the step is the binding
    constraint the answer is a better *sharp* (preconditioning, cf.
    ``rvect.inverse_hessian``) or an implicit stage loop, not more explicit stages.
    """
    KAPPA0, C0 = 0.5, 0.02
    triv = lambda _o: (jnp.zeros(1), jnp.array([HI]))

    def euler_ok(h, T=800):
        one = stepper(inverter(c=C0 * h, kappa=KAPPA0 * h))
        s = inverter_state(0.5)
        for _ in range(T):
            _, _, s = one(s, jnp.array([HI]), jnp.zeros(1))
            if not jnp.isfinite(s).all() or float(jnp.abs(s).max()) > 20:
                return False
        return abs(float(s[4])) < 0.02

    def rk4_ok(h, T=800):
        O = Phirk4(inverter(c=C0, kappa=KAPPA0), h=h)
        s = inverter_state(0.5)
        for _ in range(T):
            s = O.with_state(s).run_one(_IN_POS, triv)[-1]
            if not jnp.isfinite(s).all() or float(jnp.abs(s).max()) > 20:
                return False
        return abs(float(s[4])) < 0.02

    assert euler_ok(3.5) and not euler_ok(4.5)      # Euler's limit is near h = 4.0
    assert rk4_ok(5.0)                              # RK4 is stable past it ...
    assert not rk4_ok(6.0)                          # ... but only by ~1.39x,
    # so 4 gradients per tick buy well under 2x the step: net throughput < 1.


def test_frozen_modes_are_exactly_frozen():
    """Three unit eigenvalues of the one-tick map: two gate ``q1``'s and the rail."""
    one = stepper(inverter(c=0.02))
    upd = jax.jit(lambda s: one(s, jnp.array([HI]), jnp.zeros(1))[2])
    st = inverter_state(0.5)
    for _ in range(2000):
        st = upd(st)
    ev = np.sort(np.abs(np.linalg.eigvals(np.asarray(jax.jacobian(upd)(st)))))[::-1]
    np.testing.assert_allclose(ev[:3], 1.0, atol=1e-6)   # sharp kernel + sharp = 0
    assert ev[3] < 0.99                                  # everything else contracts


def test_explicit_euler_limit_cycles_when_the_node_step_is_too_big():
    """The integrator is the binding constraint: it degrades to a cycle, it does not diverge.

    The channel's ``tanh`` bounds the current, so an over-large node step
    ``c = dt/C`` does not blow up -- it settles onto a period-4 cycle at a wrong
    logic level.  A silent wrong answer, not a NaN.
    """
    one = stepper(nand_gate(c=0.10))
    st, n_in, xi = nand_state(0.5), jnp.array([HI, HI]), jnp.zeros(1)
    for _ in range(2000):
        out, _, st = one(st, n_in, xi)
    tail = []
    for _ in range(8):
        out, _, st = one(st, n_in, xi)
        tail.append(float(out[0]))
    assert max(tail) - min(tail) > 0.1                      # cycling, not settled
    np.testing.assert_allclose(tail[:4], tail[4:], atol=1e-4)  # with period 4
    assert min(tail) > 0.01                                 # and at the wrong level
