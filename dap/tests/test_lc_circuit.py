"""The LC circuit via Phiphase/Phiconf: the electrical reading acquires magnetism.

The arrangement (dap/lc_circuit.py) stores the capacitor charge ``q`` with
reactive sharp ``xi |-> xi/L`` (inductance in the mass slot) and potential
``U(q) = q^2/(2C)`` (capacitor energy). The claims under audit:

* Phiphase: the conjugate momentum is the flux linkage ``lambda = L dq/dt``,
  the kinetic term ``xi^2/(2L)`` is the inductor's magnetic energy, and the
  circuit oscillates at ``omega = 1/sqrt(LC)``.
* Phiconf: the same arrangement relaxes RC-style with no oscillation; the
  number in the sharp then functions as a resistance ``R := L`` (mobility
  ``1/R``), not an inductance.
* Coupled pair: two capacitor boxes wired through a stateless coupling box
  exhibit the two normal modes ``omega_+ = 1/sqrt(LC)`` and
  ``omega_- = sqrt(1/(LC) + 2/(L C_c))``.

Discrete relations, derived from the code (not from the continuum):

* ``phase_integrator`` (integrator.py) steps ``(q, xi) |-> (q + sharp(xi),
  xi - xi_Q)`` with ``xi_Q = dU`` at the presented position ``q~ = q + sharp(xi)``.
  For the LC cell: ``q_{n+1} = q_n + xi_n/L`` and ``xi_{n+1} = xi_n - q_{n+1}/C``
  -- symplectic (semi-implicit) Euler at tick ``dt = 1``. Hence EXACTLY
  ``xi_n = L (q_{n+1} - q_n)/dt``, the flux linkage (test 2), and eliminating
  ``xi``: ``L (q_{n+1} - 2 q_n + q_{n-1}) = -q_n/C``, a linear two-step
  recurrence whose rotation angle per tick is ``theta = 2 arcsin(omega/2)``,
  ``omega := 1/sqrt(LC)``. So the discrete period is ``2 pi / theta`` ticks:
  shorter than the continuum ``2 pi sqrt(LC)`` by the relative frequency bias
  ``omega^2/24 + O(omega^4)`` -- an O(dt^2) error (the map's trace ``2 - omega^2``
  coincides with velocity Verlet's, so leapfrog has the SAME period; it differs
  only in the energy band, test 3).
* ``leapfrog_integrator`` (leapfrog.py) stores integer-tick ``(q, xi)`` with
  half-kicks; eliminating the half-step momenta gives the exact centered
  relation ``xi_n = L (q_{n+1} - q_{n-1})/2`` (test 2).
* ``configuration_integrator`` steps ``q |-> q - sharp(dU(q)) = (1 - 1/(LC)) q``:
  exact geometric decay, forward Euler for ``dq/dt = -q/(LC)`` (test 4).

There is no ``dt`` in the framework (the tick is unit time), so "small dt"
means ``omega = 1/sqrt(LC)`` small in tick units; physical ``(L, C)`` at step
``dt`` correspond to ``(L/dt, C/dt)`` here, and then ``xi`` is the physical
flux linkage.
"""

import jax.numpy as jnp
import numpy as np

from dap.functors import Phiconf, Phiphase
from dap.interpretation import trivial_omega
from dap.leapfrog import Phileap
from dap.lc_circuit import coupled_lc_pair, coupling_capacitor, lc_cell

_IN_POS = (jnp.zeros(0), trivial_omega(0))

# Unloaded output port: the environment exerts the zero covector on the cell's
# single terminal (and there are no input ports). The pair is fully closed.
_DIR_CELL = lambda _out_pos: (jnp.zeros(1), jnp.zeros(0))
_DIR_PAIR = lambda _out_pos: (jnp.zeros(0), jnp.zeros(0))


def _run(O, q0, xi0, steps, dir_from):
    """Iterate a 1-stage pc morphism from phase state ``(q0, xi0)``; return q, xi paths."""
    state = (jnp.asarray(q0, float), jnp.asarray(xi0, float))
    qs, xis = [state[0]], [state[1]]
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one(_IN_POS, dir_from)
        qs.append(state[0])
        xis.append(state[1])
    return np.stack([np.asarray(x) for x in qs]), np.stack([np.asarray(x) for x in xis])


def _run2(O2, q0, xi0, steps, dir_from):
    """Same, for a 2-stage (leapfrog) pc^(2) morphism."""
    state = (jnp.asarray(q0, float), jnp.asarray(xi0, float))
    qs, xis = [state[0]], [state[1]]
    for _ in range(steps):
        _, _, state = O2.with_state(state).run_one(_IN_POS, dir_from)
        qs.append(state[0])
        xis.append(state[1])
    return np.stack([np.asarray(x) for x in qs]), np.stack([np.asarray(x) for x in xis])


def _measured_period(q):
    """Average tick-count between upward zero crossings (linear interpolation).

    The trajectory is an exact discrete sinusoid (linear symplectic map), so
    interpolation error in each crossing is O(theta^2) ticks and averaging over
    several periods leaves a relative error ~1e-7 at omega = 0.05 (measured).
    """
    ts = [n + q[n] / (q[n] - q[n + 1]) for n in range(len(q) - 1) if q[n] < 0.0 <= q[n + 1]]
    assert len(ts) >= 3, "trajectory too short to measure a period"
    return (ts[-1] - ts[0]) / (len(ts) - 1)


# ---------------------------------------------------------------------------
# 1. Frequency: omega = 1/sqrt(LC), three (L, C) pairs, LC spanning 400..10^4.
# ---------------------------------------------------------------------------


def test_lc_frequency_is_one_over_sqrt_LC():
    """Measured period = 2 pi sqrt(LC) within the integrator's known O(dt^2) bias.

    Tolerance justification: the discrete rotation angle is
    ``theta = 2 arcsin(omega/2)``, so the relative period deviation is
    ``omega/theta - 1 = -omega^2/24 + O(omega^4)``. We assert (a) agreement
    with the continuum within ``omega^2/12`` (twice the known bias -- NOT a
    loose tolerance: (c) asserts the bias is actually resolved), (b) agreement
    with the exact discrete prediction ``2 pi/theta`` to 1e-6 (measurement
    error only), (c) the deviation from the continuum has the predicted sign
    and at least half the predicted size, so (a)'s margin is doing no work.
    """
    for L, C in [(20.0, 20.0), (80.0, 20.0), (100.0, 100.0)]:
        w = 1.0 / np.sqrt(L * C)                     # omega * dt, dt = 1 tick
        T_cont = 2.0 * np.pi / w                     # 2 pi sqrt(LC)
        T_disc = 2.0 * np.pi / (2.0 * np.arcsin(w / 2.0))
        steps = int(5.2 * T_cont)
        qs, _ = _run(Phiphase(lc_cell(L, C)), [1.0], [0.0], steps, _DIR_CELL)
        T_meas = _measured_period(qs[:, 0])

        assert abs(T_meas - T_cont) / T_cont <= w * w / 12.0            # (a)
        assert abs(T_meas - T_disc) / T_disc <= 1e-6                    # (b)
        assert T_cont - T_meas >= (w * w / 48.0) * T_cont               # (c)


def test_leapfrog_has_the_same_period():
    """Phileap measures the SAME discrete period (equal trace 2 - omega^2), so
    using Phiphase above loses no frequency accuracy -- leapfrog only tightens
    the energy band (test 3)."""
    L, C = 20.0, 20.0
    w = 1.0 / np.sqrt(L * C)
    T_disc = 2.0 * np.pi / (2.0 * np.arcsin(w / 2.0))
    qs, _ = _run2(Phileap(lc_cell(L, C)), [1.0], [0.0], int(5.2 * 2 * np.pi / w), _DIR_CELL)
    assert abs(_measured_period(qs[:, 0]) - T_disc) / T_disc <= 1e-6


# ---------------------------------------------------------------------------
# 2. Keystone: the stored momentum IS the flux linkage lambda = L dq/dt.
# ---------------------------------------------------------------------------


def test_momentum_is_flux_linkage():
    """Phiphase: ``xi_n = L (q_{n+1} - q_n)/dt`` EXACTLY (dt = 1 tick).

    Read off ``phase_integrator``: ``q_{n+1} = q_n + sharp(xi_n) = q_n + xi_n/L``
    before any force acts, so the identity is exact in floating point (~1e-14),
    not an approximation: the momentum slot stores inductance times current,
    i.e. the flux linkage conjugate to charge.
    """
    L, C = 20.0, 20.0
    qs, xis = _run(Phiphase(lc_cell(L, C)), [1.0], [0.0], 300, _DIR_CELL)
    np.testing.assert_allclose(xis[:-1, 0], L * (qs[1:, 0] - qs[:-1, 0]), atol=1e-12)


def test_leapfrog_momentum_is_centered_flux_linkage():
    """Phileap: the integer-tick momentum is the CENTERED difference,
    ``xi_n = L (q_{n+1} - q_{n-1})/2``, exactly.

    Derivation from leapfrog.py: ``xi_{n+1/2} = xi_n - (1/2) xi_Q(q_n)`` and
    ``xi_{n+1} = xi_{n+1/2} - (1/2) xi_Q(q_{n+1}) = xi_{n+3/2} + ... `` give
    ``xi_{n+1} = (xi_{n+1/2} + xi_{n+3/2})/2`` with ``q_{n+1} - q_n = xi_{n+1/2}/L``.
    """
    L, C = 20.0, 20.0
    qs, xis = _run2(Phileap(lc_cell(L, C)), [1.0], [0.0], 300, _DIR_CELL)
    np.testing.assert_allclose(
        xis[1:-1, 0], L * (qs[2:, 0] - qs[:-2, 0]) / 2.0, atol=1e-12
    )


# ---------------------------------------------------------------------------
# 3. Energy exchange: q^2/(2C) + xi^2/(2L) bounded over many periods.
# ---------------------------------------------------------------------------


def _energy(qs, xis, L, C):
    return qs[:, 0] ** 2 / (2.0 * C) + xis[:, 0] ** 2 / (2.0 * L)


def test_energy_exchange_is_bounded_phase():
    """Phiphase over 12 periods: electric + magnetic energy oscillates in a band
    of relative half-width ~ omega/2 (symplectic Euler's O(dt) energy wobble;
    it preserves a nearby quadratic invariant, so no secular drift).

    Measured: max relative deviation = 0.513 * omega (at omega = 0.05); we
    assert the band is [0.45, 0.55] * omega -- two-sided, so the bound is the
    measured O(dt) wobble, not slack -- and that the last quarter of the run
    stays in the same band (bounded, not drifting).
    """
    L, C = 20.0, 20.0
    w = 1.0 / np.sqrt(L * C)
    steps = int(12.0 * 2.0 * np.pi / w)
    qs, xis = _run(Phiphase(lc_cell(L, C)), [1.0], [0.0], steps, _DIR_CELL)
    H = _energy(qs, xis, L, C)
    dev = np.abs(H / H[0] - 1.0)
    assert dev.max() <= 0.55 * w                       # bounded by the O(dt) wobble
    assert dev.max() >= 0.45 * w                       # ... and the bound is tight
    assert dev[3 * len(dev) // 4:].max() <= 0.55 * w   # no secular drift


def test_energy_exchange_is_bounded_leapfrog():
    """Phileap over 10 periods: the band tightens to O(dt^2).

    Measured: max relative deviation = 0.25 * omega^2 (vs 0.513 * omega for
    Phiphase -- a factor ~40 at omega = 0.05). Asserted two-sided at
    [0.2, 0.3] * omega^2; this is the 'report the measured bound' number.
    """
    L, C = 20.0, 20.0
    w = 1.0 / np.sqrt(L * C)
    steps = int(10.0 * 2.0 * np.pi / w)
    qs, xis = _run2(Phileap(lc_cell(L, C)), [1.0], [0.0], steps, _DIR_CELL)
    H = _energy(qs, xis, L, C)
    dev = np.abs(H / H[0] - 1.0)
    assert dev.max() <= 0.30 * w * w
    assert dev.max() >= 0.20 * w * w


# ---------------------------------------------------------------------------
# 4. Phiconf on the SAME arrangement: RC relaxation, no oscillation.
# ---------------------------------------------------------------------------


def test_phiconf_is_rc_relaxation():
    """Phiconf: ``q_{n+1} = (1 - 1/(LC)) q_n`` exactly; monotone decay to 0.

    From ``configuration_integrator``: ``q -> q - sharp(dU(q)) = q - q/(LC)``.
    This is forward Euler at dt = 1 for ``dq/dt = -q/(LC)``. Electrical
    reading, honestly stated: RC discharge ``dq/dt = -q/(RC)`` with ``R := L``
    -- under the first-order semantics the sharp is a mobility/conductance
    ``1/R`` (Ohm's law ``I = -V/R``, ``V = dU/dq = q/C``), NOT an inductance;
    the same stored number changes physical role (ohms vs henries) with the
    integrator. Asserted: (a) the exact geometric law at machine precision,
    (b) strictly monotone decay, no sign change (no oscillation; requires
    ``LC > 1``, i.e. dt inside the forward-Euler stability region),
    (c) the continuum exponential within the forward-Euler O(dt) bound
    ``max_n |(1-x)^n - e^{-nx}| ~ x/(2e)`` with ``x = 1/(LC)``.
    """
    L, C = 4.0, 25.0                                   # x = 1/(LC) = 0.01
    x = 1.0 / (L * C)
    O = Phiconf(lc_cell(L, C))
    state = jnp.array([1.0])
    qs = [1.0]
    for _ in range(400):
        _, _, state = O.with_state(state).run_one(_IN_POS, _DIR_CELL)
        qs.append(float(state[0]))
    qs = np.asarray(qs)
    n = np.arange(len(qs))

    np.testing.assert_allclose(qs, (1.0 - x) ** n, atol=1e-12)          # (a)
    assert np.all(np.diff(qs) < 0.0) and np.all(qs > 0.0)               # (b)
    assert qs[-1] < 0.02                                                # decayed to ~0
    assert np.max(np.abs(qs - np.exp(-x * n))) <= 0.2 * x               # (c)


# ---------------------------------------------------------------------------
# 5. Coupled LC pair: two capacitor boxes + stateless coupling box.
# ---------------------------------------------------------------------------


def test_pair_construction_is_the_advertised_composite():
    """The wired pair's data emerge from genuine sarr composition:
    parameter R^2, sharp diag(1/L, 1/L), and the composite potential
    ``q1^2/(2C) + q2^2/(2C) + (q1 - q2)^2/(2 C_c)`` -- not written by hand."""
    L, C, Cc = 20.0, 20.0, 5.0
    pair = coupled_lc_pair(L, C, Cc)
    assert pair.Q.dim == 2
    assert (pair.out_dim_M, pair.in_dim_M, pair.out_dim_N, pair.in_dim_N) == (0, 0, 0, 0)
    assert coupling_capacitor(Cc).Q.dim == 0           # the coupling box is stateless
    np.testing.assert_allclose(
        np.asarray(pair.Q.sharp_at(jnp.zeros(2))), np.eye(2) / L, atol=1e-14
    )
    rng = np.random.default_rng(7)
    for _ in range(5):
        q = rng.standard_normal(2)
        U = float(pair.U(jnp.asarray(q), jnp.zeros(0), jnp.zeros(0)))
        U_ref = q[0] ** 2 / (2 * C) + q[1] ** 2 / (2 * C) + (q[0] - q[1]) ** 2 / (2 * Cc)
        np.testing.assert_allclose(U, U_ref, atol=1e-12)


def test_coupled_pair_normal_modes():
    """Both predicted normal-mode frequencies appear:

        omega_+ = 1/sqrt(LC)                  (in phase: coupling capacitor idle),
        omega_- = sqrt(1/(LC) + 2/(L C_c))    (antiphase).

    With (L, C, C_c) = (20, 20, 5): omega_+ = 0.05, omega_- = 0.15 exactly.
    The discrete map is linear and preserves the symmetric/antisymmetric
    subspaces, so starting in an eigenmode we measure that mode's period
    (tolerances as in test 1: known bias omega^2/24, factor-2 margin, plus the
    sharp discrete check) and assert the trajectory stays in the eigenspace to
    machine precision -- both modes are exhibited, cleanly separated.
    """
    L, C, Cc = 20.0, 20.0, 5.0
    pair = coupled_lc_pair(L, C, Cc)
    w_plus = 1.0 / np.sqrt(L * C)
    w_minus = np.sqrt(1.0 / (L * C) + 2.0 / (L * Cc))
    np.testing.assert_allclose([w_plus, w_minus], [0.05, 0.15], atol=1e-15)

    for q0, w, purity in [
        ([1.0, 1.0], w_plus, lambda q: q[:, 0] - q[:, 1]),
        ([1.0, -1.0], w_minus, lambda q: q[:, 0] + q[:, 1]),
    ]:
        T_cont = 2.0 * np.pi / w
        T_disc = 2.0 * np.pi / (2.0 * np.arcsin(w / 2.0))
        qs, _ = _run(Phiphase(pair), q0, [0.0, 0.0], int(5.2 * T_cont), _DIR_PAIR)
        T_meas = _measured_period(qs[:, 0])
        assert abs(T_meas - T_cont) / T_cont <= w * w / 12.0
        assert abs(T_meas - T_disc) / T_disc <= 1e-5
        assert np.max(np.abs(purity(qs))) <= 1e-12     # stays in the eigenmode


# ---------------------------------------------------------------------------
# 6. The two-box Phiconf encoding (ex.further_reach, "LC circuit" item).
# ---------------------------------------------------------------------------

from dap.lc_circuit import capacitor_cell, current_voltage_coupler, inductor_cell, lc_conf

# The composite has one outer output port (the terminal voltage), unloaded.
_DIR_CONF = lambda _out_pos: (jnp.zeros(1), jnp.zeros(0))


def test_lc_conf_is_the_advertised_composite():
    """The paper item's data emerge from genuine sarr composition: parameter
    R^2 = (i, v); sharp diag(-1/L, +1/C) -- OPPOSITE signs; composite
    potential i*v (not written by hand); outer output reporting v."""
    L, C = 5.0, 20.0
    arr = lc_conf(L, C)
    assert arr.Q.dim == 2
    assert (arr.out_dim_N, arr.in_dim_N) == (1, 0)
    assert current_voltage_coupler().Q.dim == 0        # the coupler is stateless
    np.testing.assert_allclose(
        np.asarray(arr.Q.sharp_at(jnp.zeros(2))),
        np.diag([-1.0 / L, 1.0 / C]),
        atol=1e-14,
    )
    rng = np.random.default_rng(11)
    for _ in range(5):
        q = rng.standard_normal(2)
        U = float(arr.U(jnp.asarray(q), jnp.zeros(0), jnp.zeros(0)))
        np.testing.assert_allclose(U, q[0] * q[1], atol=1e-12)
        v = np.asarray(arr.out_f(jnp.asarray(q), jnp.zeros(0)))
        np.testing.assert_allclose(v, q[1:2], atol=1e-14)


def test_lc_conf_one_tick_is_forward_euler():
    """One Phiconf tick is EXACTLY  i -> i + v/L,  v -> v - i/C  (the
    forward-Euler step of the inductor law di/dt = v/L and the capacitor law
    dv/dt = -i/C), for random states and several (L, C)."""
    rng = np.random.default_rng(3)
    for L, C in [(5.0, 20.0), (2.0, 2.0), (30.0, 7.0)]:
        O = Phiconf(lc_conf(L, C))
        for _ in range(4):
            iv = rng.standard_normal(2)
            state = jnp.asarray(iv)
            _, _, state = O.with_state(state).run_one(_IN_POS, _DIR_CONF)
            expect = np.array([iv[0] + iv[1] / L, iv[1] - iv[0] / C])
            np.testing.assert_allclose(np.asarray(state), expect, atol=1e-13)


def test_lc_conf_energy_law_and_period():
    """The lossless forward-Euler signature, exactly and in aggregate:

    (a) the energy H = (L i^2 + C v^2)/2 grows by EXACTLY 1 + 1/(LC) per tick
        (in scaled coordinates the map is sqrt(1 + omega^2) times a rotation,
        omega = 1/sqrt(LC); cf. dap/dc_motor.py's lossless law);
    (b) the rotation angle per tick is arctan(omega), so the measured period
        is 2 pi / arctan(omega) ticks, within the continuum 2 pi sqrt(LC) up
        to the forward-Euler period bias omega^2/3 (arctan(w) = w - w^3/3 + ...).
    """
    L, C = 20.0, 20.0                                  # omega = 0.05
    w = 1.0 / np.sqrt(L * C)
    O = Phiconf(lc_conf(L, C))
    state = jnp.array([0.0, 1.0])
    ivs = [np.asarray(state)]
    for _ in range(400):
        _, _, state = O.with_state(state).run_one(_IN_POS, _DIR_CONF)
        ivs.append(np.asarray(state))
    ivs = np.stack(ivs)
    H = (L * ivs[:, 0] ** 2 + C * ivs[:, 1] ** 2) / 2.0

    np.testing.assert_allclose(H[1:] / H[:-1], 1.0 + w**2, atol=1e-12)      # (a)

    T = _measured_period(ivs[:, 1])
    T_disc = 2.0 * np.pi / np.arctan(w)
    T_cont = 2.0 * np.pi * np.sqrt(L * C)
    np.testing.assert_allclose(T, T_disc, rtol=1e-4)                        # (b)
    assert abs(T - T_cont) / T_cont < 0.5 * w**2                            # bias ~ w^2/3


def test_lc_conf_picks_up_a_radio_signal():
    """The circuit tunes: a sinusoidal covector fed at the terminal is an
    antenna, and the steady-state response peaks at omega = 1/sqrt(LC).

    The drive enters as ``xi_N(t) = sin(Omega t)`` at the single outer port;
    its pullback along the report ``(i, v) |-> v`` lands in the ``v``-slot, so
    the tick becomes ``v -> v - (i + xi_N)/C``: an injected current. A small
    series resistance ``R`` (potential ``-(R/2) i^2`` in the negative-sharp
    inductor cell) supplies the damping that makes the peak finite and the
    transient die; it also swamps the forward-Euler energy growth
    (``R/(2L) = 0.005 > omega^2/2 = 0.00125``), so the undriven map is stable.
    Asserted: over a sweep of drive frequencies the largest steady-state
    amplitude of ``v`` occurs exactly at ``Omega = omega``, and beats the
    +-25% detunings by more than 2x (measured ~3x, Q = omega L / R = 5).
    """
    L, C, R = 20.0, 20.0, 0.2
    w = 1.0 / np.sqrt(L * C)
    O = Phiconf(lc_conf(L, C, R))
    freqs = [0.6 * w, 0.8 * w, w, 1.25 * w, 1.6 * w]
    amps = []
    for W in freqs:
        state = jnp.zeros(2)
        vs = []
        for t in range(2000):
            drive = lambda _out_pos, t=t: (jnp.array([np.sin(W * t)]), jnp.zeros(0))
            _, _, state = O.with_state(state).run_one(_IN_POS, drive)
            vs.append(float(state[1]))
        amps.append(float(np.max(np.abs(np.asarray(vs[1000:])))))
    assert int(np.argmax(amps)) == freqs.index(w)
    assert amps[freqs.index(w)] > 2.0 * max(amps[1], amps[3])
