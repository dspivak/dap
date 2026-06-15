"""Leapfrog as a two-stage integrator org^(2) (leapfrog.py; cf. rmk.multistage).

Same harmonic chain as the wave test, run with the symplectic two-stage
integrator (velocity Verlet):
* eliminating momentum gives the centered discrete wave recurrence
      m (q_{n+1} - 2 q_n + q_{n-1}) = kappa * Lap(q_n);   [force at the center]
* the energy stays in a bounded band (symplectic).

The single-stage phase integrator is itself symplectic now (presented-position
readout), so it stays bounded too; leapfrog is a higher-order alternative.
"""

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phiphase
from dap.leapfrog import Phileap
from dap.rvect import diagonal
from dap.wiring import compose_chain


def _harmonic(m, kappa):
    return SmoothArrangement(
        diagonal(jnp.array([1.0 / m])), 0, 0, 1, 1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2,
        label="Part",
    )


_IN_POS = (jnp.zeros(0), (jnp.zeros((0, 0)), jnp.zeros(0)))


def _boundary(kappa):
    # pinned ends: q_prev = 0, xi_N = kappa * (emitted q_K).  Reads the emitted position,
    # so it is re-evaluated correctly at both leapfrog stages.
    return lambda out_pos: (kappa * out_pos[0], jnp.array([0.0]))


def _lap_pinned(v):
    aug = np.concatenate([[0.0], np.asarray(v), [0.0]])
    return aug[:-2] - 2.0 * aug[1:-1] + aug[2:]


def _energy(q, xi, m, kappa):
    ke = 0.5 * float(np.sum(np.asarray(xi) ** 2)) / m
    aug = np.concatenate([[0.0], np.asarray(q), [0.0]])
    pe = 0.5 * kappa * float(np.sum(np.diff(aug) ** 2))
    return ke + pe


def test_leapfrog_and_phase_are_both_bounded():
    """The presented-position readout makes the single-stage phase integrator
    symplectic, so it stays bounded like leapfrog (kappa/m = 0.6)."""
    K, m, kappa = 5, 1.5, 0.9
    arr = compose_chain([_harmonic(m, kappa)] * K)
    rng = np.random.default_rng(0)
    q0 = jnp.asarray(rng.standard_normal(K))
    peak0 = float(np.max(np.abs(np.asarray(q0))))
    bdy = _boundary(kappa)

    # leapfrog: many steps, bounded
    O = Phileap(arr)
    state = (q0, jnp.zeros(K))
    leap_peak = peak0
    for _ in range(300):
        _, _, state = O.with_state(state).run_one(_IN_POS, bdy)
        leap_peak = max(leap_peak, float(np.max(np.abs(np.asarray(state[0])))))

    # single-stage phase: now symplectic, also bounded
    Oe = Phiphase(arr)
    es = (q0, jnp.zeros(K))
    phase_peak = peak0
    for _ in range(300):
        _, _, es = Oe.with_state(es).run_one(_IN_POS, bdy)
        phase_peak = max(phase_peak, float(np.max(np.abs(np.asarray(es[0])))))

    assert leap_peak < 10.0 * peak0
    assert phase_peak < 10.0 * peak0


def test_leapfrog_centered_wave_recurrence():
    """m (q_{n+1} - 2 q_n + q_{n-1}) = kappa * Lap(q_n), force at the CENTER point."""
    K, m, kappa = 5, 1.5, 0.9
    O = Phileap(compose_chain([_harmonic(m, kappa)] * K))
    rng = np.random.default_rng(1)
    state = (jnp.asarray(rng.standard_normal(K)), jnp.asarray(rng.standard_normal(K)))
    bdy = _boundary(kappa)

    traj = [np.asarray(state[0])]
    for _ in range(12):
        _, _, state = O.with_state(state).run_one(_IN_POS, bdy)
        traj.append(np.asarray(state[0]))
    a = np.stack(traj)
    for t in range(len(a) - 2):
        centered = m * (a[t + 2] - 2.0 * a[t + 1] + a[t])
        np.testing.assert_allclose(centered, kappa * _lap_pinned(a[t + 1]), atol=1e-10)


def test_leapfrog_energy_is_bounded():
    """Symplectic: the energy stays in a bounded band (here well within a factor of 2)."""
    K, m, kappa = 5, 1.5, 0.9
    O = Phileap(compose_chain([_harmonic(m, kappa)] * K))
    rng = np.random.default_rng(2)
    state = (jnp.asarray(rng.standard_normal(K)), jnp.asarray(rng.standard_normal(K)))
    bdy = _boundary(kappa)

    H0 = _energy(state[0], state[1], m, kappa)
    for _ in range(150):
        _, _, state = O.with_state(state).run_one(_IN_POS, bdy)
        H = _energy(state[0], state[1], m, kappa)
        assert 0.5 * H0 < H < 2.0 * H0
