"""Molecular dynamics IS leapfrog phase dynamics of a wired arrangement.

Audit of the claim (moleculardyn.py): velocity-Verlet MD with pairwise
Lennard-Jones forces is ``Phileap`` applied to the closed composite

    compose_seq( tensor(Part_1..Part_N, LJ_1..LJ_P), graph_wire(N+P, edges) )

of particle boxes (state = position, constant mass-like sharp) and stateless
LJ pair boxes, with the physical timestep ``dt`` carried by the sharp
``(dt^2/m) I`` and momentum coordinate ``xi = (m/dt) v`` (see the module
docstring of moleculardyn.py for the identification).

Checks:

1. the composite potential ``sum_p U_LJ`` EMERGES from the wiring;
2. the force -- one covector evaluation ``-xi_Q`` through the smooth
   interpretation -- vanishes at exactly ``2^(1/6) sigma``, is repulsive
   inside and attractive outside; two particles released near the minimum
   oscillate around it;
3. ONE ``Phileap`` macro-tick equals ONE hand-computed velocity-Verlet step
   (independent analytic LJ force) to numerical precision;
4. over 10^4 steps of a 2-D LJ trimer, the total energy (kinetic via the
   sharp + emergent potential) under ``Phileap`` stays in a bounded
   oscillating band with no secular growth, and that band is more than an
   order of magnitude narrower than under the single-stage ``Phiphase``.

Timestep choice: with ``eps = sigma = m = 1``, the pair-mode frequency at the
LJ minimum is ``omega = sqrt(2 U''(r0)/m) = sqrt(2 * 36 * 2^(2/3)) ~ 10.7``
(``U''(r0) = 36 * 2^(2/3) eps / sigma^2 ~ 57.1``), so ``dt = 0.005`` gives
``omega dt ~ 0.05`` -- far inside leapfrog's stability threshold
``omega dt < 2`` -- and initial conditions stay near equilibrium, away from
the stiff short-range wall.

The long runs wrap the framework step in ``jax.jit``: jit *traces through*
``run_one`` of the actual ``pc``/``pc^(2)`` morphisms (the framework code is
the traced program); it does not replace them.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.functors import Phiphase
from dap.interpretation import trivial_omega
from dap.leapfrog import Phileap
from dap.moleculardyn import md_arrangement, md_energy, md_force

EPS, SIGMA, MASS = 1.0, 1.0, 1.0
R0 = 2.0 ** (1.0 / 6.0)  # the LJ minimum separation 2^(1/6) sigma
DT = 0.005

# trivial boundary data of a closed (0-ary) arrangement, as in the wave tests
_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def _lj_energy_hand(r: float) -> float:
    return 4.0 * EPS * ((SIGMA / r) ** 12 - (SIGMA / r) ** 6)


def _run(morphism, state, n_steps):
    """Iterate a closed pc/pc^(2) morphism; jit traces through run_one."""

    def one(s):
        *_, new = morphism.with_state(s).run_one(_IN_POS, _TRIV)
        return new

    step = jax.jit(one)
    out = [state]
    for _ in range(n_steps):
        state = step(state)
        out.append(state)
    return out


# ---------------------------------------------------------------------------
# 1. The potential emerges from the wiring.
# ---------------------------------------------------------------------------


def test_lj_potential_emerges_from_wiring():
    """The closed composite's U is the sum of LJ pair energies -- produced by
    compose_seq(tensor(particles + pair boxes), graph_wire), not written by hand."""
    pairs = [(0, 1), (0, 2), (1, 2)]
    arr = md_arrangement(3, pairs, MASS, EPS, SIGMA, DT, dim=2)
    assert (arr.out_dim_M, arr.in_dim_M, arr.out_dim_N, arr.in_dim_N) == (0, 0, 0, 0)
    assert arr.Q.dim == 6  # three 2-D positions; LJ boxes are stateless
    rng = np.random.default_rng(0)
    for _ in range(5):
        pts = R0 * (np.eye(3, 2) + 0.1 * rng.standard_normal((3, 2)))
        q = jnp.asarray(pts.reshape(-1))
        got = float(arr.U(q, jnp.zeros(0), jnp.zeros(0)))
        want = sum(
            _lj_energy_hand(float(np.linalg.norm(pts[i] - pts[j]))) for i, j in pairs
        )
        np.testing.assert_allclose(got, want, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Pair equilibrium at exactly 2^(1/6) sigma.
# ---------------------------------------------------------------------------


def test_force_vanishes_at_lj_minimum():
    """One covector evaluation through the interpretation: -xi_Q = 0 at exactly
    r = 2^(1/6) sigma, repulsive inside, attractive outside."""
    arr = md_arrangement(2, [(0, 1)], MASS, EPS, SIGMA, DT, dim=1)

    F0 = np.asarray(md_force(arr, jnp.array([0.0, R0])))
    np.testing.assert_allclose(F0, np.zeros(2), atol=1e-12)

    # inside the minimum: particle 0 (at the origin) is pushed AWAY (negative)
    F_in = np.asarray(md_force(arr, jnp.array([0.0, 0.99 * R0])))
    assert F_in[0] < -1e-2 and F_in[1] > 1e-2

    # outside the minimum: particle 0 is pulled TOWARD particle 1 (positive)
    F_out = np.asarray(md_force(arr, jnp.array([0.0, 1.01 * R0])))
    assert F_out[0] > 1e-2 and F_out[1] < -1e-2

    # Newton's third law comes with the construction: dU is a gradient
    np.testing.assert_allclose(F_in[0] + F_in[1], 0.0, atol=1e-12)


def test_pair_oscillates_around_lj_minimum():
    """Two particles released at separation R0 + 0.02 oscillate around R0:
    bounded amplitude, mean separation within 0.01 of R0 (the small outward
    bias of the mean is LJ anharmonicity, not drift)."""
    delta = 0.02
    arr = md_arrangement(2, [(0, 1)], MASS, EPS, SIGMA, DT, dim=1)
    states = _run(
        Phileap(arr), (jnp.array([0.0, R0 + delta]), jnp.zeros(2)), 2000
    )
    seps = np.array([float(s[0][1] - s[0][0]) for s in states[1:]])
    assert seps.max() <= R0 + 2.0 * delta  # bounded, near the release amplitude
    assert seps.min() >= R0 - 2.0 * delta
    assert seps.min() < R0 < seps.max()  # genuinely oscillates around R0
    assert abs(seps.mean() - R0) < 0.01  # measured: mean - R0 ~ +1.6e-3


# ---------------------------------------------------------------------------
# 3. One Phileap macro-tick == one velocity-Verlet step (pins conventions).
# ---------------------------------------------------------------------------


def _lj_force_hand(q_flat: np.ndarray) -> np.ndarray:
    """Independent analytic LJ force for two 2-D particles (no autodiff)."""
    qs = q_flat.reshape(2, 2)
    d = qs[0] - qs[1]
    r2 = float(np.dot(d, d))
    r = np.sqrt(r2)
    s6 = (SIGMA**2 / r2) ** 3
    dUdr = (4.0 * EPS / r) * (-12.0 * s6 * s6 + 6.0 * s6)
    f0 = -dUdr * d / r  # force on particle 0
    return np.concatenate([f0, -f0])


def test_one_phileap_step_is_velocity_verlet():
    """One pc^(2) macro-tick from (q, (m/dt) v) equals the hand-computed
    velocity-Verlet step  v+ = v + (dt/2m)F(q); q' = q + dt v+;
    v' = v+ + (dt/2m)F(q')  to numerical precision."""
    arr = md_arrangement(2, [(0, 1)], MASS, EPS, SIGMA, DT, dim=2)
    q0 = np.array([0.0, 0.0, 1.25, 0.3])  # r ~ 1.286: nonzero force
    v0 = np.array([0.1, -0.2, 0.05, 0.15])

    # hand velocity Verlet, independent force law
    v_half = v0 + (DT / (2.0 * MASS)) * _lj_force_hand(q0)
    q1 = q0 + DT * v_half
    v1 = v_half + (DT / (2.0 * MASS)) * _lj_force_hand(q1)

    # one Phileap macro-tick (two pc^(2) rounds = the two force evaluations)
    O = Phileap(arr)
    _, _, (q_new, xi_new) = O.with_state(
        (jnp.asarray(q0), (MASS / DT) * jnp.asarray(v0))
    ).run_one(_IN_POS, _TRIV)

    np.testing.assert_allclose(np.asarray(q_new), q1, atol=1e-14)
    np.testing.assert_allclose(np.asarray(xi_new) * DT / MASS, v1, atol=1e-13)


# ---------------------------------------------------------------------------
# 4. Energy over 10^4 steps: leapfrog band vs symplectic-Euler band.
# ---------------------------------------------------------------------------


def test_trimer_energy_bounded_and_beats_phase():
    """2-D LJ trimer, dt = 0.005, 10^4 steps. Total energy = kinetic via the
    sharp + emergent potential.

    Tolerance justification: the leapfrog shadow-Hamiltonian error scale is
    (omega dt)^2 * E_osc with omega ~ 16 (stiffest trimer mode), E_osc =
    H0 - U_min ~ 6.5e-4 eps here, giving ~4e-6 eps; measured max |H - H0| is
    6.0e-7 eps, so 5e-6 eps is a comfortable bound. Phiphase (symplectic
    Euler) is first order -- band scale (omega dt) * E_osc -- measured
    2.1e-5 eps, > 30x wider; we assert a >= 10x separation. Both integrators
    are symplectic, so BOTH bands are flat (no secular growth): the last-10%
    band must not exceed twice the first-10% band.
    """
    pairs = [(0, 1), (0, 2), (1, 2)]
    arr = md_arrangement(3, pairs, MASS, EPS, SIGMA, DT, dim=2)
    tri = R0 * np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    rng = np.random.default_rng(0)
    q0 = jnp.asarray((tri + 0.005 * rng.standard_normal(tri.shape)).reshape(-1))
    state0 = (q0, jnp.zeros(6))
    n_steps = 10_000

    energy = jax.jit(lambda s: md_energy(arr, s[0], s[1]))

    def drift_profile(morphism):
        states = _run(morphism, state0, n_steps)
        H = np.array([float(energy(s)) for s in states])
        d = np.abs(H - H[0])
        return d.max(), d[: n_steps // 10].max(), d[-n_steps // 10 :].max()

    leap_max, leap_early, leap_late = drift_profile(Phileap(arr))
    phase_max, phase_early, phase_late = drift_profile(Phiphase(arr))
    print(
        f"\nleapfrog: max|dH| = {leap_max:.3e} (early {leap_early:.3e}, "
        f"late {leap_late:.3e}); phase: max|dH| = {phase_max:.3e} "
        f"(early {phase_early:.3e}, late {phase_late:.3e}); "
        f"ratio = {phase_max / leap_max:.1f}x"
    )

    # leapfrog: bounded oscillating error, no secular growth
    assert leap_max < 5e-6  # measured 6.0e-7
    assert leap_late <= 2.0 * leap_early  # band flat over the run

    # dramatically narrower band than single-stage symplectic Euler
    assert phase_max > 10.0 * leap_max  # measured ~35x

    # honesty check: Phiphase is also symplectic -- ITS band is flat too;
    # the win is bandwidth (2nd order vs 1st), not "drift vs no drift"
    assert phase_late <= 2.0 * phase_early


# ---------------------------------------------------------------------------
# 6. Phiconf: energy minimization (structure relaxation) to the LJ minimum.
# ---------------------------------------------------------------------------


def test_conf_relaxes_pair_to_lj_minimum():
    """Under the configuration dynamics the particles descend the emergent
    total potential -- the energy-minimization ("structure relaxation") mode of
    an MD code -- so a pair released near the LJ minimum settles at separation
    exactly ``2^(1/6) sigma``. Asserted: from 1.05 R0 and from 0.97 R0 the
    separation converges monotonically to R0 (tolerance 1e-8 after 12000
    ticks; the conf step is the small ``dt^2/m``), and the emergent energy
    decreases at every tick."""
    from dap.functors import Phiconf

    arr = md_arrangement(2, [(0, 1)], MASS, EPS, SIGMA, DT, dim=1)
    O = Phiconf(arr)
    for r_start in (1.05 * R0, 0.97 * R0):
        states = _run(O, jnp.array([0.0, r_start]), 12000)
        seps = np.array([float(s[1] - s[0]) for s in states])
        err = np.abs(seps - R0)
        assert np.all(np.diff(err) <= 1e-15)                 # monotone approach
        assert err[-1] < 1e-8                                # settled at 2^(1/6) sigma
        Us = np.array([float(arr.U(s, jnp.zeros(0), jnp.zeros(0))) for s in states])
        assert np.all(np.diff(Us) <= 1e-15)                  # energy descends
