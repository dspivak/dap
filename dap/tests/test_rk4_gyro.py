"""RK4 on the gyro phase system as org^(4) (rk4.rk4_gyro_integrator / Phirk4gyro; Phase 3).

The faithful machine's integrator: classical RK4 stepping the phase ODE
``qdot = sharp(xi), xidot = -dU - drag*F(v) - gamma*(J v)`` -- the same forces as
``Phigyro`` but the blog's RK4 time-stepping instead of symplectic Euler. We verify:

* 4th-order accuracy: on a harmonic oscillator (closed form via matrix exp) the global
  error falls like h^4 (ratios -> 16) -- the falsifiable proof it is genuine RK4 run
  through the four org^(4) rounds on the PHASE state;
* one macro-tick equals a hand-rolled RK4 step of the full phase ODE *including* the
  quadratic-drag and per-gyro precession 1-forms.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phirk4gyro
from dap.gyroscope import complex_structure
from dap.interpretation import trivial_omega
from dap.rvect import constant

_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda op: (jnp.zeros(0), jnp.zeros(0))


def _oscillator(d, m, k):
    """Closed harmonic oscillator on R^d: U = (k/2)|q|^2, sharp = (1/m) I (dU = k q)."""
    return SmoothArrangement(
        constant(jnp.eye(d) / m), 0, 0, 0, 0,
        out_f=lambda q, mo: jnp.zeros(0),
        in_f=lambda q, mo, ni: jnp.zeros(0),
        U=lambda q, mo, ni: 0.5 * k * jnp.sum(q ** 2),
        label="osc",
    )


def test_rk4_gyro_is_fourth_order_on_oscillator():
    """Global error vs the exact phase flow e^{AT} falls like h^4 (ratios -> 16)."""
    d, m, k, T = 2, 1.3, 2.0, 1.0
    arr = _oscillator(d, m, k)
    q0 = np.array([1.0, -0.5])
    xi0 = np.array([0.2, 0.7])
    # phase generator A on (q, xi): qdot = xi/m, xidot = -k q
    A = np.block([[np.zeros((d, d)), np.eye(d) / m], [-k * np.eye(d), np.zeros((d, d))]])
    z0 = np.concatenate([q0, xi0])
    exact = np.asarray(jax.scipy.linalg.expm(jnp.asarray(A * T))) @ z0

    def err(N):
        O = Phirk4gyro(arr, T / N)
        s = (jnp.asarray(q0), jnp.asarray(xi0))
        for _ in range(N):
            _, s = O.with_state(s).run_one(_IN_POS, _TRIV)
        got = np.concatenate([np.asarray(s[0]), np.asarray(s[1])])
        return float(np.linalg.norm(got - exact))

    e8, e16, e32 = err(8), err(16), err(32)
    assert e32 < 1e-5
    assert e8 / e16 > 12.0
    assert e16 / e32 > 12.0


def test_rk4_gyro_one_step_matches_hand_written_with_drag_and_precession():
    """One Phirk4gyro tick = a hand-rolled RK4 step of the full phase ODE, drag + gamma in."""
    V, m, k, h = 2, 1.0, 0.5, 0.1
    dim = 2 * V
    arr = _oscillator(dim, m, k)  # dU = k q
    J = complex_structure(V)
    drag = 0.3
    gamma = jnp.array([0.4, 0.4, 0.0, 0.0])  # per-gyro
    O = Phirk4gyro(arr, h, drag=drag, gamma=gamma, J=J, gyro_block=2)

    q0 = jnp.array([0.3, -0.2, 0.1, 0.5])
    xi0 = jnp.array([0.2, 0.1, -0.3, 0.4])
    _, (q1, xi1) = O.with_state((q0, xi0)).run_one(_IN_POS, _TRIV)

    Jm = np.asarray(J)
    gam = np.asarray(gamma)

    def Fdrag(v):
        vg = v.reshape(V, 2)
        sp = np.sqrt((vg ** 2).sum(axis=1, keepdims=True))
        return (sp * vg).reshape(-1)

    def f(z):  # full phase vector field
        q, xi = z[:dim], z[dim:]
        v = xi / m
        dU = k * q
        return np.concatenate([v, -dU - drag * Fdrag(v) - gam * (Jm @ v)])

    z = np.concatenate([np.asarray(q0), np.asarray(xi0)])
    k1 = f(z)
    k2 = f(z + h / 2 * k1)
    k3 = f(z + h / 2 * k2)
    k4 = f(z + h * k3)
    z_new = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    got = np.concatenate([np.asarray(q1), np.asarray(xi1)])
    np.testing.assert_allclose(got, z_new, atol=1e-10)
