"""Faithful-machine physics pieces (Phase 3): nonlinear rod gravity + per-gyro gamma.

Two small in-framework closures of blog deviations:

* ``rod_gravity(g, L)`` -- the nonlinear rod-geometry on-site potential
  ``-g sqrt(L^2 - |q|^2)``: a restoring well, *nonlinear* (stiffens toward horizontal),
  whose small-tilt limit is the harmonic well the surrogate used. Plugs into the
  ``onsite`` hook of ``compose_graph``.
* per-gyro ``gamma``: a per-component vector gamma makes ``Phigyro`` precess each gyro
  at its own rate (the integrator just broadcasts it against ``J @ v``).
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phigyro
from dap.gyroscope import complex_structure, rod_gravity
from dap.interpretation import trivial_omega
from dap.rvect import constant

_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda op: (jnp.zeros(0), jnp.zeros(0))


def test_rod_gravity_is_a_restoring_well_with_harmonic_limit():
    """Minimum upright (r=0), and the small-tilt force is the harmonic ``(g/L) q``."""
    g, L = 2.0, 3.0
    U = rod_gravity(g, L)
    assert abs(float(U(jnp.zeros(2))) - (-g * L)) < 1e-12          # minimum value -gL
    assert float(U(jnp.array([0.5, 0.3]))) > float(U(jnp.zeros(2)))  # a well
    q_small = jnp.array([1e-3, -2e-3])
    grad = jax.grad(U)(q_small)
    np.testing.assert_allclose(np.asarray(grad), np.asarray((g / L) * q_small), rtol=1e-3)


def test_rod_gravity_stiffens_nonlinearly():
    """The restoring force per unit displacement grows toward the rod's horizontal limit
    -- genuinely nonlinear, unlike the constant-stiffness harmonic surrogate."""
    g, L = 1.0, 1.0
    U = rod_gravity(g, L)
    near = float(jnp.linalg.norm(jax.grad(U)(jnp.array([0.1, 0.0])))) / 0.1   # ~ g/L
    far = float(jnp.linalg.norm(jax.grad(U)(jnp.array([0.9, 0.0])))) / 0.9    # >> g/L
    assert far > 1.5 * near


def _free_gyros(V, m=1.0):
    """A closed V-gyro free system (Q = R^{2V}, sharp = (1/m) I, U = 0)."""
    return SmoothArrangement(
        constant(jnp.eye(2 * V) / m), 0, 0, 0, 0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: jnp.array(0.0),
        label="free_gyros",
    )


def test_per_gyro_gamma_precesses_independently():
    """A per-gyro gamma vector precesses each gyro at its own rate: with gamma=(g0,0),
    gyro 0 gets the skew kick -g0 J v_0 and gyro 1 (gamma=0) is untouched."""
    V, m, g0 = 2, 1.0, 0.5
    arr = _free_gyros(V, m)
    J = complex_structure(V)
    gamma = jnp.array([g0, g0, 0.0, 0.0])  # per-gyro: gyro 0 -> g0, gyro 1 -> 0
    O = Phigyro(arr, gamma=gamma, J=J)

    xi = jnp.array([0.0, 1.0, 0.0, 1.0])  # both gyros start with the same momentum
    _, _, (q1, xi1) = O.with_state((jnp.zeros(2 * V), xi)).run_one(_IN_POS, _TRIV)

    Jb = np.array([[0.0, -1.0], [1.0, 0.0]])
    v = np.asarray(xi) / m
    exp0 = np.asarray(xi[0:2]) - g0 * (Jb @ v[0:2])  # gyro 0 precesses
    np.testing.assert_allclose(np.asarray(xi1[0:2]), exp0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(xi1[2:4]), np.asarray(xi[2:4]), atol=1e-10)  # gyro 1 fixed
