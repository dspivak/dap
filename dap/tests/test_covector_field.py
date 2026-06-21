"""Covector fields are general (eqn.Phi_on_obs: ``Omega(M)``), not affine.

For a box with a *nonlinear* potential, the output covector field ``omega_N``
(eqn.omegaprime) is nonlinear in the input. The interpretation returns it as the
exact field (a callable); the old affine ``(A, b)`` reconstruction from value and
Jacobian at zero would be wrong away from zero. We check exactness here.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.interpretation import smooth_interpretation, trivial_omega
from dap.rvect import euclidean


def test_nonlinear_omega_field_is_exact():
    # An open box <R^0|R> -> <R|R> with NONLINEAR potential U(q, m_out, n_in) = sum cos(n_in).
    arr = SmoothArrangement(
        euclidean(1), out_dim_M=0, in_dim_M=0, out_dim_N=1, in_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: jnp.sum(jnp.cos(n_in)),
    )
    q = jnp.array([0.3])
    position_action, _ = smooth_interpretation(arr)(q)
    out_n, omega_N = position_action(jnp.zeros(0), trivial_omega(0))

    np.testing.assert_allclose(np.asarray(out_n), np.asarray(q), atol=1e-12)
    # omega_N(n_in) = d/d(n_in) sum cos(n_in) = -sin(n_in): exact at every point,
    # including ones the affine-at-zero reconstruction (slope -1, i.e. -n_in) gets wrong.
    for x in (-1.3, 0.0, 0.7, 2.1):
        n_in = jnp.array([x])
        np.testing.assert_allclose(np.asarray(omega_N(n_in)), np.asarray(-jnp.sin(n_in)), atol=1e-10)
    assert abs(float(omega_N(jnp.array([2.1]))[0]) - (-2.1)) > 0.5  # genuinely nonlinear
