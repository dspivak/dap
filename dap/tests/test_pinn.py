"""A physics-informed network (deep Ritz), via Phiconf.

A field network ``u_theta(x) = x(1-x) g_theta(x)`` (hard Dirichlet BC) is trained
under ``Phiconf`` to minimize the discrete Dirichlet energy of ``-u'' = f`` on
``[0,1]``. We check the energy decreases and the learned field matches the discrete
Poisson solution ``L u = b`` -- i.e. the network solved the BVP.
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from dap.pinn import solve_deep_ritz


def test_deep_ritz_solves_poisson():
    u, u_star, hist, x = solve_deep_ritz(steps=4000, seed=0)
    rel = float(
        np.linalg.norm(np.asarray(u) - np.asarray(u_star))
        / np.linalg.norm(np.asarray(u_star))
    )

    assert hist[-1] < hist[0]  # the Dirichlet energy decreased
    assert rel < 0.08          # the field matches the discrete Poisson solution
