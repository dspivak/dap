"""System identification (rmk.graph_environment, "Predicting a closed system").

``Phiphase`` runs a nonlinear pendulum; a network trained on its consecutive-state
pairs -- the ``<o, o.c>`` environment with ``o = id`` -- learns the system's
one-step flow map. We check the held-out one-step error is small and far below the
naive "no change" baseline, so the net genuinely identified the (nonlinear) dynamics.
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from dap.system_id import PENDULUM, identify, one_step_error


def test_identifies_nonlinear_pendulum_one_step_map():
    params, F, hist, O, dim, m = identify(PENDULUM, steps=3000, seed=1)
    err, base = one_step_error(params, F, O, dim, m, seed=101)

    assert np.mean(hist[-200:]) < 0.5 * np.mean(hist[:200])  # training drove the loss down
    assert err < 0.15                                        # accurate one-step prediction
    assert err < 0.25 * base                                 # far better than 'no change'
