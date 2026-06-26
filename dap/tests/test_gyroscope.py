"""Gyroscopes-on-springs as a classifier (EXTENSION; dap/gyroscope.py).

A network of 2-D harmonic gyros read by ``Phigyro`` is a harmonic surrogate of the
system of Bull & Achour (unconv.ai, 2026). We check four things: the gyroscopic phase
step matches the closed-form Hamilton update; damping dissipates energy out of a free
network; the skew gyroscopic momentum kick is perpendicular to the velocity, so the
*force* does no work (``(J v).v = 0``) -- though its explicit discretization still
needs the damping to stay bounded; and the whole classifier trains -- the loss drops
and a learnable synthetic task is classified well above chance, exercising backprop
through the full ``Phigyro`` rollout.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.functors import Phigyro
from dap.gyroscope import (
    complex_structure,
    gyro_arrangement,
    make_config,
    init_params,
    classify,
    batch_loss,
    accuracy,
    train,
)
from dap.interpretation import trivial_omega

_IN_POS = (jnp.zeros(0), trivial_omega(0))


def test_gyro_step_matches_closed_form():
    """One ``Phigyro`` step of a single 2-D gyro equals the hand-written symplectic
    Euler + gyroscopic update ``(q + v, xi - (k*q~ - f) - gamma*J*v)``."""
    cfg = make_config(rows=1, cols=1)  # one gyro; in == out
    m, k, gamma = 2.0, 0.7, 0.4
    arr = gyro_arrangement(cfg, inv_mass=jnp.array([1.0 / m]), kappa=jnp.zeros(0), grav_k=jnp.array(k))
    J = complex_structure(1)
    O = Phigyro(arr, damping=0.0, gamma=gamma, J=J)

    q = jnp.array([0.3, -0.5])
    xi = jnp.array([0.1, 0.2])
    f = jnp.array([0.05, -0.02])
    _, _, (q_new, xi_new) = O.with_state((q, xi)).run_one(_IN_POS, lambda op: (jnp.zeros(2), f))

    v = xi / m
    q_tilde = q + v
    xi_Q = k * q_tilde - f
    q_exp = q + v
    xi_exp = xi - xi_Q - gamma * (J @ v)
    assert np.allclose(np.asarray(q_new), np.asarray(q_exp), atol=1e-10)
    assert np.allclose(np.asarray(xi_new), np.asarray(xi_exp), atol=1e-10)


def _free_energy_trace(cfg, params, q0, xi0, steps):
    """Energy ``H = 1/2 xi.M^{-1}.xi + U(q)`` along the unforced rollout."""
    from dap.gyroscope import coalgebra

    O = coalgebra(cfg, params)
    inv_diag = jnp.repeat(jnp.exp(params["log_inv_mass"]), 2)
    kappa = jnp.exp(params["log_kappa"])
    grav = jnp.exp(params["log_grav"])
    E = jnp.asarray(cfg.edges)

    def energy(q, xi):
        qv = q.reshape(cfg.V, 2)
        springs = 0.5 * jnp.sum(kappa * jnp.sum((qv[E[:, 0]] - qv[E[:, 1]]) ** 2, axis=1))
        return 0.5 * jnp.sum(inv_diag * xi ** 2) + springs + 0.5 * grav * jnp.sum(qv ** 2)

    state = (q0, xi0)
    trace = [float(energy(*state))]
    zero_force = jnp.zeros(cfg.n_in)
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one(_IN_POS, lambda op: (jnp.zeros(cfg.n_out), zero_force))
        trace.append(float(energy(*state)))
    return np.array(trace)


def test_damping_dissipates_energy():
    """Damping bleeds energy out of a free (unforced) network."""
    rng = np.random.default_rng(0)
    cfg = make_config(rows=2, cols=3, damping=0.1)
    params = init_params(cfg, seed=0, kappa0=0.2, grav0=0.1, gamma0=0.0)
    q0 = jnp.asarray(0.5 * rng.standard_normal(2 * cfg.V))
    xi0 = jnp.asarray(0.5 * rng.standard_normal(2 * cfg.V))

    damped = _free_energy_trace(cfg, params, q0, xi0, steps=60)
    assert damped[-1] < 0.5 * damped[0]  # damping dissipates


def test_gyroscopic_force_does_no_work():
    """The gyroscopic momentum kick ``-gamma*J*v`` is perpendicular to the velocity
    ``v`` (its defining no-work property, rmk.symplectic_perpendicular): per 2-D gyro
    block, ``(J v) . v = 0``. (The explicit discretization still needs damping to stay
    bounded -- see the integrator docstring -- but the *force* itself does no work.)"""
    rng = np.random.default_rng(2)
    V = 6
    J = complex_structure(V)
    for _ in range(20):
        v = jnp.asarray(rng.standard_normal(2 * V))
        assert abs(float(jnp.dot(J @ v, v))) < 1e-10


def _synthetic(n, T, rng):
    """A 2-class task: class 0 strokes drift right (+x), class 1 drift left (-x)."""
    y = rng.integers(0, 2, n)
    sign = np.where(y == 0, 1.0, -1.0)[:, None, None]
    x = 0.5 * sign + 0.25 * rng.standard_normal((n, T, 2))
    return jnp.asarray(x), jnp.asarray(y)


def test_classifier_trains_and_separates():
    """Backprop through the full Phigyro rollout reduces the loss and learns a
    linearly-separable synthetic stroke task well above chance."""
    rng = np.random.default_rng(1)
    cfg = make_config(rows=2, cols=3, n_classes=2, settle=6)
    Xtr, Ytr = _synthetic(400, 6, rng)
    Xva, Yva = _synthetic(200, 6, rng)

    p0 = init_params(cfg, seed=0)  # cfg.n_classes=2 sizes the decoder head
    loss0 = float(batch_loss(p0, Xtr[:128], Ytr[:128], cfg))

    params, hist = train(cfg, Xtr, Ytr, Xva, Yva, epochs=15, batch=64, lr=5e-3, seed=0, verbose=False)
    best = max(h["val_acc"] for h in hist)

    assert hist[-1]["loss"] < 0.7 * loss0    # training drove the loss down
    assert best > 0.8                        # well above chance (0.5) on a learnable task
