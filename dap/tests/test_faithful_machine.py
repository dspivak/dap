"""The faithful gyroscope machine (EXTENSION, Phase 3; gyroscope_faithful.py).

The Bull & Achour classifier assembled entirely from the paper's constructions. The
point (GYRO_BUILD_HANDOFF goal) is the *factorization* and the *springs ablation*, not
accuracy. We check:

* **factorization** -- the classifier's potential is the *wired* R^2 graph Laplacian
  plus rod gravity minus the open-port drive, matching an independent oracle (the spring
  coupling EMERGES from ``compose_graph``, it is not a hand-written U);
* **the springs->0 ablation (the headline)** -- with springs the input signal reaches the
  output gyros and depends on the input; freeze the stiffness to ~0 and *nothing* reaches
  the output (it stays at rest regardless of the input). The blog's "Take 1 failed",
  here a one-line categorical wiring fact;
* the classifier **runs end-to-end and is differentiable** (backprop through the full
  ``Phirk4gyro`` rollout gives finite gradients).
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.gyroscope_faithful import (
    batch_loss,
    classify,
    faithful_arrangement,
    init_params,
    make_hex_config,
    make_strokes,
    output_readout,
    train,
)


def test_faithful_potential_is_wired_laplacian_plus_gravity_minus_drive():
    """The classifier's U = (wired R^2 spring Laplacian) + (rod gravity) - (drive)."""
    cfg = make_hex_config(3, 3, n_classes=4)
    m, kappa, g, L = 1.2, 0.6, 0.3, 3.0
    arr = faithful_arrangement(cfg, m, kappa, g, L)
    assert (arr.out_dim_M, arr.in_dim_M) == (0, 0)
    assert (arr.out_dim_N, arr.in_dim_N) == (cfg.n_out, cfg.n_in)  # open readout + drive ports
    assert arr.Q.dim == 2 * cfg.V

    rng = np.random.default_rng(0)
    in_idx = list(cfg.in_idx)
    for _ in range(4):
        q = jnp.asarray(0.3 * rng.standard_normal(2 * cfg.V))  # small tilts (|q| < L)
        force = jnp.asarray(rng.standard_normal(cfg.n_in))
        got = float(arr.U(q, jnp.zeros(0), force))
        qv = np.asarray(q).reshape(cfg.V, 2)
        springs = sum(0.5 * kappa * float(np.sum((qv[t] - qv[s]) ** 2)) for (s, t) in cfg.edges)
        gravity = sum(-g * float(np.sqrt(L ** 2 - np.sum(qv[v] ** 2))) for v in range(cfg.V))
        drive = float(np.dot(np.asarray(force), np.asarray(q)[in_idx]))
        np.testing.assert_allclose(got, springs + gravity - drive, atol=1e-9)


def test_springs_zero_collapses_information_flow():
    """Headline: springs carry the input signal to the output; freeze them and nothing
    reaches the output. Input gyros = left column, output = right column (disjoint), so
    with no spring path the output gyros stay at rest regardless of the input drive."""
    cfg = make_hex_config(3, 3, n_classes=4, settle=10)
    p = init_params(cfg, seed=0)
    rng = np.random.default_rng(1)
    T = 8
    driveA = jnp.asarray(0.2 * rng.standard_normal((T, cfg.n_in)))
    driveB = jnp.asarray(0.2 * rng.standard_normal((T, cfg.n_in)))

    outA = np.asarray(output_readout(p, driveA, cfg))
    outB = np.asarray(output_readout(p, driveB, cfg))
    assert np.linalg.norm(outA) > 1e-3                       # with springs: signal reaches the output
    assert not np.allclose(outA, outB, atol=1e-5)            # ...and it is input-dependent

    p0 = {**p, "log_kappa": jnp.array(-100.0)}               # freeze stiffness to ~0
    outA0 = np.asarray(output_readout(p0, driveA, cfg))
    outB0 = np.asarray(output_readout(p0, driveB, cfg))
    np.testing.assert_allclose(outA0, 0.0, atol=1e-9)        # springs off: nothing reaches the output
    np.testing.assert_allclose(outB0, 0.0, atol=1e-9)        # ...independent of the input


def test_classifier_runs_and_is_differentiable():
    """One stroke -> (n_classes,) logits; backprop through the full Phirk4gyro rollout
    gives finite gradients (the classifier is trainable end-to-end)."""
    cfg = make_hex_config(2, 3, n_classes=3, settle=6)
    p = init_params(cfg, seed=0)
    X, Y = make_strokes(8, T=10, n_classes=3, seed=0)

    logits = classify(p, X[0], cfg)
    assert logits.shape == (3,)

    loss, grads = jax.value_and_grad(lambda pp: batch_loss(pp, X, Y, cfg))(p)
    assert np.isfinite(float(loss))
    assert all(bool(np.all(np.isfinite(np.asarray(g)))) for g in grads.values())


def test_faithful_machine_trains_loss_decreases():
    """It trains end-to-end: minibatch Adam through the full Phirk4gyro rollout reduces the
    loss. Per the goal this is only a trainability sanity -- NO accuracy claim (the machine
    exists for the factorization + the springs ablation, not to hit a benchmark number)."""
    cfg = make_hex_config(3, 3, n_classes=3, settle=8)
    Xtr, Ytr = make_strokes(300, T=12, n_classes=3, seed=0)
    _p, hist = train(cfg, Xtr, Ytr, epochs=10, batch=60, lr=5e-3, seed=0)
    assert np.isfinite(hist[-1])
    assert hist[-1] < hist[0]  # the loss genuinely decreases under training

