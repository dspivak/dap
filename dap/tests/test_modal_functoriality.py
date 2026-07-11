"""Functor-law evidence for ``Phi_modal`` (dap/modal.py), the mode-dependent extension.

``Phi_modal`` compiles a *modal arrangement* (a family of smooth arrangements + a
state-dependent mode ``transition``) into an ``org`` coalgebra on a ``Sum`` interface. It
is a hybrid dynamical system: smooth ``Phi``-flow inside a mode, discrete jumps between
modes. These tests are the falsifiable evidence that the extension is *functorial* -- that
mode-switching does not break the compositional structure of ``Phi``:

1. **Conservativity** -- a single never-switching mode is exactly ``functors.Phi(arr)``.
   ``Phi_modal`` restricted to the single-mode subcategory *is* the base dynamics functor.
2. **Gluing (temporal)** -- the hybrid trajectory equals ``Phi(mode0)`` spliced with
   ``Phi(mode1)`` at the switch: the hybrid run is the sequential-in-time composition of
   the smooth runs. This is 'compose-then-run = run-then-compose' surviving a mode jump.
3. **Tensor law (spatial)** -- ``Phi_modal(A (x) B) == Phi_modal(A) || Phi_modal(B)``
   (product modes, independent switching): the parallel composition the tissue rests on.

Sequential ``;`` composition of modal arrangements (wiring one modal box into another) is
not audited here -- it needs a modal ``compose_seq`` and is the natural next step
(rmk.multistage-style: datatype + instances + these laws, the full proof still open).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from dap.arrangement import SmoothArrangement
from dap.functors import Phi
from dap.integrator import configuration_integrator
from dap.interpretation import trivial_omega
from dap.modal import (DECAY, PROD0, PRODA, THETA, ModalArrangement, Phi_modal,
                       Phi_modal_soft, fate_transition, morphogen_cell, tensor_modal)
from dap.rvect import euclidean

_IN = (jnp.zeros(0), trivial_omega(0))
_DIR1 = lambda _o: (jnp.zeros(1), jnp.zeros(0))   # closed drive on a 1-D readout


def _readout_cell(prod: float, eta: float = 0.15) -> SmoothArrangement:
    """A 1-D cell that reports its state (out_dim_N=1), so behaviour is observable."""
    return SmoothArrangement(
        euclidean(1, eta), 0, 0, 1, 0,
        out_f=lambda q, m: q,
        in_f=lambda q, m, n: jnp.zeros(0),
        U=lambda q, m, n: -prod * q[0] + 0.5 * DECAY * q[0] ** 2,
    )


def _run_modal(O, steps, s0):
    s, rec = s0, []
    for _ in range(steps):
        out_pos, _od, s = O.with_state(s).run_one(_IN, _DIR1)
        m, (out_n, _om) = out_pos
        rec.append((int(m), float(out_n[0]), s[1]))
    return rec


def _run_phi(O, steps, s0):
    s, rec = s0, []
    for _ in range(steps):
        out_pos, _od, s = O.with_state(s).run_one(_IN, _DIR1)
        out_n, _om = out_pos
        rec.append((float(out_n[0]), s))
    return rec


def test_single_cell_differentiates():
    """Basic behaviour: a progenitor crosses threshold and the coalgebra switches mode."""
    O = Phi_modal(morphogen_cell(), configuration_integrator())
    s, modes = (0, jnp.array([0.0])), []
    for _ in range(20):
        _op, _od, s = O.with_state(s).run_one(_IN, lambda _o: (jnp.zeros(0), jnp.zeros(0)))
        modes.append(int(s[0]))
    assert modes[0] == 0 and modes[-1] == 1        # progenitor -> differentiated
    assert modes == sorted(modes)                  # terminal: never un-differentiates


def test_conservativity_single_mode_is_phi():
    """Law 1: one never-switching mode == functors.Phi(arr), run-for-run."""
    arr = _readout_cell(PROD0)
    phi = _run_phi(Phi(arr, configuration_integrator()), 25, jnp.array([0.0]))
    marr = ModalArrangement((arr,), lambda m, q: (m, q))
    mod = _run_modal(Phi_modal(marr, configuration_integrator()), 25, (0, jnp.array([0.0])))
    for p, m in zip(phi, mod):
        assert abs(p[0] - m[1]) < 1e-9                       # readouts agree
        assert jnp.allclose(p[1], m[2], atol=1e-9)           # states agree


def test_gluing_hybrid_is_spliced_phi():
    """Law 2: hybrid run == Phi(mode0) ++ Phi(mode1), spliced at the switch."""
    m0, m1 = _readout_cell(PROD0), _readout_cell(PRODA)
    O = Phi_modal(ModalArrangement((m0, m1), fate_transition(THETA)), configuration_integrator())
    T = 30
    mod = _run_modal(O, T, (0, jnp.array([0.0])))
    modes = [r[0] for r in mod]
    k = next(t for t in range(T) if modes[t] == 1)           # first mode-1 step
    assert 0 < k < T                                          # the switch actually happens mid-run

    phi0 = _run_phi(Phi(m0, configuration_integrator()), k, jnp.array([0.0]))
    for t in range(k):
        assert jnp.allclose(phi0[t][1], mod[t][2], atol=1e-9)

    phi1 = _run_phi(Phi(m1, configuration_integrator()), T - k, mod[k - 1][2])
    for j in range(T - k):
        assert jnp.allclose(phi1[j][1], mod[k + j][2], atol=1e-9)


def test_tensor_law_parallel_equals_product_modal():
    """Law 3: Phi_modal(A (x) B) == Phi_modal(A) || Phi_modal(B), incl. independent switching."""
    A = ModalArrangement((_readout_cell(PROD0), _readout_cell(PRODA)), fate_transition(0.40))
    B = ModalArrangement((_readout_cell(PROD0), _readout_cell(PRODA)), fate_transition(0.75))
    O_par = Phi_modal(A, configuration_integrator()).parallel(Phi_modal(B, configuration_integrator()))
    O_ten = Phi_modal(tensor_modal(A, B), configuration_integrator())
    KB = len(B.modes)

    in_par = ((jnp.zeros(0), trivial_omega(0)), (jnp.zeros(0), trivial_omega(0)))
    dir_par = lambda _o: ((jnp.zeros(1), jnp.zeros(0)), (jnp.zeros(1), jnp.zeros(0)))
    dir_ten = lambda _o: (jnp.zeros(2), jnp.zeros(0))

    sp, st, seen = O_par.state, O_ten.state, set()
    for _t in range(30):
        opp, _od, sp = O_par.with_state(sp).run_one(in_par, dir_par)
        opt, _od, st = O_ten.with_state(st).run_one(_IN, dir_ten)
        mA, outA = int(opp[0][0]), float(opp[0][1][0][0])
        mB, outB = int(opp[1][0]), float(opp[1][1][0][0])
        M, (out_n, _om) = opt
        a, b = divmod(int(M), KB)
        seen.add((mA, mB))
        assert (mA, mB) == (a, b)                             # product mode agrees
        assert abs(outA - float(out_n[0])) < 1e-9 and abs(outB - float(out_n[1])) < 1e-9
    assert (1, 0) in seen                                     # A switched before B: staggered, not simultaneous


def test_soft_modal_is_differentiable():
    """Phi_modal_soft: backprop runs through the coalgebra and a fate parameter is learnable."""
    marr = morphogen_cell()

    def final_u(theta):
        def tr(w, s):
            p1 = jax.nn.sigmoid(8.0 * (s[0] - theta))
            return jnp.array([1.0 - p1, p1])
        O = Phi_modal_soft(marr, configuration_integrator(), tr)
        s = (jnp.array([1.0, 0.0]), jnp.array([0.0]))
        for _ in range(30):
            _op, _od, s = O.with_state(s).run_one(_IN, lambda _o: (jnp.zeros(0), jnp.zeros(0)))
        return s[1][0]

    loss = lambda th: (final_u(th) - 2.0) ** 2
    L0, g0 = jax.value_and_grad(loss)(0.9)
    assert jnp.isfinite(g0) and abs(float(g0)) > 0            # finite, nonzero gradient through the functor
    assert float(loss(0.9 - 0.05 * g0)) < float(L0)          # the gradient is a valid descent direction
    theta = 0.9                                               # and a gentle descent tunes the fate rule
    for _ in range(80):
        theta = theta - 0.3 * jax.grad(loss)(theta)
    assert float(loss(theta)) < 1e-3                          # converges to the target commitment
