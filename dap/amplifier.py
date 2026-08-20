"""Black's feedback amplifier (EXTENSION -- audit artifact, not part of the paper).

This module audits the draft remark: "The same unilaterality should be the
active ingredient in Black's feedback amplifier: a high-gain unilateral stage
whose reported output is returned, attenuated, to one of its inputs by the
wiring.  The classical insensitivity of the closed-loop gain to the stage's own
gain would then be a statement about equilibria of the composite system."

The construction
----------------

**Stage** (``stage``): a ``<R^0|R^0> -> <R^1|R^1>`` arrangement with parameter
``Q = R^2``, state ``q = (v, w)``, reported output ``out_f(q) = v``, one input
port carrying the drive voltage ``e``, potential

    U(q, e) = (kappa/2) (A g(e) - v - w)^2        (g = id for the linear stage)

and the *same* degenerate, non-symmetric reaction as ``logic.py``'s MOSFET,
scaled by the timestep:

    sharpR = dt * [[0, 1],
                   [0, 0]]  =  dt * KERNEL_SHARP.

``Phiconf`` steps ``q -> q - sharpR(dU)``, i.e.

    v <- v + dt kappa (A g(e) - v - w),        w <- w  (frozen; start at 0),

so ``v`` relaxes toward ``A g(e)`` at rate ``dt kappa``, read off ``dU/dw``
alone.  The whole ``v``-component of ``dU`` -- which is where every back-action
on the stage lands (see "Which unilaterality", below) -- lies in
``ker(sharpR)``.  This is exactly ``logic.py``'s **kernel condition**, not the
paper's output-side ``unilateral`` of def.arrangement_terminology (which the
stage satisfies only in the composite, where it has no free output port left).

**Feedback wiring** (``feedback_wire``): a static arrangement
``<R^1|R^1> -> <R^1|R^1>`` with trivial parameter and ``U = 0``, routing

    out_f(m_out)        = m_out            (report v to the outer boundary),
    in_f(m_out, n_in)   = n_in - beta * m_out   (the inverting summing junction).

HONESTY FLAG: ``in_f`` is *linear*, not a port re-plumbing.  def.potlens asks
only that ``in_f`` be smooth, so this is a legitimate static morphism of
``sarr``; but it is NOT in the image of ``R^-`` of a finset lens (lem.lens_pow)
-- every wiring in ``wiring.py`` is a coordinate routing (``nand_wire`` at most
stacks the constant 0).  If "wiring" in the remark means the operadic
(finset-induced) wiring, the summing junction is outside it; the subtraction
and the attenuation ``beta`` are linear algebra that some static box must
perform.  (In Black's amplifier that box is a resistive network -- itself a
device, not a plumbing.)

**Composite** (``black_amplifier``): ``compose_seq(stage, feedback_wire)``, an
open arrangement ``<R^0|R^0> -> <R^1|R^1>`` whose outer input port carries
``e_src`` at run time (the ``control.py`` pattern) and whose outer output
reports ``v``.  ``compose_seq`` produces the composite potential

    U_comp((v, w), e_src) = (kappa/2) (A g(e_src - beta v) - v - w)^2

and the one-step update (w = 0)

    v <- v + dt kappa (A g(e_src - beta v) - v).                      (*)

For the linear stage this is ``v <- v + dt kappa (A e_src - (1 + A beta) v)``:

* fixed point   ``v* = A / (1 + A beta) e_src``   -- Black's gain formula;
* error factor  ``1 - dt kappa (1 + A beta)`` per step, so the iteration
  diverges iff ``dt kappa (1 + A beta) > 2``.

Which unilaterality is operating
--------------------------------

Two distinct mechanisms, both present, doing different jobs:

1. **Kernel condition (transistor mechanism), inside the loop.**  In the
   composite, the loop back-action -- the stage's input-port covector
   ``dU/de = -kappa r A g'(e)`` pulled back through the wiring's
   ``in_f = e_src - beta v`` and the stage's report ``out_f = v`` -- lands
   entirely in the ``v``-component of ``xi_Q``, which ``sharpR`` reads as 0.
   Likewise any external covector ``xi_N`` presented at the output port:
   ``(d_q out_f)^T xi_N = (xi_N, 0)`` is in the kernel, so the load cannot
   move the stage (ideal voltage-stiff output).  This is exactly how
   ``logic.py``'s MOSFET decouples its gate from its drain.

2. **Boundary discard (control.py mechanism), at the source only.**  The
   composite still *emits* the reaction covector
   ``omega_N(e_src) = kappa r A g'(e)`` on the drive port (eqn.omegaprime) --
   the amplifier's input current.  The runner discards it: the source is
   treated as infinitely stiff.  It vanishes at equilibrium (r = 0), but off
   equilibrium it is A-large; a non-ideal source would feel it.

The audit's negative
--------------------

``bilateral_stage`` is the reciprocal control: state ``v`` alone, symmetric
sharp ``[[dt]]``, same potential.  Its closed loop is plain gradient descent on
``U_comp``; the update gains the loop factor,

    v <- v + dt kappa (1 + A beta g'(e)) (A g(e) - v),

but the fixed-point locus ``{A g(e) = v}`` -- hence Black's formula, the
desensitization, and the quasi-static distortion suppression -- is IDENTICAL.
Unilaterality is *not* the active ingredient in the equilibrium statements; it
is the active ingredient in the *dynamics*: the unilateral loop's stability
boundary is ``dt kappa (1 + A beta) = 2`` where the bilateral loop's is
``dt kappa (1 + A beta)^2 = 2`` -- reciprocity makes the discrete loop stiffer
by the whole loop gain -- and only the unilateral stage rejects output-port
covectors.  See the tests.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .functors import Phiconf
from .interpretation import trivial_omega
from .rvect import constant, trivial
from .wiring import compose_seq

# The stage's reaction: logic.py's KERNEL_SHARP scaled by the timestep.
KERNEL = jnp.array([[0.0, 1.0], [0.0, 0.0]])

_IN_POS = (jnp.zeros(0), trivial_omega(0))


# ---------------------------------------------------------------------------
# The boxes.
# ---------------------------------------------------------------------------


def stage(
    A: float,
    kappa: float = 1.0,
    dt: float = 0.01,
    g: Optional[Callable[[Array], Array]] = None,
    label: str = "",
) -> SmoothArrangement:
    """The unilateral gain stage ``<R^0|R^0> -> <R^1|R^1>``, state ``(v, w)``.

    Reports ``v``; potential ``(kappa/2)(A g(e) - v - w)^2``; reaction
    ``dt * KERNEL`` (degenerate, non-symmetric -- def.rvect requires neither),
    so ``Phiconf`` gives ``v <- v + dt kappa (A g(e) - v - w)``, ``w`` frozen.
    ``g`` defaults to the identity (linear stage); pass a saturating ``g`` for
    the distortion study.  Start the state at ``(v0, 0)`` (``amp_state``).
    """
    gg = (lambda e: e) if g is None else g

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return 0.5 * kappa * (A * gg(n_in[0]) - q[0] - q[1]) ** 2

    return SmoothArrangement(
        Q=constant(dt * KERNEL),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=1,
        in_dim_N=1,
        out_f=lambda q, m_out: q[0:1],
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=U,
        label=label or f"stage(A={A:g})",
    )


def bilateral_stage(
    A: float,
    kappa: float = 1.0,
    dt: float = 0.01,
    g: Optional[Callable[[Array], Array]] = None,
    label: str = "",
) -> SmoothArrangement:
    """The reciprocal control: same potential, state ``v`` alone, sharp ``[[dt]]``.

    Its closed loop is genuine gradient descent on the composite potential, so
    the back-action through the report is *felt*: the update carries the loop
    factor ``(1 + A beta g'(e))``.  Same equilibria as ``stage``; different
    dynamics.  State is 1-dimensional (start at ``jnp.array([v0])``).
    """
    gg = (lambda e: e) if g is None else g

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return 0.5 * kappa * (A * gg(n_in[0]) - q[0]) ** 2

    return SmoothArrangement(
        Q=constant(dt * jnp.eye(1)),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=1,
        in_dim_N=1,
        out_f=lambda q, m_out: q[0:1],
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=U,
        label=label or f"bilateral_stage(A={A:g})",
    )


def feedback_wire(beta: float) -> SmoothArrangement:
    """The static feedback arrangement ``<R^1|R^1> -> <R^1|R^1>``.

    ``out_f = m_out`` (pass the report through), ``in_f = n_in - beta * m_out``
    (the inverting summing junction), ``U = 0``, trivial parameter.  A morphism
    of ``sarr`` (smooth ``in_f`` is all def.potlens asks) but *linear*, not a
    finset port-routing -- see the module docstring's honesty flag.
    """

    def in_f(q_w: Array, m_out: Array, n_in: Array) -> Array:
        return n_in - beta * m_out

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=1,
        in_dim_M=1,
        out_dim_N=1,
        in_dim_N=1,
        out_f=lambda q_w, m_out: m_out,
        in_f=in_f,
        U=lambda q_w, m_out, n_in: jnp.array(0.0),
        label=f"feedback_wire(beta={beta:g})",
    )


def black_amplifier(
    A: float,
    beta: float,
    kappa: float = 1.0,
    dt: float = 0.01,
    g: Optional[Callable[[Array], Array]] = None,
    unilateral: bool = True,
) -> SmoothArrangement:
    """``compose_seq(stage, feedback_wire(beta)) : <R^0|R^0> -> <R^1|R^1>``.

    Genuine composition in ``sarr``: the composite potential
    ``(kappa/2)(A g(e_src - beta v) - v - w)^2`` *emerges* from ``compose_seq``
    (it is not written by hand); the outer input port carries ``e_src`` at run
    time and the outer output reports ``v``.  ``beta = 0`` is the open loop.
    ``unilateral=False`` swaps in the reciprocal control stage.
    """
    box = (stage if unilateral else bilateral_stage)(A, kappa, dt, g)
    return replace(
        compose_seq(box, feedback_wire(beta)),
        label=f"black(A={A:g}, beta={beta:g}, {'uni' if unilateral else 'bi'})",
    )


def amp_state(v: float = 0.0, unilateral: bool = True) -> Array:
    """Initial state: ``(v, 0)`` for the unilateral stage, ``(v,)`` for the control."""
    return jnp.array([v, 0.0]) if unilateral else jnp.array([v])


# ---------------------------------------------------------------------------
# Running the loop: e_src as a run-time input (the control.py pattern).
# ---------------------------------------------------------------------------


def stepper(arr: SmoothArrangement):
    """A jitted one-tick map ``(state, e_src, xi_N) -> (out_v, react, state')``.

    One application of the ``Phiconf(arr)`` coalgebra: the reported output, the
    emitted covector field ``omega_N`` *evaluated at the driven input* -- the
    reaction on the source, which the caller may discard (the ``control.py``
    boundary situation) -- and the updated state.  ``xi_N`` is the covector the
    environment presents at the output port (the load); pass zeros for an
    unloaded output.
    """
    O = Phiconf(arr)

    def one(state: Array, e_src: Array, xi_N: Array):
        _, fiber = O.with_state(state).step(state)
        (out_n, omega_N), at_pos = fiber(_IN_POS)
        _, new_state = at_pos((xi_N, e_src))
        return out_n[0], omega_N(e_src)[0], new_state

    return jax.jit(one)


def settle(
    one, state: Array, e_src: float, steps: int, xi_N: Optional[Array] = None
) -> Tuple[Array, Array, Array]:
    """Iterate the one-tick map at fixed drive ``e_src``; return ``(v, react, state)``.

    The quasi-static primitive: hold the run-time input fixed and let the
    framework run reach its equilibrium.  ``react`` is the final emitted
    reaction covector on the drive port (0 at an exact equilibrium).
    """
    e = jnp.atleast_1d(jnp.asarray(e_src, dtype=float))
    xi = jnp.zeros(1) if xi_N is None else jnp.asarray(xi_N, dtype=float)
    v = react = None
    for _ in range(steps):
        v, react, state = one(state, e, xi)
    return v, react, state
