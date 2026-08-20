"""CMOS logic from non-reciprocal MOSFETs (EXTENSION -- not part of the paper).

A MOSFET is *non-reciprocal*: the drain current depends strongly on the gate
voltage, the gate current not at all on the drain voltage.  An arrangement's
port covectors are ``dU`` restricted to its input interface (eqn.covector_triple,
whose ``outp f`` summand is extended by ``0`` on the ``inpt N`` factor), so for a
*stateless* box the port Jacobian is ``d^2 U`` -- symmetric, hence reciprocal.
No stateless potential is a transistor.

The device below gets non-reciprocity from the **sharp** instead.  With
parameter ``Q = R^2`` and the constant, degenerate, non-symmetric reaction

    sharpR = [[0, 1],
              [0, 0]]              (def.rvect asks only for Q -> vect(Q^*, Q))

and potential

    U(q, eG, eD, eS) = (kappa/2) (vgs - q0 - q1)^2 + W(q0, vds),

where ``vgs = eG - eS`` and ``vds = eD - eS``, we get

    qdot_0 = -(sharpR dU)_0 = -dU/dq1 = kappa (vgs - q0 - q1),
    qdot_1 = 0.

The whole ``vds``-dependence of ``dU`` sits in the ``q0``-component, which
``sharpR`` reads as ``0``.  So the gate state tracks ``vgs`` with *no* drain
back-action -- exactly, for **any** channel model ``W``.  Started at ``q1 = 0``
the state settles at ``q0 = vgs`` and the drain returns ``I_D = dW/dvds``.

We call that the **kernel condition**: the drain-sensitive component of ``dU``
lies in ``ker(sharpR)``.  It is *not* the paper's ``unilateral``
(def.arrangement_terminology), which is the output-side condition
``sharpR_q((partial_q outp f)^T xi) = 0`` for covectors ``xi`` at the output.  Each
device below does satisfy that too -- but vacuously, because it has no output port
(``out_dim_N = 0``, so ``partial_q outp f = 0``), which has nothing to do with being
a transistor.  The kernel condition is input-side, and it is what buys
non-reciprocity here.

Two further structural facts, neither of them put in by hand:

* ``U`` depends on the terminal voltages only through differences, so
  ``dU/deG + dU/deD + dU/deS = 0``: the device's three port currents sum to
  zero.  Device KCL is shift-invariance of the potential.
* Composing devices with a *static* wiring in ``sarr`` (compose_seq) adds
  potentials and direct-sums sharps (prop.suboperads).  Since the devices have
  no output ports, ``dU_tot/dq_j = dU_j/dq_j``: each device's kernel condition
  survives composition unchanged, and the node states' updates are the sums of
  the currents drawn on them -- circuit KCL, emergent.

Contents: ``mosfet`` (the device), ``node``/``rail`` (a capacitor / an ideal
source, the latter a *frozen* state, ``sharpR = 0``), ``nand_gate`` (four
devices wired by ``nand_wire``), ``sr_latch`` (two ``nand_gate``s cross-coupled),
and ``run`` (a ``Phiconf`` rollout).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .functors import Phiconf
from .interpretation import trivial_omega
from .rvect import ReactiveVectorSpace, constant, trivial
from .wiring import compose_chain, compose_seq, tensor_arrangements

# The non-reciprocal reaction: degenerate and non-symmetric.  It reads only the
# q1-component of the covector and writes only the q0-component of the state, so
# the q0-component of dU -- which carries all the drain dependence -- is in its
# kernel (the "kernel condition" of the header; NOT def.arrangement_terminology's
# output-side ``unilateral``).
KERNEL_SHARP = jnp.array([[0.0, 1.0], [0.0, 0.0]])

_IN_POS = (jnp.zeros(0), trivial_omega(0))


# ---------------------------------------------------------------------------
# Devices.
# ---------------------------------------------------------------------------


def mosfet(
    kind: str = "n",
    kappa: float = 0.5,
    beta: float = 4.0,
    Vt: float = 0.3,
    Vsat: float = 0.1,
    sharpness: float = 20.0,
    label: str = "",
) -> SmoothArrangement:
    """A non-reciprocal MOSFET ``<R^3|R^0> `` system: ports ``(eG, eD, eS)``.

    Square-law saturation with a ``tanh`` triode knee:
    ``I_D = g(vgs) tanh(vds / Vsat)`` with ``g(v) = beta * relu_s(+-v - Vt)^2``
    (``+`` for ``kind="n"``, ``-`` for ``kind="p"``), realized as the potential
    ``W(q0, vds) = g(q0) * Vsat * logcosh(vds / Vsat)``.  The gate returns
    ``I_G = kappa (vgs - q0 - q1)``, which vanishes once the gate state settles.
    """
    if kind not in ("n", "p"):
        raise ValueError("kind must be 'n' or 'p'")
    sign = 1.0 if kind == "n" else -1.0

    def g(v: Array) -> Array:
        # smooth relu, sharp enough that a sub-threshold device is really off
        return beta * (jax.nn.softplus(sharpness * (sign * v - Vt)) / sharpness) ** 2

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        eG, eD, eS = n_in[0], n_in[1], n_in[2]
        vgs, vds = eG - eS, eD - eS
        gate = 0.5 * kappa * (vgs - q[0] - q[1]) ** 2
        x = vds / Vsat
        logcosh = jnp.logaddexp(x, -x) - jnp.log(2.0)
        return gate + g(q[0]) * Vsat * logcosh

    return SmoothArrangement(
        Q=constant(KERNEL_SHARP),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=0,
        in_dim_N=3,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=U,
        label=label or f"{kind}mos",
    )


def node(c: float = 0.02, label: str = "node") -> SmoothArrangement:
    """A capacitive node ``<R^0|R^1>``: reports its voltage, integrates its current.

    State is the node voltage, ``outp(q) = q``, ``sharpR = c`` (i.e. ``dt / C``),
    ``U = 0``.  The only covector reaching its state is ``(d_q outp)^T xi_N = xi_N``,
    the total current drawn on the node, so ``Phiconf`` steps ``q -> q - c * I``.
    """
    return SmoothArrangement(
        Q=constant(jnp.array([[c]])),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=1,
        in_dim_N=0,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: jnp.array(0.0),
        label=label,
    )


def rail(label: str = "rail") -> SmoothArrangement:
    """An ideal voltage source ``<R^0|R^1>``: a ``node`` with ``sharpR = 0``.

    A frozen state.  It reports its voltage and absorbs any current without
    moving, which is what makes it the circuit's power source: the composite
    potential is not bounded below, and this is where the energy enters.
    """
    return replace(node(0.0), Q=constant(jnp.zeros((1, 1))), label=label)


# ---------------------------------------------------------------------------
# The NAND gate: four devices, three voltage-reporters, one static wiring.
# ---------------------------------------------------------------------------

# tensor order of the inner boxes
_NAND_BOXES = ["P1", "P2", "N1", "N2", "out", "mid", "vdd"]


def nand_wire() -> SmoothArrangement:
    """The static wiring ``<R^12|R^3> -> <R^2|R^1>`` of a CMOS NAND.

    Stateless and potential-free (a ``wire`` in the sense of ex.tanks_connected).
    Reads the three reported node voltages ``(V_out, V_mid, V_dd)`` and the two
    outer inputs ``(a, b)``, and hands each device its ``(eG, eD, eS)``:
    ``P1 = (a, V_out, V_dd)``, ``P2 = (b, V_out, V_dd)``, ``N1 = (a, V_out, V_mid)``,
    ``N2 = (b, V_mid, 0)``.  Reports ``V_out``.
    """

    def out_f(q_w: Array, m_out: Array) -> Array:
        return m_out[0:1]  # V_out

    def in_f(q_w: Array, m_out: Array, n_in: Array) -> Array:
        v_out, v_mid, v_dd = m_out[0], m_out[1], m_out[2]
        a, b = n_in[0], n_in[1]
        zero = jnp.array(0.0)
        return jnp.stack(
            [
                a, v_out, v_dd,   # P1
                b, v_out, v_dd,   # P2
                a, v_out, v_mid,  # N1
                b, v_mid, zero,   # N2
            ]
        )

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=3,
        in_dim_M=12,
        out_dim_N=1,
        in_dim_N=2,
        out_f=out_f,
        in_f=in_f,
        U=lambda q_w, m_out, n_in: jnp.array(0.0),
        label="nand_wire",
    )


def nand_gate(c: float = 0.02, **fet) -> SmoothArrangement:
    """``nand_wire(P1, P2, N1, N2, node_out, node_mid, rail)``, by genuine composition.

    A ``<R^2|R^1>`` system: two input ports (the logic inputs) and one output
    port (the reported output-node voltage).  Its potential is *not* written by
    hand -- ``compose_seq`` sums the four device potentials at the port voltages
    the wiring delivers.
    """
    boxes = [
        mosfet("p", label="P1", **fet),
        mosfet("p", label="P2", **fet),
        mosfet("n", label="N1", **fet),
        mosfet("n", label="N2", **fet),
        node(c, "node_out"),
        node(c, "node_mid"),
        rail(),
    ]
    return replace(
        compose_seq(tensor_arrangements(boxes), nand_wire()), label="nand"
    )


def nand_state(v_out: float, v_mid: float = 0.0, vdd: float = 1.0) -> Array:
    """An initial state for ``nand_gate``: gate states zeroed, nodes/rail set.

    Layout is the tensor order ``_NAND_BOXES``: four ``R^2`` gate states, then
    ``V_out``, ``V_mid``, ``V_dd``.
    """
    return jnp.array([0.0] * 8 + [v_out, v_mid, vdd])


# ---------------------------------------------------------------------------
# The inverter, and chains of them (for depth-scaling studies).
# ---------------------------------------------------------------------------


def inverter_wire() -> SmoothArrangement:
    """The static wiring ``<R^6|R^2> -> <R^1|R^1>`` of a CMOS inverter.

    Reads ``(V_out, V_dd)`` and the outer input ``a``; hands ``P = (a, V_out, V_dd)``
    and ``N = (a, V_out, 0)``.  Reports ``V_out``.
    """

    def in_f(q_w: Array, m_out: Array, n_in: Array) -> Array:
        v_out, v_dd, a = m_out[0], m_out[1], n_in[0]
        return jnp.stack([a, v_out, v_dd, a, v_out, jnp.array(0.0)])

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=2,
        in_dim_M=6,
        out_dim_N=1,
        in_dim_N=1,
        out_f=lambda q_w, m_out: m_out[0:1],
        in_f=in_f,
        U=lambda q_w, m_out, n_in: jnp.array(0.0),
        label="inverter_wire",
    )


def inverter(c: float = 0.02, **fet) -> SmoothArrangement:
    """``inverter_wire(P, N, node_out, rail)``: a ``<R^1|R^1>`` system, ``Q = R^6``."""
    boxes = [
        mosfet("p", label="P", **fet),
        mosfet("n", label="N", **fet),
        node(c, "node_out"),
        rail(),
    ]
    return replace(compose_seq(tensor_arrangements(boxes), inverter_wire()), label="inv")


def inverter_state(v_out: float, vdd: float = 1.0) -> Array:
    """Initial state for ``inverter``: gate states zeroed, then ``V_out``, ``V_dd``."""
    return jnp.array([0.0] * 4 + [v_out, vdd])


def inverter_chain(K: int, c: float = 0.02, **fet) -> SmoothArrangement:
    """``chain_wire(K)(inv, ..., inv)``: ``K`` inverters in series, ``<R^1|R^1>``.

    Uses the paper's own chain wiring (sec.spring_first_pass), which feeds each
    box from the previous one's output and the first from the outer input.  Depth
    ``K``, so it isolates *logic depth* from *component count*.
    """
    return replace(
        compose_chain([inverter(c, **fet) for _ in range(K)]), label=f"inv_chain({K})"
    )


def chain_state(K: int, v_out: float = 0.5, vdd: float = 1.0) -> Array:
    return jnp.concatenate([inverter_state(v_out, vdd) for _ in range(K)])


# ---------------------------------------------------------------------------
# The SR latch: two NAND gates, cross-coupled.
# ---------------------------------------------------------------------------


def latch_wire() -> SmoothArrangement:
    """The static feedback wiring ``<R^4|R^2> -> <R^2|R^2>`` of an SR latch.

    Outer inputs ``(Sbar, Rbar)``, outer outputs ``(Q, Qbar)``.  Gate 1 is fed
    ``(Sbar, Qbar)``, gate 2 is fed ``(Rbar, Q)``.  The loop is well-posed
    because each gate's ``outp f`` reads only its own node state, so no output
    depends instantaneously on any input.
    """

    def in_f(q_w: Array, m_out: Array, n_in: Array) -> Array:
        Q, Qbar = m_out[0], m_out[1]
        Sbar, Rbar = n_in[0], n_in[1]
        return jnp.stack([Sbar, Qbar, Rbar, Q])

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=2,
        in_dim_M=4,
        out_dim_N=2,
        in_dim_N=2,
        out_f=lambda q_w, m_out: m_out,
        in_f=in_f,
        U=lambda q_w, m_out, n_in: jnp.array(0.0),
        label="latch_wire",
    )


def sr_latch(c: float = 0.02, **fet) -> SmoothArrangement:
    """``latch_wire(nand, nand)``: a ``<R^2|R^2>`` system, by genuine composition."""
    g = nand_gate(c, **fet)
    return replace(compose_seq(tensor_arrangements([g, g]), latch_wire()), label="sr_latch")


def latch_state(q: float, qbar: float, vdd: float = 1.0) -> Array:
    """Initial state for ``sr_latch``: the two gates' states concatenated."""
    return jnp.concatenate([nand_state(q, 0.0, vdd), nand_state(qbar, 0.0, vdd)])


# ---------------------------------------------------------------------------
# Rollout.
# ---------------------------------------------------------------------------


def stepper(arr: SmoothArrangement) -> Callable:
    """A jitted one-tick map ``(state, n_in, xi_N) -> (out_n, port_currents, state')``.

    One application of the ``Phiconf(arr)`` coalgebra at ``state``: the readout's
    position ``out_n``, the readout's covector field ``omega_N`` *evaluated at the
    driven inputs* -- i.e. the currents the circuit draws on its input ports,
    computed by the framework rather than by hand -- and the updated state.
    """
    O = Phiconf(arr)

    def one(state: Array, n_in: Array, xi_N: Array):
        _, fiber = O.with_state(state).step(state)
        (out_n, omega_N), at_pos = fiber(_IN_POS)
        _, new_state = at_pos((xi_N, n_in))
        return out_n, omega_N(n_in), new_state

    return jax.jit(one)


def run(
    arr: SmoothArrangement,
    state: Array,
    inputs: Callable[[int], Sequence[float]],
    steps: int,
    load: Sequence[float] | None = None,
) -> Tuple[Array, List[Array], List[Array]]:
    """Step ``Phiconf(arr)`` for ``steps`` ticks, driving the input ports.

    ``inputs(t)`` supplies the outer input voltages at tick ``t``; ``load`` is the
    covector presented at the output ports (the current drawn by whatever the
    circuit drives), default zero.  Returns ``(final_state, outputs, port_currents)``.
    """
    one = stepper(arr)
    xi_N = jnp.zeros(arr.out_dim_N) if load is None else jnp.asarray(load, float)
    outs: List[Array] = []
    currents: List[Array] = []
    for t in range(steps):
        out_n, cur, state = one(state, jnp.asarray(inputs(t), float), xi_N)
        outs.append(out_n)
        currents.append(cur)
    return state, outs, currents
