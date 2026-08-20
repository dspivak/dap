"""Kuramoto oscillators via edge-box wiring (compositional, in ``sarr``).

The claim under test: the Kuramoto model

    theta_i' = omega_i + K * sum_{j ~ i} sin(theta_j - theta_i)

is the configuration dynamics ``Phiconf`` of a closed arrangement whose total
potential is

    U(theta) = - K * sum_{{i,j} in edges} cos(theta_i - theta_j)
               - sum_i omega_i * theta_i,

built by *genuine composition* in ``sarr`` (no hand-written total potential):

* one **oscillator box** per node ``i``: parameter ``Q_i = R`` with constant
  scalar sharp ``c`` (Phiconf reads ``c`` as the time step ``dt``; Phiphase
  reads it as the inverse inertia ``1/m``, giving the swing equation), output
  the reported angle ``theta_i`` broadcast to the ``deg(i)`` incident edge
  ports (the diagonal, as in ``wiring.harmonic_vertex``), potential the drift
  term ``-omega_i * theta_i``;
* one stateless **coupling box** per edge ``{i,j}``: trivial parameter ``R^0``,
  two input ports, no outputs, potential ``-K * cos(theta_i - theta_j)`` read
  off its inputs;
* one **wiring**: the static arrangement (image under ``R^-``, lem.lens_pow, of
  a finite-set lens exactly as ``wiring.graph_wire``) that routes each
  oscillator's broadcast out-ports bijectively onto the edge boxes' in-ports.

``kuramoto_arrangement`` is ``compose_seq(tensor(boxes), wire)``; the total
potential above *emerges* from ``compose_seq``'s writer-monad addition.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .rvect import diagonal, trivial
from .wiring import compose_seq, tensor_arrangements


# ---------------------------------------------------------------------------
# Port bookkeeping: incidences and the routing bijection.
# ---------------------------------------------------------------------------


def _incidences(
    num_oscillators: int, edges: Sequence[Tuple[int, int]]
) -> Tuple[List[int], List[int]]:
    """Degrees and the routing permutation for the Kuramoto wire.

    Oscillator ``v`` gets one out-port per incidence ``(v, e)`` with an edge
    ``e``; edge box ``e = {i, j}`` has two in-ports, slot 0 for ``theta_i`` and
    slot 1 for ``theta_j``. Out-ports are numbered oscillator-major (all of
    oscillator 0's ports, then oscillator 1's, ...), in-ports edge-major.
    Since every edge has exactly two incidences, both sides have ``2E`` ports
    and the routing ``in-port |-> out-port`` is a bijection ``perm`` with
    ``in_f(m_out) = m_out[perm]``.
    """
    for e, (i, j) in enumerate(edges):
        if i == j:
            raise ValueError(f"edge {e} = ({i}, {j}) is a self-loop; not allowed")
        if not (0 <= i < num_oscillators and 0 <= j < num_oscillators):
            raise ValueError(f"edge {e} = ({i}, {j}) out of range")

    out_port: Dict[Tuple[int, int], int] = {}
    degrees: List[int] = []
    idx = 0
    for v in range(num_oscillators):
        deg = 0
        for e, (i, j) in enumerate(edges):
            if v in (i, j):
                out_port[(v, e)] = idx
                idx += 1
                deg += 1
        degrees.append(deg)

    perm: List[int] = []
    for e, (i, j) in enumerate(edges):
        perm.append(out_port[(i, e)])
        perm.append(out_port[(j, e)])
    return degrees, perm


# ---------------------------------------------------------------------------
# The three kinds of box.
# ---------------------------------------------------------------------------


def kuramoto_oscillator(
    omega: float, degree: int, sharp_scale: float = 1.0
) -> SmoothArrangement:
    """The oscillator box ``Osc_i : <R^0|R^0> -> <R^0 | R^{deg}>``.

    Parameter ``Q_i = R`` holding the angle ``theta_i``, with constant scalar
    sharp ``sharp_scale`` (``dt`` for Phiconf, ``1/m`` for Phiphase). Output
    map the diagonal broadcast of ``theta_i`` to the ``degree`` incident edge
    ports; input map vacuous; potential the drift term ``-omega * theta_i``
    (so ``-dU/dtheta_i = omega``, the natural frequency).
    """

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.tile(q, degree)

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return -omega * q[0]

    return SmoothArrangement(
        Q=diagonal(jnp.full(1, sharp_scale)),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=degree,
        in_dim_N=0,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label=f"Osc(omega={omega:g})",
    )


def kuramoto_coupling(coupling: float) -> SmoothArrangement:
    """The stateless coupling box ``Edge_e : <R^0|R^0> -> <R^2 | R^0>``.

    Trivial parameter ``R^0``; two input ports carrying the two reported
    angles, no outputs; potential ``-coupling * cos(theta_i - theta_j)``.
    All the dynamics it contributes flows back through the wiring as the
    cotangent pullback of this potential to the two oscillators' parameters.
    """

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return -coupling * jnp.cos(n_in[0] - n_in[1])

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=0,
        in_dim_N=2,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label=f"Edge(K={coupling:g})",
    )


def kuramoto_wire(
    num_oscillators: int, edges: Sequence[Tuple[int, int]]
) -> SmoothArrangement:
    """The routing wire as a static arrangement (cf. ``wiring.graph_wire``).

    Image under ``R^-`` (lem.lens_pow) of the finite-set lens that feeds each
    edge box's two in-ports from its endpoints' broadcast out-ports. Source
    ``<R^{2E} | R^{2E}>`` (the tensor of all boxes' interfaces: ``2E``
    oscillator out-ports, ``2E`` edge in-ports), target the unit ``<R^0|R^0>``,
    trivial parameter, ``U = 0``. The routing is the bijection ``perm`` of
    ``_incidences``.
    """
    E = len(edges)
    _, perm = _incidences(num_oscillators, edges)
    perm = jnp.asarray(perm, dtype=int)

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return m_out[perm]

    def U(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.array(0.0)

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=2 * E,
        in_dim_M=2 * E,
        out_dim_N=0,
        in_dim_N=0,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label=f"kuramoto_wire(N={num_oscillators}, E={E})",
    )


# ---------------------------------------------------------------------------
# The composite arrangement.
# ---------------------------------------------------------------------------


def kuramoto_arrangement(
    omegas: Sequence[float],
    edges: Sequence[Tuple[int, int]],
    coupling: float,
    sharp_scale: float = 1.0,
) -> SmoothArrangement:
    """The closed Kuramoto arrangement, by genuine composition in ``sarr``:

        compose_seq( tensor(Osc_1, ..., Osc_N, Edge_1, ..., Edge_E),
                     kuramoto_wire(N, edges) )  :  <R^0|R^0> -> <R^0|R^0>,

    with parameter ``R^N`` (the angles, in oscillator order; the stateless
    edge boxes contribute nothing) and sharp ``sharp_scale * I``. The total
    potential

        U(theta) = - sum_i omegas[i] * theta_i
                   - coupling * sum_{(i,j) in edges} cos(theta_i - theta_j)

    emerges from ``compose_seq`` -- it is not written anywhere by hand. Then
    ``Phiconf`` (with ``sharp_scale = dt``) steps forward-Euler Kuramoto, and
    ``Phiphase`` (with ``sharp_scale = 1/m``) steps the swing equation
    ``m * ddot(theta)_i = omega_i + K sum_j sin(theta_j - theta_i)`` by
    semi-implicit Euler (unit time step, as in the graph-wave test).
    """
    N = len(omegas)
    degrees, _ = _incidences(N, edges)
    boxes = [
        kuramoto_oscillator(float(omegas[v]), degrees[v], sharp_scale)
        for v in range(N)
    ] + [kuramoto_coupling(coupling) for _ in edges]
    wired = compose_seq(tensor_arrangements(boxes), kuramoto_wire(N, edges))
    return replace(
        wired, label=f"kuramoto(N={N}, E={len(edges)}, K={coupling:g})"
    )
