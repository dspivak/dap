"""Continuous Hopfield network via neuron/synapse-box wiring (compositional, in ``sarr``).

The claim under test: the continuous Hopfield network (Hopfield 1984, units
``R_i = C_i = 1``)

    u_i' = -u_i + sum_j W_ij g(u_j) + I_i,        g = tanh,  W symmetric,

is the configuration dynamics ``Phiconf`` of a closed arrangement built by
*genuine composition* in ``sarr`` (no hand-written total potential):

* one **neuron box** per unit ``i``: parameter ``Q_i = R`` holding the internal
  state ``u_i``, with *state-dependent* sharp

      sharpR_u = dt / g'(u) = dt * cosh(u)^2  > 0

  (positive-definite everywhere since ``g' = sech^2 > 0``; the same species of
  position-dependent reaction as ``rvect.inverse_hessian``). Output map the
  diagonal broadcast of the *reported activation* ``V_i = g(u_i)`` to the
  ``deg(i)`` incident synapse ports; input map vacuous; potential the leak and
  bias terms

      U_i(u) = u * g(u) - log cosh(u) - I_i * g(u),

  whose derivative is ``(u - I_i) * g'(u)`` (the antiderivative of ``u g'(u)``
  is ``u g(u) - log cosh(u)``; the bias is carried as the linear-in-``V`` term
  ``-I_i g(u)``).

* one stateless **synapse box** per unordered pair ``{i, j}`` with
  ``W_ij != 0``: trivial parameter ``R^0``, two input ports carrying the two
  reported activations, no outputs, potential ``-W_ij * V_i * V_j``.

* one **wiring**: the static arrangement (image under ``R^-``, lem.lens_pow,
  of a finite-set lens exactly as ``kuramoto.kuramoto_wire``) routing each
  neuron's broadcast out-ports bijectively onto the synapse boxes' in-ports.

``hopfield_arrangement`` is ``compose_seq(tensor(boxes), wire)``; the total
potential

    U(u) = sum_i [u_i g(u_i) - log cosh(u_i) - I_i g(u_i)]
           - sum_{i<j} W_ij g(u_i) g(u_j)

*emerges* from ``compose_seq``'s writer-monad addition. Then, since the
composite is closed, the parameter covector is ``xi_Q = dU/du`` and one
``Phiconf`` step is

    u_i  |->  u_i - (dt / g'(u_i)) * g'(u_i) * (u_i - I_i - sum_j W_ij g(u_j))
          =  u_i + dt * (-u_i + sum_j W_ij g(u_j) + I_i),

forward Euler for the Hopfield ODE: the ``g'`` produced by the chain rule
through the reported activation is cancelled exactly by the state-dependent
sharp. Moreover the composite potential *is* Hopfield's energy function: with
``V = g(u)``,

    integral_0^V g^{-1}(s) ds = V artanh(V) + (1/2) log(1 - V^2)
                              = u g(u) - log cosh(u),

so ``U(u) = E_Hopfield(g(u))`` exactly (additive constant zero), and energy
decrease along trajectories is inherited from the descent structure of the
configuration integrator (positive-definite sharp), up to the usual
forward-Euler ``O(dt^2)`` caveat.

Design choice: the external input ``I_i`` is a *linear term in the neuron
potential* (``-I_i g(u_i)``), not a run-time input. This keeps the arrangement
closed, so Hopfield's energy theorem is inherited directly from ``Phiconf``'s
descent structure; a run-time input would open the interface and make the
energy statement conditional on the input being held constant.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import SmoothArrangement
from .rvect import ReactiveVectorSpace, trivial
from .wiring import compose_seq, parallel_arrangements


def _log_cosh(u: Array) -> Array:
    """Numerically stable ``log cosh(u) = logaddexp(u, -u) - log 2``."""
    return jnp.logaddexp(u, -u) - jnp.log(2.0)


# ---------------------------------------------------------------------------
# Port bookkeeping: incidences and the routing bijection (as in kuramoto.py).
# ---------------------------------------------------------------------------


def _incidences(
    num_neurons: int, edges: Sequence[Tuple[int, int]]
) -> Tuple[List[int], List[int]]:
    """Degrees and the routing permutation for the Hopfield wire.

    Neuron ``v`` gets one out-port per incidence ``(v, e)`` with a synapse
    ``e``; synapse box ``e = {i, j}`` has two in-ports, slot 0 for ``V_i`` and
    slot 1 for ``V_j``. Out-ports are numbered neuron-major, in-ports
    synapse-major. Every synapse has exactly two incidences, so both sides
    have ``2E`` ports and the routing ``in-port |-> out-port`` is a bijection
    ``perm`` with ``in_f(m_out) = m_out[perm]``.
    """
    for e, (i, j) in enumerate(edges):
        if i == j:
            raise ValueError(f"synapse {e} = ({i}, {j}) is a self-loop; not allowed")
        if not (0 <= i < num_neurons and 0 <= j < num_neurons):
            raise ValueError(f"synapse {e} = ({i}, {j}) out of range")

    out_port: Dict[Tuple[int, int], int] = {}
    degrees: List[int] = []
    idx = 0
    for v in range(num_neurons):
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


def hopfield_neuron(bias: float, degree: int, dt: float) -> SmoothArrangement:
    """The neuron box ``Neu_i : <R^0|R^0> -> <R^0 | R^{deg}>``.

    Parameter ``Q_i = R`` holding the internal state ``u_i``, with the
    state-dependent sharp ``sharpR_u = dt / g'(u) = dt * cosh(u)^2``
    (positive for every ``u``, since ``g' = sech^2 > 0``). Output map the
    diagonal broadcast of the reported activation ``V_i = g(u_i) = tanh(u_i)``
    to the ``degree`` incident synapse ports; input map vacuous; potential
    the leak-plus-bias term ``u g(u) - log cosh(u) - bias * g(u)``, with
    ``d/du [u g(u) - log cosh(u)] = u g'(u)`` (the leak) and
    ``d/du [-bias * g(u)] = -bias * g'(u)`` (the external input).
    """

    def sharp_fn(q: Array) -> Array:
        return (dt * jnp.cosh(q[0]) ** 2)[None, None]  # dt / g'(u), a (1,1) matrix

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.tile(jnp.tanh(q), degree)  # report V = g(u), broadcast

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        u = q[0]
        return u * jnp.tanh(u) - _log_cosh(u) - bias * jnp.tanh(u)

    return SmoothArrangement(
        Q=ReactiveVectorSpace(dim=1, sharp_fn=sharp_fn),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=degree,
        in_dim_N=0,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label=f"Neu(I={bias:g})",
    )


def hopfield_synapse(weight: float) -> SmoothArrangement:
    """The stateless synapse box ``Syn_e : <R^0|R^0> -> <R^2 | R^0>``.

    Trivial parameter ``R^0``; two input ports carrying the two reported
    activations ``V_i, V_j``, no outputs; potential ``-weight * V_i * V_j``.
    All the dynamics it contributes flows back through the wiring as the
    cotangent pullback of this potential to the two neurons' parameters
    (which is where the chain-rule factor ``g'(u)`` arises).
    """

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return -weight * n_in[0] * n_in[1]

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=0,
        in_dim_N=2,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label=f"Syn(W={weight:g})",
    )


def hopfield_wire(
    num_neurons: int, edges: Sequence[Tuple[int, int]]
) -> SmoothArrangement:
    """The routing wire as a static arrangement (cf. ``kuramoto.kuramoto_wire``).

    Image under ``R^-`` (lem.lens_pow) of the finite-set lens that feeds each
    synapse box's two in-ports from its endpoints' broadcast out-ports. Source
    ``<R^{2E} | R^{2E}>``, target the unit ``<R^0|R^0>``, trivial parameter,
    ``U = 0``. The routing is the bijection ``perm`` of ``_incidences``.
    """
    E = len(edges)
    _, perm = _incidences(num_neurons, edges)
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
        label=f"hopfield_wire(N={num_neurons}, E={E})",
    )


# ---------------------------------------------------------------------------
# Balanced tensor fold (associativity only; needed for hundreds of boxes).
# ---------------------------------------------------------------------------


def _tensor_balanced(parts: Sequence[SmoothArrangement]) -> SmoothArrangement:
    """The monoidal product ``Part_1 (x) ... (x) Part_K``, folded as a balanced tree.

    Same composite as ``wiring.tensor_arrangements`` (the monoidal product is
    associative, and ``parallel_arrangements`` concatenates ports in order, so
    every association order yields the same flat port ordering); the balanced
    fold keeps the closure-nesting depth at ``O(log K)`` instead of ``O(K)``,
    which a Hebbian network with hundreds of synapse boxes needs to stay within
    Python's recursion limit. Each pairing is the framework's own
    ``parallel_arrangements`` -- nothing about the composition is hand-rolled.
    """
    if len(parts) < 1:
        raise ValueError("_tensor_balanced: need at least one factor")
    layer = list(parts)
    while len(layer) > 1:
        nxt = []
        for k in range(0, len(layer) - 1, 2):
            nxt.append(parallel_arrangements(layer[k], layer[k + 1]))
        if len(layer) % 2 == 1:
            nxt.append(layer[-1])
        layer = nxt
    return layer[0]


# ---------------------------------------------------------------------------
# The composite arrangement.
# ---------------------------------------------------------------------------


def hopfield_edges(W: np.ndarray) -> List[Tuple[int, int]]:
    """The unordered pairs ``i < j`` with ``W[i, j] != 0`` (one synapse box each)."""
    N = W.shape[0]
    return [(i, j) for i in range(N) for j in range(i + 1, N) if W[i, j] != 0.0]


def hopfield_arrangement(
    W: np.ndarray, biases: Sequence[float], dt: float
) -> SmoothArrangement:
    """The closed Hopfield arrangement, by genuine composition in ``sarr``:

        compose_seq( tensor(Neu_1, ..., Neu_N, Syn_1, ..., Syn_E),
                     hopfield_wire(N, edges) )  :  <R^0|R^0> -> <R^0|R^0>,

    with parameter ``R^N`` (the internal states ``u``, in neuron order; the
    stateless synapse boxes contribute nothing) and block-diagonal
    state-dependent sharp ``sharpR_u = diag(dt * cosh(u_i)^2)``. ``W`` must be
    symmetric with zero diagonal; one synapse box per unordered pair with
    ``W_ij != 0``. The total potential

        U(u) = sum_i [u_i g(u_i) - log cosh(u_i) - biases[i] g(u_i)]
               - sum_{i<j} W_ij g(u_i) g(u_j)

    emerges from ``compose_seq`` -- it is not written anywhere by hand -- and
    equals Hopfield's energy ``E(V)`` at ``V = g(u)`` exactly. ``Phiconf``
    steps forward-Euler Hopfield: ``u |-> u + dt * (-u + W g(u) + I)``.
    """
    W = np.asarray(W, dtype=float)
    N = W.shape[0]
    if W.shape != (N, N):
        raise ValueError(f"W must be square, got {W.shape}")
    if not np.allclose(W, W.T):
        raise ValueError("W must be symmetric (Hopfield's hypothesis)")
    if not np.all(np.diag(W) == 0.0):
        raise ValueError("W must have zero diagonal (one box per unordered pair i < j)")
    if len(biases) != N:
        raise ValueError(f"biases has length {len(biases)}, expected {N}")

    edges = hopfield_edges(W)
    degrees, _ = _incidences(N, edges)
    boxes = [
        hopfield_neuron(float(biases[v]), degrees[v], dt) for v in range(N)
    ] + [hopfield_synapse(float(W[i, j])) for (i, j) in edges]
    wired = compose_seq(_tensor_balanced(boxes), hopfield_wire(N, edges))
    return replace(wired, label=f"hopfield(N={N}, E={len(edges)}, dt={dt:g})")


# ---------------------------------------------------------------------------
# Hebbian storage.
# ---------------------------------------------------------------------------


def hebbian_weights(patterns: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Hebbian weights ``W = (gain/N) sum_p xi_p xi_p^T`` with zero diagonal.

    ``patterns`` is ``(P, N)`` with entries ``+-1``. At ``gain = 1`` the
    continuous network ``u' = -u + W tanh(u)`` has *no* retrieval attractors:
    the mean-field overlap equation is ``m = tanh(a m)`` with
    ``a ~ (N-1)/N < 1``, so only ``m = 0`` solves it and every stored pattern
    decays to the origin. Retrieval requires effective gain ``> 1``; we carry
    it as a scalar on ``W`` (equivalently a gain in ``g``), which changes no
    formula elsewhere -- ``W`` is an arbitrary symmetric matrix throughout.
    """
    patterns = np.asarray(patterns, dtype=float)
    P, N = patterns.shape
    W = (gain / N) * (patterns.T @ patterns)
    np.fill_diagonal(W, 0.0)
    return W
