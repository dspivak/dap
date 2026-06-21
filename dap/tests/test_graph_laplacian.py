"""Graph Laplacian on a directed graph via prism wiring (sec.graph_laplacian).

The closed system ``wire_G = R^{varphi_G}((Part_v)_v)`` (eqn.graph_wire) is built
by *genuine composition* in ``sarr`` -- the prism wiring ``graph_wire`` (image of
``varphi_G`` under ``R^-``, eqn.prism_f) composed with the per-vertex harmonic
particles ``Part_v``. We check:

* the composite potential ``sum_e (kappa/2)(q_tgt - q_src)^2`` (eqn.graph_potential)
  EMERGES from the wiring -- it is not written by hand;
* the prism input map ``inpt f`` is a bijection on ``R^E`` (eqn.prism_f);
* ``Phiphase(wire_G)`` is the discrete graph-wave equation (prop.graph_laplacian,
  eqn.graph_laplacian_dynamics) and ``Phiconf(wire_G)`` the graph-heat equation
  (rmk.graph_heat), both against an independent Laplacian oracle.

The test graph mixes in- and out-degrees and has a non-identity prism permutation,
so it exercises the broadcast (``outp f_v``) and routing (``inpt f``) non-trivially.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.interpretation import trivial_omega
from dap.functors import Phiconf, Phiphase
from dap.wiring import compose_graph, graph_wire

# directed triangle 0->1->2->0 with an extra chord 0->2:
#   out-degrees (2,1,1), in-degrees (1,1,2); the 0--2 bond is doubled.
V = 3
EDGES = [(0, 1), (1, 2), (2, 0), (0, 2)]

_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def _laplacian(num_vertices, edges):
    """Independent oracle: combinatorial (symmetrized) Laplacian, +1 per directed edge."""
    L = np.zeros((num_vertices, num_vertices))
    for (s, t) in edges:
        L[s, s] += 1.0
        L[t, t] += 1.0
        L[s, t] -= 1.0
        L[t, s] -= 1.0
    return L


def test_graph_potential_emerges_from_wiring():
    """eqn.graph_potential: the closed potential is the edge sum, produced by compose_seq."""
    m, kappa = 1.3, 2.1
    arr = compose_graph(V, EDGES, m, kappa)
    assert (arr.out_dim_M, arr.in_dim_M, arr.out_dim_N, arr.in_dim_N) == (0, 0, 0, 0)  # closed
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(V))
        got = float(arr.U(q, jnp.zeros(0), jnp.zeros(0)))
        want = sum(0.5 * kappa * float(q[t] - q[s]) ** 2 for (s, t) in EDGES)
        np.testing.assert_allclose(got, want, atol=1e-10)


def test_prism_routing_is_a_bijection():
    """eqn.prism_f: ``inpt f : sum in(v) -> sum out(v)`` is a bijection on R^E, so the
    prism routes the |E| out-ports onto the |E| in-ports with nothing lost or doubled."""
    wire = graph_wire(V, EDGES)
    E = len(EDGES)
    assert (wire.out_dim_M, wire.in_dim_M, wire.out_dim_N, wire.in_dim_N) == (E, E, 0, 0)
    m_out = jnp.arange(E, dtype=float)  # tag each out-port by its position
    routed = np.asarray(wire.in_f(jnp.zeros(0), m_out, jnp.zeros(0)))
    assert sorted(routed.tolist()) == list(range(E))  # a permutation of {0,...,E-1}


def test_graph_wave_is_laplacian_dynamics():
    """prop.graph_laplacian / eqn.graph_laplacian_dynamics: m * ddot q = -kappa * L q(center)."""
    m, kappa = 1.5, 0.9
    O = Phiphase(compose_graph(V, EDGES, m, kappa))
    L = _laplacian(V, EDGES)

    rng = np.random.default_rng(42)
    state = (jnp.asarray(rng.standard_normal(V)), jnp.zeros(V))
    traj = [np.asarray(state[0])]
    for _ in range(12):
        _, _, state = O.with_state(state).run_one(_IN_POS, _TRIV)
        traj.append(np.asarray(state[0]))
    a = np.stack(traj)
    for t in range(len(a) - 2):
        ddot_q = a[t + 2] - 2.0 * a[t + 1] + a[t]
        residual = m * ddot_q + kappa * (L @ a[t + 1])
        np.testing.assert_allclose(residual, np.zeros(V), atol=1e-10)


def test_graph_heat_is_laplacian_descent():
    """rmk.graph_heat: the SAME wire_G under Phiconf descends by q(t+1) = q(t) - (kappa/m) L q(t)."""
    m, kappa = 1.0, 0.2
    O = Phiconf(compose_graph(V, EDGES, m, kappa))
    L = _laplacian(V, EDGES)

    rng = np.random.default_rng(7)
    q = jnp.asarray(rng.standard_normal(V))
    for _ in range(50):
        q_prev = np.asarray(q)
        _, _, q = O.with_state(q).run_one(_IN_POS, _TRIV)
        want = q_prev - (kappa / m) * (L @ q_prev)
        np.testing.assert_allclose(np.asarray(q), want, atol=1e-10)
