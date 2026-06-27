"""R^d generalization of the prism graph-wiring (sec.graph_laplacian; Phase 3).

The paper's prism wiring (``compose_graph``) couples scalar (R^1) vertices; this is
its generalization to R^vdim vertices plus an optional on-site potential, so a network
of 2-D gyros is built BY COMPOSITION -- closing the audit's "U written directly, not
built by the graph-wiring construction" gap. We check that:

* the R^2 graph-Laplacian potential ``sum_e (kappa/2)|q_tgt - q_src|^2`` EMERGES from
  the prism wiring on a hex topology, matching an independent hand-written oracle (it
  is *not* the U we write -- it comes out of ``compose_seq``);
* an on-site potential per vertex is added on top;
* the scalar case (vdim=1) is unchanged (backward compatibility).
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.gyroscope import hex_graph
from dap.wiring import compose_graph


def _spring_oracle(q, edges, kappa, vdim):
    """Independent oracle: sum_e (kappa/2)|q_tgt - q_src|^2 (Euclidean per edge)."""
    qv = np.asarray(q).reshape(-1, vdim)
    return sum(0.5 * kappa * float(np.sum((qv[t] - qv[s]) ** 2)) for (s, t) in edges)


def test_r2_graph_laplacian_emerges_on_hex():
    """The 2-D spring coupling is produced by compose_graph (vdim=2), not hand-written."""
    edges, in_g, out_g = hex_graph(3, 3)
    V = 9
    m, kappa = 1.3, 0.7
    arr = compose_graph(V, edges, m, kappa, vdim=2)
    # closed system, parameter R^{2V}, mass-sharp -- everything from composition:
    assert (arr.out_dim_M, arr.in_dim_M, arr.out_dim_N, arr.in_dim_N) == (0, 0, 0, 0)
    assert arr.Q.dim == 2 * V
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(2 * V))
        got = float(arr.U(q, jnp.zeros(0), jnp.zeros(0)))
        np.testing.assert_allclose(got, _spring_oracle(q, edges, kappa, 2), atol=1e-10)


def test_onsite_potential_is_added():
    """An on-site term (a per-gyro well) adds to the emergent spring potential."""
    edges, _, _ = hex_graph(2, 3)
    V = 6
    m, kappa, g = 1.0, 0.5, 0.3
    onsite = lambda qv: 0.5 * g * jnp.sum(qv ** 2)
    arr = compose_graph(V, edges, m, kappa, vdim=2, onsite=onsite)
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(2 * V))
        got = float(arr.U(q, jnp.zeros(0), jnp.zeros(0)))
        qv = np.asarray(q).reshape(V, 2)
        want = _spring_oracle(q, edges, kappa, 2) + sum(
            0.5 * g * float(np.sum(qv[v] ** 2)) for v in range(V)
        )
        np.testing.assert_allclose(got, want, atol=1e-10)


def test_scalar_graph_unchanged():
    """vdim=1 (the default) still gives the paper's scalar graph Laplacian."""
    edges = [(0, 1), (1, 2), (2, 0), (0, 2)]
    V = 3
    m, kappa = 1.3, 2.1
    arr = compose_graph(V, edges, m, kappa)  # default vdim=1, onsite=None
    rng = np.random.default_rng(2)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(V))
        got = float(arr.U(q, jnp.zeros(0), jnp.zeros(0)))
        want = sum(0.5 * kappa * float(q[t] - q[s]) ** 2 for (s, t) in edges)
        np.testing.assert_allclose(got, want, atol=1e-10)
