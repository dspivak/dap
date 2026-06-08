"""Tests for ``wiring`` (sec.spring_first_pass, lemma.lens_rr)."""

import jax.numpy as jnp
import numpy as np

from dap.wiring import chain_wire, finset_chain_wire


def test_finset_chain_wire():
    """ex.lens_finsetop: phi_K has out_f=(K) and in_f=identity on K elements."""
    out_f, in_f = finset_chain_wire(5)
    assert out_f == [4]               # selects K-th wire (0-indexed K-1)
    assert in_f == [0, 1, 2, 3, 4]    # identity on K of K+1


def test_chain_wire_out_and_in_f():
    """lemma.lens_rr image of phi_K (sec.spring_first_pass, line 2378-2382)."""
    K = 4
    W = chain_wire(K)
    assert W.out_dim_M == K and W.in_dim_M == K
    assert W.out_dim_N == 1 and W.in_dim_N == 1

    v_wire = jnp.zeros(0)
    m_out = jnp.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(W.out_f(v_wire, m_out), jnp.array([4.0]), atol=1e-10)
    np.testing.assert_allclose(
        W.in_f(v_wire, m_out, jnp.array([10.0])),
        jnp.array([10.0, 1.0, 2.0, 3.0]),
        atol=1e-10,
    )
