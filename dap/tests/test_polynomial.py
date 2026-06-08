"""Test polynomial constructors and ``cot`` (def.cot)."""

import jax.numpy as jnp
import numpy as np

from dap.polynomial import Cot, DirichletProduct, PolyMap, Yon
from dap.functors import cot_map, cot_object


def test_cot_object():
    assert cot_object(3) == Cot(3)


def test_dirichlet_product_factors():
    p = DirichletProduct(Cot(2), Yon(), Cot(1))
    assert len(p.factors) == 3
    assert p.factors[0] == Cot(2)


def test_cot_map_position_and_direction_are_pullback():
    """Functoriality witness for cot (def.cot, prop.cot_monoidal):
    cot(f).pos = f, cot(f).dir at x is (T_x f)^T.
    """
    # f: R^2 -> R^1, f(x) = (x[0] + 2*x[1],)
    def f(x):
        return jnp.array([x[0] + 2.0 * x[1]])

    fp = cot_map(f, src_dim=2, tgt_dim=1, label="f")
    x = jnp.array([3.0, 4.0])
    np.testing.assert_allclose(fp.on_position(x), jnp.array([11.0]), atol=1e-10)

    # (T f)^T applied to xi=(7,) is (7, 14).
    xi = jnp.array([7.0])
    pulled = fp.on_direction(x, xi)
    np.testing.assert_allclose(pulled, jnp.array([7.0, 14.0]), atol=1e-10)


def test_cot_functoriality_one_composition():
    """One composition example for cot: cot(g . f) = cot(g) . cot(f)."""

    def f(x):  # R -> R^2
        return jnp.array([x[0], 3.0 * x[0]])

    def g(y):  # R^2 -> R
        return jnp.array([y[0] - y[1]])

    cf = cot_map(f, 1, 2)
    cg = cot_map(g, 2, 1)
    cgf = cot_map(lambda x: g(f(x)), 1, 1)

    x = jnp.array([2.5])
    # Position: cot(g.f) and cg . cf agree on points.
    p1 = cgf.on_position(x)
    p2 = cg.on_position(cf.on_position(x))
    np.testing.assert_allclose(p1, p2, atol=1e-10)

    # Direction: cot(g.f) at x with xi at g(f(x)) equals cf-direction at x
    # of (cg-direction at f(x) of xi).
    xi = jnp.array([5.0])
    d1 = cgf.on_direction(x, xi)
    inter = cg.on_direction(cf.on_position(x), xi)
    d2 = cf.on_direction(x, inter)
    np.testing.assert_allclose(d1, d2, atol=1e-10)
