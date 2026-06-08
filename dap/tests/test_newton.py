"""Newton's method as Phiconf of a sarr-scalar (sec.newton_warmup).

A scalar ``f = ((Q, sharpR^U), !, !, U) : I -> I`` with the position-dependent
reaction ``sharpR^U_q = (T_q dU)^{-1}`` (the inverse Hessian) has

    Phiconf(f) : q |-> q - (T_q dU)^{-1}(dU|_q)   (eqn.newton),

one Newton step for a critical point of ``U``.
"""

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phiconf
from dap.rvect import inverse_hessian


def _scalar(U_scalar, dim):
    """The sarr-scalar with potential U and Newton reaction (inverse Hessian)."""
    Q = inverse_hessian(U_scalar, dim)
    return SmoothArrangement(
        Q=Q, out_dim_M=0, in_dim_M=0, out_dim_N=0, in_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: U_scalar(q),
        label="newton",
    )


_IN_POS = (jnp.zeros(0), (jnp.zeros((0, 0)), jnp.zeros(0)))
_IN_DIR = (jnp.zeros(0), jnp.zeros(0))


def _newton_step(O, q):
    _, _, new_q = O.with_state(q).run_one(_IN_POS, lambda _o: _IN_DIR)
    return new_q


def test_newton_quadratic_lands_in_one_step():
    """Quadratic U: the inverse-Hessian step reaches the critical point in one step."""
    a = 2.0
    U = lambda q: (q[0] - a) ** 2 + 1.0
    O = Phiconf(_scalar(U, 1))
    new_q = _newton_step(O, jnp.array([5.0]))
    np.testing.assert_allclose(np.asarray(new_q), np.array([a]), atol=1e-10)


def test_newton_exp_step_formula_and_convergence():
    """U(x) = e^x - x: one step is x -> x - 1 + e^{-x}, and the orbit reaches 0."""
    U = lambda q: jnp.exp(q[0]) - q[0]
    O = Phiconf(_scalar(U, 1))

    # exact step formula
    new_q = _newton_step(O, jnp.array([2.0]))
    np.testing.assert_allclose(np.asarray(new_q), np.array([2.0 - 1.0 + np.exp(-2.0)]), atol=1e-10)

    # quadratic convergence to the minimizer x = 0
    q = jnp.array([2.0])
    for _ in range(40):
        q = _newton_step(O, q)
    assert abs(float(q[0])) < 1e-10
