"""Gradient descent and backpropagation via ``Phiconf`` (sec.dl_warmup).

A feedforward network is a smooth parameterized map ``F : Q x R^m -> R^n``
(eqn.model_map), regarded as a smooth adaptive arrangement

    f : <R^0 / R^m> -> <R^0 / R^n>

with ``out_f = F``, trivial input map, and zero potential (the learning signal
comes from whatever the network is wired into, not from ``U``). Under the
configuration functor ``Phiconf`` this is the learner ``lrn(F)`` of eqn.lrn:
a coalgebra on state ``|Q|`` whose backward pass pulls an output covector ``xi``
back to a parameter covector ``xi_Q`` and an input covector ``xi_M``
(eqn.dl_pullback), and whose state update descends, ``q |-> q - sharpR_q(xi_Q)``
(eqn.dl_gradient_update). With ``sharpR_q = eta_LR * sharpEuc`` (eqn.learning_sharp)
this is vanilla gradient descent at rate ``eta_LR``, and backpropagation is the
chain rule packaged as the functoriality of ``Phiconf``.

The training loop (``train``) realizes the *environment* of ex.training_data:
the loss is a 2-ary closed system comparing a prediction ``y`` to a label
``lambda``, and the environment streams data ``(x, lambda)`` and seeds the
output covector ``xi = dU(-, lambda)|_y`` on each step.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .functors import Phiconf
from .interpretation import smooth_interpretation, trivial_omega
from .rvect import ReactiveVectorSpace


# ---------------------------------------------------------------------------
# A parameterized map as a smooth adaptive arrangement (sec.dl_warmup).
# ---------------------------------------------------------------------------


def parameterized_map(
    F: Callable[[Array, Array], Array],
    Q: ReactiveVectorSpace,
    in_dim: int,
    out_dim: int,
    label: str = "net",
) -> SmoothArrangement:
    """The arrangement ``f`` of a feedforward network ``F : Q x R^m -> R^n``.

    The data ``x`` lives in the output port ``out_M = R^m`` and the prediction
    ``y = F(q, x)`` in ``out_N = R^n``; the input ports are trivial and ``U = 0``
    (eqn.model_map and the paragraph after it).
    """

    return SmoothArrangement(
        Q=Q,
        out_dim_M=in_dim,
        in_dim_M=0,
        out_dim_N=out_dim,
        in_dim_N=0,
        out_f=lambda q, x: F(q, x),
        in_f=lambda q, x, n_in: jnp.zeros(0),
        U=lambda q, x, n_in: jnp.array(0.0),
        label=label,
    )


def _trivial_omega(d: int):
    """The (trivial) zero covector field on ``R^d``; for the network ``d = 0``."""
    return trivial_omega(d)


def forward_backward(
    arr: SmoothArrangement, q: Array, x: Array, xi_out: Array
) -> Tuple[Array, Array, Array]:
    """One forward+backward pass of ``lrn(F)`` at state ``q`` on input ``x``.

    Returns ``(y, xi_Q, xi_M)`` where ``y = F(q, x)`` is the prediction, and
    ``(xi_Q, xi_M) = (T_{(q,x)} F)^T xi_out`` is the cotangent pullback of the
    incoming output covector (eqn.dl_pullback). The parameter covector ``xi_Q``
    is the backpropagated gradient that drives the descent step.
    """

    interp = smooth_interpretation(arr)
    position_action, direction_action = interp(q)
    omega_M = _trivial_omega(arr.in_dim_M)
    out_n, _ = position_action(x, omega_M)
    xi_Q, xi_M, _ = direction_action(x, omega_M, xi_out, jnp.zeros(arr.in_dim_N))
    return out_n, xi_Q, xi_M


# ---------------------------------------------------------------------------
# Loss (a 2-ary closed system, sec.dl_warmup) -- as covector seed + scalar.
# ---------------------------------------------------------------------------


def squared_error_grad(y: Array, lam: Array) -> Array:
    """``dU(-, lambda)|_y`` for the regression loss ``U(y, lambda) = 1/2 ||y - lambda||^2``."""
    return y - lam


def squared_error(y: Array, lam: Array) -> Array:
    """The regression loss ``U(y, lambda) = 1/2 ||y - lambda||^2``."""
    return 0.5 * jnp.sum((y - lam) ** 2)


# ---------------------------------------------------------------------------
# The training loop: Phiconf(net) closed by a data-streaming environment.
# ---------------------------------------------------------------------------


def train(
    arr: SmoothArrangement,
    init_q: Array,
    data: Sequence[Tuple[Array, Array]],
    *,
    grad_loss: Callable[[Array, Array], Array] = squared_error_grad,
    loss: Callable[[Array, Array], Array] = squared_error,
    steps: int = None,
) -> Tuple[Array, List[float]]:
    """Run gradient descent by stepping the ``Phiconf`` coalgebra of ``arr``.

    This is the closed system of ex.training_data: at each step the environment
    emits a datum ``(x, lambda)``, the network outputs ``y = F(q, x)``, the loss
    seeds the output covector ``xi = grad_loss(y, lambda)``, which is pulled back
    to ``xi_Q`` and descends the parameter by eqn.dl_gradient_update. Returns the
    final parameter and the per-step loss history.
    """

    O = Phiconf(arr)
    q = jnp.asarray(init_q, dtype=float)
    history: List[float] = []
    n_steps = len(data) if steps is None else steps
    omega_M = _trivial_omega(arr.in_dim_M)

    for t in range(n_steps):
        x, lam = data[t % len(data)]
        x = jnp.asarray(x, dtype=float)
        lam = jnp.asarray(lam, dtype=float)
        in_pos = (x, omega_M)
        org = O.with_state(q)
        out_pos, _out_dir, new_q = org.run_one(
            in_pos,
            lambda op, lam=lam: (grad_loss(op[0], lam), jnp.zeros(arr.in_dim_N)),
        )
        history.append(float(loss(out_pos[0], lam)))
        q = new_q

    return q, history
