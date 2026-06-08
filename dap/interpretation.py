"""The smooth polynomial interpretation ``Phi'_interpsm`` (sec.smooth_interpretation).

This is the *integrator-free* half of the dynamics functor (def.smooth_setup,
thm.poly_interpretation): it sends a smooth adaptive arrangement to a
``cot(Q)``-parameterized polynomial map, computing everything that depends only
on the parameter *position* ``q in Q`` and not on how the state is stored or
updated. The two dynamics functors ``Phiconf``/``Phiphase`` share this factor;
they differ only in their integrator (integrator.py).

Given the parameter position ``q``, the interpretation provides:

* ``position_action(out_m, omega_M) -> (out_n, omega_N)``
  the readout, with ``out_n`` from ``out_f`` (eqn.outpn) and the output
  covector field ``omega_N`` from the pullback of ``omega_M`` along ``in_f``
  plus the differential of ``U`` (eqn.omegaprime).

* ``direction_action(out_m, omega_M, xi_N, in_n) -> (xi_Q, xi_M, in_m)``
  with ``in_m`` from ``in_f`` (eqn.inptm) and the covector triple
  ``(xi_Q, xi_M, xi_inN)`` the sum of three cotangent pullbacks (eqn.bigtheta);
  ``xi_Q in Q^*`` is the parameter covector the integrator consumes, ``xi_M``
  is returned to the source interface.

Covector fields are represented affinely as ``(A, b)`` pairs, ``omega(z) = A z + b``
(exact for the affine/quadratic running examples).
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement


def smooth_interpretation(
    arr: SmoothArrangement,
) -> Callable[[Array], Tuple[Callable, Callable]]:
    """Return ``q |-> (position_action, direction_action)`` for ``Phi'_interpsm(arr)``."""

    out_f, in_f, U = arr.out_f, arr.in_f, arr.U
    d_in_N = arr.in_dim_N

    def at_position(q: Array):
        # ---- readout: (out_m, omega_M) |-> (out_n, omega_N) ----
        def position_action(out_m: Array, omega_M):
            A_M, b_M = omega_M
            out_n = out_f(q, out_m)  # eqn.outpn

            # omega_N(in_n) = (in_f(q, out_m, -))^* omega_M + d(U(q, out_m, -))   (eqn.omegaprime)
            def omega_N_at(in_n: Array) -> Array:
                in_f_qm = lambda y: in_f(q, out_m, y)
                in_m_at = in_f_qm(in_n)
                omega_M_val = A_M @ in_m_at + b_M
                _, vjp_in_f = jax.vjp(in_f_qm, in_n)
                (pull_back,) = vjp_in_f(omega_M_val)
                dU_at = jax.grad(lambda y: U(q, out_m, y))(in_n)
                return pull_back + dU_at

            zero_in_n = jnp.zeros(d_in_N)
            b_N = omega_N_at(zero_in_n)
            A_N = jax.jacfwd(omega_N_at)(zero_in_n)
            return out_n, (A_N, b_N)

        # ---- backward: (out_m, omega_M, xi_N, in_n) |-> (xi_Q, xi_M, in_m) ----
        def direction_action(out_m: Array, omega_M, xi_N: Array, in_n: Array):
            A_M, b_M = omega_M
            in_m = in_f(q, out_m, in_n)  # eqn.inptm
            omega_M_val = A_M @ in_m + b_M

            # The three pullbacks summed in eqn.bigtheta.
            _, vjp_out_f = jax.vjp(lambda qq, mm: out_f(qq, mm), q, out_m)
            xi_Q_a, xi_M_a = vjp_out_f(xi_N)

            _, vjp_in_f = jax.vjp(
                lambda qq, mm, nn: in_f(qq, mm, nn), q, out_m, in_n
            )
            xi_Q_b, xi_M_b, _ = vjp_in_f(omega_M_val)

            dU_q, dU_m, _ = jax.grad(U, argnums=(0, 1, 2))(q, out_m, in_n)

            xi_Q = xi_Q_a + xi_Q_b + dU_q
            xi_M = xi_M_a + xi_M_b + dU_m
            return xi_Q, xi_M, in_m

        return position_action, direction_action

    return at_position
