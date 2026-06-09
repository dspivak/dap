"""Leapfrog (velocity Verlet) as ONE instance of general ``org^(2)`` (org2.py).

Leapfrog is symplectic and evaluates the force twice per step, so it is a
two-stage ``[p,q]^{∘2}``-coalgebra (rmk.org_N) — an ``org2.OrgMorphism2``, not a
hardcoded stepper. It reuses the same polynomial interpretation as
``Phiconf``/``Phiphase``; only the integrator is two-stage:

    round 1 at q   : read force ξ_Q(q),  half-kick + drift to q' = q + sharpR_q(ξ - ½ξ_Q(q))
    round 2 at q'  : read force ξ_Q(q'), half-kick to ξ' = ξ½ - ½ξ_Q(q')

Round 2 is built as a genuine 1-stage ``org.OrgMorphism`` (the inner coalgebra
that round 1 lands in), via the existing ``functors.Phi`` with a *half-kick*
integrator. So leapfrog is literally "round 1 → an inner org-morphism → round 2",
which is exactly the substitution ``[p,q] ◁ [p,q]``.

Closed/physics case (constant ``sharpR``); ``run_one`` drives the closed loop.
"""

from __future__ import annotations

import jax.numpy as jnp

from .arrangement import SmoothArrangement
from .functors import Phi, phi_box_poly
from .integrator import Integrator
from .interpretation import smooth_interpretation
from .org2 import OrgMorphism2
from .polynomial import PolyMap


def _half_kick_integrator() -> Integrator:
    """Round-2 integrator: keep the (already drifted) position, half-kick the momentum.

    A 1-stage ``cot``-integrator whose state is ``(q', ξ½)`` and whose update on an
    incoming force ``ξ_Q`` is ``(q', ξ½ - ½ ξ_Q)``.
    """
    return Integrator(
        init=lambda Q: (jnp.zeros(Q.dim), jnp.zeros(Q.dim)),
        position=lambda s: s[0],
        step=lambda Q, s, xi_Q: (s[0], s[1] - 0.5 * xi_Q),
        label="halfkick",
    )


def Phileap(arr: SmoothArrangement) -> OrgMorphism2:
    """Leapfrog dynamics of a (closed) arrangement, as an ``org^(2)`` morphism."""
    Q = arr.Q
    interp = smooth_interpretation(arr)
    src_p = phi_box_poly(arr.out_dim_M, arr.in_dim_M)
    tgt_p = phi_box_poly(arr.out_dim_N, arr.in_dim_N)
    half_kick = _half_kick_integrator()

    def step(state):
        q, xi = state
        position_action, direction_action = interp(q)  # round-1 interpretation at q

        # round-1 action a^β(state), used only when composing (parallel/then_static)
        def act_positions(in_pos):
            out_m, omega_M = in_pos
            return position_action(out_m, omega_M)

        def act_directions(in_pos, in_dir):
            out_m, omega_M = in_pos
            xi_N, in_n = in_dir
            _, xi_M, in_m = direction_action(out_m, omega_M, xi_N, in_n)
            return (xi_M, in_m)

        act = PolyMap(src_p, tgt_p, act_positions, act_directions, label="leap")

        def fiber(in_pos):
            out_m, omega_M = in_pos
            out_pos = position_action(out_m, omega_M)  # emit at q

            def at_pos(in_dir):
                xi_N, in_n = in_dir
                xi_Q1, xi_M, in_m = direction_action(out_m, omega_M, xi_N, in_n)
                xi_half = xi - 0.5 * xi_Q1                       # half-kick
                q2 = q + Q.apply_sharp(q, xi_half)               # drift
                inner = Phi(arr, half_kick).with_state((q2, xi_half))  # round 2 (org.OrgMorphism)
                return (xi_M, in_m), inner

            return out_pos, at_pos

        return act, fiber

    return OrgMorphism2(src_p, tgt_p, (jnp.zeros(Q.dim), jnp.zeros(Q.dim)), step)
