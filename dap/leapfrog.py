"""Leapfrog (velocity Verlet) as an ``org^(2)`` coalgebra (rmk.org_N).

Phase-space Euler (``Phiphase``) is explicit Euler on a Hamiltonian flow: it is
never symplectic, so the energy grows and the wave blows up. Leapfrog is the
symplectic, second-order integrator. It is *multi-stage* -- it evaluates the
force **twice** per step -- so it does not land in ``org`` but in ``org^(2)``,
the bicategory whose homs are ``[p,p']^{∘2}``-coalgebras conjectured in
rmk.org_N. Each macro-tick is two emit/receive rounds:

    state (q, ξ)  --emit q --> [force ξ_Q(q)]  --emit q' --> [force ξ_Q(q')]  --> (q', ξ')

with the velocity-Verlet update

    ξ½ = ξ - ½ ξ_Q(q),     q' = q + sharpR_q(ξ½),     ξ' = ξ½ - ½ ξ_Q(q').

The force ``ξ_Q`` is the backward pass of the *same* polynomial interpretation
(interpretation.py) used by ``Phiconf``/``Phiphase``; only the integrator
changes. Eliminating ξ gives the **centered** discrete wave recurrence

    m (q_{n+1} - 2 q_n + q_{n-1}) = κ · Lap(q_n),

with the Laplacian at the *center* point (vs. the trailing point in the Euler
recurrence), which is stable for ``κ/m < 4/λ_max``.

Scope: this realizes the K=2 case and runs *closed* systems (the wave/heat
examples, where the only feedback is the boundary rule). General K and the
compositional (lax-monoidal) structure of ``org^(K)`` are the conjecture of
rmk.org_N and are not built here. Leapfrog assumes a constant (mass-like)
``sharpR`` -- the physics case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax.numpy as jnp

from .arrangement import SmoothArrangement
from .interpretation import smooth_interpretation


@dataclass(frozen=True)
class OrgMorphism2:
    """A ``[p,p']^{∘2}``-coalgebra: a morphism in ``org^(2)``.

    The carried state is a phase point ``(q, ξ) ∈ T^*Q``. A macro-tick performs
    two emit/receive rounds (``run_one``) and updates by velocity Verlet.
    """

    arr: SmoothArrangement
    state: Any  # (q, ξ)

    def with_state(self, state: Any) -> "OrgMorphism2":
        return OrgMorphism2(self.arr, state)

    def _force(self, q, in_pos, in_dir_from) -> Tuple[Any, Any]:
        """One emit/receive round at position ``q``: returns ``(ξ_Q, out_pos)``.

        Runs the polynomial interpretation at ``q``: emit the readout position,
        let the environment answer (``in_dir_from``), and read back the parameter
        covector ``ξ_Q`` (the force).
        """
        position_action, direction_action = smooth_interpretation(self.arr)(q)
        out_m, omega_M = in_pos
        out_pos = position_action(out_m, omega_M)
        xi_N, in_n = in_dir_from(out_pos)
        xi_Q, _, _ = direction_action(out_m, omega_M, xi_N, in_n)
        return xi_Q, out_pos

    def run_one(self, in_pos, in_dir_from):
        """One leapfrog macro-tick = two emit/receive rounds (``org^(2)``).

        ``in_dir_from`` is the environment, applied at *both* emitted positions
        (so e.g. a position-dependent boundary is re-evaluated at ``q`` and ``q'``).
        Returns ``(out_pos_stage1, out_pos_stage2, new_state)``.
        """
        Q = self.arr.Q
        q, xi = self.state

        xi_Q1, out_pos1 = self._force(q, in_pos, in_dir_from)   # stage 1: force at q
        xi_half = xi - 0.5 * xi_Q1
        q2 = q + Q.apply_sharp(q, xi_half)                      # drift

        xi_Q2, out_pos2 = self._force(q2, in_pos, in_dir_from)  # stage 2: force at q'
        xi2 = xi_half - 0.5 * xi_Q2

        return out_pos1, out_pos2, (q2, xi2)


def Phileap(arr: SmoothArrangement) -> OrgMorphism2:
    """The leapfrog dynamics of a (closed) arrangement: ``sarr -> org^(2)``.

    Same interpretation factor as ``Phiphase``; the integrator is the symplectic,
    second-order velocity-Verlet step instead of explicit Euler.
    """
    return OrgMorphism2(arr, (jnp.zeros(arr.Q.dim), jnp.zeros(arr.Q.dim)))
