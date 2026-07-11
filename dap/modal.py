"""Mode-dependent (hybrid) arrangements: the ``Sum`` polynomial + ``Phi_modal``.

EXTENSION (beyond the paper) -- the piece that lets DAP speak *mode-dependence*.

DAP's polynomials (``Yon``, ``Cot``, ``DirichletProduct``) all have **uniform fibers**:
every position of ``Cot(d)`` exposes the same direction set ``(R^d)^*``. So the interface
a box presents cannot depend on *where the box is* -- and "the rule is part of the state"
(the ALife *rule-as-state* pattern) is exactly a system whose interface/behavior depends
on a discrete **mode** carried in the state. That needs the coproduct

    Sum(p_0, p_1, ...) = p_0 + p_1 + ...            (varying fibers)

whose positions are a *tagged union*: a position ``(m, x)`` sits in summand ``m`` and
exposes ``p_m``'s fiber -- different modes, different interfaces. This is the polynomial
DAP was missing.

``Phi_modal`` is the dynamics functor for a **modal arrangement**: a finite family of
ordinary smooth arrangements (one per mode) plus a state-dependent ``transition`` that may
switch mode between ticks. Within a mode it is exactly ``functors.Phi`` -- the smooth
interpretation + integrator, unchanged; the only new thing is that the emitted interface
is a ``Sum`` and the state carries the current mode, which the transition can flip. This
is a **hybrid / mode-dependent dynamical system** (a core Poly example): smooth flow
inside each mode, discrete jumps between modes.

The worked example is **morphogenesis with cell fate**: a tissue of cells, each with a
smooth internal state and a discrete *type* (mode); the type sets the dynamics, and cells
*differentiate* (switch type) by the ``transition``. What in a closed ``State -> State``
step function is a monolithic ``if type == ...`` is here a first-class ``Sum``-polynomial
coalgebra -- composable, integrator-swappable, and (with soft switching) differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .functors import phi_box_poly
from .integrator import Integrator, configuration_integrator
from .interpretation import smooth_interpretation
from .org import OrgMorphism
from .polynomial import Poly, PolyMap
from .rvect import euclidean


# ---------------------------------------------------------------------------
# The coproduct polynomial (the varying-fiber object DAP lacked).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sum:
    """The coproduct ``p_0 + p_1 + ... + p_{k-1}`` in poly (sec.poly; new for DAP).

    A position is a pair ``(m, x)`` -- mode ``m`` and a position ``x`` of branch ``p_m`` --
    and its fiber is ``p_m``'s fiber at ``x``. Positions in different summands expose
    *different* direction sets: this is where mode-dependence lives. ``Cot`` /
    ``DirichletProduct`` can only build uniform fibers; ``Sum`` cannot be reduced to them.
    """

    branches: Tuple[Poly, ...]

    def __init__(self, *branches: Poly):
        object.__setattr__(self, "branches", tuple(branches))


@dataclass(frozen=True)
class Fin:
    """The finite polynomial ``n · y^k = Σ_{i<n} y^k`` -- the discrete counterpart of ``Cot``.

    ``n_modes`` positions (the rules/modes a cell can be in), each exposing a ``nbhd``-cell
    neighborhood as its fiber. A *rule-as-state* cell (a heterogeneous CA cell whose update
    rule is part of its state) has interface ``Fin(#rules, #neighbors)``: which summand it
    sits in is its current rule. This is the collapsed ``Sum`` of representables that the
    ``dap/rule_as_state.py`` unification is built on -- ``Cot`` is to smooth flow what ``Fin``
    is to a discrete mode-dependent CA.
    """

    n_modes: int
    nbhd: int


# A mode transition reads (current mode, post-step within-mode state) and returns the
# next (mode, state): the guard/reset that makes the coalgebra hybrid.
ModeTransition = Callable[[int, Array], Tuple[int, Array]]


@dataclass(frozen=True)
class ModalArrangement:
    """A finite family of smooth arrangements (one per mode) + a mode ``transition``.

    ``modes[m]`` is an ordinary DAP ``SmoothArrangement``; ``transition(m, q)`` reads the
    post-step parameter state and returns ``(m', q')``. Here the modes share a parameter
    space (a *switched* system -- same state space, mode selects the vector field);
    modes of different dimension (a genuinely varying interface) are the general case.
    """

    modes: Tuple[SmoothArrangement, ...]
    transition: ModeTransition
    label: str = ""


# ---------------------------------------------------------------------------
# The dynamics functor for a modal arrangement.
# ---------------------------------------------------------------------------


def Phi_modal(marr: ModalArrangement, integrator: Integrator) -> OrgMorphism:
    """``Phi`` per mode + a mode switch: the dynamics functor on a modal arrangement.

    The state is ``(m, s)`` -- current mode and within-mode integrator state. Each tick
    runs the mode-``m`` smooth interpretation and integrator (``functors.Phi`` verbatim),
    emits into summand ``m`` of the ``Sum`` interface, then applies ``transition`` to pick
    the mode for the next tick. Setting a single mode with an identity transition recovers
    ``functors.Phi`` exactly.
    """
    interps = [smooth_interpretation(a) for a in marr.modes]
    src_p = Sum(*[phi_box_poly(a.out_dim_M, a.in_dim_M) for a in marr.modes])
    tgt_p = Sum(*[phi_box_poly(a.out_dim_N, a.in_dim_N) for a in marr.modes])

    def step(state):
        m, s = state
        Q = marr.modes[m].Q
        q = integrator.position(Q, s)
        position_action, direction_action = interps[m](q)

        def act_positions(in_pos):
            out_m, omega_M = in_pos
            return (m, position_action(out_m, omega_M))  # tag with the mode

        def act_directions(in_pos, in_dir):
            out_m, omega_M = in_pos
            xi_N, in_n = in_dir
            _, xi_M, in_m = direction_action(out_m, omega_M, xi_N, in_n)
            return (xi_M, in_m)

        act = PolyMap(src_p, tgt_p, act_positions, act_directions, label=f"modal[{m}]")

        def fiber(in_pos):
            out_m, omega_M = in_pos
            out_pos = (m, position_action(out_m, omega_M))

            def at_pos(in_dir):
                xi_N, in_n = in_dir
                xi_Q, xi_M, in_m = direction_action(out_m, omega_M, xi_N, in_n)
                s2 = integrator.step(Q, s, xi_Q)          # smooth update, unchanged
                m2, s2b = marr.transition(m, s2)          # <-- the hybrid jump
                return (xi_M, in_m), (m2, s2b)

            return out_pos, at_pos

        return act, fiber

    m0 = 0
    return OrgMorphism(src_p, tgt_p, (m0, integrator.init(marr.modes[m0].Q)), step)


# A soft transition maps (membership w in the simplex, post-step state) -> new membership.
SoftTransition = Callable[[Array, Array], Array]


def Phi_modal_soft(marr: ModalArrangement, integrator: Integrator,
                   soft_transition: SoftTransition) -> OrgMorphism:
    """The **differentiable** modal functor: a soft membership ``w`` over modes.

    Instead of a hard mode, the state is ``(w, s)`` with ``w`` a distribution over modes
    (all sharing the parameter space ``Q``). The force is the membership-weighted mixture
    of the modes' forces ``xi_Q = Σ_m w_m · dU_m(s)`` -- i.e. gradient flow on the mixture
    potential ``Σ_m w_m U_m`` -- and ``soft_transition`` moves ``w`` differentiably each
    tick. Hard ``Phi_modal`` is the ``w -> one-hot`` limit. Because everything is smooth in
    ``w`` and in any parameters of the modes/transition, a rollout is differentiable: you
    can **backprop a loss on the final tissue to the fate rule** (``train`` in the demo).
    """
    interps = [smooth_interpretation(a) for a in marr.modes]
    K = len(marr.modes)
    Q = marr.modes[0].Q
    src_p = Sum(*[phi_box_poly(a.out_dim_M, a.in_dim_M) for a in marr.modes])
    tgt_p = Sum(*[phi_box_poly(a.out_dim_N, a.in_dim_N) for a in marr.modes])

    def step(state):
        w, s = state
        q = integrator.position(Q, s)
        das = [interps[m](q)[1] for m in range(K)]  # each mode's direction_action at q

        def fiber(in_pos):
            out_m, omega_M = in_pos
            pos_action0 = interps[0](q)[0]
            out_pos = (w, pos_action0(out_m, omega_M))

            def at_pos(in_dir):
                xi_N, in_n = in_dir
                xi_Q = 0.0
                xi_M = in_m = None
                for m in range(K):
                    xq, xM, im = das[m](out_m, omega_M, xi_N, in_n)
                    xi_Q = xi_Q + w[m] * xq                      # mixture force Σ_m w_m dU_m
                    xi_M, in_m = xM, im
                s2 = integrator.step(Q, s, xi_Q)
                w2 = soft_transition(w, s2)                      # differentiable fate
                return (xi_M, in_m), (w2, s2)

            return out_pos, at_pos

        act = PolyMap(src_p, tgt_p, lambda i: (w, i), lambda i, d: d, label="modal_soft")
        return act, fiber

    w0 = jnp.ones(K) / K
    return OrgMorphism(src_p, tgt_p, (w0, integrator.init(Q)), step)


# ---------------------------------------------------------------------------
# Worked example: morphogenesis with cell fate.
# ---------------------------------------------------------------------------
#
# A cell's continuous state is a scalar signal ``u``. Two modes:
#   PROGENITOR (0): u relaxes toward a low set-point PROD0/DECAY.
#   DIFFERENTIATED (1): u relaxes toward a high set-point PRODA/DECAY (a strong source).
# Each mode is a 1-D closed smooth arrangement whose potential is U_mode(u) =
# -prod*u + 0.5*decay*u^2, so the configuration flow ``u <- u - eta*U'`` relaxes u toward
# prod/decay. Fate (the transition): a progenitor differentiates once u crosses THETA.
# In the tissue, lateral inhibition (a committed cell blocks its neighbors from committing)
# turns this into a *spaced* differentiation pattern -- classic developmental biology.

DECAY, PROD0, PRODA, THETA, ETA = 1.0, 1.0, 2.2, 0.55, 0.15


def _cell_mode(prod: float, eta: float = ETA) -> SmoothArrangement:
    """A 1-D closed arrangement whose config flow relaxes ``u`` toward ``prod/DECAY``."""
    return SmoothArrangement(
        euclidean(1, eta), 0, 0, 0, 0,
        out_f=lambda q, m: jnp.zeros(0),
        in_f=lambda q, m, n: jnp.zeros(0),
        U=lambda q, m, n: (-prod * q[0] + 0.5 * DECAY * q[0] ** 2),
    )


def tensor_modal(A: ModalArrangement, B: ModalArrangement) -> ModalArrangement:
    """The monoidal product of two modal arrangements: product modes, independent switching.

    Modes are ``A.modes[a] (x) B.modes[b]`` (via ``wiring.tensor_arrangements``), indexed
    ``M = a*|B.modes| + b``; the transition splits the direct-sum state and applies each
    factor's transition on its own component. ``Phi_modal`` preserving this ``(x)`` (with the
    distributive law ``Sum (x) Sum = Sum of products``) is the parallel/spatial functor law
    the tissue rests on -- see ``test_modal_functoriality``.
    """
    from .wiring import tensor_arrangements

    KB = len(B.modes)
    nA = A.modes[0].Q.dim
    modes = tuple(tensor_arrangements([a, b]) for a in A.modes for b in B.modes)

    def transition(M: int, q: Array):
        a, b = divmod(M, KB)
        a2, qa = A.transition(a, q[:nA])
        b2, qb = B.transition(b, q[nA:])
        return a2 * KB + b2, jnp.concatenate([qa, qb])

    return ModalArrangement(modes, transition, label=f"({A.label} (x) {B.label})")


def fate_transition(theta: float = THETA) -> ModeTransition:
    """Progenitor (mode 0) -> differentiated (mode 1) once its signal crosses ``theta``."""
    def transition(m: int, q: Array):
        if m == 0 and float(q[0]) > theta:
            return 1, q
        return m, q
    return transition


def morphogen_cell(theta: float = THETA, eta: float = ETA) -> ModalArrangement:
    """A single differentiating cell: progenitor + differentiated modes + the fate rule."""
    return ModalArrangement(
        modes=(_cell_mode(PROD0, eta), _cell_mode(PRODA, eta)),
        transition=fate_transition(theta),
        label="morphogen-cell",
    )
