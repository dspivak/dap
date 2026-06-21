"""System identification: a closed system trains a predictor of itself
(rmk.graph_environment, "Predicting a closed system").

EXTENSION (beyond the paper). This reuses the paper's phase functor ``Phiphase``
and the learner of sec.dl_warmup unchanged, and the data stream is the environment
of rmk.graph_environment / ex.training_data; but the nonlinear pendulum, the
tanh-MLP predictor, and the libration-regime sampling are standard ML layered on
top, not part of the paper's formal development. It is a toy demonstration that the
primitives compose -- not an implementation of a paper result. See the "Extensions"
section of the README.

Given a closed system ``c : S -> S`` (a ``y``-coalgebra, ex.y_coalgebra_trajectory --
here ``Phiphase`` of a nonlinear oscillator) and an observation ``o : S -> X``,
the canonical *one-step-prediction stream* is the coalgebra with

    readout   <o, o.c> : S -> X x X      (observe now, and one step ahead)
    update    c        : S -> S          (advance)

i.e. the environment of ex.training_data, driven by the system itself. Closing
the learner of sec.dl_warmup with it descends the network's weights
(eqn.dl_gradient_update) toward predicting ``o.c`` from ``o``: the system's
one-step map.

The paper's particles are all quadratic, so that map is linear; in code we put in
*any* smooth potential -- a pendulum, a double well -- and the map is genuinely
nonlinear, so the net learns a real flow map. We take ``o = id`` (full state), so
the target is a deterministic function of the input and the fit is clean.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from .interpretation import trivial_omega
from .arrangement import SmoothArrangement
from .functors import Phiphase
from .learning import parameterized_map, train
from .org import OrgMorphism
from .rvect import diagonal, euclidean

_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


# ---------------------------------------------------------------------------
# The system: a closed autonomous oscillator (an arrangement, run by Phiphase).
# ---------------------------------------------------------------------------


def oscillator(
    potential: Callable[[Array], Array], dim: int, m: float = 1.0, label: str = "oscillator"
) -> SmoothArrangement:
    """A closed autonomous oscillator: parameter ``R^dim`` (position), sharp
    ``(1/m) I``, on-site potential ``U(q)``.

    Under ``Phiphase`` (sec.phase_dynamics) this is the symplectic flow of the
    Hamiltonian ``H = |p|^2 / (2m) + U(q)``; its state is ``(q, p)`` in
    ``R^dim x R^dim``. Quadratic ``U`` gives a linear flow; any other gives a
    nonlinear one.
    """

    return SmoothArrangement(
        diagonal(jnp.full(dim, 1.0 / m)), 0, 0, 0, 0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: potential(q),
        label=label,
    )


# The pendulum, as (potential, dim): U(theta) = 1 - cos(theta), genuinely nonlinear,
# and bounded in the libration regime so the symplectic step stays stable. Stiffer
# potentials (e.g. a double well, force ~q^3) need a gentler step (larger m).
PENDULUM: Tuple[Callable[[Array], Array], int] = (lambda q: 1.0 - jnp.cos(q[0]), 1)


# ---------------------------------------------------------------------------
# The trajectory of the closed coalgebra, and the <o, o.c> data stream.
# ---------------------------------------------------------------------------


def trajectory(O: OrgMorphism, q0: Array, p0: Array, T: int) -> List[Array]:
    """Iterate the closed coalgebra ``c = O`` from ``(q0, p0)`` for ``T`` steps
    (ex.y_coalgebra_trajectory); return the states ``s_t = (q_t, p_t)`` as flat
    ``R^{2 dim}`` arrays."""

    state = (jnp.asarray(q0, float), jnp.asarray(p0, float))
    out = [jnp.concatenate(state)]
    for _ in range(T):
        _, _, state = O.with_state(state).run_one(_IN_POS, _TRIV)
        out.append(jnp.concatenate(state))
    return out


def consecutive_pairs(states: Sequence[Array]) -> List[Tuple[Array, Array]]:
    """The ``<o, o.c>`` stream with ``o = id``: the pairs ``(s_t, s_{t+1})``."""
    return [(states[t], states[t + 1]) for t in range(len(states) - 1)]


# ---------------------------------------------------------------------------
# The predictor: a small tanh MLP, as a parameterized map (sec.dl_warmup).
# ---------------------------------------------------------------------------


def mlp(hidden: int, in_dim: int, out_dim: int) -> Tuple[Callable[[Array, Array], Array], int]:
    """``F(q, x) = W2 tanh(W1 x + b1) + b2`` with params flattened into ``q``;
    returns ``(F, n_params)``."""

    shapes = [(hidden, in_dim), (hidden,), (out_dim, hidden), (out_dim,)]
    sizes = [int(np.prod(s)) for s in shapes]
    n = sum(sizes)

    def unpack(q):
        out, i = [], 0
        for s, k in zip(shapes, sizes):
            out.append(q[i : i + k].reshape(s))
            i += k
        return out

    def F(q, x):
        W1, b1, W2, b2 = unpack(q)
        return W2 @ jnp.tanh(W1 @ x + b1) + b2

    return F, n


# ---------------------------------------------------------------------------
# Identify: generate trajectories, train a net to predict the next state.
# ---------------------------------------------------------------------------


def _sample_ic(rng, dim: int, ic_scale: float):
    """An initial ``(q0, p0)`` drawn uniformly from ``[-ic_scale, ic_scale]``.

    Uniform (not Gaussian) keeps the energy bounded, so a pendulum stays in the
    *libration* regime where the state is bounded and the one-step map is learnable;
    Gaussian tails would reach the rotating regime, where ``q`` grows without bound.
    """
    return (
        rng.uniform(-ic_scale, ic_scale, dim),
        rng.uniform(-ic_scale, ic_scale, dim),
    )


def identify(
    system: Tuple[Callable[[Array], Array], int] = PENDULUM,
    *,
    m: float = 1.0,
    n_traj: int = 40,
    T: int = 40,
    ic_scale: float = 1.0,
    hidden: int = 32,
    eta: float = 6e-3,
    steps: int = 8000,
    seed: int = 0,
):
    """Train a net to predict the one-step map of the oscillator ``system``.

    Returns ``(params, F, history, O, dim, m)``: the trained weights, the net,
    the per-step training loss, the system coalgebra ``O = Phiphase(...)``, and
    the configuration dimension and mass.
    """

    potential, dim = system
    O = Phiphase(oscillator(potential, dim, m))
    d = 2 * dim
    rng = np.random.default_rng(seed)

    data: List[Tuple[Array, Array]] = []
    for _ in range(n_traj):
        q0, p0 = _sample_ic(rng, dim, ic_scale)
        data += consecutive_pairs(trajectory(O, q0, p0, T))
    rng.shuffle(data)

    F, n = mlp(hidden, d, d)
    init = 0.1 * jnp.asarray(rng.standard_normal(n))
    arr = parameterized_map(F, euclidean(n, eta), in_dim=d, out_dim=d)
    params, history = train(arr, init, data, steps=steps)
    return params, F, history, O, dim, m


def one_step_error(
    params: Array,
    F: Callable[[Array, Array], Array],
    O: OrgMorphism,
    dim: int,
    m: float,
    *,
    ic_scale: float = 1.0,
    T: int = 40,
    n_traj: int = 8,
    seed: int = 999,
) -> Tuple[float, float]:
    """Mean one-step prediction error of ``F`` on held-out trajectories, and the
    error of the naive "no change" baseline ``s_{t+1} ~ s_t`` for comparison."""

    rng = np.random.default_rng(seed)
    errs, base = [], []
    for _ in range(n_traj):
        q0, p0 = _sample_ic(rng, dim, ic_scale)
        for s, s_next in consecutive_pairs(trajectory(O, q0, p0, T)):
            errs.append(float(jnp.linalg.norm(F(params, s) - s_next)))
            base.append(float(jnp.linalg.norm(s - s_next)))
    return float(np.mean(errs)), float(np.mean(base))
