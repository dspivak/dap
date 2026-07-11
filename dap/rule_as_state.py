"""Rule-as-state CA as one ``Fin``-polynomial coalgebra -- three ALife paradigms unified.

EXTENSION (beyond the paper). The ALife *rule-as-state* pattern -- "per-cell selection from
a set of update functions where selection depends on state" (the frontier the ``alife`` lab
logged across **HetCA**, **MNCA**, **DiffLogic** as *"mechanisms diverge fundamentally; code
extraction limited"*) -- is exactly a coalgebra on the finite polynomial

    Fin(#rules, #neighbors) = Σ_{rule r} y^{neighborhood}.

A cell's *position* in this polynomial is its current rule (mode); the coalgebra applies
that rule's local map (``apply``) and lets the rule *coevolve* (``evolve``). The lab's 11
grand-synthesis attempts all tried to unify paradigms by *analysis* (measures on closed
trajectories) and hit "within-paradigm works, cross-paradigm fails" -- because they compared
the outputs of structurally different ``State -> State`` machines. Here the unification is
*structural*: the machines are one coalgebra, differing only in

    (n_modes,  apply,  evolve/select).

``life`` (1 mode, homogeneous), ``hetca`` (a huge coevolving rule set), and ``mixture`` (K
rules, state-gated selection -- the MNCA structure) are three instances of ``RuleAsStateCA``.
That the *rule is a state field* -- ``CAState.mode`` -- is the whole point; in a closed
``State -> State`` step it is a monolithic ``if rule == ...``, here it is the polynomial's
position, first-class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple

import numpy as np

from .modal import Fin

_MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def moore_count(grid: np.ndarray) -> np.ndarray:
    """Count Moore neighbours equal to 1 (periodic boundaries) -- the shared ``perceive``."""
    c = np.zeros_like(grid, dtype=np.int16)
    for dy, dx in _MOORE:
        c += (np.roll(np.roll(grid, -dy, 0), -dx, 1) == 1).astype(np.int16)
    return c


@dataclass
class CAState:
    """A rule-as-state grid: a phenotype, the per-cell **rule field** (mode), and aux state."""

    pheno: np.ndarray
    mode: np.ndarray          # the rule each cell currently runs -- rule *as state*
    aux: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleAsStateCA:
    """One coalgebra on ``Fin(n_modes, 9)``: apply the cell's mode's rule, then coevolve the mode.

    Every rule-as-state CA is an instance -- only ``apply`` (the per-mode local update) and
    ``evolve`` (how the mode field changes) differ. ``interface`` records the ``Fin`` polynomial.
    """

    interface: Fin
    perceive: Callable[[np.ndarray], np.ndarray]
    apply: Callable[[np.ndarray, np.ndarray, np.ndarray, dict], np.ndarray]
    evolve: Callable[[CAState, np.ndarray, Any], Tuple[np.ndarray, dict]]
    label: str = ""

    def step(self, s: CAState, rng=None) -> CAState:
        nbr = self.perceive(s.pheno)
        pheno2 = self.apply(s.mode, s.pheno, nbr, s.aux)           # mode-dependent update
        mode2, aux2 = self.evolve(s, nbr, rng)                     # rule coevolution / re-selection
        return CAState(pheno2, mode2, aux2)


# ---------------------------------------------------------------------------
# B/S life-like rule applied per cell (the shared per-mode local map).
# ---------------------------------------------------------------------------


def _apply_bs(b_mask, s_mask, pheno, n, alive_mask):
    born = ((b_mask >> n) & 1).astype(bool) & (pheno == 0)
    survive = ((s_mask >> n) & 1).astype(bool) & (pheno == 1)
    return np.where(alive_mask, (born | survive).astype(np.int8), pheno)


# ---------------------------------------------------------------------------
# Instance 1: Conway's Life -- the degenerate single-mode case (homogeneous CA).
# ---------------------------------------------------------------------------


def life(H=64, W=64, rng=None):
    rng = np.random.default_rng(rng)
    pheno = (rng.random((H, W)) < 0.3).astype(np.int8)
    B3, S23 = 1 << 3, (1 << 2) | (1 << 3)

    def apply(mode, pheno, n, aux):
        return _apply_bs(B3, S23, pheno, n, np.ones_like(pheno, bool))

    ca = RuleAsStateCA(Fin(1, 9), moore_count, apply,
                       lambda s, n, r: (s.mode, s.aux), label="life [1 mode]")
    return ca, CAState(pheno, np.zeros((H, W), np.int64), {})


# ---------------------------------------------------------------------------
# Instance 2: HetCA -- rule genome as state, coevolving (Shrestha et al. 2024).
# ---------------------------------------------------------------------------

QUIESCENT, ALIVE, DECAY = 0, 1, 2


def hetca(H=50, W=50, a_max=10, a_dec=15, inherit_prob=0.125, mutation_prob=0.2, rng=None):
    rng = np.random.default_rng(rng)
    pheno = np.zeros((H, W), np.int8)
    b = np.zeros((H, W), np.int64)
    s = np.zeros((H, W), np.int64)
    cellst = np.zeros((H, W), np.int64)
    age = np.zeros((H, W), np.int64)
    alive0 = rng.random((H, W)) < 0.5
    pheno[alive0] = rng.integers(0, 2, size=alive0.sum())
    cellst[alive0], age[alive0] = ALIVE, 1
    b[alive0], s[alive0] = (1 << 3), (1 << 2) | (1 << 3)   # B3/S23 seed genome

    def apply(mode, pheno, n, aux):
        return _apply_bs(mode[..., 0], mode[..., 1], pheno, n, aux["cell"] == ALIVE)

    def evolve(state, n, rng):
        b, s = state.mode[..., 0].copy(), state.mode[..., 1].copy()
        cell, age = state.aux["cell"].copy(), state.aux["age"].copy()
        alive, decay = cell == ALIVE, cell == DECAY
        age = np.where(alive | decay, age + 1, age)
        cell = np.where(alive & (age >= a_max), DECAY, cell)
        gone = (cell == DECAY) & (age >= a_dec)
        cell = np.where(gone, QUIESCENT, cell); age = np.where(gone, 0, age)
        b = np.where(gone, 0, b); s = np.where(gone, 0, s)
        # inheritance: quiescent cells next to an alive cell may adopt its genome (+mutation)
        alive_now = cell == ALIVE
        has_alive = np.zeros((H, W), bool)
        for dy, dx in _MOORE:
            has_alive |= np.roll(np.roll(alive_now, -dy, 0), -dx, 1)
        take = (cell == QUIESCENT) & has_alive & (rng.random((H, W)) < inherit_prob)
        selb, sels, picked = b.copy(), s.copy(), np.zeros((H, W), bool)
        for d in rng.permutation(len(_MOORE)):
            dy, dx = _MOORE[d]
            na = np.roll(np.roll(alive_now, -dy, 0), -dx, 1)
            can = take & na & ~picked
            selb = np.where(can, np.roll(np.roll(b, -dy, 0), -dx, 1), selb)
            sels = np.where(can, np.roll(np.roll(s, -dy, 0), -dx, 1), sels)
            picked |= can
        mut = picked & (rng.random((H, W)) < mutation_prob)
        flipb = mut & (rng.random((H, W)) < 0.5)
        bit = (1 << rng.integers(0, 9, size=(H, W)))
        selb = np.where(flipb, selb ^ bit, selb)
        sels = np.where(mut & ~flipb, sels ^ bit, sels)
        cell = np.where(picked, ALIVE, cell); age = np.where(picked, 1, age)
        b = np.where(picked, selb, b); s = np.where(picked, sels, s)
        return np.stack([b, s], -1), {"cell": cell, "age": age}

    ca = RuleAsStateCA(Fin(512 * 512, 9), moore_count, apply, evolve,
                       label="hetca [rule genome as state, coevolving]")
    return ca, CAState(pheno, np.stack([b, s], -1), {"cell": cellst, "age": age})


# ---------------------------------------------------------------------------
# Instance 3: mixture -- K rules with state-dependent selection (the MNCA structure).
# ---------------------------------------------------------------------------


def mixture(H=64, W=64, rng=None):
    """K life-like rules, the rule *selected per cell by local density* (a state-dependent gate).

    This is MNCA's structure -- a mixture of update functions with a state-dependent selector
    -- with the neural nets replaced by simple B/S rules to isolate the mode-selection itself.
    """
    rng = np.random.default_rng(rng)
    pheno = (rng.random((H, W)) < 0.3).astype(np.int8)
    rules = [((1 << 3), (1 << 2) | (1 << 3)),            # 0: Conway B3/S23
             ((1 << 3) | (1 << 6), (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8))]  # 1: HighLife-ish/coral
    B = np.array([r[0] for r in rules]); S = np.array([r[1] for r in rules])

    def select(n):                                       # the gate: rule depends on local density
        return (n >= 4).astype(np.int64)

    def apply(mode, pheno, n, aux):
        return _apply_bs(B[mode], S[mode], pheno, n, np.ones_like(pheno, bool))

    def evolve(state, n, rng):
        return select(n), state.aux                      # re-select each step from the new field

    ca = RuleAsStateCA(Fin(len(rules), 9), moore_count, apply, evolve,
                       label=f"mixture [{len(rules)} rules, state-gated]")
    return ca, CAState(pheno, select(moore_count(pheno)), {})


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def n_unique_rules(state: CAState) -> int:
    """How many distinct rules are in play across the grid (genotypic diversity)."""
    m = state.mode
    if m.ndim == 3:                                       # (H,W,2) genome
        alive = state.aux.get("cell", np.ones(m.shape[:2])) != QUIESCENT
        keys = m[..., 0].astype(np.int64) * 512 + m[..., 1]
        return int(np.unique(keys[alive]).size) if alive.any() else 0
    return int(np.unique(m).size)
