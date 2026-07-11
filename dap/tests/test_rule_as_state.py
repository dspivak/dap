"""Evidence for the rule-as-state unification (dap/rule_as_state.py).

'Per-cell selection from a set of update functions where selection depends on state' -- the
ALife pattern the lab logged as un-extractable across HetCA / MNCA / DiffLogic -- is one
coalgebra on ``Fin(#rules, #neighbors)``. HetCA, the MNCA-style mixture, and Conway's Life
are instances of the *same* ``RuleAsStateCA``, differing only in (n_modes, apply, evolve).
These tests pin the anchor (Life == Conway) and each paradigm's signature.
"""

from __future__ import annotations

import numpy as np

from dap.modal import Fin
from dap.rule_as_state import (RuleAsStateCA, hetca, life, mixture, moore_count,
                               n_unique_rules)


def _conway(pheno):
    n = moore_count(pheno)
    return (((n == 3) & (pheno == 0)) | (((n == 2) | (n == 3)) & (pheno == 1))).astype(np.int8)


def test_all_three_are_one_construction():
    """The three paradigms are literally instances of the same RuleAsStateCA on a Fin interface."""
    for build in (life, hetca, mixture):
        ca, st = build(rng=0)
        assert isinstance(ca, RuleAsStateCA)
        assert isinstance(ca.interface, Fin)
    # ...and the mode set cardinality is the only structural difference:
    assert life(rng=0)[0].interface.n_modes == 1
    assert mixture(rng=0)[0].interface.n_modes == 2
    assert hetca(rng=0)[0].interface.n_modes == 512 * 512


def test_life_single_mode_is_conway():
    """Anchor: the 1-mode instance reproduces plain Conway exactly (the unification reduces right)."""
    ca, st = life(48, 48, rng=0)
    ref = st.pheno.copy()
    for _ in range(50):
        st = ca.step(st)
        ref = _conway(ref)
        assert np.array_equal(st.pheno, ref)


def test_hetca_diversifies_and_stays_active():
    """HetCA signature (Shrestha 2024): rule genomes diversify open-endedly; activity is sustained."""
    ca, st = hetca(48, 48, rng=1)
    rng = np.random.default_rng(7)
    seen, prev, act = set(), st.pheno.copy(), []
    for _ in range(120):
        st = ca.step(st, rng=rng)
        keys = st.mode[..., 0].astype(np.int64) * 512 + st.mode[..., 1]
        seen |= set(np.unique(keys[st.aux["cell"] != 0]).tolist())
        act.append(int((prev != st.pheno).sum()))
        prev = st.pheno.copy()
    assert len(seen) > 50               # far beyond the single seed genome: open-ended diversification
    assert np.mean(act[-40:]) > 10      # sustained phenotypic activity (not frozen/dead)


def test_mixture_selects_rule_by_state():
    """MNCA structure: which update rule a cell runs is chosen by its state (local density gate)."""
    ca, st = mixture(48, 48, rng=2)
    ever, before = set(), st.pheno.copy()
    for _ in range(30):
        before = st.pheno.copy()
        st = ca.step(st)
        ever |= set(np.unique(st.mode).tolist())
    # selection depends on state: the stored rule field == the density gate of the pre-step field
    assert np.array_equal(st.mode, (moore_count(before) >= 4).astype(st.mode.dtype))
    assert ever == {0, 1}                              # both rules genuinely in play over the run
