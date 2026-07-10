"""``Phi`` preserves the monoidal product ``⊗`` (sec.spring_second_pass, prop.cot_monoidal).

PR #1 audited that ``Phi`` preserves *sequential* composition ``;`` (``test_composition``,
``test_multistage_functoriality``). A *monoidal* functor must also preserve the *monoidal
product*. This module runs the complementary audit -- for every shipped functor, at every
stage count:

  ``Phi(tensor(A, B)) == Phi(A).parallel(Phi(B))``     (tensor in sarr, then Phi
                                                        == Phi each, then ∥ in pc)

for ``Phiconf``/``Phiphase`` (``org``, K = 1), ``Phileap`` (``org^(2)``) and ``Phirk4``
(``org^(4)``). This is the ``⊗``-half of monoidal functoriality; with PR #1's ``;``-half
and the symmetric-monoidal-category laws of ``test_monoidal_laws``, it is the evidence
that ``Phi : sarr -> org^(K)`` is a *symmetric monoidal functor* (the open ``rmk.multistage``
question, still not a proof).

**Up to the productor.** The two sides are not identical on the nose: they differ by the
monoidal coherence iso ``cot(R^{a+b}) ~= cot(R^a) (x) cot(R^b)`` (the *productor*,
prop.cot_monoidal) -- ``tensor(A, B)`` presents a box with *concatenated* interfaces
``R^{a+b}``, while ``Phi(A) ∥ Phi(B)`` presents the *product* of two boxes. So the test
drives the tensor side with a concatenated environment and the parallel side with the
split one, and checks the outputs agree under concatenation: the emitted position and the
whole emitted covector field agree block-for-block, and the states agree up to the same
splitting (checked by driving several macro-ticks and comparing, so a mismatch would
surface on the next tick). That the productor threads through untouched at every round is
exactly what "``Phi`` is (lax) monoidal" asserts.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phiconf, Phiphase, Phirk4
from dap.interpretation import trivial_omega
from dap.leapfrog import Phileap
from dap.rvect import diagonal
from dap.wiring import tensor_arrangements

_IN_POS0 = (jnp.zeros(0), trivial_omega(0))  # a closed-M box position


def _particle(m, kappa):
    """A ``<R^0|R^0> -> <R^1|R^1>`` harmonic particle: closed input, open output.

    The same box the functoriality chains use (``test_composition``); tensoring two of
    them gives a ``<R^0|R^0> -> <R^2|R^2>`` box whose open output is the concatenation
    of the two 1-D outputs -- exactly where the productor lives.
    """
    return SmoothArrangement(
        diagonal(jnp.array([1.0 / m])), 0, 0, 1, 1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2,
    )


def _run(O, in_pos, env):
    """One macro-tick of any org datatype; returns ``(out_poss, out_dirs, new_state)``."""
    if hasattr(O, "K"):  # OrgMorphismK
        return O.run([in_pos] * O.K, [env] * O.K)
    if hasattr(O, "run_two"):  # OrgMorphism2
        op1, od1, op2, od2, ns = O.run_two(in_pos, env, in_pos, env)
        return [op1, op2], [od1, od2], ns
    op, od, ns = O.run_one(in_pos, env)  # OrgMorphism
    return [op], [od], ns


def _assert_preserves_tensor(Phi, ms, kappas, T=4, seed=5):
    A, B = _particle(ms[0], kappas[0]), _particle(ms[1], kappas[1])
    tens = Phi(tensor_arrangements([A, B]))     # tensor in sarr, then Phi
    par = Phi(A).parallel(Phi(B))               # Phi each, then parallel in pc

    rng = np.random.default_rng(seed)
    xiA, xiB = jnp.asarray(rng.standard_normal(1)), jnp.asarray(rng.standard_normal(1))
    inA, inB = jnp.asarray(rng.standard_normal(1)), jnp.asarray(rng.standard_normal(1))
    zA, zB = jnp.array([0.31]), jnp.array([-0.22])  # probe points for the covector fields
    z2 = jnp.concatenate([zA, zB])

    # Directions: concatenated for the tensor side, split for the parallel side.
    env_tens = lambda _op: (jnp.concatenate([xiA, xiB]), jnp.concatenate([inA, inB]))  # noqa: E731
    env_par = lambda _op: ((xiA, inA), (xiB, inB))  # noqa: E731
    in_par = (_IN_POS0, _IN_POS0)

    stT, stP = tens.state, par.state
    for _ in range(T):
        opsT, _odsT, stT = _run(tens.with_state(stT), _IN_POS0, env_tens)
        opsP, _odsP, stP = _run(par.with_state(stP), in_par, env_par)
        assert len(opsT) == len(opsP)  # same number of rounds
        for (onT, omT), ((onA, omA), (onB, omB)) in zip(opsT, opsP):
            # emitted position: concatenation of the two blocks
            np.testing.assert_allclose(np.asarray(onT), np.concatenate([onA, onB]), atol=1e-10)
            # emitted covector field: block-for-block equal to the two separate fields
            np.testing.assert_allclose(
                np.asarray(omT(z2)), np.concatenate([omA(zA), omB(zB)]), atol=1e-10
            )

    # states agree under the same splitting (same multiset of scalar leaves)
    lT = sorted(float(v) for x in jax.tree_util.tree_leaves(stT) for v in np.ravel(x))
    lP = sorted(float(v) for x in jax.tree_util.tree_leaves(stP) for v in np.ravel(x))
    assert len(lT) == len(lP)
    np.testing.assert_allclose(lT, lP, atol=1e-10)


def test_phiconf_preserves_tensor():
    """org (K=1), descent: ``Phiconf(A ⊗ B) == Phiconf(A) ∥ Phiconf(B)``."""
    _assert_preserves_tensor(Phiconf, ms=(1.3, 1.7), kappas=(2.1, 0.9))


def test_phiphase_preserves_tensor():
    """org (K=1), Hamiltonian: ``Phiphase(A ⊗ B) == Phiphase(A) ∥ Phiphase(B)``."""
    _assert_preserves_tensor(Phiphase, ms=(1.1, 1.9), kappas=(1.7, 2.3))


def test_phileap_preserves_tensor():
    """org^(2), leapfrog: ``Phileap(A ⊗ B) == Phileap(A) ∥ Phileap(B)`` (both rounds)."""
    _assert_preserves_tensor(Phileap, ms=(1.5, 0.8), kappas=(2.0, 1.2))


def test_phirk4_preserves_tensor():
    """org^(4), RK4: ``Phirk4(A ⊗ B) == Phirk4(A) ∥ Phirk4(B)`` (all four stages)."""
    _assert_preserves_tensor(lambda arr: Phirk4(arr, h=0.1), ms=(1.4, 1.0), kappas=(0.9, 1.6))
