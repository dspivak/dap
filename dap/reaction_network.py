"""Mass-action reaction networks as ``Phiconf`` of arrangements.

A reversible reaction network on species ``1..n`` is a list of reactions
``r`` with reactant/product stoichiometric vectors ``alpha_r, beta_r in N^n``
and rate constants ``kf_r, kb_r > 0``; mass-action kinetics is the ODE

    x' = sum_r zeta_r * v_r(x),   zeta_r = beta_r - alpha_r,
    v_r(x) = kf_r * x^{alpha_r} - kb_r * x^{beta_r}          (net flux).

The network is *detailed balanced* if some ``x_eq > 0`` equalizes every
reaction's forward and backward flux: ``kf_r x_eq^{alpha_r} = kb_r
x_eq^{beta_r} =: w_r``. This module settles which of these ODEs are the
configuration dynamics ``Phiconf`` of an arrangement, and in which encoding.

VACUITY GUARD. ``def.rvect`` puts no symmetry or definiteness condition on a
sharp, so *every* smooth ODE ``x' = V(x)`` is trivially ``Phiconf`` of a
one-box arrangement: take ``U(x) = x_1`` and the rank-one sharp
``sharpR_x = -dt * V(x) (x) e_1``. That realization has no content -- the
"potential" certifies nothing and descends nothing. The claims below are
therefore about *structured* realizations: a symmetric positive-semidefinite
sharp together with a potential that the flow genuinely descends (an Onsager
gradient structure), which is what every dissipative example in the paper and
in this repo has.

THE VERIFIED FRONTIER.

1. MONOLITHIC (one box), any molecularity: every reversible detailed-balanced
   mass-action network is ``Phiconf`` of the closed sarr-scalar
   (``crn_arrangement``) whose parameter reactive vector space is ``R^n`` with
   the state-dependent Mielke--Onsager sharp [Mielke 2011, "A gradient
   structure for reaction-diffusion systems and for energy-drift-diffusion
   systems", Nonlinearity 24, thm 3.1 specialized to pure reaction]

       sharpR_x = dt * K(x),
       K(x)     = sum_r Lambda(f_r(x), b_r(x)) * zeta_r zeta_r^T,

   with ``f_r = kf_r x^{alpha_r}``, ``b_r = kb_r x^{beta_r}`` the one-way
   fluxes and ``Lambda(u, v) = (u - v)/(log u - log v)`` the logarithmic mean
   (``Lambda(u, u) = u``), and whose potential is the free energy

       F(x) = sum_i x_i (log(x_i / x_eq_i) - 1).

   The realization is EXACT, not asymptotic: since ``dF = log(x/x_eq)`` and
   detailed balance gives ``zeta_r . dF = log(b_r/f_r)``, the identity
   ``Lambda(f, b) log(b/f) = b - f`` makes one configuration-integrator step
   ``x -> x - sharpR_x(dF)`` (eqn.conf_integrator) literally the forward-Euler
   mass-action step ``x + dt * sum_r zeta_r v_r(x)``, to machine precision.
   ``K(x)`` is symmetric PSD on ``x > 0`` (``Lambda > 0``), so ``F`` decreases
   along the flow and the equilibrium in each stoichiometric compatibility
   class is its minimum. ``dt`` rides on the sharp exactly as in
   ``replicator.shahshahani``. Positivity of ``x`` is preserved by the flow
   but only approximately by Euler steps (keep ``dt`` small); ``F``, ``K``
   need ``x > 0`` and reversibility ``kb_r > 0`` (a one-way flux 0 puts
   ``Lambda`` on the boundary and the log-mean form degenerates).

   Not covered: networks WITHOUT detailed balance. Irreversible reactions
   break ``Lambda`` outright; reversible-but-Wegscheider-violating cycles
   (e.g. ``A<->B<->C<->A`` with cycle affinity != 0) admit no ``x_eq`` and
   ``crn_arrangement`` rejects them. Complex-balanced networks still
   dissipate this ``F`` but the flow has a non-gradient part; no structured
   realization is claimed or built here.

2. COMPOSITIONAL, unimolecular reversible networks (``A_i <-> A_j`` edges,
   detailed balanced): realized by genuine composition in ``sarr``
   (``unimolecular_arrangement``), on the kuramoto/graph-Laplacian pattern:
   one species box per species (parameter ``R`` holding ``x_i``, CONSTANT
   scalar sharp ``dt * x_eq_i``, readout the activity ``a_i = x_i / x_eq_i``
   broadcast to its reaction ports), one stateless reaction box per edge with
   potential ``(w_r / 2)(a_i - a_j)^2``, and a routing wire; the composite is
   ``compose_seq(tensor(boxes), wire)`` and its total potential

       U(x) = sum_r (w_r / 2) (x_i/x_eq_i - x_j/x_eq_j)^2

   emerges from the writer-monad addition, written nowhere by hand. One
   ``Phiconf`` step is again EXACTLY the forward-Euler mass-action step:
   the covector reaction ``r`` returns to species ``i`` is ``+- w_r (a_i -
   a_j) / x_eq_i`` (the ``1/x_eq_i`` from the readout's pullback), and the
   sharp ``dt * x_eq_i`` cancels the weight, leaving ``-+ dt * w_r (a_i -
   a_j) = dt * zeta_{r,i} v_r(x)``. Here the linear vector field is
   ``-K0 * diag(1/x_eq) x`` with CONSTANT ``K0 = sum_r w_r zeta_r zeta_r^T``;
   the constant-sharp/quadratic-potential pair is the same gradient flow as
   the Mielke pair of 1 (both equal ``sum_r zeta_r v_r`` on the nose), so the
   monolithic and compositional encodings agree step-for-step.

3. COMPOSITIONAL OBSTRUCTION, bimolecular: the species-box encoding STOPS at
   unimolecular. Precisely: in any composite built from species boxes with
   one-dimensional state (storing any diffeomorphic reparameterization of its
   own concentration -- log-coordinates included), stateless reaction boxes,
   and wiring, the composite sharp is the DIRECT SUM of the species sharps
   (``rvect.direct_sum``, via ``compose_seq``/``parallel_arrangements``), each
   block a function of its own species' state only. The realizable dynamics
   are therefore exactly

       x_i' = -g_i(x_i) * dU/dx_i (x),        g_i a function of x_i alone,

   (reaction data in the shared potential, per-species conformal factors in
   the sharp). Closedness of ``dU`` forces, for all ``i != j`` and all ``x``,

       dV_j/dx_i (x) / dV_i/dx_j (x) = g_j(x_j) / g_i(x_i),

   i.e. the ratio of opposite Jacobian entries of the vector field ``V`` must
   be SEPARABLE -- independent of every other species. For ``A + B <-> C``
   (``V = zeta * (kf x_A x_B - kb x_C)``, ``zeta = (-1,-1,+1)``) the (A, C)
   ratio is ``(dV_C/dx_A)/(dV_A/dx_C) = (kf/kb) * x_B``, which varies with the
   third species ``B``: NO choice of ``g_A, g_B, g_C`` and ``U`` works, in any
   per-species coordinates (a reparameterization ``x_i = phi_i(y_i)`` only
   changes ``g_i``, never the ratio). Irreversibly, ``A + B -> C`` is worse:
   ``dV_A/dx_C = 0`` while ``dV_C/dx_A = kf x_B != 0``, so ``C`` would need a
   potential term the closedness of ``dU`` forbids -- the covector a potential
   returns to a species is the potential's OWN derivative in that species, so
   production stoichiometry cannot be decoupled from flux dependence. This is
   also why the paper's Lotka--Volterra pattern (log states, monomial
   potential, oppositely-SIGNED sharps) works for predation yet never yields a
   produced third species: its interaction monomial couples exactly the
   species that appear in it, generating the Kolmogorov class
   ``x_i' = x_i * (sign_i) dU/d(log x_i)`` -- interaction matrix = signed
   diagonal times symmetric -- and ``A + B <-> C`` is outside that class by
   the same ratio test (``separability_ratio``).

   Scope of the no-go: species boxes with hidden state (dim > 1) and stateful
   reaction boxes are not ruled out. In particular a SINGLE reversible
   reaction is realizable in its extent coordinate ``s`` (state ``x_0 + zeta
   s``), but that is a one-box arrangement -- monolithic in disguise; with two
   reactions sharing a species each extent box's sharp would need the other
   box's state, which ``direct_sum`` forbids, and the obstruction returns.
   Item 4 occupies exactly this gap.

4. COMPOSITIONAL, ARBITRARY Petri net (the paper's "Petri nets" item): put the
   state in the TRANSITIONS, doubled, and the species in the WIRING. A Petri
   net ``(S, T, m, n, r, x0)`` has arc multiplicities ``m, n : T x S -> N``,
   rate constants ``r : T -> R_{>=0}`` and initial concentrations ``x0``; each
   transition is one-way and fires at ``f_t(x) = r(t) prod_s x_s^{m_ts}`` (a
   reversible reaction is two transitions). Transition ``t`` is a box
   ``<R^{S_t}|R> `` whose parameter is ``logic.py``'s transistor pair

       (R^2, sharpR),   sharpR(xi, xi') = (xi', 0),

   holding ``(eps_t, eps'_t)``, reporting ``eps_t`` (the extent of reaction),
   receiving the concentrations at the species ``S_t`` it touches, with

       U_t((eps_t, eps'_t), x) = -eps'_t * f_t(x).

   The species are the static wiring: an affine input map sending the reported
   extents to ``x_s = x0_s + sum_t (n_ts - m_ts) eps_t`` (affine wiring has
   precedent in ``amplifier.py``'s feedback stage). Since ``U`` is LINEAR in
   ``eps'_t``, ``dU/deps'_t = -f_t(x)`` is the rate itself rather than a
   derivative of it, and the nilpotent sharp delivers that to ``eps_t`` while
   discarding ``dU/deps_t`` (the rate sensitivities, which are what a genuine
   descent would follow). One ``Phiconf`` tick is then EXACTLY the forward-Euler
   mass-action step, at any molecularity, irreversibly, with no detailed-balance
   hypothesis; ``eps'_t`` never moves and its value never matters.

   WHAT THIS DOES AND DOESN'T BUY. The sharp is nilpotent, not symmetric PSD,
   and ``U_t`` is a rate, not an energy: nothing descends, so by the vacuity
   guard above this realization certifies no thermodynamics. What it does have,
   which the one-box vacuous realization does not, is LOCALITY: each box carries
   only its own rate constant and arc multiplicities and sees only its own
   species, and the total potential is assembled by ``compose_seq``. Moiety
   conservation is also structural rather than numerical: the state IS the
   extent vector and ``x`` is recomputed as ``x0 + Z^T eps`` at every tick, so
   any ``c`` with ``Z c = 0`` has ``c . x = c . x0`` at every step without
   rounding accumulating in ``x`` (Euler stepped on ``x`` preserves such laws
   in exact arithmetic too, but drifts by accumulated float error).

5. COMPOSITIONAL, ARBITRARY Petri net IN LOG COORDINATES: positivity by
   construction. Dual to 4: the state moves into doubled SPECIES boxes and the
   transitions become stateless. Species ``s`` is a box ``<R^0|R^0> ->
   <R^0 | R^{2 deg(s)}>`` with the transistor pair

       (R^2, sharpR),   sharpR(xi, xi') = (dt * xi', 0),

   holding ``(y_s, y'_s)``, broadcasting the pair ``(x_s, y'_s)``,
   ``x_s = exp(y_s)``, to each incident transition; transition ``t`` is a
   STATELESS box receiving those pairs at the species it touches, with

       V_t = -f_t(x) * sum_s zeta_ts * y'_s / x_s,   zeta = n - m;

   the wiring is an honest prism (a port bijection -- no affine map, and
   ``x0`` is genuinely the initial state rather than wiring data). Since
   ``dV_t/dy'_s = -zeta_ts f_t(x)/x_s`` is ``y'``-free, and the exp Jacobian
   of the readout lands only in the discarded ``xi``-slot, one ``Phiconf``
   tick is EXACTLY

       y_s -> y_s + dt * sum_t zeta_ts f_t(x)/x_s,

   the forward-Euler step of mass action written in log-concentration
   coordinates ``y = log x`` (equivalently ``x_s -> x_s * exp(dt *
   (Z^T f)_s / x_s)``, a multiplicative Euler step). Concentrations are
   POSITIVE BY CONSTRUCTION at every tick and any ``dt``, where encoding 4
   can overshoot negative; the price is exactly dual: 4 preserves every
   moiety conservation law structurally and positivity only approximately,
   5 preserves positivity structurally and conservation laws only
   approximately (``O(dt^2)`` drift per tick). ``y'_s`` is frozen and its
   value never matters, as in 4. POSITIVITY IS NOT STABILITY: near the
   orthant boundary the exponent ``(Z^T f)_s/x_s`` is unbounded, so at large
   ``dt`` the iteration can still blow up (in floats, overflowing to
   ``inf``/``0``); what the construction removes is only the orthant-exit
   failure mode, and ``dt`` must still be small for stability and accuracy.
   The clean win is stiff pure decay (``A -> 0`` fast): Euler exits the
   orthant at ``dt > 1/rate`` while the log tick ``x e^{-rate dt}`` is
   positive, monotone, and stable at every ``dt``.

In short: detailed-balanced mass-action = ``Phiconf`` with an Onsager
(symmetric-PSD, free-energy-descending) structure, monolithically, exactly;
compositionally with SPECIES boxes exactly the unimolecular ones, the first
bimolecular reaction already unwireable because the cross-species Onsager
coupling ``zeta_r zeta_r^T`` cannot arise from a direct-sum sharp; and
compositionally with doubled TRANSITION boxes every Petri net, exactly, at the
price of a nilpotent sharp and a potential that is not an energy -- or with
doubled SPECIES boxes in log coordinates, exactly Euler-in-log, positive by
construction, trading structural conservation for structural positivity.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .functors import Phiconf
from .interpretation import trivial_omega
from .pc import PCMorphism
from .rvect import ReactiveVectorSpace, constant, diagonal, trivial
from .wiring import compose_seq, tensor_arrangements


# ---------------------------------------------------------------------------
# Mass-action ground truth (used by the tests as the reference; the dynamics
# below never call it -- they arise from Phiconf).
# ---------------------------------------------------------------------------


def one_way_fluxes(
    alpha: Array, beta: Array, kf: Array, kb: Array
) -> Callable[[Array], Tuple[Array, Array]]:
    """``x |-> (f, b)`` with ``f_r = kf_r x^{alpha_r}``, ``b_r = kb_r x^{beta_r}``."""
    alpha = jnp.asarray(alpha, float)
    beta = jnp.asarray(beta, float)
    kf = jnp.asarray(kf, float)
    kb = jnp.asarray(kb, float)

    def fluxes(x: Array) -> Tuple[Array, Array]:
        f = kf * jnp.prod(x[None, :] ** alpha, axis=1)
        b = kb * jnp.prod(x[None, :] ** beta, axis=1)
        return f, b

    return fluxes


def mass_action_rhs(
    alpha: Array, beta: Array, kf: Array, kb: Array
) -> Callable[[Array], Array]:
    """The mass-action vector field ``x |-> sum_r zeta_r v_r(x)``."""
    Z = jnp.asarray(beta, float) - jnp.asarray(alpha, float)
    fluxes = one_way_fluxes(alpha, beta, kf, kb)

    def rhs(x: Array) -> Array:
        f, b = fluxes(x)
        return Z.T @ (f - b)

    return rhs


def check_detailed_balance(
    alpha: Array, beta: Array, kf: Array, kb: Array, x_eq: Array, tol: float = 1e-9
) -> Array:
    """Verify ``kf_r x_eq^{alpha_r} = kb_r x_eq^{beta_r}`` for every ``r``.

    Returns the equilibrium fluxes ``w_r`` (the common value); raises
    ``ValueError`` if ``x_eq`` is not a detailed-balance point (relative
    residual above ``tol``), or if positivity/reversibility fails.
    """
    x_eq = jnp.asarray(x_eq, float)
    if not bool(jnp.all(x_eq > 0)):
        raise ValueError("check_detailed_balance: x_eq must be strictly positive")
    if not (bool(jnp.all(jnp.asarray(kf, float) > 0)) and bool(jnp.all(jnp.asarray(kb, float) > 0))):
        raise ValueError("check_detailed_balance: need kf, kb > 0 (reversibility)")
    wf, wb = one_way_fluxes(alpha, beta, kf, kb)(x_eq)
    resid = jnp.abs(wf - wb) / jnp.maximum(jnp.maximum(wf, wb), 1e-300)
    if not bool(jnp.all(resid <= tol)):
        raise ValueError(
            f"check_detailed_balance: x_eq is not a detailed-balance point "
            f"(max relative flux residual {float(jnp.max(resid)):.3e} > {tol:g})"
        )
    return wf


# ---------------------------------------------------------------------------
# 1. The monolithic encoding: Mielke--Onsager sharp + free-energy potential.
# ---------------------------------------------------------------------------


def log_mean(u: Array, v: Array) -> Array:
    """The logarithmic mean ``Lambda(u, v) = (u - v)/(log u - log v)``, ``Lambda(u,u) = u``.

    Guarded at ``u ~ v``: when ``|log(u/v)| < 1e-10`` the midpoint ``(u+v)/2``
    is used; the two branches then differ by relative ``O(log(u/v)^2) < 1e-20``,
    below rounding, so the guard costs no exactness.
    """
    r = jnp.log(u / v)
    close = jnp.abs(r) < 1e-10
    safe_r = jnp.where(close, 1.0, r)
    return jnp.where(close, 0.5 * (u + v), (u - v) / safe_r)


def onsager_rvect(
    alpha: Array, beta: Array, kf: Array, kb: Array, dt: float = 1.0
) -> ReactiveVectorSpace:
    """The Mielke--Onsager reactive vector space ``(R^n, dt * K(x))`` (module docstring, 1).

    ``K(x) = sum_r Lambda(f_r, b_r) zeta_r zeta_r^T`` is symmetric PSD on
    ``x > 0``; state-dependent, like ``replicator.shahshahani``. ``dt`` is the
    Euler step, carried by the sharp.
    """
    Z = jnp.asarray(beta, float) - jnp.asarray(alpha, float)
    n = int(Z.shape[1])
    fluxes = one_way_fluxes(alpha, beta, kf, kb)

    def sharp_fn(x: Array) -> Array:
        f, b = fluxes(x)
        lam = log_mean(f, b)
        return dt * (Z.T * lam) @ Z

    return ReactiveVectorSpace(dim=n, sharp_fn=sharp_fn)


def free_energy(x_eq: Array) -> Callable[[Array], Array]:
    """The free energy ``F(x) = sum_i x_i (log(x_i/x_eq_i) - 1)`` relative to ``x_eq``."""
    x_eq = jnp.asarray(x_eq, float)

    def F(x: Array) -> Array:
        return jnp.sum(x * (jnp.log(x / x_eq) - 1.0))

    return F


def crn_arrangement(
    alpha: Array, beta: Array, kf: Array, kb: Array, x_eq: Array, dt: float = 1.0
) -> SmoothArrangement:
    """The closed sarr-scalar ``((R^n, dt*K(x)), !, !, F) : I -> I`` (module docstring, 1).

    Verifies detailed balance at ``x_eq`` (raising ``ValueError`` otherwise),
    since without it the Onsager pair ``(K, F)`` does not reproduce the
    mass-action field and the construction would silently change the kinetics.
    """
    check_detailed_balance(alpha, beta, kf, kb, x_eq)
    F = free_energy(x_eq)
    Q = onsager_rvect(alpha, beta, kf, kb, dt)

    return SmoothArrangement(
        Q=Q,
        out_dim_M=0, in_dim_M=0, out_dim_N=0, in_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: F(q),
        label="crn",
    )


def crn_dynamics(
    alpha: Array, beta: Array, kf: Array, kb: Array, x_eq: Array, dt: float = 1.0
) -> PCMorphism:
    """``Phiconf`` of the monolithic reaction-network arrangement."""
    return Phiconf(crn_arrangement(alpha, beta, kf, kb, x_eq, dt))


_IN_POS = (jnp.zeros(0), trivial_omega(0))
_IN_DIR = (jnp.zeros(0), jnp.zeros(0))


def crn_step(O: PCMorphism, x: Array) -> Array:
    """One Phiconf Euler step of a closed system from concentrations ``x``."""
    _, _, new_x = O.with_state(x).run_one(_IN_POS, lambda _o: _IN_DIR)
    return new_x


# ---------------------------------------------------------------------------
# 2. The compositional encoding for unimolecular reversible networks
#    (kuramoto pattern: species boxes + reaction boxes + routing wire).
# ---------------------------------------------------------------------------


def unimolecular_stoichiometry(
    n_species: int, edges: Sequence[Tuple[int, int]]
) -> Tuple[Array, Array]:
    """``(alpha, beta)`` for the network with one reaction ``A_i <-> A_j`` per edge."""
    R = len(edges)
    alpha = jnp.zeros((R, n_species))
    beta = jnp.zeros((R, n_species))
    for r, (i, j) in enumerate(edges):
        alpha = alpha.at[r, i].set(1.0)
        beta = beta.at[r, j].set(1.0)
    return alpha, beta


def _incidences(
    n_species: int, edges: Sequence[Tuple[int, int]]
) -> Tuple[List[int], List[int]]:
    """Species degrees and the routing bijection (cf. ``kuramoto._incidences``).

    Species ``s`` gets one out-port per reaction it takes part in (out-ports
    species-major); reaction box ``r = (i, j)`` has in-port 0 for ``a_i`` and
    in-port 1 for ``a_j`` (in-ports reaction-major). Both sides have ``2R``
    ports; ``perm`` satisfies ``in_f(m_out)[p] = m_out[perm[p]]``.
    """
    for r, (i, j) in enumerate(edges):
        if i == j:
            raise ValueError(f"reaction {r} = ({i}, {j}) is a self-loop; not allowed")
        if not (0 <= i < n_species and 0 <= j < n_species):
            raise ValueError(f"reaction {r} = ({i}, {j}) out of range")

    out_port: Dict[Tuple[int, int], int] = {}
    degrees: List[int] = []
    idx = 0
    for s in range(n_species):
        deg = 0
        for r, (i, j) in enumerate(edges):
            if s in (i, j):
                out_port[(s, r)] = idx
                idx += 1
                deg += 1
        degrees.append(deg)

    perm: List[int] = []
    for r, (i, j) in enumerate(edges):
        perm.append(out_port[(i, r)])
        perm.append(out_port[(j, r)])
    return degrees, perm


def species_box(x_eq_i: float, degree: int, dt: float = 1.0) -> SmoothArrangement:
    """The species box ``Sp_i : <R^0|R^0> -> <R^0 | R^{degree}>``.

    Parameter ``Q_i = R`` holding the concentration ``x_i``, with CONSTANT
    scalar sharp ``dt * x_eq_i`` (the species' own equilibrium weight -- the
    only equilibrium datum the species carries); readout the activity
    ``a_i = x_i / x_eq_i`` broadcast to the ``degree`` incident reaction
    ports; no inputs, no potential.
    """
    x_eq_i = float(x_eq_i)
    if x_eq_i <= 0:
        raise ValueError("species_box: x_eq_i must be positive")

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.tile(q / x_eq_i, degree)

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.array(0.0)

    return SmoothArrangement(
        Q=diagonal(jnp.full(1, dt * x_eq_i)),
        out_dim_M=0, in_dim_M=0,
        out_dim_N=degree, in_dim_N=0,
        out_f=out_f, in_f=in_f, U=U,
        label=f"Sp(xeq={x_eq_i:g})",
    )


def unimolecular_reaction_box(w: float) -> SmoothArrangement:
    """The stateless reaction box ``Rxn_r : <R^0|R^0> -> <R^2 | R^0>``.

    Trivial parameter; two input ports carrying the endpoint activities
    ``(a_i, a_j)``; potential ``(w/2)(a_i - a_j)^2``, ``w`` the reaction's
    equilibrium flux (its only datum). All its dynamics flows back through
    the wiring as the cotangent pullback of this potential to the two
    species' parameters (cf. ``kuramoto.kuramoto_coupling``).
    """
    w = float(w)

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return 0.5 * w * (n_in[0] - n_in[1]) ** 2

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=0, in_dim_M=0,
        out_dim_N=0, in_dim_N=2,
        out_f=out_f, in_f=in_f, U=U,
        label=f"Rxn(w={w:g})",
    )


def unimolecular_wire(
    n_species: int, edges: Sequence[Tuple[int, int]]
) -> SmoothArrangement:
    """The routing wire (image of a finite-set lens under ``R^-``, lem.lens_pow).

    Source ``<R^{2R} | R^{2R}>`` (all boxes' ports), target the unit
    ``<R^0|R^0>``, trivial parameter, ``U = 0``; routes each reaction box's
    two in-ports from its endpoint species' broadcast out-ports by the
    bijection ``perm`` of ``_incidences`` (cf. ``kuramoto.kuramoto_wire``).
    """
    R = len(edges)
    _, perm = _incidences(n_species, edges)
    perm = jnp.asarray(perm, dtype=int)

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return m_out[perm]

    def U(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.array(0.0)

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=2 * R, in_dim_M=2 * R,
        out_dim_N=0, in_dim_N=0,
        out_f=out_f, in_f=in_f, U=U,
        label=f"uni_wire(n={n_species}, R={R})",
    )


def unimolecular_arrangement(
    n_species: int,
    edges: Sequence[Tuple[int, int]],
    kf: Sequence[float],
    kb: Sequence[float],
    x_eq: Array,
    dt: float = 1.0,
) -> SmoothArrangement:
    """The closed unimolecular network, by genuine composition in ``sarr``:

        compose_seq( tensor(Sp_1, ..., Sp_n, Rxn_1, ..., Rxn_R),
                     unimolecular_wire(n, edges) )  :  <R^0|R^0> -> <R^0|R^0>,

    with parameter ``R^n`` (the concentrations, species order; the stateless
    reaction boxes contribute nothing) and CONSTANT diagonal sharp
    ``dt * diag(x_eq)``. The total potential ``sum_r (w_r/2)(a_i - a_j)^2``
    emerges from ``compose_seq``'s writer-monad addition -- it is written
    nowhere by hand. Detailed balance ``kf_r x_eq_i = kb_r x_eq_j`` is
    verified per reaction (this fixes ``w_r``); with cycles present this is
    exactly the Wegscheider condition, and violating rates are rejected.
    """
    x_eq = jnp.asarray(x_eq, float)
    alpha, beta = unimolecular_stoichiometry(n_species, edges)
    w = check_detailed_balance(alpha, beta, kf, kb, x_eq)

    degrees, _ = _incidences(n_species, edges)
    boxes = [
        species_box(float(x_eq[s]), degrees[s], dt) for s in range(n_species)
    ] + [unimolecular_reaction_box(float(w[r])) for r in range(len(edges))]
    wired = compose_seq(tensor_arrangements(boxes), unimolecular_wire(n_species, edges))
    return replace(wired, label=f"uni_crn(n={n_species}, R={len(edges)})")


def unimolecular_dynamics(
    n_species: int,
    edges: Sequence[Tuple[int, int]],
    kf: Sequence[float],
    kb: Sequence[float],
    x_eq: Array,
    dt: float = 1.0,
) -> PCMorphism:
    """``Phiconf`` of the compositional unimolecular arrangement."""
    return Phiconf(unimolecular_arrangement(n_species, edges, kf, kb, x_eq, dt))


# ---------------------------------------------------------------------------
# 3. The obstruction certificate (module docstring, 3).
# ---------------------------------------------------------------------------


def separability_ratio(
    rhs: Callable[[Array], Array], i: int, j: int
) -> Callable[[Array], Array]:
    """``x |-> (dV_j/dx_i)(x) / (dV_i/dx_j)(x)`` for the vector field ``V = rhs``.

    In any species-box realization ``x_k' = -g_k(x_k) dU/dx_k`` (the general
    compositional form: direct-sum sharp, shared potential) the closedness of
    ``dU`` forces this ratio to equal ``g_j(x_j)/g_i(x_i)`` -- a separable
    function, independent of every OTHER coordinate, in every per-species
    coordinate system. A network whose ratio varies with a third species
    therefore has NO species-box realization; ``A + B <-> C`` fails this test
    (the (A, C) ratio is ``(kf/kb) x_B``). Only meaningful where the
    denominator entry is nonzero; for the irreversible failure mode (a zero
    denominator against a nonzero numerator) compare the Jacobian entries
    directly.
    """
    import jax

    def ratio(x: Array) -> Array:
        J = jax.jacobian(rhs)(x)
        return J[j, i] / J[i, j]

    return ratio


# ---------------------------------------------------------------------------
# 4. Arbitrary Petri nets: doubled transition boxes, species as wiring
#    (module docstring, 4; the paper's "Petri nets" item).
# ---------------------------------------------------------------------------


# The transistor pair of logic.py: sharpR(xi, xi') = (xi', 0). Repeated here
# rather than imported so that this module does not depend on an extension.
_TRANSISTOR_SHARP = jnp.array([[0.0, 1.0], [0.0, 0.0]])


def petri_check(m: Array, n: Array, r: Array, x0: Array) -> Tuple[Array, Array, Array, Array]:
    """Validate a Petri net ``(m, n, r, x0)`` and return it as float arrays."""
    m = jnp.asarray(m, float)
    n = jnp.asarray(n, float)
    r = jnp.asarray(r, float)
    x0 = jnp.asarray(x0, float)
    if m.shape != n.shape or m.ndim != 2:
        raise ValueError("petri_check: m, n must be (T, S) of the same shape")
    if r.shape != (m.shape[0],) or x0.shape != (m.shape[1],):
        raise ValueError("petri_check: r must be (T,) and x0 must be (S,)")
    if not bool(jnp.all(m >= 0) and jnp.all(n >= 0)):
        raise ValueError("petri_check: arc multiplicities must be nonnegative")
    if not bool(jnp.all(m == jnp.round(m)) and jnp.all(n == jnp.round(n))):
        raise ValueError("petri_check: arc multiplicities must be integers")
    if not bool(jnp.all(r >= 0)):
        raise ValueError("petri_check: rate constants must be nonnegative")
    if not bool(jnp.all(x0 >= 0)):
        raise ValueError("petri_check: initial concentrations must be nonnegative")
    return m, n, r, x0


def petri_incidences(m: Array, n: Array) -> List[List[int]]:
    """``S_t``, the species each transition touches (as reactant or product)."""
    m, n = jnp.asarray(m, float), jnp.asarray(n, float)
    return [
        [s for s in range(m.shape[1]) if bool(m[t, s] > 0 or n[t, s] > 0)]
        for t in range(m.shape[0])
    ]


def petri_transition_box(
    m_local: Array, r_t: float, dt: float = 1.0, label: str = "Trans"
) -> SmoothArrangement:
    """The doubled transition box ``<R^0|R^0> -> <R^{S_t} | R>`` (module docstring, 4).

    Parameter ``(R^2, sharpR)`` with the transistor sharp ``(xi, xi') |->
    (dt * xi', 0)``, state ``(eps_t, eps'_t)``; readout the extent ``eps_t``;
    inputs the concentrations at the ``len(m_local)`` species the transition
    touches, in the order given by ``petri_incidences``; potential

        U_t((eps_t, eps'_t), x_local) = -eps'_t * r_t * prod_s x_s^{m_local_s}.

    ``m_local`` is the transition's REACTANT multiplicities at those species
    (its product multiplicities live in the wiring, not here). ``dt`` rides on
    the sharp, as in ``onsager_rvect``.
    """
    m_local = jnp.asarray(m_local, float)
    r_t = float(r_t)
    if r_t < 0:
        raise ValueError("petri_transition_box: rate constant must be nonnegative")
    k = int(m_local.shape[0])

    def out_f(q: Array, m_out: Array) -> Array:
        return q[:1]  # report the extent eps_t

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return -q[1] * r_t * jnp.prod(n_in ** m_local)

    return SmoothArrangement(
        Q=constant(dt * _TRANSISTOR_SHARP),
        out_dim_M=0, in_dim_M=0,
        out_dim_N=1, in_dim_N=k,
        out_f=out_f, in_f=in_f, U=U,
        label=f"{label}(r={r_t:g})",
    )


def petri_wire(m: Array, n: Array, x0: Array) -> SmoothArrangement:
    """The species wiring ``(x)_t <R^{S_t}|R> -> <R^0|R^0>`` (module docstring, 4).

    Static (trivial parameter, ``U = 0``) but not a prism: its input map is the
    AFFINE map carrying the transitions' reported extents to the concentrations

        x_s = x0_s + sum_t (n_ts - m_ts) eps_t,

    gathered to each transition's in-ports (cf. the feedback stage of
    ``amplifier.py``, whose input map is likewise affine rather than a
    permutation).
    """
    m, n, _, x0 = petri_check(m, n, jnp.zeros(jnp.asarray(m).shape[0]), x0)
    Z = n - m
    idx = jnp.asarray([s for St in petri_incidences(m, n) for s in St], dtype=int)

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return (x0 + Z.T @ m_out)[idx]  # m_out = the reported extents

    def U(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.array(0.0)

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=int(Z.shape[0]), in_dim_M=int(idx.shape[0]),
        out_dim_N=0, in_dim_N=0,
        out_f=out_f, in_f=in_f, U=U,
        label=f"species_wire(S={int(Z.shape[1])}, T={int(Z.shape[0])})",
    )


def petri_arrangement(
    m: Array, n: Array, r: Array, x0: Array, dt: float = 1.0
) -> SmoothArrangement:
    """The closed Petri net, by genuine composition in ``sarr``:

        compose_seq( tensor(Trans_t for t in T), petri_wire(m, n, x0) ),

    a ``<R^0|R^0> -> <R^0|R^0>`` arrangement with parameter ``R^{2T}`` holding
    ``(eps_t, eps'_t)_t`` and block-diagonal nilpotent sharp. The total
    potential ``-sum_t eps'_t f_t(x)`` emerges from ``compose_seq``'s writer
    addition; it is written nowhere by hand.
    """
    m, n, r, x0 = petri_check(m, n, r, x0)
    St = petri_incidences(m, n)
    boxes = [
        petri_transition_box(m[t, jnp.asarray(St[t], dtype=int)], float(r[t]), dt)
        for t in range(int(m.shape[0]))
    ]
    wired = compose_seq(tensor_arrangements(boxes), petri_wire(m, n, x0))
    return replace(wired, label=f"petri(S={int(m.shape[1])}, T={int(m.shape[0])})")


def petri_dynamics(
    m: Array, n: Array, r: Array, x0: Array, dt: float = 1.0
) -> PCMorphism:
    """``Phiconf`` of the Petri-net arrangement."""
    return Phiconf(petri_arrangement(m, n, r, x0, dt))


def petri_initial_state(num_transitions: int, eps_prime: float = 1.0) -> Array:
    """The state ``(eps_t, eps'_t)_t = (0, eps_prime)_t`` at time zero.

    ``eps_prime`` is arbitrary and never changes: it only scales ``U``, whose
    ``eps'``-derivative -- the only component the sharp reads -- is independent
    of it.
    """
    return jnp.tile(jnp.array([0.0, float(eps_prime)]), num_transitions)


def petri_extents(state: Array) -> Array:
    """The extents ``(eps_t)_t`` of a Petri-net state (its even coordinates)."""
    return jnp.asarray(state)[0::2]


def petri_concentrations(m: Array, n: Array, x0: Array, state: Array) -> Array:
    """``x = x0 + Z^T eps`` read off a Petri-net state."""
    m, n = jnp.asarray(m, float), jnp.asarray(n, float)
    return jnp.asarray(x0, float) + (n - m).T @ petri_extents(state)


def petri_step(O: PCMorphism, state: Array) -> Array:
    """One ``Phiconf`` tick of a closed Petri-net coalgebra."""
    _, _, new_state = O.with_state(state).run_one(_IN_POS, lambda _o: _IN_DIR)
    return new_state


# ---------------------------------------------------------------------------
# 5. Arbitrary Petri nets in log coordinates: doubled species boxes,
#    stateless transitions, prism wiring; positive by construction
#    (module docstring, 5).
# ---------------------------------------------------------------------------


def petri_log_incidences(m: Array, n: Array) -> Tuple[List[int], List[int]]:
    """Species degrees and the routing bijection for the log encoding.

    Species ``s`` gets one PAIR of out-ports ``(x_s, y'_s)`` per transition it
    takes part in (out-ports species-major, ``x`` before ``y'``); transition
    ``t`` expects its touched species' pairs in ``petri_incidences`` order
    (in-ports transition-major). ``perm`` satisfies
    ``in_f(m_out)[p] = m_out[perm[p]]`` and is a bijection: the wire is a
    prism, unlike the affine ``petri_wire``.
    """
    St = petri_incidences(m, n)
    T, S = jnp.asarray(m).shape

    out_port: Dict[Tuple[int, int], int] = {}
    degrees: List[int] = []
    idx = 0
    for s in range(int(S)):
        deg = 0
        for t in range(int(T)):
            if s in St[t]:
                out_port[(s, t)] = idx
                idx += 2
                deg += 1
        degrees.append(deg)

    perm: List[int] = []
    for t in range(int(T)):
        for s in St[t]:
            perm.append(out_port[(s, t)])
            perm.append(out_port[(s, t)] + 1)
    return degrees, perm


def petri_log_species_box(degree: int, dt: float = 1.0) -> SmoothArrangement:
    """The doubled species box ``<R^0|R^0> -> <R^0 | R^{2 degree}>``.

    Parameter ``(R^2, sharpR)`` with the transistor sharp ``(xi, xi') |->
    (dt * xi', 0)``, state ``(y_s, y'_s)``; readout the pair ``(x_s, y'_s)``,
    ``x_s = exp(y_s)``, broadcast to the ``degree`` incident transitions; no
    inputs, no potential. The exp Jacobian of the readout pulls incoming
    ``x``-port covectors back into the ``xi``-slot, which the sharp discards;
    the ``y'``-port covectors land untouched in the ``xi'``-slot, which drives
    ``y_s``. The box carries NO reaction data -- not even its own initial
    concentration, which is the initial state, not a structural constant.
    """

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.tile(jnp.stack([jnp.exp(q[0]), q[1]]), degree)

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.array(0.0)

    return SmoothArrangement(
        Q=constant(dt * _TRANSISTOR_SHARP),
        out_dim_M=0, in_dim_M=0,
        out_dim_N=2 * degree, in_dim_N=0,
        out_f=out_f, in_f=in_f, U=U,
        label=f"LogSp(deg={degree})",
    )


def petri_log_transition_box(
    m_local: Array, zeta_local: Array, r_t: float
) -> SmoothArrangement:
    """The STATELESS transition box ``<R^0|R^0> -> <R^0 | R^{2k}>``-dual,
    ``k = len(m_local)``: trivial parameter, in-ports the pairs
    ``(x_s, y'_s)`` at the ``k`` species it touches, potential

        V_t = -f_t(x) * sum_s zeta_local_s * y'_s / x_s,
        f_t(x) = r_t * prod_s x_s^{m_local_s}.

    ``dV_t/dy'_s = -zeta_local_s f_t(x)/x_s`` is the (signed, per-species)
    log-rate itself, independent of ``y'``; all ``x``-derivatives flow to
    discarded slots. Unlike ``petri_transition_box``, the box carries its
    PRODUCT multiplicities too (inside ``zeta_local``), since delivery no
    longer routes through affine wiring.
    """
    m_local = jnp.asarray(m_local, float)
    zeta_local = jnp.asarray(zeta_local, float)
    r_t = float(r_t)
    if r_t < 0:
        raise ValueError("petri_log_transition_box: rate constant must be nonnegative")
    k = int(m_local.shape[0])

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        x, yp = n_in[0::2], n_in[1::2]
        f = r_t * jnp.prod(x ** m_local)
        return -f * jnp.sum(zeta_local * yp / x)

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=0, in_dim_M=0,
        out_dim_N=0, in_dim_N=2 * k,
        out_f=out_f, in_f=in_f, U=U,
        label=f"LogTrans(r={r_t:g})",
    )


def petri_log_wire(m: Array, n: Array) -> SmoothArrangement:
    """The routing prism: a port BIJECTION, cf. ``unimolecular_wire``.

    No affine map and no ``x0``: species compute their own concentrations,
    so the wire merely routes each species' broadcast ``(x_s, y'_s)`` pair to
    the transitions touching it.
    """
    _, perm = petri_log_incidences(m, n)
    perm_arr = jnp.asarray(perm, dtype=int)
    P = int(perm_arr.shape[0])

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return m_out[perm_arr]

    def U(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.array(0.0)

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=P, in_dim_M=P,
        out_dim_N=0, in_dim_N=0,
        out_f=out_f, in_f=in_f, U=U,
        label=f"log_wire(P={P})",
    )


def petri_log_arrangement(m: Array, n: Array, r: Array, dt: float = 1.0) -> SmoothArrangement:
    """The closed log-coordinate Petri net, by genuine composition in ``sarr``:

        compose_seq( tensor(LogSp_s for s in S, LogTrans_t for t in T),
                     petri_log_wire(m, n) )  :  <R^0|R^0> -> <R^0|R^0>,

    with parameter ``R^{2S}`` holding ``(y_s, y'_s)_s`` (the stateless
    transition boxes contribute nothing) and block-diagonal nilpotent sharp.
    The total potential ``-sum_t f_t(x) sum_s zeta_ts y'_s/x_s`` emerges from
    ``compose_seq``'s writer addition. Note ``x0`` is NOT an argument: initial
    concentrations enter through ``petri_log_initial_state``.
    """
    m, n, r, _ = petri_check(m, n, r, jnp.zeros(jnp.asarray(m).shape[1]))
    Z = n - m
    St = petri_incidences(m, n)
    degrees, _ = petri_log_incidences(m, n)
    boxes = [petri_log_species_box(degrees[s], dt) for s in range(int(m.shape[1]))] + [
        petri_log_transition_box(
            m[t, jnp.asarray(St[t], dtype=int)],
            Z[t, jnp.asarray(St[t], dtype=int)],
            float(r[t]),
        )
        for t in range(int(m.shape[0]))
    ]
    wired = compose_seq(tensor_arrangements(boxes), petri_log_wire(m, n))
    return replace(wired, label=f"petri_log(S={int(m.shape[1])}, T={int(m.shape[0])})")


def petri_log_dynamics(m: Array, n: Array, r: Array, dt: float = 1.0) -> PCMorphism:
    """``Phiconf`` of the log-coordinate Petri-net arrangement."""
    return Phiconf(petri_log_arrangement(m, n, r, dt))


def petri_log_initial_state(x0: Array, y_prime: float = 1.0) -> Array:
    """The state ``(y_s, y'_s)_s = (log x0_s, y_prime)_s`` at time zero.

    Requires ``x0 > 0`` strictly: the chart is ``y = log x``.  ``y_prime`` is
    arbitrary and never changes, as in ``petri_initial_state``.
    """
    x0 = jnp.asarray(x0, float)
    if not bool(jnp.all(x0 > 0)):
        raise ValueError("petri_log_initial_state: initial concentrations must be strictly positive")
    return jnp.stack(
        [jnp.log(x0), jnp.full(x0.shape[0], float(y_prime))], axis=1
    ).reshape(-1)


def petri_log_concentrations(state: Array) -> Array:
    """``x = exp(y)`` read off a log-Petri state (its even coordinates)."""
    return jnp.exp(jnp.asarray(state)[0::2])


def log_mass_action_step(
    m: Array, n: Array, r: Array, dt: float = 1.0
) -> Callable[[Array], Array]:
    """Ground truth for the tests: ``x |-> x * exp(dt * (Z^T f)(x) / x)``,
    the forward-Euler step of mass action in log coordinates."""
    m = jnp.asarray(m, float)
    Z = jnp.asarray(n, float) - m
    r = jnp.asarray(r, float)

    def step(x: Array) -> Array:
        f = r * jnp.prod(x[None, :] ** m, axis=1)
        return x * jnp.exp(dt * (Z.T @ f) / x)

    return step
