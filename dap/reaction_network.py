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

In short: detailed-balanced mass-action = ``Phiconf`` with an Onsager
(symmetric-PSD, free-energy-descending) structure, monolithically, exactly;
compositionally (species boxes + stateless reaction boxes) exactly the
unimolecular ones; the first bimolecular reaction already cannot be wired,
because wiring couples potentials only and the cross-species Onsager coupling
``zeta_r zeta_r^T`` cannot arise from a direct-sum sharp.
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
from .rvect import ReactiveVectorSpace, diagonal, trivial
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
