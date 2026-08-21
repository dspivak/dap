"""The port-Hamiltonian DC motor in ``sarr``: audit of the paper's "first pass".

TARGET (van der Schaft, "Port-Hamiltonian systems theory: an introductory
overview", 2014, Example 2.5). States: flux linkage ``phi`` (electrical,
inductance ``L``) and angular momentum ``p`` (mechanical, inertia ``J``);
Hamiltonian ``H = phi^2/(2L) + p^2/(2J)``; dynamics

    phi' = -(R/L) phi - K (p/J) + V         (KVL: back-EMF K*omega)
    p'   =  K (phi/L) - (b/J) p - tau        (torque K*i, friction b)

i.e. ``[phi'; p'] = (Jgyr - Rdiss) grad H + u`` with ``Jgyr = [[0,-K],[K,0]]``,
``Rdiss = diag(R, b)``, ``u = (V, -tau)``. We write ``i = phi/L`` (current) and
``omega = p/J`` (angular velocity) for the two *flows*.

THE SENTENCE UNDER AUDIT (the paper's conclusion): "We expect sarr to
accommodate the DC motor by coupling an electrical and a mechanical system so
that the torque is proportional to current and the voltage is proportional to
angular velocity. A first pass suggests that configuration dynamics Phiconf
suffices when applied to two systems, each with a quadratic potential and with
reactive vector space T^*R equipped with the opposite -sharpS of the symplectic
sharp; the electrical system reports its momentum and the mechanical reports
its position."

VERDICT, derived against the code (``configuration_integrator`` steps
``q -> q - sharpR_q(xi_Q)`` with ``xi_Q = d(U_total)`` for a closed composite;
``compose_seq`` adds potentials; sharps direct-sum blockwise):

1. "sarr accommodates the DC motor by coupling an electrical and a mechanical
   system, torque prop. to current, voltage prop. to angular velocity" -- TRUE,
   and Phiconf suffices, but with ONE-dimensional boxes, not T^*R:

       dc_motor = compose_seq( electrical_cell (x) mechanical_cell, gyrator )

   * ``electrical_cell``: parameter ``phi``, sharp ``+dt*L``, potential
     ``(R/2) i^2 - V i`` (Ohmic dissipation + source), reports ``i = phi/L``;
   * ``mechanical_cell``: parameter ``p``, sharp ``-dt*J`` (NEGATIVE), potential
     ``-((b/2) omega^2 + tau*omega)`` (the NEGATED dissipation + load), reports
     ``omega = p/J``;
   * ``gyrator``: stateless, potential ``K * i * omega`` -- the gyrator
     constant is literally ``K``.

   One Phiconf step is EXACTLY the forward-Euler step of the target ODE (all
   parameters, all states, sources included; test 5). Each box carries only its
   own physical constants. The covector the coupling pulls back to ``phi`` is
   the back-EMF ``K*omega / L`` and to ``p`` the torque covector ``K*i / J``.

2. The OPPOSITE-SIGN sharps are FORCED, not a choice: any coupling mediated by
   a potential contributes a SYMMETRIC cross-Hessian, so the one-step map's
   cross-Jacobian entries are ``(-s_e * H12, -s_m * H12)`` -- their sign ratio
   is ``s_e / s_m``.  The motor's gyroscopic coupling has OPPOSITE-signed cross
   terms (``-K/J`` back-EMF vs ``+K/L`` torque), hence ``s_e * s_m < 0``: with
   1-dim boxes, exactly one of the two boxes must carry a negative sharp (and,
   to keep its own friction dissipative, the negated potential; the flow
   ``-sharp(dU)`` is invariant under ``(sharp, U) -> (-sharp, -U)``, so the
   negation is only visible through the coupling).  This is the Lotka-Volterra
   sign mechanism. The ratio is pinned: ``s_e / s_m = -L/J``.

3. The sentence's OWN encoding -- two ``T^*R`` boxes with sharp ``-sharpS``
   (per the paper's eqn.sharp_S, ``sharpS(xi_x, xi_y) = (xi_y, -xi_x)``, so
   Phiconf with ``-sharpS`` steps each box by +Hamilton's equations of its
   potential), electrical reporting its momentum ``y1``, mechanical its
   position ``x2``, quadratic potentials -- is WRONG on the nose, twice over:

   (a) STATE MISMATCH / MISSING TERM. The coupling potential ``Uc(y1, x2)``
       pulls back only to the ``x1``- and ``y2``-updates (each box's report is
       the coordinate the coupling can push on, and ``-sharpS`` rotates that
       push onto the CONJUGATE coordinate). The flux update is
       ``y1' = y1 - dt * dU1/dx1(x1, y1)``: it sees box 1 alone.  For
       ``(y1, y2) = (phi, p)`` to evolve autonomously the ``x1``-dependence
       must vanish, and then ``phi'`` can contain NO ``p``-term: the back-EMF
       ``-K omega`` cannot be produced (test 9).
   (b) VOLUME. With every box carrying an antisymmetric sharp the closed
       composite's field is ``(S (+) S) dU_total``: divergence-free for EVERY
       potential (``tr(antisym * sym) = 0``, test 10), while the motor
       contracts at rate ``R/L + b/J > 0``. No choice of quadratic potentials,
       reports, or wiring with all-antisymmetric sharps yields the lossy motor
       as the dynamics of the box states.

   What survives (test 11): a BATEMAN-STYLE EMBEDDING. For ``R > 0``, ``K != 0``
   there are quadratic potentials (``bateman_coefficients``) making the motor
   the dynamics on a 2-dim invariant subspace ``x1 = f1 phi + f2 p``,
   ``x2 = g1 phi + g2 p`` of the 4-dim composite, with ``(phi, p) = (y1, y2)``
   -- exactly per the sentence's reports -- while the complementary 2-dim
   "adjoint" subspace EXPANDS at the mirror rate (the spectrum is
   ``{lam, -lam}``-symmetric, as it must be for a Hamiltonian flow). The
   emulation therefore needs the hidden coordinates initialized on the
   subspace (a measure-zero condition) and is exponentially non-robust.

4. "each with a quadratic potential" -- TRUE in the corrected encoding
   (quadratic-plus-linear once sources are on), but the mechanical one is
   CONCAVE (negated), per point 2.

5. "the electrical system reports its momentum and the mechanical reports its
   position" -- HALF WRONG. Electrical: ``phi`` is the electrical momentum (the
   flux linkage, conjugate to charge -- cf. dap/lc_circuit.py) and the cell
   reports ``i = phi/L``, its momentum up to the fixed scalar ``1/L``: right.
   Mechanical: back-EMF is proportional to angular VELOCITY, so the mechanical
   cell must report ``omega = p/J`` -- its MOMENTUM (up to ``1/J``), not its
   position. Both boxes report their momentum/flow; only the Bateman embedding
   realizes the literal "mechanical reports position", in the weak sense of 3.

MONOLITHIC CONTROL (test 1): one closed 2-dim box, sharp
``dt*[[R, K], [-K, b]]`` (constant, non-symmetric; its symmetric part is the
dissipation ``Rdiss``, its antisymmetric part ``-Jgyr``), potential ``U = H``:
one Phiconf step is exactly Euler for the sourceless motor. Constant sources as
linear potential terms must use the coefficient ``c = -A^{-1} u``,
``A = Rdiss - Jgyr`` -- every potential term is a covector and passes through
the sharp, so the naive ``U_src = -V phi + tau p`` (i.e. ``c = -u``) produces
the flow ``dt*A u`` instead of ``dt*u``, off by exactly ``dt*(A - I) u``
(test 2; ``naive_sources=True`` is that negative control). In the two-box
encoding the sharps are scalar per box, so the sources sit in the cells'
potentials with no such correction.

Lossless case ``R = b = 0`` (tests 3, 4): the sharp is purely antisymmetric and
one derives EXACTLY ``H_{n+1} = (1 + (omega0*dt)^2) H_n`` with
``omega0 = K/sqrt(LJ)`` (for the step matrix ``I + M``: ``Q M`` is antisymmetric
so the O(dt) terms cancel and ``M^T Q M = (omega0*dt)^2 Q``): geometric energy
growth at the explicit-Euler rate -- ``O(dt)`` in the exponent over a fixed
horizon -- while the energy trades fully between ``phi^2/(2L)`` and
``p^2/(2J)`` at frequency ``omega0``.

Units: the coalgebra tick is unit time; the physical time step ``dt`` lives in
the sharps (as in dap/kuramoto.py). ``locked_rotor=True`` freezes the
mechanical state by the degenerate sharp ``0`` (as dap/logic.py freezes its
voltage rails), which holds ``omega = 0``: the stall configuration, in-framework.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import SmoothArrangement
from .rvect import constant, trivial
from .wiring import compose_seq, tensor_arrangements


# ---------------------------------------------------------------------------
# 1. The monolithic control: one closed box, state (phi, p).
# ---------------------------------------------------------------------------


def motor_monolithic(
    R: float,
    L: float,
    K: float,
    J: float,
    b: float,
    V: float = 0.0,
    tau: float = 0.0,
    dt: float = 0.01,
    naive_sources: bool = False,
) -> SmoothArrangement:
    """The DC motor as ONE closed box ``<R^0|R^0> -> <R^0|R^0>``.

    Parameter ``Q = R^2 = (phi, p)`` with constant NON-symmetric sharp

        sharpR = dt * A,   A = Rdiss - Jgyr = [[R, K], [-K, b]],

    and potential ``U = H + c . (phi, p)`` where ``H`` is the motor Hamiltonian
    and ``c = -A^{-1} (V, -tau)`` encodes the constant sources (module
    docstring; requires ``det A = R b + K^2 != 0`` when a source is on). Then
    ``q - sharpR(dU) = q + dt*(-A grad H + u)``: exactly one forward-Euler step
    of the target ODE.

    ``naive_sources=True`` is the audit's negative control: it uses the sketch
    coefficient ``c = (-V, tau)`` instead, whose one-step flow is off by the
    constant ``dt*(A - I) u``.
    """
    A = np.array([[R, K], [-K, b]], dtype=float)
    u = np.array([V, -tau], dtype=float)
    if naive_sources:
        c = -u
    elif V == 0.0 and tau == 0.0:
        c = np.zeros(2)
    else:
        c = -np.linalg.solve(A, u)  # det A = R b + K^2 must be nonzero
    c = jnp.asarray(c)

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return q[0] ** 2 / (2.0 * L) + q[1] ** 2 / (2.0 * J) + jnp.dot(c, q)

    return SmoothArrangement(
        Q=constant(dt * jnp.asarray(A)),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=0,
        in_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=U,
        label=f"motor_mono(R={R:g},L={L:g},K={K:g},J={J:g},b={b:g})",
    )


# ---------------------------------------------------------------------------
# 2. The TRUE compositional encoding: two 1-dim cells + stateless gyrator.
# ---------------------------------------------------------------------------


def electrical_cell(
    R: float, L: float, V: float = 0.0, dt: float = 0.01
) -> SmoothArrangement:
    """The electrical (armature) cell ``<R^0|R^0> -> <R^1_out | R^0_in>``.

    Parameter ``phi`` (flux linkage -- the electrical MOMENTUM, conjugate to
    charge, cf. dap/lc_circuit.py), constant scalar sharp ``+dt*L``, potential

        U_e(phi) = (R/2) i^2 - V i,     i = phi/L,

    reporting the current ``i`` on its single output port. Self-dynamics:
    ``phi -> phi - dt*L * dU_e/dphi = phi + dt*(V - R i)``.
    """

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        i = q[0] / L
        return (R / 2.0) * i**2 - V * i

    return SmoothArrangement(
        Q=constant(jnp.array([[dt * L]])),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=1,
        in_dim_N=0,
        out_f=lambda q, m_out: q / L,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=U,
        label=f"armature(R={R:g},L={L:g},V={V:g})",
    )


def mechanical_cell(
    b: float,
    J: float,
    tau: float = 0.0,
    dt: float = 0.01,
    locked: bool = False,
    sharp_sign: float = -1.0,
) -> SmoothArrangement:
    """The mechanical (rotor) cell ``<R^0|R^0> -> <R^1_out | R^0_in>``.

    Parameter ``p`` (angular MOMENTUM), constant scalar sharp ``-dt*J`` -- the
    OPPOSITE sign to the electrical cell's, which point 2 of the module
    docstring shows is forced -- and the correspondingly NEGATED potential

        U_m(p) = -((b/2) omega^2 + tau*omega),     omega = p/J,

    reporting the angular velocity ``omega``. Self-dynamics:
    ``p -> p - (-dt*J) * dU_m/dp = p + dt*(-b omega - tau)`` -- the negations
    cancel in ``-sharp(dU)``, so friction stays dissipative and the load
    opposes; only the gyrator coupling sees the sign.

    ``locked=True`` replaces the sharp by the degenerate ``0`` (def.rvect
    allows it; cf. the frozen rails of dap/logic.py): the rotor is held, the
    stall configuration. ``sharp_sign=+1.0`` is the audit's negative control
    (same-signed sharps), under which the torque comes out REVERSED (test 8).
    """
    s = 0.0 if locked else sharp_sign * dt * J

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        omega = q[0] / J
        return -((b / 2.0) * omega**2 + tau * omega)

    return SmoothArrangement(
        Q=constant(jnp.array([[s]])),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=1,
        in_dim_N=0,
        out_f=lambda q, m_out: q / J,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=U,
        label=f"rotor(b={b:g},J={J:g},tau={tau:g})" + ("[locked]" if locked else ""),
    )


def gyrator(K: float) -> SmoothArrangement:
    """The stateless electromechanical coupling ``<R^2_out | R^0_in> -> <R^0|R^0>``.

    Trivial parameter, no outer ports: it closes the pair. Potential

        U_c(i, omega) = K * i * omega

    read off its two input ports (the reported current and angular velocity) --
    the gyrator, with its literal constant ``K``. Its differential pulls back
    (eqn.bigtheta) to the covectors ``K*omega / L`` on ``phi`` (the back-EMF,
    scaled by the report map ``phi/L``) and ``K*i / J`` on ``p`` (the torque
    covector); the cells' opposite sharps turn this SYMMETRIC pair into the
    motor's ANTISYMMETRIC gyroscopic flow.
    """

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return K * m_out[0] * m_out[1]

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=2,
        in_dim_M=0,
        out_dim_N=0,
        in_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=U,
        label=f"gyrator(K={K:g})",
    )


def dc_motor(
    R: float,
    L: float,
    K: float,
    J: float,
    b: float,
    V: float = 0.0,
    tau: float = 0.0,
    dt: float = 0.01,
    locked_rotor: bool = False,
) -> SmoothArrangement:
    """The DC motor by genuine composition in ``sarr``:

        dc_motor = compose_seq( electrical_cell (x) mechanical_cell, gyrator ),

    a closed 0-ary arrangement with parameter ``R^2 = (phi, p)``, sharp
    ``diag(dt*L, -dt*J)``, and composite potential

        (R/2) i^2 - V i - (b/2) omega^2 - tau*omega + K i omega

    emerging from ``compose_seq`` (not written by hand). One Phiconf step is
    exactly one forward-Euler step of van der Schaft's Example 2.5.
    """
    cells = tensor_arrangements(
        [electrical_cell(R, L, V, dt), mechanical_cell(b, J, tau, dt, locked=locked_rotor)]
    )
    return compose_seq(cells, gyrator(K))


# ---------------------------------------------------------------------------
# 3. The sentence's own encoding: T^*R boxes with -sharpS, and its Bateman salvage.
# ---------------------------------------------------------------------------

#: The paper's symplectic sharp on T^*R (eqn.sharp_S): (xi_x, xi_y) |-> (xi_y, -xi_x).
_SHARP_S = jnp.array([[0.0, 1.0], [-1.0, 0.0]])


def tstar_cell(
    a: float, bq: float, d: float, report: str, dt: float = 0.01, label: str = ""
) -> SmoothArrangement:
    """A ``T^*R`` box of the quoted sentence: ``<R^0|R^0> -> <R^1_out | R^0_in>``.

    Parameter ``(x, y)`` (position, momentum) with sharp ``-dt * sharpS``
    (Phiconf then steps ``(x, y) -> (x, y) + dt*(dU/dy, -dU/dx)``: explicit
    Euler for +Hamilton's equations of ``U``), quadratic potential

        U(x, y) = (a/2) x^2 + bq * x y + (d/2) y^2,

    reporting ``x`` (``report="position"``) or ``y`` (``report="momentum"``).
    """
    if report not in ("position", "momentum"):
        raise ValueError(f"report must be 'position' or 'momentum', got {report!r}")
    k = 0 if report == "position" else 1

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return (a / 2.0) * q[0] ** 2 + bq * q[0] * q[1] + (d / 2.0) * q[1] ** 2

    return SmoothArrangement(
        Q=constant(-dt * _SHARP_S),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=1,
        in_dim_N=0,
        out_f=lambda q, m_out: q[k : k + 1],
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=U,
        label=label or f"tstar({report})",
    )


def bilinear_coupling(c: float) -> SmoothArrangement:
    """The stateless coupling of the sentence encoding: ``U_c = c * m_1 * m_2``
    read off the two reported coordinates. Same shape as ``gyrator``."""

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return c * m_out[0] * m_out[1]

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=2,
        in_dim_M=0,
        out_dim_N=0,
        in_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=U,
        label=f"bilinear({c:g})",
    )


def sentence_pair(
    coeffs_e: Tuple[float, float, float],
    coeffs_m: Tuple[float, float, float],
    c: float,
    dt: float = 0.01,
) -> SmoothArrangement:
    """The quoted sentence's encoding, verbatim, by genuine composition:

        compose_seq( tstar_cell(momentum) (x) tstar_cell(position),
                     bilinear_coupling(c) ),

    a closed arrangement with parameter ``R^4 = (x1, y1, x2, y2)``, blockwise
    sharp ``-dt*sharpS (+) -dt*sharpS``, and composite potential
    ``U1(x1,y1) + U2(x2,y2) + c*y1*x2`` (electrical reports its momentum
    ``y1``, mechanical its position ``x2``). ``coeffs_e = (a1, b1, d1)`` and
    ``coeffs_m = (a2, b2, d2)`` are the two quadratic forms of ``tstar_cell``.
    """
    boxes = tensor_arrangements(
        [
            tstar_cell(*coeffs_e, report="momentum", dt=dt, label="elec(T*R)"),
            tstar_cell(*coeffs_m, report="position", dt=dt, label="mech(T*R)"),
        ]
    )
    return compose_seq(boxes, bilinear_coupling(c))


def bateman_coefficients(
    R: float, L: float, K: float, J: float, b: float
) -> Dict[str, float]:
    """Quadratic potentials making ``sentence_pair`` carry the motor on an
    invariant subspace (the Bateman-style salvage; module docstring point 3).

    With the motor matrix entries ``e11 = -R/L``, ``e12 = -K/J``, ``e21 = K/L``,
    ``e22 = -b/J``, cross terms ``b1 = b2 = 0``, and the normalization
    ``g2 = 1``, the invariance conditions for the subspace

        x1 = f1*phi + f2*p,   x2 = g1*phi + g2*p,   (phi, p) = (y1, y2)

    solve in closed form (requires ``R > 0`` and ``K != 0``):

        g1 = -e21/e11,          a2 = -e22,
        c  = -e21*(e11+e22)/e11,  a1 = e11*e12/e21,
        f1 = -e11/a1,           f2 = -e12/a1,
        d1 = f1*e11 + f2*e21 - c*g1,   d2 = g1*e12 + e22.

    On that subspace the composite's Phiconf step restricts to the exact Euler
    step of the motor; the complementary subspace expands at the mirror rate.
    """
    if R <= 0.0 or K == 0.0:
        raise ValueError("bateman_coefficients requires R > 0 and K != 0")
    e11, e12, e21, e22 = -R / L, -K / J, K / L, -b / J
    g2 = 1.0
    g1 = -e21 / e11 * g2
    a2 = -e22 / g2
    c = -e21 * (e11 + e22) / e11
    a1 = e11 * e12 / (e21 * g2)
    f1 = -e11 / a1
    f2 = -e12 / a1
    d1 = f1 * e11 + f2 * e21 - c * g1
    d2 = g1 * e12 + g2 * e22
    return dict(a1=a1, d1=d1, a2=a2, d2=d2, c=c, f1=f1, f2=f2, g1=g1, g2=g2)


def bateman_motor(
    R: float, L: float, K: float, J: float, b: float, dt: float = 0.01
) -> Tuple[SmoothArrangement, Callable[[Array], Array], Callable[[Array], Array]]:
    """The Bateman salvage of the sentence encoding: ``(arrangement, embed, project)``.

    ``arrangement`` is ``sentence_pair`` with the ``bateman_coefficients``
    potentials (sourceless motor); ``embed(phi, p)`` places the hidden
    coordinates on the invariant subspace; ``project`` reads ``(phi, p) =
    (y1, y2)`` back off the 4-dim state. The match holds ONLY from embedded
    initial states, and the complementary directions expand.
    """
    C = bateman_coefficients(R, L, K, J, b)
    arr = sentence_pair(
        (C["a1"], 0.0, C["d1"]), (C["a2"], 0.0, C["d2"]), C["c"], dt=dt
    )

    def embed(y: Array) -> Array:
        phi, p = y[0], y[1]
        return jnp.array(
            [C["f1"] * phi + C["f2"] * p, phi, C["g1"] * phi + C["g2"] * p, p]
        )

    def project(z: Array) -> Array:
        return jnp.array([z[1], z[3]])

    return arr, embed, project


# ---------------------------------------------------------------------------
# Flow-coordinate variant (the paper's ex.further_reach "DC motor" item): the
# same motor with states (i, omega) instead of the momenta (phi, p) -- the
# constant rescaling phi = L i, p = J omega -- so its cells match the LC
# item's convention (state = the reported flow, sharp = +-dt/mass).
# ---------------------------------------------------------------------------


def electrical_flow_cell(R: float, L: float, V: float = 0.0, dt: float = 0.01) -> SmoothArrangement:
    """State ``i`` (the current), sharp ``+dt/L``, potential ``(R/2) i^2 - V i``,
    reporting ``i``."""
    return SmoothArrangement(
        Q=constant(jnp.array([[dt / L]])),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=1,
        in_dim_N=0,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: (R / 2.0) * q[0] ** 2 - V * q[0],
        label=f"armature-flow(R={R:g},L={L:g},V={V:g})",
    )


def mechanical_flow_cell(b: float, J: float, tau: float = 0.0, dt: float = 0.01) -> SmoothArrangement:
    """State ``omega`` (the angular velocity), oppositely-signed sharp
    ``-dt/J``, NEGATED potential ``-((b/2) omega^2 + tau*omega)`` (the
    ``(sharp, U) -> (-sharp, -U)`` invariance keeps friction dissipative),
    reporting ``omega``."""
    return SmoothArrangement(
        Q=constant(jnp.array([[-dt / J]])),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=1,
        in_dim_N=0,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: -((b / 2.0) * q[0] ** 2 + tau * q[0]),
        label=f"rotor-flow(b={b:g},J={J:g},tau={tau:g})",
    )


def dc_motor_flow(
    R: float,
    L: float,
    K: float,
    J: float,
    b: float,
    V: float = 0.0,
    tau: float = 0.0,
    dt: float = 0.01,
) -> SmoothArrangement:
    """``compose_seq( electrical_flow_cell (x) mechanical_flow_cell, gyrator )``:
    one Phiconf tick is exactly the Euler step

        i     -> i     + dt (V - R i - K omega) / L,
        omega -> omega + dt (K i - b omega - tau) / J.
    """
    cells = tensor_arrangements(
        [electrical_flow_cell(R, L, V, dt), mechanical_flow_cell(b, J, tau, dt)]
    )
    return compose_seq(cells, gyrator(K))
