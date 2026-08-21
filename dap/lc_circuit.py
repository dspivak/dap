"""The LC circuit: the electrical reading of one arrangement under two semantics.

An electrical instance of the wave/heat dichotomy (cf. test_wave_equation.py /
test_heat_equation.py): ONE smooth adaptive arrangement whose parameter is the
capacitor charge ``q``, with

* reactive sharp  ``sharpR(xi) = xi / L``   -- the inductance ``L`` in the mass slot,
* potential       ``U(q) = q^2 / (2 C)``    -- the capacitor's electrostatic energy,

read by the two dynamics functors of cor.functor:

* ``Phiphase`` stores ``(q, xi)`` and steps ``q -> q + xi/L``, ``xi -> xi - q~/C``
  (eqn.phase_update, forces at the presented position ``q~``). Because
  ``q_{n+1} - q_n = xi_n / L`` exactly, the stored momentum is
  ``xi_n = L (q_{n+1} - q_n) / dt``: the *flux linkage* ``lambda = L I``
  conjugate to charge, whose kinetic term ``xi^2/(2L)`` is the magnetic energy
  of the inductor. The circuit is a harmonic oscillator with angular frequency
  ``omega = 1/sqrt(LC)`` -- the same arrangement shape as the mechanical
  particle, with ``(mass, stiffness) = (L, 1/C)``.

* ``Phiconf`` stores ``q`` alone and steps ``q -> q - (1/L)(q/C)``: forward
  Euler for the first-order relaxation ``dq/dt = -q/(L C)``. Electrically this
  is RC discharge ``dq/dt = -q/(R C)`` with ``R := L``: under the first-order
  semantics the sharp is a *mobility* (a conductance ``1/R``, Ohm's law
  ``I = -V/R`` with ``V = dU/dq = q/C``), not an inductance. The arrangement
  fixes one number; the integrator decides whether it plays the reactive role
  (henries) or the dissipative one (ohms) -- exactly the paper's mass-vs-mobility
  reading of the wave/heat pair.

Units and time step: the coalgebra tick is one unit of time -- the integrators
of integrator.py carry no ``dt`` (eqn.phase_update steps by the sharp itself).
Working in tick units, physical data ``(L_phys, C_phys)`` at step ``dt``
correspond to ``(L, C) = (L_phys/dt, C_phys/dt)`` here, and then the stored
``xi`` *is* the physical flux linkage. Resolution therefore means choosing
``omega = 1/sqrt(LC)`` small in tick units.

The coupled pair (the multi-box variant): two capacitor boxes, each exposing
its charge on an output port, composed in ``sarr`` with a *stateless* coupling
box -- trivial parameter, no state -- carrying the coupling capacitor's energy
``(q_1 - q_2)^2/(2 C_c)`` read off its two input ports. The composite potential
emerges from ``compose_seq`` (it is not written by hand), and the normal modes
are ``omega_+ = 1/sqrt(LC)`` (charges in phase; the coupling capacitor carries
no charge) and ``omega_- = sqrt(1/(LC) + 2/(L C_c))`` (charges in antiphase).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .rvect import diagonal, trivial
from .wiring import compose_seq, tensor_arrangements


def lc_cell(L: float, C: float) -> SmoothArrangement:
    """A single LC cell ``<R^0|R^0> -> <R^1_out | R^0_in>``.

    Parameter ``Q = R`` is the capacitor charge ``q``, with constant reactive
    sharp ``xi |-> xi/L`` (the inductance in the mass slot) and potential
    ``U(q) = q^2/(2C)`` (the capacitor energy). The output port exposes the
    charge; there are no input ports. Run closed, feed the port the zero
    covector ``xi_N = 0`` (an unloaded terminal).
    """
    return SmoothArrangement(
        Q=diagonal(jnp.array([1.0 / L])),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=1,
        in_dim_N=0,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: q[0] ** 2 / (2.0 * C),
        label=f"LC(L={L:g},C={C:g})",
    )


def coupling_capacitor(C_c: float) -> SmoothArrangement:
    """The stateless coupling box ``<R^2_out | R^0_in> -> <R^0|R^0>``.

    Trivial parameter ``R^0`` (no state), no ports on the outside: it closes
    the pair. Its potential is the coupling capacitor's energy
    ``(m_1 - m_2)^2/(2 C_c)`` read off the two charges presented at its input
    ports; ``compose_seq`` adds it to the cells' own potentials and the
    interpretation's cotangent pullback (eqn.bigtheta) routes its differential
    back to both cells' parameter covectors.
    """
    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=2,
        in_dim_M=0,
        out_dim_N=0,
        in_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: (m_out[0] - m_out[1]) ** 2 / (2.0 * C_c),
        label=f"Cc({C_c:g})",
    )


def coupled_lc_pair(L: float, C: float, C_c: float) -> SmoothArrangement:
    """Two LC cells wired through the stateless coupling box, in ``sarr``:

        coupled = compose_seq( lc_cell (x) lc_cell,  coupling_capacitor ),

    a closed 0-ary arrangement with parameter ``R^2 = (q_1, q_2)``, sharp
    ``diag(1/L, 1/L)``, and composite potential

        q_1^2/(2C) + q_2^2/(2C) + (q_1 - q_2)^2/(2 C_c)

    emerging from genuine composition (no hand-specialization).
    """
    cells = tensor_arrangements([lc_cell(L, C), lc_cell(L, C)])
    return compose_seq(cells, coupling_capacitor(C_c))


# ---------------------------------------------------------------------------
# The two-box Phiconf encoding (ex.further_reach, "LC circuit"): an inductor
# cell and a capacitor cell with OPPOSITELY-SIGNED sharps, wired by a stateless
# arrangement whose potential is the product i*v and which reports v.
# ---------------------------------------------------------------------------


def inductor_cell(L: float, R: float = 0.0) -> SmoothArrangement:
    """The inductor as a Phiconf cell ``<R^0|R^0> -> <R^1_out | R^0_in>``.

    Parameter ``i`` (the current through the inductor), constant scalar sharp
    ``-1/L`` (NEGATIVE: a potential coupling has a symmetric cross-Hessian, so
    producing a rotation forces the two cells' sharps to differ in sign -- the
    Lotka-Volterra / DC-motor mechanism, cf. dap/dc_motor.py point 2),
    reporting ``i`` on its single output port. An optional series resistance
    ``R`` enters as the potential ``-(R/2) i^2`` -- NEGATED, because the cell's
    sharp is negative (dap/dc_motor.py's mechanical-cell convention: the flow
    ``-sharp(dU)`` is invariant under ``(sharp, U) -> (-sharp, -U)``); the
    self-dynamics is then the damped ``i -> i + (v - R i)/L``.
    """
    return SmoothArrangement(
        Q=diagonal(jnp.array([-1.0 / L])),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=1,
        in_dim_N=0,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: -(R / 2.0) * q[0] ** 2,
        label=f"ind(L={L:g},R={R:g})",
    )


def capacitor_cell(C: float) -> SmoothArrangement:
    """The capacitor as a Phiconf cell ``<R^0|R^0> -> <R^1_out | R^0_in>``:
    parameter ``v`` (the voltage across it), constant scalar sharp ``+1/C``,
    zero potential, reporting ``v``. (Same convention as dap/logic.py's
    ``node``: state = voltage, sharp = 1/C.)"""
    return SmoothArrangement(
        Q=diagonal(jnp.array([1.0 / C])),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=1,
        in_dim_N=0,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.0 * q[0],
        label=f"cap(C={C:g})",
    )


def current_voltage_coupler() -> SmoothArrangement:
    """The stateless arrangement ``f: <R^2_out|R^0> -> <R^1_out|R^0>`` of the
    paper's LC-circuit item: potential ``W(i, v) = i * v`` read off the two
    reports, output map ``(i, v) |-> v`` (the circuit's terminal voltage)."""
    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=2,
        in_dim_M=0,
        out_dim_N=1,
        in_dim_N=0,
        out_f=lambda q, m_out: m_out[1:2],
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: m_out[0] * m_out[1],
        label="iv",
    )


def lc_conf(L: float, C: float, R: float = 0.0) -> SmoothArrangement:
    """``compose_seq( inductor_cell (x) capacitor_cell, current_voltage_coupler )``.

    Parameter ``R^2 = (i, v)``, sharp ``diag(-1/L, +1/C)``, composite potential
    ``i*v`` (emerging from composition, not written by hand). One Phiconf tick
    is exactly the forward-Euler step of the LC equations

        i -> i + v/L,        v -> v - i/C,

    the discrete rotation-with-growth whose energy law is pinned in
    tests/test_lc_circuit.py. With ``R > 0`` the ``i``-update gains the damping
    ``-R i / L``, and a time-varying covector fed at the outer port (an
    antenna: it enters the ``v``-update through the report's pullback) drives
    a resonance peaked at ``omega = 1/sqrt(LC)`` -- the tuning of a radio.
    """
    cells = tensor_arrangements([inductor_cell(L, R), capacitor_cell(C)])
    return compose_seq(cells, current_voltage_coupler())
