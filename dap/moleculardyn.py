"""Molecular dynamics as leapfrog phase dynamics of a wired arrangement.

EXTENSION (an *instance*, not new machinery). The claim under test: the
standard MD simulation loop -- velocity-Verlet integration of point particles
under pairwise Lennard-Jones forces -- is the leapfrog phase dynamics
``Phileap`` (leapfrog.py, rmk.multistage) of an arrangement wired from

* **particle boxes** ``Part_i : <R^0|R^0> -> <R^{d*c_i}|R^0>`` -- parameter
  ``Q_i = R^d`` (the position, ``d in {1, 2}``) with *constant* mass-like sharp
  ``sharpR(xi) = (dt^2/m) xi`` (see below), output map the diagonal broadcast
  of the position to the ``c_i`` pair boxes the particle participates in,
  vacuous input map, ``U_i = 0``;
* **stateless LJ pair boxes** ``LJ_p : <R^0|R^0> -> <R^0|R^{2d}>`` -- trivial
  parameter ``R^0`` (no state), no output, input the two reported positions
  ``(r, r')``, and potential the Lennard-Jones pair energy

      U_LJ(r, r') = 4 eps ((sigma/|r - r'|)^12 - (sigma/|r - r'|)^6);

* the **prism wiring** ``graph_wire`` (eqn.prism_f) on the bipartite graph
  particles -> pair-boxes, one edge per membership, routing each particle's
  broadcast out-port onto the pair boxes' in-ports.

The closed composite (``compose_seq(tensor(parts), graph_wire(...))``, exactly
the ``compose_graph`` pattern of sec.graph_laplacian) is a 0-ary arrangement
with parameter ``Q = (R^d)^N`` whose total potential

    U(q) = sum_{pairs p = (i,j)} U_LJ(q_i, q_j)

*emerges* from the composition -- it is not written by hand.

Where the timestep lives
------------------------

The framework's integrators take a *unit* step; there is no ``dt`` slot. The
physical timestep therefore enters through the reactive metric: the particle
sharp is the constant ``(dt^2/m) I`` and the stored momentum coordinate is
``xi = (m/dt) v``. Under this identification one ``Phileap`` macro-tick

    xi_half = xi - (1/2) dU(q);  q' = q + sharpR(xi_half);
    xi'     = xi_half - (1/2) dU(q')

is *literally* one velocity-Verlet step of timestep ``dt``,

    v_half = v - (dt/2m) dU(q);  q' = q + dt v_half;
    v'     = v_half - (dt/2m) dU(q'),

with the two force evaluations at ``q`` and ``q'`` being the two ``pc^(2)``
rounds; and the kinetic energy *via the sharp* is exact:

    (1/2) <xi, sharpR(xi)> = (1/2) (m/dt)^2 |v|^2 (dt^2/m) = (1/2) m |v|^2.

Equivalently: the sharp is ``xi / m_eff`` with effective mass ``m_eff = m/dt^2``
(time measured in ticks). The arrangement's potential is the *physical* LJ
energy, unscaled, so the parameter covector ``xi_Q`` that the interpretation
hands the integrator (eqn.bigtheta) is exactly ``dU`` and ``-xi_Q`` is the
physical force.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import SmoothArrangement
from .interpretation import smooth_interpretation, trivial_omega
from .rvect import diagonal, trivial
from .wiring import compose_seq, graph_wire, tensor_arrangements


# ---------------------------------------------------------------------------
# The two generators: particle box and stateless LJ pair box.
# ---------------------------------------------------------------------------


def particle_box(mass: float, dt: float, n_pairs: int, dim: int = 1) -> SmoothArrangement:
    """The particle ``Part_i : <R^0|R^0> -> <R^{dim*n_pairs}|R^0>``.

    Parameter ``Q_i = R^dim`` (the position) with constant sharp
    ``(dt^2/mass) I`` (mass-like; see module docstring for why ``dt^2`` lives
    here). Output map broadcasts the position to the ``n_pairs`` pair boxes the
    particle feeds; input map vacuous; ``U = 0`` -- all interaction energy
    lives in the pair boxes.
    """

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.tile(q, n_pairs)  # diagonal broadcast, as in harmonic_vertex

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)  # bang

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.array(0.0)

    return SmoothArrangement(
        Q=diagonal(jnp.full(dim, dt * dt / mass)),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=dim * n_pairs,
        in_dim_N=0,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label=f"Part(m={mass:g})",
    )


def lj_pair_energy(r: Array, rp: Array, eps: float, sigma: float) -> Array:
    """The Lennard-Jones pair energy ``4 eps ((sigma/|r-r'|)^12 - (sigma/|r-r'|)^6)``.

    Computed via ``s6 = (sigma^2/|r-r'|^2)^3`` so no square root appears
    (smooth away from the ``r = r'`` singularity, which LJ genuinely has).
    """
    d = r - rp
    r2 = jnp.sum(d * d)
    s6 = (sigma * sigma / r2) ** 3
    return 4.0 * eps * (s6 * s6 - s6)


def lj_pair_box(eps: float, sigma: float, dim: int = 1) -> SmoothArrangement:
    """The stateless pair-interaction box ``LJ_p : <R^0|R^0> -> <R^0|R^{2*dim}>``.

    Trivial parameter ``R^0`` (no state, nothing for an integrator to update),
    no output ports, ``2*dim`` input ports receiving the two reported particle
    positions, and ``U`` the Lennard-Jones energy of their separation. The
    entire content of the box is its potential.
    """

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.zeros(0)  # bang

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)  # bang

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return lj_pair_energy(n_in[:dim], n_in[dim:], eps, sigma)

    return SmoothArrangement(
        Q=trivial(),
        out_dim_M=0,
        in_dim_M=0,
        out_dim_N=0,
        in_dim_N=2 * dim,
        out_f=out_f,
        in_f=in_f,
        U=U,
        label=f"LJ(eps={eps:g}, sigma={sigma:g})",
    )


# ---------------------------------------------------------------------------
# The wired composite: particles + pair boxes + prism wiring.
# ---------------------------------------------------------------------------


def md_arrangement(
    num_particles: int,
    pairs: Sequence[Tuple[int, int]],
    mass: float,
    eps: float,
    sigma: float,
    dt: float,
    dim: int = 1,
) -> SmoothArrangement:
    """The closed MD arrangement: particles wired into LJ pair boxes.

    Vertices of the wiring graph are the ``N = num_particles`` particle boxes
    followed by the ``P = len(pairs)`` pair boxes; for pair ``p = (i, j)`` the
    edges ``(i, N+p)`` and ``(j, N+p)`` carry the two reported positions (each
    a ``dim``-block) into the pair box. The composite

        compose_seq( tensor(Part_1, ..., Part_N, LJ_1, ..., LJ_P),
                     graph_wire(N + P, edges, vdim=dim) )

    is a closed 0-ary arrangement with parameter ``(R^dim)^N``, block sharp
    ``(dt^2/mass) I``, and emergent potential ``sum_p U_LJ(q_i, q_j)``.
    """
    N, P = num_particles, len(pairs)
    counts = [0] * N
    edges: List[Tuple[int, int]] = []
    for p, (i, j) in enumerate(pairs):
        if not (0 <= i < N and 0 <= j < N and i != j):
            raise ValueError(f"pair {p} = ({i}, {j}) is not a pair of distinct particles")
        edges.append((i, N + p))
        edges.append((j, N + p))
        counts[i] += 1
        counts[j] += 1

    parts = [particle_box(mass, dt, counts[i], dim=dim) for i in range(N)]
    ljs = [lj_pair_box(eps, sigma, dim=dim) for _ in range(P)]
    wired = compose_seq(
        tensor_arrangements(parts + ljs), graph_wire(N + P, edges, vdim=dim)
    )
    return replace(wired, label=f"md(N={N}, P={P}, dim={dim})")


# ---------------------------------------------------------------------------
# Framework-level observables: the force covector and the total energy.
# ---------------------------------------------------------------------------


def md_force(arr: SmoothArrangement, q: Array) -> Array:
    """The physical force ``-xi_Q`` at ``q``: ONE covector evaluation through
    the framework.

    Runs the direction action of the smooth interpretation ``Phi'_interpsm``
    (eqn.bigtheta) of the closed arrangement at parameter position ``q`` with
    the trivial boundary data, and returns ``-xi_Q`` -- the negated parameter
    covector every integrator consumes. Since the arrangement's potential is
    the physical LJ energy, this is the physical force, with no ``dt`` factor.
    """
    _, direction_action = smooth_interpretation(arr)(jnp.asarray(q))
    xi_Q, _, _ = direction_action(
        jnp.zeros(0), trivial_omega(0), jnp.zeros(0), jnp.zeros(0)
    )
    return -xi_Q


def md_energy(arr: SmoothArrangement, q: Array, xi: Array) -> Array:
    """Total energy: kinetic via the sharp plus the composite potential.

        H(q, xi) = (1/2) <xi, sharpR_q(xi)> + U(q)

    With the particle sharp ``(dt^2/m) I`` and ``xi = (m/dt) v`` the kinetic
    term equals ``(1/2) m |v|^2`` exactly, and ``U`` is the emergent sum of LJ
    pair energies, so ``H`` is the physical total energy.
    """
    ke = 0.5 * jnp.dot(xi, arr.Q.apply_sharp(q, xi))
    pe = arr.U(q, jnp.zeros(0), jnp.zeros(0))
    return ke + pe
