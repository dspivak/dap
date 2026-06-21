"""The integrator is the optimizer (damped_phase_integrator).

One convex arrangement -- a quadratic well U(q) = 1/2 q.A.q -- read by three
integrators:

* ``Phiconf``   -- configuration / descent: gradient descent, converges.
* ``Phiphase``  -- conservative phase flow: oscillates forever, never converges.
* ``Phidamped`` -- phase with friction: heavy-ball momentum, converges.

So the same syntactic object yields SGD, conservative dynamics, or momentum,
selected by the integrator alone -- the optimizer analogue of wave-vs-heat.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.interpretation import trivial_omega
from dap.arrangement import SmoothArrangement
from dap.functors import Phiconf, Phidamped, Phiphase
from dap.rvect import euclidean

_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def _quadratic_well(A, eta=1.0):
    """Closed autonomous system with potential U(q) = 1/2 q.A.q and sharp eta*I."""
    A = jnp.asarray(A, float)
    d = int(A.shape[0])
    return SmoothArrangement(
        euclidean(d, eta), 0, 0, 0, 0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * q @ (A @ q),
        label="quadratic_well",
    )


def _conf_norms(O, q0, steps):
    q = jnp.asarray(q0, float)
    out = [float(jnp.linalg.norm(q))]
    for _ in range(steps):
        _, _, q = O.with_state(q).run_one(_IN_POS, _TRIV)
        out.append(float(jnp.linalg.norm(q)))
    return np.array(out)


def _phase_norms(O, q0, steps):
    state = (jnp.asarray(q0, float), jnp.zeros(len(q0)))
    out = [float(jnp.linalg.norm(state[0]))]
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one(_IN_POS, _TRIV)
        out.append(float(jnp.linalg.norm(state[0])))
    return np.array(out)


def test_integrator_is_the_optimizer():
    A = jnp.diag(jnp.array([1.0, 3.0, 9.0]))  # convex, ill-conditioned
    q0 = jnp.array([1.0, 1.0, 1.0])
    arr = _quadratic_well(A, eta=0.1)

    conf = _conf_norms(Phiconf(arr), q0, 600)
    phase = _phase_norms(Phiphase(arr), q0, 600)
    damped = _phase_norms(Phidamped(arr, 0.15), q0, 600)

    # configuration integrator: descent, converges to the minimum q = 0.
    assert conf[-1] < 1e-3
    # phase integrator: conservative, never converges -- still oscillating.
    assert phase[-100:].max() > 0.3
    # damped phase: momentum, converges.
    assert damped[-1] < 1e-3
