"""Quadratic-drag 1-form (EXTENSION; integrator.quadratic_drag_kick / gyro_phase_integrator).

The blog's air drag -- force ``~ -|v| v`` -- as a 1-form ``omega_drag(q,xi)=(|v| v, 0)``
taken *per gyro*. We verify the rmk.adam split that makes it a legitimate integrator:

* MONOIDAL over ``(+)`` -- each gyro drags on its own velocity (the non-negotiable
  compositionality of rmk.adam);
* NATURAL only over per-gyro orthogonal maps -- a "smaller Q", which rmk.adam
  explicitly sanctions; the third test shows the restriction is *real* (a non-orthogonal
  per-gyro map breaks naturality), so it is not a vacuous claim;
* and the drag dissipates energy in a free rollout.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.functors import Phigyro
from dap.gyroscope import gyro_arrangement, make_config
from dap.integrator import quadratic_drag_kick
from dap.interpretation import trivial_omega

_IN_POS = (jnp.zeros(0), trivial_omega(0))


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _block_diag(mats):
    return np.asarray(jax.scipy.linalg.block_diag(*[jnp.asarray(m) for m in mats]))


def test_drag_is_monoidal_over_oplus():
    """Each gyro drags on its own velocity: F is block-diagonal, so F on the whole
    space is the concatenation of F on each 2-D gyro block -- monoidality over (+)."""
    rng = np.random.default_rng(0)
    V = 4
    v = jnp.asarray(rng.standard_normal(2 * V))
    whole = np.asarray(quadratic_drag_kick(v, 2))
    per_block = np.concatenate(
        [np.asarray(quadratic_drag_kick(v[2 * i : 2 * i + 2], 2)) for i in range(V)]
    )
    np.testing.assert_allclose(whole, per_block, atol=1e-12)


def test_drag_is_natural_over_block_orthogonal():
    """F(O v) = O F(v) for per-gyro orthogonal O (block O(2)): natural over the
    orthogonal subcategory -- exactly the restriction rmk.adam permits."""
    rng = np.random.default_rng(1)
    V = 4
    v = jnp.asarray(rng.standard_normal(2 * V))
    O = jnp.asarray(_block_diag([_rot(t) for t in rng.uniform(0, 2 * np.pi, V)]))
    lhs = np.asarray(quadratic_drag_kick(O @ v, 2))
    rhs = np.asarray(O @ quadratic_drag_kick(v, 2))
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_drag_is_NOT_natural_over_nonorthogonal():
    """A non-orthogonal per-gyro map (here area-preserving but not an isometry) breaks
    naturality -- the subcategory restriction is real, not vacuous, because |v_i| is
    not scale-invariant. This is why drag lives over the *orthogonal* subcategory."""
    rng = np.random.default_rng(2)
    V = 3
    v = jnp.asarray(rng.standard_normal(2 * V))
    D = jnp.asarray(_block_diag([np.diag([2.0, 0.5]) for _ in range(V)]))  # det 1, not orthogonal
    lhs = np.asarray(quadratic_drag_kick(D @ v, 2))
    rhs = np.asarray(D @ quadratic_drag_kick(v, 2))
    assert not np.allclose(lhs, rhs, atol=1e-6)


def test_drag_dissipates_energy():
    """A free (unforced) gyro network with drag > 0 bleeds out kinetic + potential
    energy -- the dissipative content of the 1-form, run through Phigyro."""
    cfg = make_config(rows=2, cols=3)  # 6 gyros
    V = cfg.V
    kappa = jnp.full(len(cfg.edges), 0.2)
    grav = jnp.array(0.1)
    arr = gyro_arrangement(cfg, inv_mass=jnp.ones(V), kappa=kappa, grav_k=grav)
    O = Phigyro(arr, damping=0.0, gamma=0.0, J=None, drag=0.3, gyro_block=2)

    rng = np.random.default_rng(0)
    q0 = jnp.asarray(0.5 * rng.standard_normal(2 * V))
    xi0 = jnp.asarray(0.8 * rng.standard_normal(2 * V))
    E = jnp.asarray(cfg.edges)

    def energy(q, xi):
        qv = q.reshape(V, 2)
        springs = 0.5 * jnp.sum(kappa * jnp.sum((qv[E[:, 0]] - qv[E[:, 1]]) ** 2, axis=1))
        return float(0.5 * jnp.sum(xi ** 2) + springs + 0.5 * grav * jnp.sum(qv ** 2))

    state = (q0, xi0)
    H0 = energy(*state)
    zero_force = jnp.zeros(cfg.n_in)
    for _ in range(60):
        _, _, state = O.with_state(state).run_one(
            _IN_POS, lambda op: (jnp.zeros(cfg.n_out), zero_force)
        )
    assert energy(*state) < 0.7 * H0  # drag bled energy out
