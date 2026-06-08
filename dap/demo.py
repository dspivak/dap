"""Run the paper's worked examples and print what happens.

Usage (from the ``misc`` directory):

    ../.venv/bin/python -m dap.demo

Each example builds a smooth adaptive arrangement and applies one of the two
dynamics functors (cor.functor): Phiconf for descent (Newton, gradient descent,
the heat equation) and Phiphase for Hamilton flow (the wave equation).
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phiconf, Phiphase
from dap.rvect import diagonal, euclidean, inverse_hessian
from dap.wiring import compose_chain
from dap.learning import parameterized_map, train, squared_error

_IN_POS = (jnp.zeros(0), (jnp.zeros((0, 0)), jnp.zeros(0)))


def _harmonic(m, kappa):
    return SmoothArrangement(
        diagonal(jnp.array([1.0 / m])), 0, 0, 1, 1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2,
    )


def _laplacian_pinned(v):
    aug = np.concatenate([[0.0], np.asarray(v), [0.0]])
    return aug[:-2] - 2.0 * aug[1:-1] + aug[2:]


def demo_gradient_descent():
    m, n = 3, 2
    dim = n * m + n
    F = lambda q, x: q[: n * m].reshape(n, m) @ x + q[n * m:]
    rng = np.random.default_rng(7)
    W_true, b_true = rng.standard_normal((n, m)), rng.standard_normal(n)
    data = [(jnp.asarray(x := rng.standard_normal(m)), jnp.asarray(W_true) @ x + jnp.asarray(b_true))
            for _ in range(20)]
    arr = parameterized_map(F, euclidean(dim, eta=0.05), m, n)
    q, hist = train(arr, jnp.zeros(dim), data, steps=4000)
    W_err = float(jnp.linalg.norm(q[: n * m].reshape(n, m) - jnp.asarray(W_true)))
    print(f"  gradient descent (Phiconf): loss {hist[0]:.3f} -> {hist[-1]:.1e},  weight error {W_err:.1e}")


def demo_newton():
    U = lambda q: jnp.exp(q[0]) - q[0]   # minimized at 0
    O = Phiconf(SmoothArrangement(
        inverse_hessian(U, 1), 0, 0, 0, 0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: U(q)))
    q, xs = jnp.array([2.0]), [2.0]
    for _ in range(6):
        _, _, q = O.with_state(q).run_one(_IN_POS, lambda _o: (jnp.zeros(0), jnp.zeros(0)))
        xs.append(float(q[0]))
    print("  newton (Phiconf):           x = " + " -> ".join(f"{v:.1e}" for v in xs) + "   (-> min at 0)")


def demo_wave():
    K, m, kappa = 5, 1.5, 0.9
    O = Phiphase(compose_chain([_harmonic(m, kappa)] * K))
    rng = np.random.default_rng(0)
    state = (jnp.asarray(rng.standard_normal(K)), jnp.zeros(K))
    traj = [np.asarray(state[0])]
    for _ in range(12):
        q = state[0]
        in_dir = (jnp.array([kappa * float(q[K - 1])]), jnp.array([0.0]))
        _, _, state = O.with_state(state).run_one(_IN_POS, lambda _o: in_dir)
        traj.append(np.asarray(state[0]))
    a = np.stack(traj)
    res = max(float(np.abs(m * (a[t + 2] - 2 * a[t + 1] + a[t]) - kappa * _laplacian_pinned(a[t])).max())
              for t in range(len(a) - 2))
    print(f"  wave (Phiphase):            discrete-wave residual {res:.1e}  (exact recurrence, not stable to run)")


def demo_heat():
    K, T, m, kappa = 9, 500, 1.0, 0.2
    O = Phiconf(compose_chain([_harmonic(m, kappa)] * K))
    rng = np.random.default_rng(4)
    q = jnp.asarray(rng.standard_normal(K))
    peak0 = float(np.max(np.abs(np.asarray(q))))
    for _ in range(T):
        in_dir = (jnp.array([kappa * float(q[K - 1])]), jnp.array([0.0]))
        _, _, q = O.with_state(q).run_one(_IN_POS, lambda _o: in_dir)
    print(f"  heat (Phiconf):             peak |q| {peak0:.2f} -> {float(np.max(np.abs(np.asarray(q)))):.1e}  (dissipates to equilibrium)")


def main():
    print("dynamic-algebra-potentials: worked examples\n")
    print("Phiconf -- descent dynamics:")
    demo_gradient_descent()
    demo_newton()
    demo_heat()
    print("\nPhiphase -- Hamiltonian dynamics:")
    demo_wave()


if __name__ == "__main__":
    main()
