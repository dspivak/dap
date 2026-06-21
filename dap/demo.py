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

from dap.interpretation import trivial_omega
from dap.arrangement import SmoothArrangement
from dap.functors import Phiconf, Phidamped, Phiphase
from dap.rvect import diagonal, euclidean, inverse_hessian
from dap.wiring import compose_chain
from dap.learning import parameterized_map, train, squared_error

_IN_POS = (jnp.zeros(0), trivial_omega(0))


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
        q, p = state
        qK = float(q[K - 1] + p[K - 1] / m)  # pinned end at the PRESENTED position q~_K (eqn.wave_bc)
        in_dir = (jnp.array([kappa * qK]), jnp.array([0.0]))
        _, _, state = O.with_state(state).run_one(_IN_POS, lambda _o: in_dir)
        traj.append(np.asarray(state[0]))
    a = np.stack(traj)
    # eqn.recurrence: m * ddot q(t+1) = kappa * Lap q(t+1), Laplacian at the CENTER t+1.
    res = max(float(np.abs(m * (a[t + 2] - 2 * a[t + 1] + a[t]) - kappa * _laplacian_pinned(a[t + 1])).max())
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


def demo_leapfrog():
    from dap.leapfrog import Phileap
    K, m, kappa = 5, 1.5, 0.9
    O = Phileap(compose_chain([_harmonic(m, kappa)] * K))
    rng = np.random.default_rng(0)
    state = (jnp.asarray(rng.standard_normal(K)), jnp.zeros(K))
    peak = peak0 = float(np.max(np.abs(np.asarray(state[0]))))
    bdy = lambda op: (kappa * op[0], jnp.array([0.0]))
    for _ in range(400):
        _, _, state = O.with_state(state).run_one(_IN_POS, bdy)
        peak = max(peak, float(np.max(np.abs(np.asarray(state[0])))))
    print(f"  wave (leapfrog, org^(2)):   peak |q| {peak0:.2f} -> max {peak:.2f} over 400 steps  (bounded -- stable)")


def demo_optimizers():
    # One convex arrangement, three integrators: the optimizer IS the integrator.
    A = jnp.diag(jnp.array([1.0, 3.0, 9.0]))
    arr = SmoothArrangement(
        euclidean(3, 0.1), 0, 0, 0, 0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * q @ (A @ q))
    q0, triv = jnp.ones(3), lambda _o: (jnp.zeros(0), jnp.zeros(0))

    def run(O, phase):
        s = (q0, jnp.zeros(3)) if phase else q0
        for _ in range(600):
            _, _, s = O.with_state(s).run_one(_IN_POS, triv)
        return float(jnp.linalg.norm(s[0] if phase else s))

    c, p, d = run(Phiconf(arr), False), run(Phiphase(arr), True), run(Phidamped(arr, 0.15), True)
    print(f"  optimizers (one well, 3 integrators):  |q| after 600 -> descent {c:.0e},  phase {p:.2f} (oscillates),  momentum {d:.0e}")


def demo_pinn():
    from dap.pinn import solve_deep_ritz
    u, u_star, hist, x = solve_deep_ritz(steps=3000, seed=0)
    rel = float(np.linalg.norm(np.asarray(u) - np.asarray(u_star)) / np.linalg.norm(np.asarray(u_star)))
    print(f"  pinn (deep Ritz, Poisson, Phiconf):     energy {hist[0]:.3f} -> {hist[-1]:.3f},  rel error vs discrete solution {rel:.1%}")


def demo_system_id():
    from dap.system_id import identify, one_step_error, PENDULUM
    params, F, hist, O, dim, m = identify(PENDULUM, steps=2000, seed=1)
    err, base = one_step_error(params, F, O, dim, m, seed=101)
    print(f"  system id (pendulum: Phiphase -> net):  one-step error {err:.3f}  vs no-change baseline {base:.3f}")


def main():
    print("dynamic-algebra-potentials: worked examples\n")
    print("Phiconf -- descent dynamics:")
    demo_gradient_descent()
    demo_newton()
    demo_heat()
    print("\nPhiphase -- Hamiltonian dynamics (org):")
    demo_wave()
    print("\nLeapfrog -- symplectic two-stage (org^(2)):")
    demo_leapfrog()
    print("\nExtensions (beyond the paper; the same pieces, composed):")
    demo_optimizers()
    demo_pinn()
    demo_system_id()


if __name__ == "__main__":
    main()
