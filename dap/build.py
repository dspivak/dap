"""Interactive builder: compose your own adaptive arrangement and run it.

Usage (from the ``misc`` directory):

    ../.venv/bin/python -m dap.build

You pick the dynamics (``phase`` = Hamilton/conservative, ``conf`` = descent/
dissipative), pick a system from a small palette of prebuilt boxes (harmonic
particles wired as a chain or a graph, or a linear model trained by gradient
descent), give numeric parameters, and the tool composes the arrangement,
applies ``Phiconf``/``Phiphase`` (cor.functor), runs it, and prints a check
(the discrete wave/heat residual, or the training loss).

This is "Tier 1": the boxes come from a palette, so no smooth functions need to
be typed. Custom potentials/maps require a few lines of Python (see learning.py
and the worked examples).
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.interpretation import trivial_omega
from dap.arrangement import SmoothArrangement
from dap.functors import Phiconf, Phiphase
from dap.rvect import diagonal, euclidean
from dap.wiring import compose_chain, compose_graph
from dap.learning import parameterized_map, train, squared_error

_IN_POS_CLOSED = (jnp.zeros(0), trivial_omega(0))
_IN_DIR_CLOSED = (jnp.zeros(0), jnp.zeros(0))


# ---------------------------------------------------------------------------
# Prompt helpers.
# ---------------------------------------------------------------------------


def ask(prompt, default=None, parse=str, choices=None):
    """Prompt until a valid value is given; blank input takes the default."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            raw = str(default)
        try:
            val = parse(raw)
        except Exception as exc:  # noqa: BLE001 - re-prompt on any parse error
            print(f"  ? {exc}")
            continue
        if choices is not None and val not in choices:
            print(f"  ? choose one of {sorted(choices)}")
            continue
        return val


def _initial(K, spec):
    parts = spec.split()
    kind = parts[0]
    if kind == "zeros":
        return jnp.zeros(K)
    if kind == "sine":  # smooth standing-wave mode n (looks like a wave)
        n = int(parts[1]) if len(parts) > 1 else 1
        js = np.arange(1, K + 1)
        return jnp.asarray(np.sin(js * n * np.pi / (K + 1)))
    if kind == "bump":  # localized pulse -> splits into two travelling waves
        js = np.arange(1, K + 1)
        return jnp.asarray(np.exp(-0.5 * ((js - (K + 1) / 2.0) / max(K / 8.0, 1.0)) ** 2))
    seed = None if kind == "random" else int(kind)  # integer => reproducible draw
    return jnp.asarray(np.random.default_rng(seed).standard_normal(K))


def parse_init(spec):
    """'random' (fresh) / 'zeros' / 'sine [n]' (smooth mode) / 'bump' / integer seed."""
    spec = spec.strip().lower()
    parts = spec.split()
    if parts and parts[0] in ("random", "zeros", "bump"):
        return spec
    if parts and parts[0] == "sine":
        if len(parts) > 1:
            int(parts[1])  # validate the mode number
        return spec
    int(spec)  # otherwise an integer seed; raises -> ask() re-prompts
    return spec


def _laplacian_pinned(v):
    aug = np.concatenate([[0.0], np.asarray(v), [0.0]])
    return aug[:-2] - 2.0 * aug[1:-1] + aug[2:]


_BLOCKS = "▁▂▃▄▅▆▇█"


def _spark(v):
    """A one-line block-glyph profile of a vector (so you can see the shape)."""
    v = np.asarray(v, dtype=float)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return _BLOCKS[len(_BLOCKS) // 2] * len(v)
    idx = np.clip(((v - lo) / (hi - lo) * (len(_BLOCKS) - 1)).round().astype(int),
                  0, len(_BLOCKS) - 1)
    return "".join(_BLOCKS[i] for i in idx)


def _frames(a, dynamics, height=13):
    """Render a precomputed chain trajectory to (label, [frame_body, ...])."""
    T, K = a.shape
    amax = max(float(np.max(np.abs(a))), 1e-12)
    mid = height // 2               # axis row
    up, down = mid, height - 1 - mid  # rows available above / below the axis
    label = "Phileap (org^(2))" if dynamics == "leapfrog" else f"Phi{dynamics}"
    bodies = []
    for t in range(T):
        grid = [[" "] * K for _ in range(height)]
        for col in range(K):
            lv = int(np.clip(round(float(a[t, col]) / amax * mid), -down, up))
            grid[mid - lv][col] = "●"
        for col in range(K):  # baseline axis
            if grid[mid][col] == " ":
                grid[mid][col] = "·"
        bodies.append("\n".join("".join(row) for row in grid))
    return label, bodies


def _animate(a, dynamics, height=13, fps=18):
    """Play a precomputed chain trajectory as a terminal animation (Ctrl-C to stop)."""
    import sys
    import time

    T = a.shape[0]
    label, bodies = _frames(a, dynamics, height)
    try:
        for t, body in enumerate(bodies):
            sys.stdout.write("\033[H\033[2J")  # cursor home + clear screen
            sys.stdout.write(f"  {label}   wave on {a.shape[1]} particles   t = {t:>4}/{T - 1}   (Ctrl-C to stop)\n\n")
            sys.stdout.write(body + "\n")
            sys.stdout.flush()
            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        pass
    sys.stdout.write("\n")


def _save_animation(a, dynamics, path, height=13):
    """Write every animation frame to a text file, one labeled block per frame."""
    import os

    T, K = a.shape
    if os.path.isdir(path):  # a bare directory -> drop a default filename inside it
        path = os.path.join(path, f"dap_{dynamics}_K{K}.txt")
    label, bodies = _frames(a, dynamics, height)
    with open(path, "w") as fh:
        for t, body in enumerate(bodies):
            fh.write(f"  {label}   wave on {K} particles   t = {t:>4}/{T - 1}\n\n")
            fh.write(body + "\n\n")
    print(f"  saved {T} frames to {path}")


def _vec(v):
    return np.array2string(np.asarray(v), precision=2, suppress_small=True, max_line_width=70)


# ---------------------------------------------------------------------------
# Palette: prebuilt boxes.
# ---------------------------------------------------------------------------


def harmonic_particle(m, kappa):
    """A harmonic particle box: Q = R, sharp p|->p/m, out_f = id, U = (k/2)(q-y)^2."""
    return SmoothArrangement(
        diagonal(jnp.array([1.0 / m])), 0, 0, 1, 1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2,
        label="Part",
    )


def _graph_laplacian(num_vertices, edges):
    L = np.zeros((num_vertices, num_vertices))
    for (i, j) in edges:
        L[i, i] += 1.0
        L[j, j] += 1.0
        L[i, j] -= 1.0
        L[j, i] -= 1.0
    return L


def parse_graph(spec):
    """Parse 'path N' / 'ring N' / 'complete N' / explicit 'i-j i-j ...'."""
    parts = spec.split()
    if parts and parts[0] in ("path", "ring", "complete"):
        N = int(parts[1])
        if parts[0] == "path":
            edges = [(i, i + 1) for i in range(N - 1)]
        elif parts[0] == "ring":
            edges = [(i, (i + 1) % N) for i in range(N)]
        else:
            edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        return N, edges
    edges = [tuple(int(x) for x in p.split("-")) for p in parts]
    V = max(max(e) for e in edges) + 1
    return V, edges


# ---------------------------------------------------------------------------
# Runners.
# ---------------------------------------------------------------------------


def _chain_trajectory(dynamics, K, m, kappa, q0, steps):
    """Step the harmonic chain under the chosen dynamics; return the (steps+1, K) q-trajectory."""
    arr = compose_chain([harmonic_particle(m, kappa)] * K)
    if dynamics == "leapfrog":
        from .leapfrog import Phileap
        O = Phileap(arr)
        state = (q0, jnp.zeros(K))
        bdy = lambda op: (kappa * op[0], jnp.array([0.0]))  # pinned ends, from emitted q_K
        traj = [np.asarray(q0)]
        for _ in range(steps):
            _, _, state = O.with_state(state).run_one(_IN_POS_CLOSED, bdy)
            traj.append(np.asarray(state[0]))
        return np.stack(traj)

    O = Phiphase(arr) if dynamics == "phase" else Phiconf(arr)
    state = (q0, jnp.zeros(K)) if dynamics == "phase" else q0
    bdy = lambda op: (kappa * op[0], jnp.array([0.0]))  # pinned ends, from emitted q_K (presented)
    traj = [np.asarray(q0)]
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one(_IN_POS_CLOSED, bdy)
        traj.append(np.asarray(state[0] if dynamics == "phase" else state))
    return np.stack(traj)


def run_chain(dynamics, K, m, kappa, init_kind, steps, animate=False):
    q0 = _initial(K, init_kind)
    a = _chain_trajectory(dynamics, K, m, kappa, q0, steps)

    if animate:
        _animate(a, dynamics)
        while ask("watch again? (y/n)", "n", parse=str.lower, choices={"y", "n"}) == "y":
            _animate(a, dynamics)
        while True:
            path = ask("save frames to file? (path, or blank to skip)", "", parse=str.strip)
            if not path:
                break
            try:
                _save_animation(a, dynamics, path)
                break
            except OSError as exc:
                print(f"  ? could not write {path!r}: {exc}")

    space = "T*R^%d" % K if dynamics in ("phase", "leapfrog") else "R^%d" % K
    name = "Phileap (org^(2))" if dynamics == "leapfrog" else f"Phi{dynamics}"
    print(f"\nbuilt  wire_{K}(Part,...,Part) : I -> box,  {name},  state = {space}")
    print(f"  q(0)   = {_vec(a[0])}")
    print(f"  q({steps:>3}) = {_vec(a[-1])}")
    print(f"  shape  {_spark(a[0])}  ->  {_spark(a[-1])}")
    peaks = np.abs(a).max(axis=1)
    if dynamics == "leapfrog":
        res = max((float(np.abs(m * (a[t + 2] - 2 * a[t + 1] + a[t]) - kappa * _laplacian_pinned(a[t + 1])).max())
                   for t in range(len(a) - 2)), default=0.0)
        print(f"  centered WAVE recurrence  m*q'' = kappa*Lap(center) :  residual {res:.1e}")
        print(f"  peak |q|: {peaks[0]:.2f} -> {peaks.max():.2f}  (bounded -- symplectic, stays stable)")
    elif dynamics == "phase":
        res = max((float(np.abs(m * (a[t + 2] - 2 * a[t + 1] + a[t]) - kappa * _laplacian_pinned(a[t + 1])).max())
                   for t in range(len(a) - 2)), default=0.0)
        print(f"  centered WAVE recurrence  m*q'' = kappa*Lap(center) :  residual {res:.1e}  (exact identity)")
        bounded = peaks.max() < 10.0 * max(float(peaks[0]), 1e-12)
        print(f"  peak |q|: {peaks[0]:.2f} -> {peaks.max():.2f}  ({'bounded -- symplectic' if bounded else 'growing (step too large)'})")
    else:
        res = max((float(np.abs(m * (a[t + 1] - a[t]) - kappa * _laplacian_pinned(a[t])).max())
                   for t in range(len(a) - 1)), default=0.0)
        print(f"  discrete HEAT equation  m*q' = kappa*Lap :  residual {res:.1e}  (exact identity)")
        print(f"  peak |q|: {peaks[0]:.2f} -> {peaks[-1]:.1e}  ({'dissipating' if peaks[-1] < peaks[0] else 'growing (step too large)'})")


def run_graph(dynamics, V, edges, m, kappa, init_kind, steps):
    arr = compose_graph(V, edges, m, kappa)  # wire_G = R^{varphi_G}((Part_v)_v), eqn.graph_wire
    q0 = _initial(V, init_kind)
    L = _graph_laplacian(V, edges)

    if dynamics == "leapfrog":
        from .leapfrog import Phileap
        O = Phileap(arr)
        state = (q0, jnp.zeros(V))
        print(f"\nbuilt  {arr.label} : I -> I,  Phileap (org^(2)),  state = T*R^{V}")
        traj = [np.asarray(q0)]
        for _ in range(steps):
            _, _, state = O.with_state(state).run_one(_IN_POS_CLOSED, lambda op: _IN_DIR_CLOSED)
            traj.append(np.asarray(state[0]))
        a = np.stack(traj)
        peaks = np.abs(a).max(axis=1)
        print(f"  q(0)   = {_vec(a[0])}")
        print(f"  q({steps:>3}) = {_vec(a[-1])}")
        res = max((float(np.abs(m * (a[t + 2] - 2 * a[t + 1] + a[t]) + kappa * (L @ a[t + 1])).max())
                   for t in range(len(a) - 2)), default=0.0)
        print(f"  graph WAVE (leapfrog)  m*q'' = -kappa*L q(center) :  residual {res:.1e}")
        print(f"  peak |q|: {peaks[0]:.2f} -> {peaks.max():.2f}  (bounded -- symplectic)")
        return

    O = Phiphase(arr) if dynamics == "phase" else Phiconf(arr)
    state = (q0, jnp.zeros(V)) if dynamics == "phase" else q0
    space = "T*R^%d" % V if dynamics == "phase" else "R^%d" % V
    print(f"\nbuilt  {arr.label} : I -> I,  Phi{dynamics},  state = {space}")

    traj = [np.asarray(q0)]
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one(_IN_POS_CLOSED, lambda _o: _IN_DIR_CLOSED)
        traj.append(np.asarray(state[0] if dynamics == "phase" else state))
    a = np.stack(traj)

    print(f"  q(0)   = {_vec(a[0])}")
    print(f"  q({steps:>3}) = {_vec(a[-1])}")
    peaks = np.abs(a).max(axis=1)
    if dynamics == "phase":
        res = max(float(np.abs(m * (a[t + 2] - 2 * a[t + 1] + a[t]) + kappa * (L @ a[t + 1])).max())
                  for t in range(len(a) - 2))
        print(f"  graph WAVE equation  m*q'' = -kappa*L q(center) :  residual {res:.1e}  (exact identity)")
        bounded = peaks.max() < 10.0 * max(float(peaks[0]), 1e-12)
        print(f"  peak |q|: {peaks[0]:.2f} -> {peaks.max():.2f}  ({'bounded -- symplectic' if bounded else 'growing (step too large)'})")
    else:
        res = max(float(np.abs(m * (a[t + 1] - a[t]) + kappa * (L @ a[t])).max())
                  for t in range(len(a) - 1))
        print(f"  graph HEAT equation  m*q' = -kappa*L q :  residual {res:.1e}  (exact identity)")
        print(f"  peak |q|: {peaks[0]:.2f} -> {peaks[-1]:.1e}  ({'dissipating' if peaks[-1] < peaks[0] else 'growing (step too large)'})")


def run_gd(in_dim, out_dim, eta, ndata, steps):
    dim = out_dim * in_dim + out_dim

    def F(q, x):
        return q[: out_dim * in_dim].reshape(out_dim, in_dim) @ x + q[out_dim * in_dim:]

    rng = np.random.default_rng(0)
    W_true, b_true = rng.standard_normal((out_dim, in_dim)), rng.standard_normal(out_dim)
    data = [(jnp.asarray(x := rng.standard_normal(in_dim)), jnp.asarray(W_true) @ x + jnp.asarray(b_true))
            for _ in range(ndata)]

    arr = parameterized_map(F, euclidean(dim, eta), in_dim, out_dim)
    print(f"\nbuilt  linear model R^{in_dim} -> R^{out_dim} : I-open,  Phiconf,  state = R^{dim} (weights)")
    q, hist = train(arr, jnp.zeros(dim), data, steps=steps)
    full = float(np.mean([float(squared_error(F(q, x), lam)) for x, lam in data]))
    w_err = float(jnp.linalg.norm(q[: out_dim * in_dim].reshape(out_dim, in_dim) - jnp.asarray(W_true)))
    print(f"  gradient descent (backprop = lens backward pass)")
    print(f"  loss: {hist[0]:.3f} -> {hist[-1]:.1e}   full-batch {full:.1e},  weight error {w_err:.1e}")


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------


def main():
    print("dynamic-algebra-potentials: build your own arrangement\n")
    try:
        dynamics = ask("dynamics (phase=symplectic Hamilton, conf=descent, leapfrog=higher-order)", "phase",
                       parse=str.lower, choices={"phase", "conf", "leapfrog"})
        print("system?\n  1) chain of harmonic particles  (wave / heat)"
              "\n  2) graph of harmonic particles  (graph Laplacian)"
              "\n  3) gradient descent on a linear model  (conf)")
        system = ask("choose", 1, parse=int, choices={1, 2, 3})

        if system == 3 and dynamics != "conf":
            print("  (gradient descent is configuration dynamics; using conf)")
            dynamics = "conf"

        if system == 1:
            K = ask("K particles", 7, int)
            m = ask("mass m", 1.0, float)
            kappa = ask("spring kappa", 0.9 if dynamics in ("phase", "leapfrog") else 0.2, float)
            init = ask("initial displacement (random / zeros / sine [n] / bump / seed)", "random",
                       parse=parse_init)
            steps = ask("steps", 60, int)
            animate = ask("animate? (y/n)", "y", parse=str.lower, choices={"y", "n"}) == "y"
            run_chain(dynamics, K, m, kappa, init, steps, animate)
        elif system == 2:
            V, edges = ask("graph ('path N' / 'ring N' / 'complete N' / 'i-j i-j ...')",
                           "ring 6", parse=parse_graph)
            m = ask("mass m", 1.0, float)
            kappa = ask("spring kappa", 0.5 if dynamics in ("phase", "leapfrog") else 0.15, float)
            init = ask("initial displacement (random / zeros / sine [n] / bump / seed)", "random",
                       parse=parse_init)
            steps = ask("steps", 12, int)
            run_graph(dynamics, V, edges, m, kappa, init, steps)
        else:
            in_dim = ask("input dim", 3, int)
            out_dim = ask("output dim", 2, int)
            eta = ask("learning rate", 0.05, float)
            ndata = ask("num data points", 20, int)
            steps = ask("steps", 2000, int)
            run_gd(in_dim, out_dim, eta, ndata, steps)
    except (EOFError, KeyboardInterrupt):
        print("\nbye")
        return


if __name__ == "__main__":
    main()
