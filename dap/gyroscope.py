"""Gyroscopes-on-springs as a classifier (EXTENSION, beyond the paper).

A harmonic surrogate of the system of M. S. Bull & S. Achour, *Machine learning with
dynamics* (unconv.ai, 2026): a network of gyroscopes-on-rods coupled by springs,
trained to classify handwriting. Each force of that physical system is mapped to one
slot of the smooth adaptive-arrangement framework -- the forward dynamics is
*exactly* ``Phigyro`` (functors.py) applied to a single harmonic arrangement:

    gravity restoring   -> on-site potential ``U``     (harmonic well)
    spring coupling     -> edge potential ``U``        (graph Laplacian, sec.graph_laplacian)
    inertia / mass      -> the reactive sharp          (``sharpR = 1/m``)
    damping             -> the damping 1-form          (``c*zeta``; ex.damping_one_form, Phidamped)
    gyroscopic torque   -> the skew 1-form             (``gamma*omega_gyro``; Phigyro)
    external nudge      -> the open input port         (encoder forces drive the input gyros)

EXTENSION caveat (cf. system_id.py / pinn.py): only the *physics* is the paper's
construction. Each gyro is a 2-D harmonic particle (Q = R^2, sharp 1/m); the
network is their graph-Laplacian arrangement at R^2, read by ``Phigyro``. The
*linear encoder/decoder and the Adam loop are standard ML layered on top*, not part
of the paper's formal development. And the gyroscopic 1-form is itself beyond the
paper (see ``gyro_phase_integrator``).

What factors through the framework: the only thing that moves the gyros is
``Phigyro``'s state update -- a symplectic-Euler step of the Hamiltonian whose
potential is the spring+gravity arrangement and whose 1-form carries damping and
precession. Training backprops through the rollout with ``jax.grad``; because
``cot``'s backward part is reverse-mode AD (functors.cot_map), this gradient realizes
the same chain-rule pullbacks that ``cot`` encodes -- an AD-equivalence, *not* a
literal ``OrgMorphism.then`` composition. (Reserve ``org^(K)`` / rmk.multistage for
genuine multi-stage *integrators* like leapfrog/RK4; the training rollout is not one.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import SmoothArrangement
from .functors import Phigyro
from .interpretation import trivial_omega
from .rvect import constant

_IN_POS = (jnp.zeros(0), trivial_omega(0))


# ---------------------------------------------------------------------------
# Topology: a grid of gyros, input column on the left, output column on the right.
# ---------------------------------------------------------------------------


def grid_graph(rows: int, cols: int) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """A ``rows x cols`` grid of gyros with 4-neighbour springs.

    Returns ``(edges, in_gyros, out_gyros)``; the input gyros are the left column
    and the output gyros the right column, so the encoder drives the left edge and
    the decoder reads the right edge -- information must propagate across the
    springs to be classified, the coupling the blog found essential.
    """
    def idx(r, c):
        return r * cols + c

    edges: List[Tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    in_gyros = [idx(r, 0) for r in range(rows)]
    out_gyros = [idx(r, cols - 1) for r in range(rows)]
    return edges, in_gyros, out_gyros


def hex_graph(rows: int, cols: int) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """A ``rows x cols`` triangular/hex lattice of gyros (up to 6 neighbours interior).

    Axial offset: each ``(i, j)`` bonds east ``(i, j+1)``, south ``(i+1, j)``, and the
    hex diagonal south-west ``(i+1, j-1)`` (directed, so each undirected bond appears
    once). As in ``grid_graph`` the input gyros are the left column and the output
    gyros the right column, so information must cross the lattice -- the coupling the
    blog found essential. The blog's network is this shape; here at small scale (the
    goal is the factorization + the springs ablation, not their accuracy number).
    """
    def idx(i, j):
        return i * cols + j

    edges: List[Tuple[int, int]] = []
    for i in range(rows):
        for j in range(cols):
            if j + 1 < cols:
                edges.append((idx(i, j), idx(i, j + 1)))           # east
            if i + 1 < rows:
                edges.append((idx(i, j), idx(i + 1, j)))           # south
            if i + 1 < rows and j - 1 >= 0:
                edges.append((idx(i, j), idx(i + 1, j - 1)))       # south-west (hex diagonal)
    in_gyros = [idx(i, 0) for i in range(rows)]
    out_gyros = [idx(i, cols - 1) for i in range(rows)]
    return edges, in_gyros, out_gyros


def complex_structure(V: int) -> Array:
    """Block-diagonal 90-degree rotation ``J``, one ``[[0,-1],[1,0]]`` per 2-D gyro
    (the complex structure the gyroscopic 1-form needs; see ``gyro_phase_integrator``)."""
    blk = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    return jax.scipy.linalg.block_diag(*([blk] * V))


@dataclass(frozen=True)
class GyroConfig:
    """Static topology + fixed hyperparameters of the gyro classifier."""

    rows: int
    cols: int
    edges: Tuple[Tuple[int, int], ...]
    in_idx: Tuple[int, ...]   # 2-D coordinate indices of the input gyros (len 2*rows)
    out_idx: Tuple[int, ...]  # 2-D coordinate indices of the output gyros
    J: Array
    n_classes: int = 10
    settle: int = 10          # free-evolution steps after the stroke (the blog's "5 seconds")
    damping: float = 0.05     # the damping 1-form coefficient c (fixed)
    force_scale: float = 0.3  # scales encoder forces into a stable regime

    @property
    def V(self) -> int:
        return self.rows * self.cols

    @property
    def n_in(self) -> int:
        return len(self.in_idx)

    @property
    def n_out(self) -> int:
        return len(self.out_idx)


def make_config(rows: int = 3, cols: int = 4, **kw) -> GyroConfig:
    edges, in_gyros, out_gyros = grid_graph(rows, cols)
    in_idx = tuple(2 * v + c for v in in_gyros for c in (0, 1))
    out_idx = tuple(2 * v + c for v in out_gyros for c in (0, 1))
    return GyroConfig(
        rows=rows, cols=cols, edges=tuple(edges),
        in_idx=in_idx, out_idx=out_idx, J=complex_structure(rows * cols), **kw,
    )


# ---------------------------------------------------------------------------
# The physics: one 2-D harmonic arrangement, read by Phigyro.
# ---------------------------------------------------------------------------


def rod_gravity(g: float, L: float):
    """Nonlinear rod-gravity on-site potential for a 2-D gyro (the blog's rod geometry).

    A gyro tip displaced by ``q`` (horizontal, ``r = |q|``) on a rod of length ``L`` sits
    at height ``sqrt(L^2 - r^2)``; gravity gives the on-site potential

        U_grav(q) = -g * sqrt(L^2 - |q|^2),

    a restoring well with its minimum upright (``r = 0``). It is genuinely *nonlinear*:
    the restoring force ``-dU = -g q / sqrt(L^2 - r^2)`` stiffens as the rod nears
    horizontal (``r -> L``), and its small-tilt limit is ``(g/L) q`` -- exactly the
    harmonic well the surrogate used (spring constant ``g/L``). Plug into
    ``wiring.compose_graph(..., onsite=rod_gravity(g, L))`` so each gyro carries it.
    Valid for ``|q| < L`` (a rod cannot tilt past horizontal); run with tilts in range.
    """

    def U(q: Array) -> Array:
        return -g * jnp.sqrt(L ** 2 - jnp.sum(q ** 2))

    return U


def gyro_arrangement(
    cfg: GyroConfig, inv_mass: Array, kappa: Array, grav_k: Array
) -> SmoothArrangement:
    """The 2-D gyro network as one ``SmoothArrangement`` (sec.graph_laplacian at R^2,
    written directly), with an open force port on the input gyros.

    Parameter ``Q = R^{2V}`` (positions of all gyros), constant sharp ``1/m`` per
    gyro. Codomain box ``<R^{2 n_in} (encoder force in) | R^{2 n_out} (positions out)>``.
    Potential ``U = springs + on-site gravity - drive``; its gradient (computed by the
    framework's backward pass) is the force ``Phigyro`` integrates.
    """
    V = cfg.V
    inv_diag = jnp.repeat(inv_mass, 2)  # both components of each gyro share its mass
    Q = constant(jnp.diag(inv_diag))
    E = jnp.asarray(cfg.edges)          # (#edges, 2)
    in_idx = jnp.asarray(cfg.in_idx)
    out_idx = jnp.asarray(cfg.out_idx)

    def out_f(q, m_out):
        return q[out_idx]

    def in_f(q, m_out, n_in):
        return jnp.zeros(0)

    has_edges = E.shape[0] > 0

    def U(q, m_out, n_in):
        qv = q.reshape(V, 2)
        if has_edges:
            diff = qv[E[:, 0]] - qv[E[:, 1]]
            springs = 0.5 * jnp.sum(kappa * jnp.sum(diff ** 2, axis=1))
        else:
            springs = jnp.array(0.0)
        gravity = 0.5 * grav_k * jnp.sum(qv ** 2)
        drive = jnp.dot(n_in, q[in_idx])  # linear forcing: -dU/dq_in = +force
        return springs + gravity - drive

    return SmoothArrangement(
        Q=Q, out_dim_M=0, in_dim_M=0,
        out_dim_N=len(cfg.out_idx), in_dim_N=len(cfg.in_idx),
        out_f=out_f, in_f=in_f, U=U, label="gyro_net",
    )


def physics_params(cfg: GyroConfig, kappa0: float = 0.3, grav0: float = 0.1, gamma0: float = 0.2) -> Dict[str, Array]:
    """Initial trainable physics: per-gyro log inverse-mass, per-edge log stiffness,
    log gravity, and gyroscopic coefficient. Logs keep masses/stiffness/gravity > 0."""
    return {
        "log_inv_mass": jnp.zeros(cfg.V),
        "log_kappa": jnp.full(len(cfg.edges), float(np.log(kappa0))),
        "log_grav": jnp.array(float(np.log(grav0))),
        "gamma": jnp.array(float(gamma0)),
    }


def coalgebra(cfg: GyroConfig, phys: Dict[str, Array]):
    """``Phigyro`` of the gyro arrangement at the given physics params."""
    arr = gyro_arrangement(
        cfg,
        inv_mass=jnp.exp(phys["log_inv_mass"]),
        kappa=jnp.exp(phys["log_kappa"]),
        grav_k=jnp.exp(phys["log_grav"]),
    )
    return Phigyro(arr, damping=cfg.damping, gamma=phys["gamma"], J=cfg.J)


# ---------------------------------------------------------------------------
# Encoder / decoder (standard ML, layered on top) and the classifier.
# ---------------------------------------------------------------------------


def init_params(cfg: GyroConfig, seed: int = 0, **phys_kw) -> Dict[str, Array]:
    """All trainable parameters: encoder, decoder, and the physics."""
    rng = np.random.default_rng(seed)
    p = dict(physics_params(cfg, **phys_kw))
    p["W_enc"] = jnp.asarray(0.3 * rng.standard_normal((cfg.n_in, 2)))
    p["b_enc"] = jnp.zeros(cfg.n_in)
    p["W_dec"] = jnp.asarray(0.1 * rng.standard_normal((cfg.n_classes, 2 * cfg.n_out)))
    p["b_dec"] = jnp.zeros(cfg.n_classes)
    return p


def classify(params: Dict[str, Array], x_seq: Array, cfg: GyroConfig) -> Array:
    """Logits for one stroke sequence ``x_seq`` of shape ``(T, 2)``.

    Encoder maps each stroke point to forces on the input gyros; ``Phigyro`` runs
    the network forced by them for ``T`` steps, then freely for ``cfg.settle`` steps;
    the decoder reads the output gyros' positions and velocities into logits.
    """
    O = coalgebra(cfg, params)
    inv_mass = jnp.exp(params["log_inv_mass"])
    inv_diag = jnp.repeat(inv_mass, 2)
    out_idx = jnp.asarray(cfg.out_idx)
    zeros_out = jnp.zeros(cfg.n_out)

    def step(state, force):
        org = O.with_state(state)
        _out_pos, _out_dir, new_state = org.run_one(_IN_POS, lambda op: (zeros_out, force))
        return new_state, None

    # encoder forces during the stroke, then zero forcing while the network settles
    forces = cfg.force_scale * (x_seq @ params["W_enc"].T + params["b_enc"])  # (T, n_in)
    forces = jnp.concatenate([forces, jnp.zeros((cfg.settle, cfg.n_in))], axis=0)

    state0 = (jnp.zeros(2 * cfg.V), jnp.zeros(2 * cfg.V))
    (q, xi), _ = jax.lax.scan(step, state0, forces)

    pos_out = q[out_idx]
    vel_out = (inv_diag * xi)[out_idx]
    readout = jnp.concatenate([pos_out, vel_out])  # output gyros' positions + velocities
    return params["W_dec"] @ readout + params["b_dec"]


def cross_entropy(logits: Array, label: int) -> Array:
    logp = logits - jax.scipy.special.logsumexp(logits)
    return -logp[label]


def batch_loss(params: Dict[str, Array], xs: Array, ys: Array, cfg: GyroConfig, weight_decay: float = 1e-4) -> Array:
    """Mean cross-entropy over a batch ``xs:(B,T,2)``, ``ys:(B,)`` + L2 on encoder/decoder."""
    logits = jax.vmap(lambda x: classify(params, x, cfg))(xs)
    ce = jnp.mean(jax.vmap(cross_entropy)(logits, ys))
    l2 = sum(jnp.sum(params[k] ** 2) for k in ("W_enc", "W_dec"))
    return ce + weight_decay * l2


def accuracy(params: Dict[str, Array], xs: Array, ys: Array, cfg: GyroConfig) -> float:
    logits = jax.vmap(lambda x: classify(params, x, cfg))(xs)
    return float(jnp.mean(jnp.argmax(logits, axis=1) == ys))


# ---------------------------------------------------------------------------
# Training: minibatch Adam (hand-rolled; no optax dependency).
# ---------------------------------------------------------------------------


def adam_init(params):
    return {k: jnp.zeros_like(v) for k, v in params.items()}, {k: jnp.zeros_like(v) for k, v in params.items()}


def adam_step(params, grads, m, v, t, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
    new_p, new_m, new_v = {}, {}, {}
    for k in params:
        new_m[k] = b1 * m[k] + (1 - b1) * grads[k]
        new_v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
        mhat = new_m[k] / (1 - b1 ** t)
        vhat = new_v[k] / (1 - b2 ** t)
        new_p[k] = params[k] - lr * mhat / (jnp.sqrt(vhat) + eps)
    return new_p, new_m, new_v


def train(
    cfg: GyroConfig,
    Xtr: Array, Ytr: Array, Xva: Array, Yva: Array,
    *, epochs: int = 30, batch: int = 128, lr: float = 3e-3, seed: int = 0, verbose: bool = True,
    freeze: Sequence[str] = (), init_kw: Dict = None,
):
    """Train the gyro classifier with minibatch Adam; return ``(params, history)``.

    ``freeze`` holds parameter keys whose gradient is zeroed (held at their init) --
    used for ablations: ``freeze=("gamma",)`` turns off gyroscopic precession,
    ``freeze=("log_kappa",)`` with tiny initial stiffness decouples the gyros (the
    blog's "Take 1": input gyros cannot reach the output gyros without springs).
    """
    params = init_params(cfg, seed=seed, **(init_kw or {}))
    freeze = set(freeze)
    m, v = adam_init(params)
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]

    loss_and_grad = jax.jit(jax.value_and_grad(lambda p, xb, yb: batch_loss(p, xb, yb, cfg)))

    history = []
    t = 0
    for ep in range(epochs):
        perm = rng.permutation(n)
        losses = []
        for i in range(0, n - batch + 1, batch):
            idx = perm[i : i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            t += 1
            loss, grads = loss_and_grad(params, xb, yb)
            if freeze:
                grads = {k: (jnp.zeros_like(g) if k in freeze else g) for k, g in grads.items()}
            params, m, v = adam_step(params, grads, m, v, t, lr=lr)
            losses.append(float(loss))
        va = accuracy(params, Xva, Yva, cfg)
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "val_acc": va})
        if verbose:
            print(f"  epoch {ep:3d}  loss {np.mean(losses):.4f}  val_acc {va:.4f}")
    return params, history


# ---------------------------------------------------------------------------
# PenDigits data (the blog's dataset) + a runnable surrogate.
# ---------------------------------------------------------------------------

_UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits"


def load_pendigits(cache_dir: str = None):
    """Load UCI PenDigits, downloading/caching on first use.

    Each sample is a stroke of 8 ``(x, y)`` pen points (the dataset's resampled
    form), reshaped to ``(8, 2)`` and rescaled from ``[0,100]`` to ``~[-1,1]``.
    Returns ``(Xtr, Ytr, Xte, Yte)``; the test split is the blog's "validation" set.
    """
    import os
    import urllib.request

    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "dap")
    os.makedirs(cache_dir, exist_ok=True)

    def fetch(split):
        path = os.path.join(cache_dir, f"pendigits.{split}")
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{_UCI}.{split}", path)
        raw = np.loadtxt(path, delimiter=",")
        X = raw[:, :16].reshape(-1, 8, 2).astype(np.float64) / 50.0 - 1.0
        return jnp.asarray(X), jnp.asarray(raw[:, 16].astype(int))

    Xtr, Ytr = fetch("tra")
    Xte, Yte = fetch("tes")
    return Xtr, Ytr, Xte, Yte


def render_stroke(points: Array, width: int = 22, height: int = 11) -> str:
    """An ASCII sketch of one pen stroke: the 8 ``(x, y)`` points joined in order.

    This is what the *input* looks like -- the path the pen traced. The classifier
    never sees this picture; it only feels the points as forces on the input gyros.
    """
    P = np.asarray(points, float)
    P = (P - P.min(0)) / (np.ptp(P, axis=0) + 1e-9)
    xs = (P[:, 0] * (width - 1)).astype(int)
    ys = ((1 - P[:, 1]) * (height - 1)).astype(int)  # flip y so the digit reads upright
    grid = [[" "] * width for _ in range(height)]
    for i in range(len(P) - 1):
        for t in np.linspace(0, 1, 30):
            x = int(round(xs[i] + t * (xs[i + 1] - xs[i])))
            y = int(round(ys[i] + t * (ys[i + 1] - ys[i])))
            grid[y][x] = "#"
    return "\n".join("".join(row) for row in grid)


def show_predictions(params: Dict[str, Array], cfg: GyroConfig, X: Array, Y: Array, n: int = 6, seed: int = 7):
    """Print per-digit accuracy and a few held-out strokes with ``true -> guess``."""
    pred = np.asarray(jnp.argmax(jax.vmap(lambda x: classify(params, x, cfg))(X), axis=1))
    true = np.asarray(Y)
    print("\nper-digit test accuracy:")
    for d in range(cfg.n_classes):
        mask = true == d
        if mask.sum():
            print(f"  {d}: {100 * np.mean(pred[mask] == d):5.1f}%  ({int(mask.sum())} samples)")
    print("\nheld-out test strokes (the pen path; true digit -> network's guess):")
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(X), n, replace=False):
        flag = "OK" if pred[i] == true[i] else "XX"
        print(f"\n[{flag}] true {true[i]}  ->  guessed {pred[i]}")
        print(render_stroke(X[i]))


def main(rows: int = 4, cols: int = 5, epochs: int = 60, seed: int = 0):
    """Train the harmonic gyro surrogate on PenDigits inside dap (``python -m dap.gyroscope``).

    Trains the ``Phigyro`` gyro network on PenDigits and prints its validation
    accuracy. The blog's reported numbers are shown for context only -- this is a
    harmonic surrogate of a richer machine, not a reproduction of their result. Pass
    ``--show`` to also print per-digit accuracy and a few pen strokes with the guess.

    Usage: ``python -m dap.gyroscope [rows cols epochs] [--show]``.
    """
    import sys

    show = "--show" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    if pos:
        rows, cols = int(pos[0]), int(pos[1])
        if len(pos) > 2:
            epochs = int(pos[2])

    print(f"PenDigits via Phigyro: {rows}x{cols} gyro grid, {epochs} epochs\n")
    Xtr, Ytr, Xte, Yte = load_pendigits()
    cfg = make_config(rows=rows, cols=cols)
    params, hist = train(cfg, Xtr, Ytr, Xte, Yte, epochs=epochs, seed=seed, verbose=True)
    best = max(h["val_acc"] for h in hist)
    print(f"\nbest val_acc {best:.4f}   (harmonic surrogate; for context only -- blog's richer "
          f"machine: linear 0.562, gyroscopes 0.834, LSTM 0.896; not a reproduction)")
    if show:
        show_predictions(params, cfg, Xte, Yte)


if __name__ == "__main__":
    main()
