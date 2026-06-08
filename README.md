# dap — dynamic algebra potentials, executable

Companion implementation for the paper *dynamic algebra potentials*. It turns the
paper's constructions into runnable code: a **smooth adaptive arrangement** — a
wiring of boxes carrying smooth maps, reactive parameters, and potentials — is
compiled by a **dynamics functor** into a discrete dynamical system you can step.

There are two dynamics functors (the paper's `cor.functor`), differing only in
their *integrator*:

- **`Phiconf`** — configuration / descent dynamics (gradient flow). Genuinely
  convergent algorithms: gradient descent, Newton's method, the heat equation.
- **`Phiphase`** — phase-space / Hamiltonian dynamics: the wave and
  Klein–Gordon equations, as exact recurrences.

## Quickstart

```bash
git clone git@github.com:dspivak/dap.git && cd dap
python -m venv .venv && . .venv/bin/activate
pip install -e .

python -m dap.demo      # run the worked examples
python -m dap.build     # build & run your OWN arrangement (interactive)
pytest dap/tests        # the test suite
```

Requires Python ≥ 3.10; pulls in JAX and NumPy.

## What's here

| module | role |
|---|---|
| `rvect.py` | reactive vector spaces (position-dependent sharp) |
| `arrangement.py` | a smooth adaptive arrangement (a morphism of `sarr`) |
| `polynomial.py`, `org.py` | polynomials and `[p,q]`-coalgebras (Moore form) |
| `interpretation.py` | the shared, integrator-free polynomial interpretation |
| `integrator.py` | the configuration and phase integrators |
| `functors.py` | `Phiconf`, `Phiphase` |
| `wiring.py` | compose boxes (chains, graphs, tensor) |
| `learning.py` | gradient descent with backpropagation |
| `demo.py`, `build.py` | run the examples / build your own |

## The one idea

A morphism of polynomials **is** a pair of programs: a forward map on positions
and a backward map on directions. The cotangent functor sends a smooth map `f`
to the pair `(f, (Tf)ᵀ)`, where the backward part is **reverse-mode autodiff**.
Every structure map of the construction — each natural transformation, monoidal
productor/unitor, and the lens internalization — is one such `(forward, backward)`
pair, and a dynamics functor is just their composite, followed by an *integrator*
saying how an incoming covector updates the stored state.

## Reproduce this yourself

This implementation was generated from the paper's definitions by an LLM (Claude).
To regenerate it from scratch, the recipe is roughly:

1. Read the syntax and semantics chapters (`ch.smooth_rwd`, `ch.smooth_dynamics`).
2. Represent a polynomial map as a pair `(forward_on_positions, backward_on_directions)`.
3. Implement the cotangent functor with forward `= f` and backward `= (Tf)ᵀ`, the
   latter via reverse-mode autodiff.
4. Realize **every** natural transformation / productor / unitor / lens
   internalization as such a `(forward, backward)` pair, and compose them into the
   polynomial interpretation.
5. Carry `[p,q]`-coalgebras in Moore form (state + step closure); never
   materialize the internal hom.
6. Pick an **integrator** — a state space plus an update from an incoming covector
   — configuration or phase.
7. Instantiate the worked examples and check they reproduce the paper's recurrences.

Four representation choices the mathematics does not dictate, which this code
makes explicit: manifolds = ℝᵈ; covector fields stored as affine `(A, b)` pairs;
coalgebras in Moore form; autodiff backend = JAX.

## Runnable "for real"?

`Phiconf` (gradient descent, Newton, heat) is explicit Euler on a *gradient*
flow: stable for small steps, a genuine algorithm. `Phiphase` (wave) is explicit
Euler on a *Hamiltonian* flow: the recurrence is exact, but as a time-stepper it
is not symplectic, so the energy grows — `python -m dap.build` shows this
directly. A stable wave simulation needs a symplectic, multi-stage integrator
(the paper's `org^(K)` remark).

## License

MIT — see [LICENSE](LICENSE).
