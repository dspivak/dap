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
| `org2.py` | general two-stage coalgebras `[p,q]^{∘2}` (`org^(2)`) + composition |
| `leapfrog.py` | leapfrog as one `org^(2)` instance (stable wave) |
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

This repository *is* the result of pointing a language model (Claude) at the
paper and asking it to implement the constructions — no human wrote the code. To
make your own version (which you can then modify), give an LLM the paper and a
prompt like:

> Read this paper. Implement the constructions of the chapters "Smooth adaptive
> arrangements" and "Smooth adaptive dynamics" as runnable Python. Represent a
> morphism of polynomials as a pair of programs — a forward map on positions and
> a backward map on directions — and implement the cotangent functor with its
> backward part as reverse-mode automatic differentiation. Realize every natural
> transformation, monoidal productor/unitor, and the lens internalization as such
> a (forward, backward) pair, and compose them into the dynamics functor; carry
> coalgebras in Moore form (a state plus a step function), never materializing the
> internal hom. Provide two integrators — configuration (descent) and phase
> (Hamiltonian). For a symplectic integrator like leapfrog (velocity Verlet),
> which evaluates the force twice per step, first build the general multi-stage
> target org^(K): a [p,q]^{∘K}-coalgebra is K emit/receive rounds per tick, where
> each round but the last lands in an inner (K−1)-stage coalgebra (the
> substitution [p,q]^{∘K}), and give it composition (parallel, then_static); then
> obtain leapfrog as ONE org^(2) instance via a two-stage integrator — do not
> hardcode it as a bespoke stepper. Then build the worked examples — Newton's
> method, gradient descent with backpropagation, the wave equation (with its
> stable leapfrog version), the heat equation — and check that each reproduces the
> paper's recurrences. Use ℝᵈ for manifolds, store covector fields as affine
> (A, b) pairs, and use JAX for autodiff.

Spelled out, the recipe is:

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
7. For symplectic / multi-stage schemes (leapfrog), build the **general `org^(K)`**
   (K emit/receive rounds per tick; the substitution `[p,q]^{∘K}`, each round but
   the last landing in an inner `(K−1)`-stage coalgebra) **with composition**, and
   derive leapfrog as a **two-stage instance** — not a hardcoded stepper.
8. Instantiate the worked examples and check they reproduce the paper's recurrences.

Four representation choices the mathematics does not dictate, which this code
makes explicit: manifolds = ℝᵈ; covector fields stored as affine `(A, b)` pairs;
coalgebras in Moore form; autodiff backend = JAX.

## Runnable "for real"?

`Phiconf` (gradient descent, Newton, heat) is explicit Euler on a *gradient*
flow: stable for small steps, a genuine algorithm. `Phiphase` (wave) is explicit
Euler on a *Hamiltonian* flow: the recurrence is exact, but as a time-stepper it
is not symplectic, so the energy grows — `python -m dap.build` shows this
directly. For a *stable* wave, choose `leapfrog` — symplectic velocity Verlet,
which evaluates the force twice per step, so it lands in `org^(2)` rather than
`org`. `org2.py` builds the general two-stage coalgebra `[p,q]^{∘2}` with
composition; `leapfrog.py` is one instance of it. Same diagram, bounded energy.
(The general *functor* `sarr → org^(K)` of rmk.org_N is still a conjecture; the
code provides the datatype, a leapfrog instance, and composition — tested, not a
proof.)

## License

MIT — see [LICENSE](LICENSE).
