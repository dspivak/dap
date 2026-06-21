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

dap            # build & run your OWN arrangement (interactive)
dap-demo       # run the worked examples
pytest dap/tests
```

After `pip install -e .` the `dap` and `dap-demo` commands are on your PATH (no
`python -m …`). Requires Python ≥ 3.10; pulls in JAX and NumPy.

## What's here

| module | role |
|---|---|
| `rvect.py` | reactive vector spaces (position-dependent sharp) |
| `arrangement.py` | a smooth adaptive arrangement (a morphism of `sarr`) |
| `polynomial.py`, `org.py` | polynomials and `[p,q]`-coalgebras (Moore form) |
| `interpretation.py` | the shared, integrator-free polynomial interpretation |
| `integrator.py` | configuration and phase integrators, and the two-stage `Integrator2` |
| `functors.py` | the dynamics functors `Phiconf`, `Phiphase` |
| `org2.py` | general two-stage coalgebras `[p,q]^{∘2}` (`org^(2)`) + composition |
| `leapfrog.py` | leapfrog as a two-stage integrator → `org^(2)` (higher-order symplectic) |
| `wiring.py` | compose boxes in `sarr` (chains, graphs, tensor) |
| `learning.py` | gradient descent with backpropagation |
| `demo.py`, `build.py` | run the worked examples / build your own |

These modules implement the paper's two chapters. The worked examples of the
paper's final section each have a test in `dap/tests` checking it reproduces the
paper's recurrence: Newton's method and gradient descent (`Phiconf`,
`sec.newton_warmup`/`sec.dl_warmup`), the wave equation (`Phiphase`,
`sec.wave_equation`, computed two ways to exhibit functoriality), and the graph
Laplacian (`compose_graph`, the prism wiring of `sec.graph_laplacian`) — plus the
heat equation and leapfrog.

## Extensions (beyond the paper)

Three further constructions **reuse** the functorial core above but add content
that is *not* part of the paper's formal development. They are toy-scale
demonstrations that the primitives compose — not implementations of paper results,
and not claims to beat dedicated tools. `dap-demo` prints them under a separate
**“Extensions”** heading.

| construction | what it reuses (paper) | what it adds (not in the paper) |
|---|---|---|
| `Phidamped` (`integrator.py`, `functors.py`) | the phase integrator | a dissipation 1-form `ζ(q,ξ)=(ξ,0)` giving heavy-ball momentum; the paper's `ex.dissipation_one_form` is **commented out** |
| `system_id.py` | `Phiphase` + the learner of `sec.dl_warmup` | a nonlinear pendulum, a tanh-MLP predictor, libration sampling — identifying a flow map |
| `pinn.py` | `Phiconf` + the learner of `sec.dl_warmup` | a deep-Ritz Dirichlet energy and a coordinate MLP — a physics-informed net for 1D Poisson |

The integrator alone turns one convex arrangement into gradient descent,
conservative oscillation, or heavy-ball momentum (`Phiconf`/`Phiphase`/`Phidamped`)
— the same selection that distinguishes wave from heat.

## Build your own

`dap` (the interactive builder, `build.py`) composes and runs an arrangement with no
code. You choose:

- **dynamics** — `phase` (Hamiltonian/symplectic Euler), `conf` (descent), or
  `leapfrog` (higher-order symplectic);
- **system** — a chain of harmonic particles, a graph of them (`path N` / `ring N` /
  `complete N` / explicit `i-j` edges), or gradient descent on a linear model;
- **initial condition** — `random`, `zeros`, `sine [n]` (a smooth standing-wave
  mode), `bump` (a localized pulse that splits into travelling waves), or an integer
  seed.

It prints the start/end state, a `▁▂▃` shape sparkline, the residual of the relevant
discrete equation, and whether the energy stays bounded. Start a wave from `random`
and it looks like noise — broadband data, every mode dephasing; start from `sine` or
`bump` to see a clean wave. On a chain, answer **`animate? y`** to watch the wave
evolve as a terminal animation (Ctrl-C to stop).

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
> (Hamiltonian). Store each covector field as a callable `ω: ℝᵈ → ℝᵈ` evaluated
> exactly (no affine/quadratic approximation). Also build the two-stage semantics org^(2): a [p,q]^{∘2}-
> coalgebra is two emit/receive rounds per macro-tick, where the first round lands
> in an inner one-stage coalgebra (the substitution [p,q] ◁ [p,q]) rather than a
> new state, with composition (parallel, then_static). A two-stage integrator then
> gives leapfrog (velocity Verlet) as one org^(2) instance — derived from org^(2),
> not hardcoded. Then build the worked examples — Newton's method, gradient descent
> with backpropagation, the wave equation (symplectic phase, plus leapfrog), the heat
> equation — and check that each reproduces the paper's recurrences. Use ℝᵈ for
> manifolds and JAX for autodiff.

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
7. Build the two-stage semantics **`org^(2)`** (two emit/receive rounds; the
   substitution `[p,q]^{∘2} = [p,q] ◁ [p,q]`, the first round landing in an inner
   one-stage coalgebra) **with composition**; leapfrog is then one instance.
8. Instantiate the worked examples and check they reproduce the paper's recurrences.

Four representation choices the mathematics does not dictate, which this code
makes explicit: manifolds = ℝᵈ; covector fields stored as callables `ω: ℝᵈ → ℝᵈ`
(the exact field, not an affine approximation); coalgebras in Moore form; autodiff
backend = JAX.

## Runnable "for real"?

`Phiconf` (gradient descent, Newton, heat) is explicit Euler on a *gradient*
flow: stable for small steps, a genuine algorithm. `Phiphase` (wave) is the
*symplectic* (semi-implicit) Euler step on a *Hamiltonian* flow: it reads out the
presented position `q~ = q + sharp(p)` and evaluates the force there, so the energy
stays bounded — `dap` shows the wave staying stable directly. `leapfrog` (velocity
Verlet) is a higher-order symplectic alternative that evaluates the force twice per
step, so it lands in `org^(2)` rather than `org`. `org2.py` builds the general
two-stage coalgebra `[p,q]^{∘2}` with composition; `leapfrog.py` is one instance of
it. (Whether `sarr → org^(2)` is a *functor* — cf. rmk.multistage — is left open;
the code provides the datatype, a leapfrog instance, and composition — tested, not
a proof.)

## License

MIT — see [LICENSE](LICENSE).
