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
| `integrator.py` | configuration and phase integrators, the two-stage `Integrator2`, the K-stage `IntegratorK` |
| `functors.py` | the dynamics functors `Phiconf`, `Phiphase` (and `Phirk4`, RK4 as `org^(4)`) |
| `org2.py` | general two-stage coalgebras `[p,q]^{∘2}` (`org^(2)`) + composition |
| `orgK.py` | general K-stage coalgebras `[p,q]^{∘K}` (`org^(K)`) + composition |
| `leapfrog.py` | leapfrog as a two-stage integrator → `org^(2)` (higher-order symplectic) |
| `rk4.py` | classical RK4 as a four-stage integrator → `org^(4)` (`Phirk4`; non-symplectic) |
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

Five further constructions **reuse** the functorial core above but add content
that is *not* part of the paper's formal development. They demonstrate that the
primitives compose — toy-scale (`gyroscope.py` is a harmonic surrogate inspired by
Bull & Achour, and `gyroscope_faithful.py` is the equation-faithful version of it) —
not implementations of paper results, and not claims to beat dedicated tools.
`dap-demo` prints them under a separate **“Extensions”** heading.

| construction | what it reuses (paper) | what it adds (not in the paper) |
|---|---|---|
| `Phidamped` (`integrator.py`, `functors.py`) | the phase integrator + the paper's damping 1-form `ex.damping_one_form` | wiring `c·ζ`, `ζ(q,ξ)=(ξ,0)`, into the integrator as heavy-ball momentum (the paper records `ζ` but it "plays no further role" there) |
| `system_id.py` | `Phiphase` + the learner of `sec.dl_warmup` | a nonlinear pendulum, a tanh-MLP predictor, libration sampling — identifying a flow map |
| `pinn.py` | `Phiconf` + the learner of `sec.dl_warmup` | a deep-Ritz Dirichlet energy and a coordinate MLP — a physics-informed net for 1D Poisson |
| `gyroscope.py` (`Phigyro`) | `Phiphase` + the graph-Laplacian spring potential of `sec.graph_laplacian` (written directly, **not** assembled by the prism wiring) + the learner of `sec.dl_warmup` | 2-D harmonic gyros with on-site gravity, a gyroscopic skew 1-form, and a linear encoder/decoder + Adam — a harmonic surrogate of the Bull & Achour gyroscope digit-classifier |
| `gyroscope_faithful.py` (`Phirk4gyro`) | the prism graph-wiring of `sec.graph_laplacian` (now at `R²`) + RK4 as `org^(4)` (`rmk.multistage`) + the 1-form vector space (`prop.one_forms_vector_space`, `rmk.adam`) | the Bull & Achour machine built **equation-faithfully**: hex springs *from the wiring*, nonlinear rod gravity, quadratic-drag + per-gyro precession 1-forms, RK4 phase integration — **no accuracy claim**; the result is the springs→0 ablation |

The integrator alone turns one convex arrangement into gradient descent,
conservative oscillation, or heavy-ball momentum (`Phiconf`/`Phiphase`/`Phidamped`)
— the same selection that distinguishes wave from heat.

`gyroscope.py` is a **harmonic surrogate** of the gyroscope-and-springs digit
classifier of [Bull & Achour, *Machine learning with
dynamics*](https://unconv.ai/blog/machine-learning-with-dynamics/) (2026). Each
physical force is mapped to one slot of the framework: gravity and springs in the
potential `U`, mass in the reactive sharp, damping and gyroscopic precession in the
phase integrator's 1-form (`Phigyro`), and the input "nudge" through the open port.
The forward dynamics *is* `Phigyro` of one harmonic arrangement. Training backprops
through the length-`T` rollout with `jax.grad`; since `cot`'s backward part is
reverse-mode AD, this realizes the same chain-rule pullbacks that `cot` encodes — an
AD-equivalence, not a literal `OrgMorphism.then` composition. On UCI PenDigits
(`python -m dap.gyroscope`) the surrogate reaches **~0.83** validation accuracy with a
3×4 gyro grid and **~0.87** with 4×5; the blog reports 0.834 for its (richer) machine
(linear baseline 0.562, LSTM 0.896), shown for context — we do **not** claim to
reproduce that number (different machine, a harmonic approximation, and a simplified
PenDigits form). The interesting result is the **ablation**: freezing the springs to
zero collapses it to chance — the blog's "Take 1" failure (no coupling, no
information flow), here a one-line wiring fact — while the gyroscopic term turns out
not to be needed for this task.

### `gyroscope_faithful.py` — the faithful machine (no accuracy claim)

`gyroscope_faithful.py` is a *supplement* to the harmonic surrogate above: the same
Bull & Achour machine, but built **equation-faithfully and entirely from the paper's
constructions**, so you can read the ODEs and check that it factors through `Phi`. Its
point is the **factorization** and the **springs→0 ablation** — *not* accuracy. Unlike
the surrogate (which reports its own measured number, qualified), the faithful machine
makes **no accuracy claim** and runs on purely synthetic data. Each force is one slot,
assembled categorically:

- **spring coupling** — the prism graph-wiring (`compose_graph`, `sec.graph_laplacian`)
  generalized to `R²` gyros: the graph-Laplacian potential *emerges from composition*,
  it is not a hand-written `U`;
- **rod gravity** — the nonlinear on-site potential `−g·√(L²−|q|²)`;
- **mass** in the reactive sharp; **quadratic air drag** and **gyroscopic precession**
  as two 1-forms; **RK4** as `Phirk4gyro` (RK4 on the phase state, an `org^(4)` morphism);
  the **input nudge** through the open input port.

The rollout genuinely runs `Phirk4gyro → orgK_from_integrator → smooth_interpretation`,
with the force `= jax.grad(U)` (the framework's backward pass) — no physics is computed
by hand outside the functor.

**The headline is the ablation, not a number.** Input gyros are the left column, output
gyros the right column. With springs, the input signal reaches the output and depends on
it; **freeze the stiffness to zero and the output stays at rest regardless of the
input** — there is no spring path to carry information across. That is the blog's "Take 1
failed", recast as a one-line categorical wiring fact (a tested result).

**Deviations from the blog** (deliberate — the goal is the factorization and the
ablation, not their experiment):

| blog | faithful machine here |
|---|---|
| ~100 gyros, 10×10 hex, ~261 springs | a small hex lattice (scale isn't needed for the factorization or the ablation) |
| per-gyro mass, per-edge stiffness | uniform mass / stiffness (one scalar each) |
| ~100-step 2-channel PenDigits strokes | short synthetic `make_strokes` (generator-as-spec, no download) |
| trained to a benchmark accuracy | **no accuracy claim** — only that the loss decreases under training |
| RK4 ODE solver | RK4 as `org^(4)` — matches the solver *and* factors through the framework |

**Beyond-paper caveats** (banner-labeled in the code): the quadratic-drag and
gyroscopic 1-forms are monoidal over `⊕` but natural only over a *subcategory* of
`rvect` (per-gyro `O(2)`) — which `rmk.adam` explicitly sanctions ("it just lives over a
smaller `Q`"); whether `sarr → org^(K)` is a *functor* is left open (datatype +
instances + composition, not a proof); and the encoder/decoder **drive and readout are
the hand-added open I/O interface** — the *coupled physics* factors through the
framework, the I/O does not claim to.

See it in `dap-demo` (the springs-ablation line under Extensions), or import
`make_hex_config` / `make_strokes` / `train` / `output_readout` from
`dap.gyroscope_faithful`.

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
two-stage coalgebra `[p,q]^{∘2}` with composition (`leapfrog.py` is one instance),
and `orgK.py` generalizes it to the K-fold substitution `[p,q]^{∘K}` — the `ℓ`-round
coalgebras of `rmk.multistage`. `rk4.py` realizes classical RK4 as one `org^(4)`
instance: its global error falls like `h⁴` (a test checks the rate against the closed
form `e^{-At}`), the falsifiable evidence that the four rounds carry the right
intermediate stages and Butcher weights. RK4 is non-symplectic — fine here; the
multi-stage construction does not require it. (Whether `sarr → org^(K)` is a
*functor* — cf. `rmk.multistage` — is left open for every `K`; the code provides the
datatype, instances at `K = 2` (leapfrog) and `K = 4` (RK4), and composition —
tested, not a proof.)

## License

MIT — see [LICENSE](LICENSE).
