# Real-world examples (the 2019 PyBNF-paper corpus, on the edition-2 surface)

These are the biological case studies from the original PyBNF paper —

> Mitra ED, Suderman R, Colvin J, Ionkov A, Hu A, Sauro HM, Posner RG, Hlavacek WS.
> **PyBioNetFit and the Biological Property Specification Language.** *iScience* 2019,
> 19:1012–1036.

— re-expressed on PyBNF's **edition-2 (new-era) config surface** (ADR-0028/0034/0046/0052).
They complement two neighbouring collections: the tiny teaching fits in
[`../tutorial/`](../tutorial/) and the interactive notebooks in
[`../notebooks/`](../notebooks/). Where those are small and pedagogical, these are the
*real* rule-based models — Kozer's EGFR, the FcεRI γ-chain network, the trivalent-ligand
aggregation model — fit to real or realistic data, exactly the workflows the paper
benchmarked.

Their purpose here is also to **validate PyBNF's bngsim-backed default simulator path on
representative deterministic, stochastic, and network-free models** (issue #380): targeted
unit tests and the tutorial fits are necessary but not sufficient; these paper-scale
examples catch integration issues in realistic workflows.

## What changed from the classic examples

Each example is the edition-2 twin of a classic `examples/<name>/` job. The upgrade is
mechanical and faithful (same model, same data, same fit); only the *surface* changes:

| classic | edition 2 |
|---|---|
| `model = m.bngl : data.exp` | `model: m.bngl` + `experiment: …, data: data.exp` |
| hand-written `begin actions` block | simulation **synthesized** from `experiment:`/`condition:` (ADR-0028) |
| `fit_type` / `objfunc` | `job_type` / `objective` |
| free params via `X X__FREE` aliases | free params **bind to model ids by name** (ADR-0034) |
| ligand added mid-run with `setConcentration` | ligand gated by a parameter across a `preequilibrate:` (ADR-0052) |
| dose-response as a `parameter_scan` action | `experiment: …, type: parameter_scan` over the data's dose column (ADR-0046) |
| SSA / NFsim chosen in the actions block | `experiment: …, method: ssa` / `method: nf` |

## Coverage matrix

The set spans the three simulator paths #380 asks be validated — deterministic ODE,
stochastic SSA, and network-free NFsim. **Status** is the end-to-end result through the
bngsim backend (see "How these are validated" and "Known limitations" below):

| example | paper mapping | simulator | edition-2 features exercised | status |
|---|---|---|---|---|
| [`receptor`](receptor/) | ligand/receptor, BioNetFit 1 ex 5 | **ODE** | pre-equilibration (`preequilibrate:` + gate parameter), `sos` | ✅ validated |
| [`igf1r`](igf1r/) | IGF1R competition binding, Erickson et al. | **ODE** (network-generating) | dose-response scan, whole-fit `normalization = init`, `chi_sq`, refinement | ✅ validated |
| [`tlbr`](tlbr/) | trivalent-ligand aggregation, BioNetFit 1 ex 3 | **NF** (NFsim) | `method: nf` dose-response scan (fixed `t_end`) | ✅ validated |
| [`egfr_nf`](egfr_nf/) | EGFR clustering, Kozer 2013 (BioNetFit 1 ex 2) | **NF** (NFsim) | `method: nf`, time course + dose-response scan, `gml:`/`complex:` | 🔶 runs; heavy (NFsim) |
| [`egfr_ode`](egfr_ode/) | EGFR activation, Kozer 2013 (Problem 2) | **ODE** (network-generating) | time course + dose-response scan, `chi_sq`, scaled-observable functions | 🔶 cluster-scale network |
| [`fceri_gamma`](fceri_gamma/) | FcεRI γ-chain, Gupta & Mendes 2018 (Problem 3) | **SSA** (Gillespie) | `method: ssa`, `smoothing` over replicate trajectories | 🔶 cluster-scale (58k rxns) |
| [`receptor_nf`](receptor_nf/) | ligand/receptor, BioNetFit 1 ex 6 | **NF** (NFsim) | `method: nf` + pre-equilibration (`equil_t_end:`) | 🔶 runs; heavy (NFsim) |

✅ validated end-to-end through bngsim (runs in the `recovery` test tier) · 🔶 builds and runs
but too heavy to run routinely (reference only, backend-free tier only): `egfr_nf`/`receptor_nf`
run real NFsim but a full fit is cluster-scale (NFsim on ~10³ molecules can exceed
`wall_time_sim`), and `egfr_ode`/`fceri_gamma` have cluster-scale networks whose generation is
impractical in CI.

## A bug this surfaced (and its fix)

Expressing the NFsim examples on the edition-2 surface exposed a real integration gap — the
kind #380 exists to catch — and drove three fixes.

1. **NF experiments could not route to the NF bridge.** Edition-2 synthesizes, for every
   experiment, a leading `resetConcentrations()` (IC hygiene) and sets `generates_network=True`.
   Both are wrong for a network-free model: the bngsim NF bridge *rejects* `resetConcentrations()`
   (NFsim re-seeds each run, so it is a no-op), and an NF model's reaction network is unbounded
   and cannot be generated. So **no** edition-2 `method: nf` experiment could run through bngsim.
   Fixed in `pybnf/pset.py`: both are suppressed on the NF path, so an NF experiment classifies as
   the NF bridge and routes to `writeXML` → `BngsimNfModel`, as a hand-written NF actions block does.
2. **NF pre-equilibration hung.** Edition-2 pre-equilibration equilibrates *to steady state*, but
   NFsim has no steady-state solve, so it integrated the equilibration to t≈1e6 event-by-event and
   never finished. Added the **`equil_t_end:`** field: a `method: nf` pre-equilibration runs its
   equilibration for that fixed duration instead (`receptor_nf` uses the classic 600 s); omitting it
   on the NF path is now a clear config-time error, not a hang.
3. **NF actions dropped NFsim options.** Added the **`gml:`** (global molecule limit) and
   **`complex:`** (track complexes) experiment fields, carried into the synthesized NFsim
   `simulate`/`parameter_scan` — `egfr_nf` now sets `gml=>1000000, complex=>1` as its classic model did.

ODE/SSA synthesis is unchanged throughout; guarded by the backend-free tests in
`tests/test_real_world_examples.py` (`…_nf_synthesis_is_network_free`,
`test_receptor_nf_equilibrates_for_a_fixed_time`, `test_egfr_nf_carries_nfsim_options`,
`test_nf_preequilibration_without_equil_t_end_errors`).

## Known limitations (examples too heavy to run routinely)

- **`receptor_nf`, `egfr_nf` — NFsim is cluster-scale.** Both now run real NFsim through bngsim,
  but a full fit is a cluster job: NFsim steps event-by-event over ~10³ molecules, and some
  parameter sets exceed the (default 1-hour) `wall_time_sim` per simulation. They stay in the
  backend-free tier, matching the paper's use of a cluster for these problems.
- **`egfr_ode`, `fceri_gamma` — cluster-scale networks.** The Kozer EGFR crosslinking network takes
  >10 min to generate; the FcεRI network is ~58k reactions and its SSA fit is a cluster job. Both
  build correctly but are impractical to run in CI. SSA-through-bngsim itself is covered at the
  fixture level by `tests/test_bngsim_ssa_replaces_rr.py` (#379).

## Running an example

Each subdirectory is self-contained (edition-2 `.bngl`, `.exp` data, `.conf`). From the
repo root, with `BNGPATH` set to your BioNetGen install:

```bash
pybnf -c examples/real-world/receptor/receptor.conf
```

`egfr_ode` and `fceri_gamma` are **cluster-scale**: the Kozer EGFR crosslinking network
and the 58k-reaction FcεRI network take minutes to generate and their published fits used
large populations on a cluster (see each conf's header). The other five run on a
workstation.

## How these are validated

`tests/test_real_world_examples.py` (with `examples/real-world/_manifest.py`) runs two
tiers against these committed confs:

* **default CI, backend-free** — for *every* example: the conf parses, is edition 2, selects
  its documented simulator, and binds its data; and each NF example synthesizes a
  network-free action set (the regression guard for the fix above). No bngsim/BNG2.pl needed.
* **opt-in `recovery`** — the ✅ examples (`receptor`, `igf1r`, `tlbr`) are built through the
  real bngsim backend and driven through a short bounded fit; the check is that the whole
  simulate → score → propose loop runs and yields a finite objective — observables map, the
  objective scores, and the optimizer advances, on a real paper model through bngsim. The
  cluster-scale / blocked examples are deliberately excluded from this tier (see above).
  Per-observable numerics are covered at the fixture level by `test_bngsim_bngl_e2e` /
  `test_bngsim_nf_e2e` / `test_bngsim_ssa_replaces_rr` (#379).

```bash
pytest tests/test_real_world_examples.py -m 'not recovery'   # backend-free tier (default CI)
BNGPATH=... pytest tests/test_real_world_examples.py -m recovery   # + real bngsim (the ✅ examples)
```
