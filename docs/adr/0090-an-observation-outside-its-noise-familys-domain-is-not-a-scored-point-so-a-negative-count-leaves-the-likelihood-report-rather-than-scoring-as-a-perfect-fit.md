# An observation outside its noise family's domain is not a scored point, so a negative count leaves the likelihood report rather than scoring as a perfect fit (issue #523)

**Status: Accepted and implemented (2026-07-30).** A measurement a noise family cannot assign a
probability to — today only a **negative count** under `neg_bin` — is now excluded from the
pointwise likelihood the way a NaN observation already was: off the `evaluate_pointwise` observation
axis, out of `n` for AIC/BIC, out of LOO/WAIC, and reported once per observable with a count. The
**cost** path is untouched: such a point still contributes nothing to the objective, which is the
correct fitting behavior and is exactly why the density path had to stop reading that "nothing" as
`log p = 0`.

## Problem

The count family guards against a negative observation by having it contribute nothing:

```python
def data_fit(self, prediction, observation, noise, extra=None):
    if observation < 0:
        return 0
```

The guard is right. A negative value has no negative-binomial probability, and real surveillance
data contains them: a jurisdiction revising its cumulative total downward produces a negative daily
increment. But "contributes nothing to the *cost*" becomes something else one layer up.
`NoiseModel.log_density` is `-(data_fit + normalizers + constant)`, and a PMF is self-normalizing,
so for `NegBinomial(location=MEAN)` with `r = 9.06`:

```pycon
>>> nb.log_density(3000.0, -5948.0, r)      # negative count
-0.0
>>> nb.log_density(3000.0,  3000.0, r)      # a perfect prediction
-7.83
```

The negative observation is assigned **probability 1** — a better per-point density than any real
observation can achieve, including one the model predicts exactly. And it was a *scored point*:
`_pointwise_suffix` skipped a point only on `np.isnan(observation)`, so these entered `n` in
`likelihood_information_criteria` and the `ids`/`values` arrays backing LOO/WAIC (ADR-0056).

For AIC/BIC the distortion is small where it surfaced — fitting NYT MSA daily case counts with
`noise_model = neg_bin, ..., location = mean`, the New York series has one negative count in 649
rows, and correcting `n` to 648 moves BIC by `k·ln(649/648) ≈ 0.003`. Two things still make it worth
fixing:

1. **It scales badly and silently.** Each such point contributes the *maximum possible* per-point
   log-density. A series with many downward revisions — not unusual in surveillance data — looks
   increasingly well-fit for a reason that has nothing to do with the model, and nothing in the
   output said so.
2. **LOO/WAIC are more exposed than AIC.** They consume the per-point densities directly, so points
   pinned at `log p = 0` distort the pointwise distribution — and the Pareto-k diagnostics computed
   from it — more than they distort a summed criterion.

## Decision

### The observation domain is a property of the family, asked of the data alone

A new predicate on `NoiseModel`:

```python
def observation_in_domain(self, observation):
    return True
```

`True` for every family whose support is the whole real line (Gaussian, Laplace, Student-t —
they exclude nothing and are byte-identical). `NegBinomial` overrides it with `observation >= 0`
and names its domain in a companion attribute, `observation_domain = 'a non-negative count'`, which
the warning quotes.

The predicate reads **only the observation** — not the prediction, the noise parameters, or the
draw. That is the load-bearing property: it partitions a data set once and identically for every
parameter set, so excluding on it preserves the invariant `evaluate_pointwise` documents and the
ArviZ bridge depends on — *the emitted observation set is fixed by the experimental data, so every
draw yields the same ids in the same order*, the rectangular `chain × draw × obs` array (ADR-0056).
That docstring now names its second skip condition rather than claiming NaN is the only one.

### It gates the density path, not the cost path

`_pointwise_suffix` skips an out-of-domain observation immediately after resolving the point's
`(family, sources)` spec, and `_aligned_prediction_suffix` — the Kalman twin (ADR-0067 Stage 3) —
carries the same skip in the same position, so the two walks stay point-for-point aligned by
construction. It is inert there today (that walk admits only a linear-scale Gaussian) and
load-bearing the moment the gate widens.

`data_fit` and every derivative guard are **unchanged**. This is the deliberate asymmetry:

| path | a negative count | why |
|---|---|---|
| objective / gradient | contributes 0 | it is not evidence for or against any parameter value; the fit should neither reward nor punish it |
| pointwise density | not a point at all | a normalized density of 0 log-units is a *claim* (probability 1), not an abstention |

An alternative — returning `-inf` or `nan` from `log_density` for such a point — was rejected: it
propagates into the AIC's finiteness check and into `az.loo`, converting a small silent bias into a
loud failure of the whole diagnostic, and it still misstates the situation. The point is missing
data for the likelihood, not infinitely improbable data.

Nothing gates on the count family specifically. `neg_bin` is simply the only family in PyBNF today
with a bounded observation domain; a future Poisson, binomial, or beta family declares its own and
inherits the exclusion, the `n` correction, and the warning.

### The exclusion is reported

Silently scoring fewer points is the failure mode this issue is about, so `_warn_out_of_domain`
prints once per observable, with the count:

```text
Warning: excluded 1 measurement(s) of 'nyc' in nyc_model/nyc from the likelihood: this
observable's noise model scores only a non-negative count. They contribute nothing to the
objective, are not counted in n for AIC/BIC, and are off the LOO/WAIC observation axis.
```

Deduplicated through `SummationObjective.warned` (the set the row-matching warning already uses),
because `evaluate_pointwise` runs once per recorded draw in a sampler while the count — a property
of the experimental data — is the same every time. Every fit with a likelihood objective computes
information criteria at the end (`_compute_information_criteria`), so the warning reaches a plain
optimizer run too, not only a Bayesian one.

## Consequences

- `n` in `Results/information_criteria.txt`, the lnL behind AIC/BIC/AICc, and the LOO/WAIC
  observation axis now count only in-domain measurements. A `neg_bin` fit on data containing
  negative counts reports exactly what it would report on the same data with those rows deleted.
- A user learns the exclusion happened; previously there was no way to.
- Every other family, and every objective value on every path, is byte-identical.
- The cost path keeps its own guard, so the two behaviors can drift only if a future family
  implements one without the other. The predicate's docstring states the pairing.
