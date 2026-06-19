# The native noise_model surface exposes the location axis (mean vs median); default median

ADR-0011 made the **location interpretation** -- which summary of the noise
distribution the deterministic prediction is taken to be (mean / median / mode) --
a first-class axis of a `NoiseModel`, but left it engine-only: it was reachable
only as a hard-coded class default per objfunc (`lognormal` = `Gaussian(LOG10,
MEDIAN)`; everything else effectively median). No config could *select* it. Two
camps in the field set the prediction differently for a **lognormal** observable
(the one case where it matters -- see below):

* **median** (`prediction = median(data)`, i.e. `log(prediction)` is the mean of
  `log(data)`): the "transform both sides, additive Gaussian on the transform"
  convention. PEtab v2 mandates it; Data2Dynamics / pyPESTO use it; PyBNF's own
  legacy `lognormal` objfunc has always been median (no moment correction).
* **mean** (`prediction = E[data]`): the ODE solution is the *expectation* of the
  underlying stochastic process (the mean-field limit of the CME), so aligning the
  prediction with the mean is the principled choice for multiplicative noise. For a
  lognormal this is `mu = log10(prediction) - sigma**2*ln10/2`.

Both are legitimate; PyBNF is PEtab-defaulted, not PEtab-bound (ADR-0004), so it
should support **both** without forcing the PEtab choice on native users.

The distinction **only bites for a Gaussian on a log scale** (lognormal). Linear
Gaussian and Laplace (on any native scale) are symmetric, so mean = median and the
axis is trivial there.

- **Expose the location axis as an optional `location = mean|median` field on the
  `noise_model` line.** Chosen over a new family token (e.g. `lognormal_mean`): the
  token approach is a code proxy for what is genuinely the *location axis*, and it
  fails to generalize (every asymmetric family would need a `_mean`/`_median`
  twin). The field names the axis directly. The grammar adds the field via
  MatchFirst on the `location` literal, so a real noise-parameter name falls
  through to a source field; the parsed `noise_model` value extends from
  `(family, {param: (verb, arg)})` (ADR-0021) to
  `(family, {param: (verb, arg)}, location)`.

- **Omitting `location` defaults to median.** This is the safe default on three
  counts, not just PEtab parity: (1) it is **backward-compatible** -- the legacy
  `lognormal` objfunc is already median, so no existing config changes; (2) it is
  **byte-identical today** -- mean and median diverge only for lognormal, which
  already defaults median, and the symmetric families coincide; (3) it matches
  PEtab v2. Mean is therefore purely **opt-in** -- the non-default, principled
  interpretation you must ask for. Implementation note: an omitted `location`
  keeps the family's built-in default (median for `lognormal`; the symmetric
  families are unaffected), rather than force-overriding -- the two are equivalent
  for every family, and keeping the default leaves the native `normal`/`laplace`
  tokens byte-identical.

- **`neg_bin` accepts `mean` (redundant) and rejects `median` as unimplemented --
  not as impossible.** A negative binomial has both a mean and a median; centering
  the prediction on either is a coherent model. PyBNF parameterizes `neg_bin`
  **directly by its mean** (the prediction *is* the mean), so `location = mean` is
  the current interpretation and is accepted as a no-op. `location = median` is a
  coherent model that PyBNF does **not implement** (the neg_bin median has no closed
  form; placing the prediction at it would need a numeric CDF inversion), so it
  raises a `PybnfError` characterized as *not implemented*, leaving room for it as a
  future feature. The location-scale families (Gaussian/Laplace) implement both via
  the offset machinery.

  **Superseded by ADR-0031 / issue #419:** the `neg_bin` median *is* now implemented.
  The prediction is interpreted as the count distribution's continuous 0.5-quantile,
  and the mean placing the median there is recovered by a per-point bounded CDF
  inversion (`scipy.special.betainc` + `scipy.optimize.brentq`); the count family owns
  this realization rather than going through the additive-offset machinery. So
  `location = median` no longer raises -- it runs (and, under a modern edition where it
  resolves *implicitly*, warns that the value changed from the legacy mean).

- **`mean` is only ever the correct Gaussian moment correction on the native
  surface.** The moment offset (`scale.mean_offset`, ADR-0011/0022) is the
  **Gaussian** correction. On the native surface the only log-scale family token is
  `lognormal` (a Gaussian), and `normal`/`laplace` are linear (offset 0), so a
  `mean` correction is never applied where it would be wrong. (A mean-aligned
  *log-Laplace* would need a Laplace-specific moment correction -- its mean is
  `e^m/(1-b**2)` -- but there is no native log-Laplace token to construct one, so
  the gap is unreachable here. It is the same "generalize when a second
  location-scale family arrives" boundary ADR-0011 records.)

- **PEtab import is unaffected -- it stays median.** PEtab v2 specifies the median,
  so `petab/observables.py` continues to build every noise model with
  `location=MEDIAN` (ADR-0023); a mean-aligned problem is simply not valid PEtab
  v2. This ADR is the *native* `.conf` surface giving the choice PEtab does not.

## Considered Options

- **A new family token (`lognormal_mean`).** Rejected: it encodes the location
  axis as a code, the parallel-table smell ADR-0019/0021 reject, and would need a
  `_mean` twin per asymmetric family. The field is the honest representation of the
  existing axis.

- **Change the default to mean (the more principled convention).** Rejected: it
  would silently change every existing lognormal fit by `sigma**2*ln10/2` (a quiet
  numerical shift, the units-trap failure mode ADR-0022 fought) and diverge from
  PEtab and PyBNF's own history. Median-default + opt-in mean gets the capability
  with zero silent change.

- **Expose the whole-fit default via a standalone `noise_location` key, not inline
  on `objfunc`.** Done as a follow-up to the per-observable field. A standalone
  global key (`noise_location = mean|median`, a `GlobalConfig` field beside
  `neg_bin_r` -- an objfunc/noise param read regardless of fit_type) needs **zero
  grammar change** and composes cleanly: it sets the default location of the
  objfunc's noise model, which a per-observable `noise_model ... location =` field
  overrides -- mirroring how `objfunc` is the global default family×source and
  `noise_model` overrides per observable. Chosen over inline `objfunc = lognormal,
  location = mean`, which would need pulling `objfunc` out of the simple-string-key
  grammar into its own branch (touching the most-referenced config key and every
  golden). Applied in `Configuration._load_obj_func` via
  `LikelihoodObjective.set_default_location`; rejected (with a clear error) on a
  non-likelihood objfunc (`sos`/`kl`/...) that has no noise model. (`neg_bin + median`
  was the same unimplemented path as the per-observable field; ADR-0031 / #419 now
  implements it, so it runs rather than raising.) Default unset -> each family keeps
  its own default (median), so existing configs are byte-identical (the golden gains
  only `noise_location: null`).

Relevant ADRs: **0011** (the location axis this exposes, and the Gaussian-only
moment-correction boundary), **0021** (the native `noise_model` surface and its
parsed-value shape this extends), **0022** (the log-base convention the moment
correction uses), **0023** (the PEtab observables adapter, which stays median),
**0004** (PEtab-defaulted not PEtab-bound -- why native users get the choice).
Follow-up: the global `objfunc` location default.
