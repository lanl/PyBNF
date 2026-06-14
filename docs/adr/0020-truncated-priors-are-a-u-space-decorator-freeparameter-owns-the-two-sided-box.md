# Truncated priors are a u-space decorator; FreeParameter owns the two-sided reflecting box (issue #411)

PEtab v2 says *"prior distributions are truncated by the lowerBound and
upperBound if the prior's domain exceeds the parameter bounds"* — so a `normal`
prior with finite bounds **is** a truncated normal, the common case. PyBNF could
not represent it: `FreeParameter` attached reflecting bounds only to
finite-support (Uniform) families, so Normal / Laplace / log-* priors were always
unbounded, and the importer's Step 1 raised `NotImplementedError` (ADR-0019,
`_reject_truncation`) on exactly this row. This ADR adds truncation as a
capability and unblocks that boundary.

Truncation is a **hybrid**: it changes both the *support* (the density's nonzero
region) and the *normalization* (`p(θ)/Z` over `[lb, ub]`). ADR-0010 split those
two concerns deliberately — *Support* lives on the prior family, *Reflecting
Bounds* live on `FreeParameter`. A truncated prior touches both, so we place each
half where ADR-0010 already put it:

- **`TruncatedPrior` is a u-space decorator over a family** (`pybnf/priors/truncated.py`).
  It wraps any inner `Prior` and confines it to `[lo_u, hi_u]` in the sampling
  space `u` (the box's bounds mapped through the owning `FreeParameter`'s
  `Scale`). It is itself a `Prior` — `logpdf = inner.logpdf(u) − log Z` inside the
  range and `−inf` outside; `rvs`/`ppf` by **inverse-CDF over the truncated
  region** (`inner.cdf(lo) + q·Z`), so a draw lands *inside the box by
  construction* rather than being drawn unbounded and folded back. It reports
  `has_bounded_support = True`. `Z = inner.cdf(hi_u) − inner.cdf(lo_u)` is computed
  once at construction; the inner family's math is untouched (the M2.3 ethos — one
  decorator, family-agnostic, works for Normal/Laplace and any future scipy-backed
  family without per-family code).

- **`FreeParameter` owns the box.** When constructed with finite truncation bounds
  on an unbounded-support family, it sets the same finite `lower_bound`/
  `upper_bound` the Uniform families already use and wraps its prior in a
  `TruncatedPrior` carrying that box in `u`. The reflection fold (`_reflect`) is
  already **family-agnostic in `u`** — it was only *gated off* for unbounded
  families — so it composes unchanged: a random-walk proposal folded into the box
  stays symmetric, so plain Metropolis still targets the correct truncated
  posterior (ADR-0003). Latin-hypercube seeding picks the parameter up for free
  via the now-`True` `has_bounded_support` + the truncated `ppf`.

- **The normalizer `Z` is computed but cancels for inference.** `Z` is
  parameter-independent (the bounds are fixed), so it cancels in the MCMC
  acceptance ratio and in MAP optimization (ADR-0003, "target defined in `u`"). We
  still subtract `log Z` so `prior_logpdf` reports the *true* normalized truncated
  density — consistent with scipy's already-normalized untruncated `logpdf` — at
  the cost of one constant evaluated once. The real behavioral change is the
  **sampling**: it now draws from the truncated region, not an unbounded draw
  folded in.

- **Two-sided only; one-sided truncation still raises.** The reflecting-box fold
  needs *two* finite bounds (a finite width to fold into). A one-sided truncation
  (one bound infinite — e.g. a normal restricted to `[0, ∞)`, or a log-* prior
  whose lower bound is its natural domain `θ > 0`) cannot form a box, so the
  importer continues to raise `NotImplementedError` with a message naming the
  missing finite bound. Two-sided truncation — PEtab's common `normal` prior with
  finite `[lb, ub]` — is what this ADR delivers.

- **`FreeParameter`'s public surface gains two optional, backward-compatible
  params.** ADR-0010 froze the constructor at `(name, type, p1, p2, value,
  bounded)`; this extends it with `lb=None, ub=None` (the truncation box, distinct
  from `p1`/`p2`, which remain the family's location/scale). Omitting them is
  today's behavior exactly; the Uniform and unbounded-untruncated paths stay
  byte-identical, and `set_value` threads the box through reconstruction.

- **Native `.conf` grammar is a deferred follow-up.** The capability is exposed on
  the `FreeParameter` constructor and consumed by the PEtab importer (the
  motivating path for #411). A native `.conf` surface for a bounded `normal_var` /
  `laplace_var` (extra `lb ub` tokens) is a separate, lower-value grammar change
  (and would touch the `parse.py` token lists flagged in #402); it is not required
  to unblock #407 Step 1 and is left for when a native config needs it.

## Considered Options

- **Truncation on the `FreeParameter` alone (extend reflecting bounds + override
  sampling there).** Rejected: the renormalized density and the truncated
  inverse-CDF sampling are *distribution* math, and putting them on `FreeParameter`
  would re-smear family knowledge across the class ADR-0010 just cleaned up. The
  decorator keeps the family math in the priors package; `FreeParameter` only owns
  the box, as before.

- **A per-family `truncated()` method (e.g. scipy `truncnorm` for Normal).**
  Rejected for Step 1: scipy `truncnorm` is more robust in extreme tails, but a
  single generic inverse-CDF decorator is exact for the typical (non-extreme) PEtab
  bounds, works for every current and future family with one implementation, and
  matches the "one seam, family-agnostic" shape of the Laplace proof (ADR-0010). A
  family may override later if extreme-tail precision is ever needed.

- **Track only the unnormalized density (skip `Z`).** Rejected: `Z` is one cheap
  constant computed once, and computing it makes reported log-prior / log-posterior
  values genuinely those of a truncated distribution, matching scipy's normalized
  untruncated densities. It cancels for inference either way, so there is no
  correctness reason to omit it and a reporting reason to keep it.

- **Support one-sided truncation now.** Rejected for this step: it would require a
  one-sided proposal scheme (the triangle-wave fold has no finite width to fold
  into), a larger change than the common two-sided case needs. Left as a follow-up;
  the boundary raises explicitly rather than silently importing a different prior.

Relevant ADRs: **0003** (prior in the parameter's own scale, no Jacobian — where
`Z`-cancels lives), **0010** (Prior family vs FreeParameter scale/bounds split; the
Support-vs-Reflecting-Bounds seam this extends), **0019** (importer Step 1, whose
`_reject_truncation` raise this turns into a real two-sided mapping).
