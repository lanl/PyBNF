# An SBML entity an initialAssignment derives has no value the model file settles, so the scanner drops the declared attribute and the measurement layer inlines the derivation (issue #795)

## Status

Accepted and implemented (2026-09-22), for every edition. It changes the numbers PyBNF reads
out of an SBML or antimony model only when that model uses an `initialAssignment`, which no
model committed to this repository does on a parameter.

## The problem

SBML lets a model give an entity its starting value in two places. A parameter carries a
`value` attribute, a compartment a `size`, and a species an `initialAmount` or an
`initialConcentration`. A `<listOfInitialAssignments>` entry may then supersede any of them,
and when it does, the attribute is a placeholder the model never starts from. Tool-exported
SBML writes both routinely, and antimony writes the assignment with no attribute at all, so
`k_derived = k_base + 1` becomes `<parameter id="k_derived" constant="true"/>` plus an
assignment.

The import-time scanner, `pybnf/petab/_sbml.py`, read only the attributes. Two consumers used
the result, and both were wrong in the same way:

- `Configuration._model_expression_namespace` puts the scanned values into the measurement
  layer's constant snapshot, so an observable or noise formula naming such a parameter was
  scored with the placeholder. On Bertozzi_PNAS2020, where `beta_N` carries `value="0.0"` and
  an assignment computing it from three parameters, that is a fit scored with a rate constant
  of zero, silently.
- The exporter's `_numeric_nominal` feeds `conditions.mutation_target_value`, which turns a
  relative condition into an absolute number, so a `condition: kon * 2` on such a parameter
  wrote a wrong number into `conditions.tsv`.

Where no attribute is written the failure was different and not much better. The symbol stayed
in the namespace with no value behind it, so a formula naming it reached the branch in
`pybnf/measurement/base.py` that reports the symbol as "neither a simulation-output column nor
a fit/model parameter", a message that is wrong and that sits under a comment saying validation
should have made it unreachable.

Measured on the public PEtab benchmark collection, 16 of 25 models use initial assignments, 31
of them on a parameter or compartment, and 28 of those still handed out a stale attribute.

## The decision

An entity whose initial value an assignment derives has **no value the model file settles**, so
the scanner reports none, and the measurement layer inlines the derivation instead of binding a
number.

Concretely, for a parameter or a compartment:

- An assignment that is arithmetic over numbers alone **is** the value. It replaces the
  attribute. `_evaluate_mathml_number` walks the MathML and refuses a `<ci>`, which makes
  "evaluates to a number" and "reads no other entity" one test rather than two.
- An assignment computed from other entities means the entity leaves `parameter_values` and
  leaves `namespace_symbols`, and its serialized right-hand side is recorded in
  `derived_initial_values`. A formula naming it is rewritten to the entities it is computed
  from, which is what `#465` already does for an `assignmentRule` target, through the same
  inliner and the same error shape.

A **species** is deliberately different. An initial assignment pins t=0 only, so the species is
still a dynamical state and still a simulation-output column. It keeps its place in the
namespace and is never inlined; it only loses a stale declared initial. Treating it like a rule
target would break a committed fixture on the first run, since Boehm's observables name
`STAT5A` and `STAT5B`, both set by initial assignment.

Inlining is gated on soundness. The substituted expression is read at a measurement time, while
the assignment fixed a value from its inputs at t=0, so the two agree only when every input
holds still for the whole simulation. The scanner therefore refuses to inline an assignment
that reads a species, a `constant="false"` parameter, an assignment-rule or rate-rule target,
an event-assignment target, or any symbol an algebraic rule touches. A refused assignment is
recorded with no expression and a clause naming the offender, which the loader composes into
one error.

## Why not the alternatives

Dropping the value and leaving the symbol in the namespace with a better error message was
rejected because it turns 28 measured silent wrong numbers into 28 refusals when a correct
answer is available, and because the better message would live in a branch the code says is
unreachable.

Dropping the value and excluding the symbol with no inlining was rejected for the same first
reason. Both alternatives also leave a user whose model simulates correctly unable to write an
observable over it without rewriting the expression by hand.

Evaluating a derived parameter from the file's own values, which is the obvious third option,
is the defect this repository has just argued against upstream. A PEtab parameter table may
override or estimate what the value is computed from, so a number settled from file defaults is
handed to a simulator as a constant and goes stale. The same rule now holds on both sides:
petab's `SbmlModel` reports an initial assignment only when it reduces to a number, and
`PEtab-dev/libpetab-python#517` applies it to BioNetGen parameter expressions.

## Consequences

The import-time view now agrees with the runtime. ADR-0094 already recomputes a parameter an
initial assignment fixes whenever a dependency moves, and because `MeasurementModel.materialize`
resolves a symbol from the parameter set before the constant snapshot, an inlined dependency
that the fit estimates tracks the fit. An observable over Bertozzi's `beta_N` is scored with
the value the model starts from, and it moves when `R0_` is estimated.

Two adjacent defects are fixed in the same pass, because they are the same sentence. An
`assignmentRule` target that carries a vestigial `value` attribute no longer reports that
number either, and `sigma = prediction_formula` now inlines the same map the measurement layer
does, so the two layers no longer disagree about which symbols a model offers.

The refusal in `conditions.mutation_target_value` no longer blames BNGL alone. It names the
parameter, states both languages' version of the same cause, and says what to write instead.
