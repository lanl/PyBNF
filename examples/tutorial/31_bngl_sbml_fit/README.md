# Lesson 31 — One fit across two model languages (BNGL + SBML)

**Feature:** a joint fit whose experiments' models are in *different formats* — BNGL and SBML — sharing parameters · **Difficulty:** ★★★ · **Tier:** recovery

[Lesson 16](../16_joint_fit) fit one parameter set across several *models*; [lesson
11](../11_interop) fit the same model written three ways (BNGL / Antimony / SBML),
one language at a time. This lesson combines the two ideas: a single joint fit in
which **one experiment's model is BNGL and another's is SBML**, both sharing the
fitted parameters. PyBNF does not care what language each model is written in.

## The scenario

A reversible bimolecular association `A + B ⇌ C` (association `kf`, dissociation
`kr`) measured at two ligand levels — two conditions of the same reaction, which
happen to be modeled in two formats:

| experiment | model | format | condition |
|---|---|---|---|
| `low_ligand`  | [binding_low.bngl](binding_low.bngl)  | BNGL | `B0 = 75`  |
| `high_ligand` | [binding_high.xml](binding_high.xml) | SBML | `B0 = 150` |

The [SBML file](binding_high.xml) is generated from a readable
[Antimony source](binding_high.ant) (`A + B -> C` etc.) — the same reaction as the
BNGL model. Both models declare parameters `kf` and `kr`, so those bind to the same
fitted values (bare-name binding, as in [lesson 16](../16_joint_fit)).

## The conf

```
model: binding_low.bngl          # a BNGL model  ...
model: binding_high.xml          # ... and an SBML model
bngl_backend = bngsim
sbml_backend = bngsim            # both simulated by bngsim

experiment: low_ligand,  model: binding_low.bngl,  data: bind_low.exp
experiment: high_ligand, model: binding_high.xml, data: bind_high.exp

uniform_var = kf  0.0002  0.05   # shared by BOTH models, bound by name
uniform_var = kr  0.02    3.0
```

`bngl_backend` and `sbml_backend` are both `bngsim`, so one fit drives both models.
The fit recovers `kf ≈ 0.002` and `kr ≈ 0.3` from the two binding curves together —
verified by [`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py).

## The one gotcha: observable names differ by format

A BNGL model reports its declared **observables** (`Obs_A`, `Obs_C`, …). The bngsim
SBML path instead reports the **raw species** by their SBML ids (`A`, `B`, `C`). So
each experiment's data column is named to match *its* model's output:

- `bind_low.exp` (BNGL model) → column **`Obs_C`**;
- `bind_high.exp` (SBML model) → column **`C`** (the species id).

That is the only concession to mixing formats. (Two model *names* must also differ —
here `binding_low` vs `binding_high` — just as in any multi-model fit.) See
[lesson 11](../11_interop) for the single-model interop details and the alternative
`observable: Obs_C, formula: C` measurement-layer mapping.

## Run it

```bash
pybnf -c bngl_sbml_fit.conf
```

## Regenerating the data (and the SBML)

```bash
python ../regenerate_data.py 31_bngl_sbml_fit    # writes bind_low.exp + bind_high.exp
```

`regenerate_data` drives the SBML model through the `sbml_backend` and reads its raw
species column. To regenerate `binding_high.xml` from its Antimony source:

```bash
python -c "import antimony as a; a.loadAntimonyFile('binding_high.ant'); \
           open('binding_high.xml','w').write(a.getSBMLString(a.getMainModuleName()))"
```

## Where this sits

- [Lesson 11](../11_interop) — one model, three languages, fit separately.
- [Lesson 16](../16_joint_fit) — a joint fit across multiple BNGL models.
- This lesson — a joint fit *mixing* BNGL and SBML models in one run.
