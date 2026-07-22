#!/usr/bin/env python
"""Generate a PEtab v2 bundle for the Salazar-Cavazos-2019 egfr_simpull job.

The runnable ``egfr_simpull.conf`` is the single source of truth. Its PEtab.v2 compliance
is a verified round-trip property: ``pybnf.petab.export_job`` transcribes the native edition-2
job into a PEtab v2 problem that lints clean under ``petab.v2`` and re-imports. This
script writes a local ``petab/`` bundle on demand; generated bundles are not tracked in
the real-world gallery.

Do NOT hand-edit anything under ``petab/``: it is generated. Re-run this script after
any change to ``egfr_simpull.conf`` / ``egfr_simpull.bngl`` / ``*.exp`` so the bundle
can't drift from the conf. For Salazar-Cavazos-2019/egfr_simpull the export emits 3
observables (``observableFormula`` = bare function name), 6 estimated parameters and 18
measurements; because the job perturbs (an ``EGFconc`` dose-response scan) it also emits
the conditions and experiments tables -- 3 conditions over 3 experiments. Both ``.exp``
carry ``_SD`` columns, so the noise is a per-point Gaussian sigma and the objective is
weighted least squares (``chi_sq``).

The output is normalized to the repo's file conventions (LF endings, no trailing
whitespace, single final newline) so it is byte-stable under pre-commit and a fresh
run reproduces stable bytes exactly. Normalization only drops the empty
delimiter of trailing all-empty columns (``noisePlaceholders`` / ``noiseParameters``);
the bundle still lints clean and re-imports (verified).

Requires the PyBNF environment's interpreter (imports ``pybnf``; the lint self-check
also needs the optional ``pybnf[petab]`` extra) and BNGPATH set:
    BNGPATH=... python make_petab.py
Exit 0 = exported + linted clean (or lint skipped because the petab extra is missing);
1 = a step failed.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# A model's ``begin actions`` block and the simulation actions PEtab drives itself.
_ACTIONS_RE = re.compile(
    r"^[ \t]*begin\s+actions\b.*?^[ \t]*end\s+actions\b[^\n]*\n?", re.S | re.I | re.M
)
_SIM_ACTION_RE = re.compile(r"\b(simulate\w*|parameter_scan|bifurcate)\b", re.I)


def _preserve_network_directives(src_dir: Path, out_dir: Path) -> list[str]:
    """Keep each model's ``generate_network`` directive in its exported copy.

    ``pybnf.petab.export_job`` renders a "PEtab-clean" model by dropping the whole
    ``begin actions`` block -- correct for *simulation* actions (PEtab drives simulation
    via the measurement times / experiments), but a ``generate_network`` directive is a
    network *definition*, not a simulation action. When it carries a finiteness cap
    (``max_stoich`` / ``max_agg`` / ``max_iter``) that cap is the only thing keeping the
    reaction network finite, and PEtab v2 has no field for it -- so a consumer who
    network-generates the stripped copy would fall back to the unbounded default. This
    restores the directive into the exported copy so the generated bundle stays
    self-contained and finite.

    If the source's ``begin actions`` block is network-definition-only (no ``simulate`` /
    ``parameter_scan`` / ``bifurcate``), the faithful PEtab model is the source model
    verbatim, so the copy is restored to it byte-for-byte. If the block also holds
    simulation actions, only the ``generate_network`` line(s) are carried across.

    Returns the names of the model copies amended (empty when no source model carries a
    ``generate_network`` -- the common case, a no-op)."""
    amended = []
    for copy in sorted(out_dir.glob("*.bngl")):
        src = src_dir / copy.name
        if not src.is_file():
            continue
        src_text = src.read_text()
        block = _ACTIONS_RE.search(src_text)
        if not block or "generate_network" not in block.group(0):
            continue
        # Decide on code only -- a comment may mention "simulate" (egfr_ode's does).
        block_code = re.sub(r"#[^\n]*", "", block.group(0))
        if _SIM_ACTION_RE.search(block_code):
            gen = [
                ln
                for ln in block.group(0).splitlines()
                if ln.strip().lower().startswith("generate_network")
            ]
            body = copy.read_text().rstrip("\n")
            copy.write_text(body + "\n\nbegin actions\n" + "\n".join(gen) + "\nend actions\n")
        else:
            copy.write_text(src_text)
        amended.append(copy.name)
    return amended


def _normalize(out_dir: Path) -> None:
    """Rewrite generated text files to the repo's conventions so the generated
    bundle is byte-stable under pre-commit and identical to a fresh regeneration:
    LF endings, no trailing whitespace, exactly one final newline."""
    for f in sorted(out_dir.iterdir()):
        if f.suffix not in (".tsv", ".yaml"):
            continue
        lines = f.read_text().replace("\r\n", "\n").replace("\r", "\n").split("\n")
        lines = [ln.rstrip() for ln in lines]
        while lines and lines[-1] == "":
            lines.pop()
        f.write_text("\n".join(lines) + "\n")


def _lint(problem_yaml: Path) -> tuple[str, str]:
    """Return (status, detail); status in {'clean', 'errors', 'skipped'}."""
    try:
        from petab.v2 import Problem
        from petab.v2.lint import lint_problem
        from pybnf.petab.bngl_model import register_bngl
    except Exception as exc:  # noqa: BLE001 -- petab extra not installed
        return "skipped", f"petab lint unavailable ({exc!r}); install pybnf[petab] to enforce it"

    register_bngl()
    report = lint_problem(Problem.from_yaml(str(problem_yaml)))
    has_errors = report.has_errors() if hasattr(report, "has_errors") else bool(report)
    return ("errors", str(report)) if has_errors else ("clean", "")


def _counts(out_dir: Path) -> str:
    """Human-readable summary of the exported tables (header row excluded)."""

    def _rows(name: str) -> list[str]:
        p = out_dir / name
        if not p.is_file():
            return []
        body = p.read_text().splitlines()
        return body[1:] if body else []

    n_obs = len(_rows("observables.tsv"))
    n_meas = len(_rows("measurements.tsv"))
    n_est = sum(1 for r in _rows("parameters.tsv") if r.split("\t")[1:2] == ["true"])
    parts = [f"{n_obs} observables", f"{n_est} estimated parameters", f"{n_meas} measurements"]
    if (out_dir / "conditions.tsv").is_file():
        parts.append(f"{len(_rows('conditions.tsv'))} conditions")
    return ", ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--conf",
        default=str(HERE / f"{HERE.name}.conf"),
        help="edition-2 .conf to export (default: <slug>.conf next to this script)",
    )
    ap.add_argument(
        "--out", default=str(HERE / "petab"), help="output bundle directory (default: ./petab)"
    )
    args = ap.parse_args()

    from pybnf.petab import export_job

    conf = Path(args.conf).resolve()
    out_dir = Path(args.out).resolve()
    if not conf.is_file():
        print(f"FAIL  no such conf: {conf}")
        return 1

    # Regenerate from clean so a removed/renamed table can't linger.
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # export_job resolves conf-relative paths (model, .exp) against the CWD.
    home = os.getcwd()
    os.chdir(conf.parent)
    try:
        try:
            export_job(str(conf), str(out_dir))
        except NotImplementedError as exc:
            print(f"FAIL  export: job uses a non-PEtab-exportable feature -- {exc}")
            print("      -> this slug is native-only; do not commit a petab/ bundle for it")
            return 1
    finally:
        os.chdir(home)

    problem_yaml = out_dir / "problem.yaml"
    if not problem_yaml.is_file():
        print(f"FAIL  export: no problem.yaml under {out_dir}")
        return 1

    amended = _preserve_network_directives(conf.parent, out_dir)
    _normalize(out_dir)

    wrote = sorted(p.name for p in out_dir.iterdir())
    print(
        f"ok    export -> {out_dir.relative_to(HERE) if out_dir.is_relative_to(HERE) else out_dir}"
    )
    print(f"        wrote: {', '.join(wrote)}")
    print(f"        {_counts(out_dir)}")
    if amended:
        print(f"        kept generate_network directive in: {', '.join(amended)}")

    status, detail = _lint(problem_yaml)
    if status == "errors":
        print(f"FAIL  lint: petab.v2 reported errors\n{detail}")
        return 1
    if status == "skipped":
        print(f"warn  lint: {detail}")
    else:
        print("ok    lint  -> petab.v2 clean (no errors)")

    print(f"\nDONE  local petab/ bundle generated from {conf.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
