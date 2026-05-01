#!/usr/bin/env python3
"""Launch-smoke runner for the PyBNF fitting jobs in pybnf_fitting_jobs.pdf.

The PDF table lists twelve scatter-search jobs and compares two engines:

* run network: force the legacy PyBNF/BioNetGen network subprocess path
* BNGsim auto: use PyBNF's default BNGsim auto-detection path

This script is intentionally a launch smoke harness first. By default each run
gets a short timeout; a timeout is counted as a successful launch if PyBNF got
far enough to create output/log files. Use a larger timeout for real timing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS_ROOT = REPO_ROOT / "dev" / "pybnf_fitting_jobs"
DEFAULT_RUN_ROOT = REPO_ROOT / "dev" / "pybnf_fitting_job_runs"


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    table_label: str
    aliases: tuple[str, ...]
    attempts: int


JOBS: tuple[JobSpec, ...] = (
    JobSpec("egg-ss", "egg-ss", ("egg-ss", "egg_ss", "egg"), 448),
    JobSpec("elephant-ss", "elephant ss", ("elephant-ss", "elephant_ss", "elephant"), 2092),
    JobSpec("fit-ss", "fit ss", ("fit-ss", "fit_ss"), 392),
    JobSpec("raf-ss-392", "raf-ss", ("raf-ss-392", "raf_ss_392", "raf-ss-a"), 392),
    JobSpec("raf-ss-280", "raf-ss", ("raf-ss-280", "raf_ss_280", "raf-ss-b"), 280),
    JobSpec("rec-ss", "rec-ss", ("rec-ss", "rec_ss", "rec"), 448),
    JobSpec("igf1r-ss", "igf1r-ss", ("igf1r-ss", "igf1r_ss", "igf1r"), 448),
    JobSpec("degran-ss-144", "degran-ss-144", ("degran-ss-144", "degran_ss_144", "degran"), 504),
    JobSpec("egfr-ss", "egfr-ss", ("egfr-ss", "egfr_ss"), 140),
    JobSpec("mapk-ss", "mapk-ss", ("mapk-ss", "mapk_ss", "mapk"), 504),
    JobSpec("egfr-ss-288", "egfr-ss-288", ("egfr-ss-288", "egfr_ss_288"), 728),
    JobSpec("fey-ss", "fey-ss", ("fey-ss", "fey_ss", "fey"), 504),
)


ENGINES = {
    "run_network": {
        "label": "run network",
        "env": {"PYBNF_NO_BNGSIM": "1"},
    },
    "bngsim_auto": {
        "label": "BNGsim auto",
        "env": {"PYBNF_NO_BNGSIM": None},
    },
}


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def iter_conf_files(root: Path) -> Iterable[Path]:
    skip_parts = {
        ".git",
        ".pytest_cache",
        ".venv",
        "FailedSimLogs",
        "Initialize",
        "Results",
        "Simulations",
        "build",
        "output",
        "runs",
    }
    for path in root.rglob("*.conf"):
        if any(part in skip_parts for part in path.parts):
            continue
        yield path


def conf_value(conf_path: Path, key: str) -> str | None:
    key_re = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.*?)\s*(?:#.*)?$")
    try:
        lines = conf_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        match = key_re.match(line)
        if match:
            return match.group(1).strip()
    return None


def find_job_config(root: Path, spec: JobSpec) -> tuple[Path | None, list[Path]]:
    aliases = {normalize(alias) for alias in spec.aliases}
    candidates: list[tuple[int, Path]] = []

    for conf in iter_conf_files(root):
        fit_type = conf_value(conf, "fit_type")
        if fit_type is not None and fit_type.strip().lower() != "ss":
            continue

        rel = conf.relative_to(root)
        norm_rel = normalize(str(rel))
        norm_stem = normalize(conf.stem)
        norm_parent = normalize(conf.parent.name)

        score = 0
        for alias in aliases:
            if alias == norm_stem or alias == norm_parent:
                score = max(score, 100 + len(alias))
            elif alias in norm_rel:
                score = max(score, 50 + len(alias))

        if score:
            candidates.append((score, conf))

    candidates.sort(key=lambda item: (-item[0], len(str(item[1])), str(item[1])))
    paths = [path for _, path in candidates]
    if len(paths) == 1:
        return paths[0], paths
    if len(paths) > 1 and candidates[0][0] > candidates[1][0]:
        return paths[0], paths
    return None, paths


def patch_config(conf_path: Path, dest_path: Path, overrides: dict[str, str]) -> None:
    lines = conf_path.read_text(encoding="utf-8", errors="replace").splitlines()
    key_re = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=.*$")
    seen: set[str] = set()
    patched: list[str] = []

    for line in lines:
        match = key_re.match(line)
        if not match:
            patched.append(line)
            continue

        key = match.group(2)
        if key in overrides:
            patched.append(f"{key} = {overrides[key]}")
            seen.add(key)
        else:
            patched.append(line)

    missing = [key for key in overrides if key not in seen]
    if missing:
        patched.append("")
        patched.append("# Added by benchmarks/run_pybnf_fitting_jobs.py")
        for key in missing:
            patched.append(f"{key} = {overrides[key]}")

    dest_path.write_text("\n".join(patched) + "\n", encoding="utf-8")


def process_started(run_dir: Path, output_dir: Path) -> bool:
    markers = (
        run_dir / "pybnf.log",
        output_dir / "Results",
        output_dir / "Simulations",
        output_dir / "alg_backup.bp",
    )
    return any(path.exists() for path in markers)


def terminate_process_group(proc: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        proc.terminate()
    else:
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run_one(
    *,
    python: Path,
    conf_path: Path,
    run_dir: Path,
    engine: str,
    timeout: float,
    parallel_count: str | None,
    log_level: str,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir / "output"
    simulation_dir = run_dir / "simulation_work"
    patched_conf = run_dir / "config.conf"

    overrides = {
        "output_dir": str(output_dir),
        "simulation_dir": str(simulation_dir),
    }
    if parallel_count is not None:
        overrides["parallel_count"] = parallel_count

    patch_config(conf_path, patched_conf, overrides)

    cmd = [
        str(python),
        "-m",
        "pybnf",
        "-c",
        str(patched_conf),
        "-o",
        "-l",
        str(run_dir / "pybnf"),
        "-L",
        log_level,
    ]

    env = os.environ.copy()
    for key, value in ENGINES[engine]["env"].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    started_at = time.time()
    timed_out = False

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(
            cmd,
            cwd=str(conf_path.parent),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=(os.name != "nt"),
        )
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(proc)
            returncode = proc.returncode

    elapsed = time.time() - started_at
    started = process_started(run_dir, output_dir)

    if timed_out and started:
        status = "started_timeout"
    elif timed_out:
        status = "timeout_no_start"
    elif returncode == 0:
        status = "completed"
    else:
        status = "failed"

    return {
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": round(elapsed, 3),
        "timed_out": timed_out,
        "started": started,
        "cmd": cmd,
        "cwd": str(conf_path.parent),
        "patched_config": str(patched_conf),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def selected_jobs(job_ids: list[str] | None) -> list[JobSpec]:
    if not job_ids:
        return list(JOBS)
    known = {job.job_id: job for job in JOBS}
    unknown = [job_id for job_id in job_ids if job_id not in known]
    if unknown:
        valid = ", ".join(job.job_id for job in JOBS)
        raise SystemExit(f"Unknown job id(s): {', '.join(unknown)}\nValid jobs: {valid}")
    return [known[job_id] for job_id in job_ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch-smoke the PyBNF fitting jobs listed in dev/pybnf_fitting_jobs.pdf.",
    )
    parser.add_argument(
        "--jobs-root",
        type=Path,
        default=DEFAULT_JOBS_ROOT,
        help=f"Root containing the archived job config directories (default: {DEFAULT_JOBS_ROOT})",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=f"Directory for isolated run outputs (default: {DEFAULT_RUN_ROOT})",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to run `-m pybnf`.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-job launch timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Replicates per job/engine (default: 1).",
    )
    parser.add_argument(
        "--engines",
        choices=tuple(ENGINES),
        nargs="+",
        default=list(ENGINES),
        help="Engines to run (default: run_network bngsim_auto).",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=None,
        help="Subset of PDF job ids to run.",
    )
    parser.add_argument(
        "--config-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON object mapping job ids to .conf paths. Relative paths "
            "are resolved under --jobs-root. Useful for duplicate/ambiguous rows."
        ),
    )
    parser.add_argument(
        "--parallel-count",
        default="1",
        help="Override parallel_count in patched configs; use 'original' to preserve configs (default: 1).",
    )
    parser.add_argument(
        "--log-level",
        default="i",
        choices=("debug", "info", "warning", "error", "critical", "none", "d", "i", "w", "e", "c", "n"),
        help="PyBNF log level passed with -L (default: i).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configs and print planned runs without launching PyBNF.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not exit nonzero when a PDF job config cannot be resolved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = selected_jobs(args.jobs)
    jobs_root = args.jobs_root.resolve()
    run_root = args.run_root.resolve()
    python = args.python.expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).resolve()
    parallel_count = None if args.parallel_count == "original" else args.parallel_count

    print("PyBNF fitting jobs launch smoke")
    print(f"  jobs root: {jobs_root}")
    print(f"  run root:  {run_root}")
    print(f"  python:    {python}")
    print(f"  timeout:   {args.timeout:.1f}s")
    print(f"  engines:   {', '.join(args.engines)}")
    print(f"  parallel:  {args.parallel_count}")
    print()

    if not jobs_root.is_dir():
        print(f"ERROR: jobs root does not exist: {jobs_root}")
        print("Place the archived PyBioNetFit job config set there, or pass --jobs-root PATH.")
        return 0 if args.allow_missing else 2

    config_map: dict[str, str] = {}
    if args.config_map is not None:
        config_map_path = args.config_map.expanduser()
        if not config_map_path.is_absolute():
            config_map_path = (Path.cwd() / config_map_path).resolve()
        config_map = json.loads(config_map_path.read_text(encoding="utf-8"))
        if not isinstance(config_map, dict):
            raise SystemExit(f"--config-map must contain a JSON object: {config_map_path}")

    resolved: dict[str, Path] = {}
    missing: list[dict[str, object]] = []

    for job in jobs:
        candidates: list[Path] = []
        if job.job_id in config_map:
            conf = Path(config_map[job.job_id]).expanduser()
            if not conf.is_absolute():
                conf = jobs_root / conf
            if not conf.is_file():
                conf = None
                candidates = []
        else:
            conf, candidates = find_job_config(jobs_root, job)
        if conf is None:
            missing.append({
                "job_id": job.job_id,
                "table_label": job.table_label,
                "candidates": [str(path) for path in candidates[:10]],
                "candidate_count": len(candidates),
            })
            if candidates:
                print(f"MISSING/AMBIGUOUS {job.job_id}: {len(candidates)} candidates")
                for path in candidates[:5]:
                    print(f"  candidate: {path}")
            else:
                print(f"MISSING {job.job_id}: no matching scatter-search .conf found")
        else:
            resolved[job.job_id] = conf
            print(f"FOUND {job.job_id}: {conf}")

    print()
    if missing and not args.allow_missing:
        print("ERROR: not all PDF jobs resolved. Use --allow-missing to run the resolved subset.")
        return 2

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_dir = run_root / timestamp
    results: list[dict[str, object]] = []

    for job in jobs:
        conf = resolved.get(job.job_id)
        if conf is None:
            results.append({
                "job_id": job.job_id,
                "table_label": job.table_label,
                "attempts": job.attempts,
                "status": "missing_config",
            })
            continue

        for engine in args.engines:
            for replicate in range(1, args.replicates + 1):
                run_dir = batch_dir / job.job_id / engine / f"rep{replicate}"
                label = f"{job.job_id} / {engine} / rep{replicate}"
                if args.dry_run:
                    print(f"DRY RUN {label}: {conf}")
                    continue

                print(f"RUN {label}", flush=True)
                result = run_one(
                    python=python,
                    conf_path=conf,
                    run_dir=run_dir,
                    engine=engine,
                    timeout=args.timeout,
                    parallel_count=parallel_count,
                    log_level=args.log_level,
                )
                result.update({
                    "job_id": job.job_id,
                    "table_label": job.table_label,
                    "attempts": job.attempts,
                    "engine": engine,
                    "engine_label": ENGINES[engine]["label"],
                    "replicate": replicate,
                    "source_config": str(conf),
                })
                results.append(result)
                print(
                    "  {status} in {elapsed_seconds}s -> {run_dir}".format(**result),
                    flush=True,
                )

    if not args.dry_run:
        batch_dir.mkdir(parents=True, exist_ok=True)
        summary_path = batch_dir / "summary.json"
        summary = {
            "created_at": timestamp,
            "jobs_root": str(jobs_root),
            "timeout_seconds": args.timeout,
            "engines": args.engines,
            "replicates": args.replicates,
            "parallel_count": args.parallel_count,
            "missing": missing,
            "results": results,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"Wrote summary: {summary_path}")

    bad_statuses = {"missing_config", "failed", "timeout_no_start"}
    bad = [result for result in results if result.get("status") in bad_statuses]
    if bad and not args.allow_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
