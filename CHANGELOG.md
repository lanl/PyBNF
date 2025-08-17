# Changelog

Notable changes to this project are documented here. Historical entries prior to 2025 use publication dates when exact release dates are not available.

## [1.2.3] — 2025-08-16
### Added
- CLI flag `--roadrunner-version` to report the libRoadRunner version (with clear guidance if missing).
- Unified number formatting helper `printing.format_numbers(seq, prec=3, sci=True)`, now used in algorithm progress output (e.g., Scatter Search, Particle Swarm, Simplex) to avoid `np.float64(...)` in console logs.
- Dask preload hook `pybnf.dask_preload.silence_kbi` auto-injected via `DASK_DISTRIBUTED__WORKER__PRELOAD` and `...__NANNY__PRELOAD` to suppress noisy KeyboardInterrupt stack traces from workers/nannies.
- Safety: the CLI gently traps `Ctrl-Z` (SIGTSTP) to prevent orphaned/zombie workers. Use `Ctrl-C` instead.

### Changed
- Packaging and dependency refresh; validated for Python 3.10–3.13; modernized `pyproject.toml`.
- Refactor: split the monolithic `algorithms.py` into modules (`pybnf/algorithms/optimizers/` and `pybnf/algorithms/samplers/`) for maintainability.
- Quieter aborts: tone down Dask/Tornado chatter during interrupts; reduce verbosity of the “port 8787 in use” HTTP server warning.
- Cleanup ordering: defer deletion of simulation/output folders until after `client.close()`/cluster teardown to avoid `FileNotFoundError`s seen in full tests.

### Fixed
- `NameError: SimplexAlgorithm not defined` by removing the hard dependency from `base.py` (uses class-name check/import-local pattern).
- Console output now shows plain numbers (with optional scientific notation) instead of `np.float64(...)`.
- More robust interruption path: cancel outstanding futures, close client, and teardown cluster cleanly on `Ctrl-C`.

_Attribution:_ W.S. Hlavacek.

## [1.1.9] — 2022-01-05
### Added
- Practical MCMC sampler improvements (implementation details in the 2022 Bioinformatics article).  
_Attribution:_ J. Neumann (lead), Y.-T. Lin, A. Mallela, E.F. Miller, J. Colvin, A.T. Duprat, Y. Chen, W.S. Hlavacek, R.G. Posner.

## [1.1.0] — 2020-02-12
### Added
- Support for **qualitative observations** in parameter inference (integrated with BPSL), as described in the 2020 Bioinformatics article.  
_Attribution:_ E.D. Mitra.

## [1.0.0] — 2019-09-27
### Added
- **Python rewrite** of BioNetFit as **PyBioNetFit**, introducing the **Biological Property Specification Language (BPSL)** and improved parallel metaheuristic optimization for BNGL and SBML workflows.  
_Attribution:_ E.D. Mitra, R. Suderman (leads), J. Colvin, A. Ionkov, A. Hu, H.M. Sauro, R.G. Posner, W.S. Hlavacek.

## [0.x] — 2016
### Added
- Original **BioNetFit** implementation in **Perl**, compatible with BioNetGen/NFsim and distributed environments.  
_Attribution:_ B.R. Thomas (lead), L.A. Chylek, J. Colvin, S. Sirimulla, A.H.A. Clayton, W.S. Hlavacek.

---

## References (reverse chronological order)

- **2022 (Bioinformatics):** *Implementation of a practical Markov chain Monte Carlo sampling algorithm in PyBioNetFit.* J. Neumann, Y.-T. Lin, A. Mallela, E.F. Miller, J. Colvin, A.T. Duprat, Y. Chen, W.S. Hlavacek, R.G. Posner. DOI: 10.1093/bioinformatics/btac004.
- **2020 (Bioinformatics):** *Bayesian inference using qualitative observations of underlying continuous variables.* E.D. Mitra, W.S. Hlavacek. DOI: 10.1093/bioinformatics/btaa084.
- **2019 (iScience):** *PyBioNetFit and the Biological Property Specification Language.* E.D. Mitra, R. Suderman, J. Colvin, A. Ionkov, A. Hu, H.M. Sauro, R.G. Posner, W.S. Hlavacek. DOI: 10.1016/j.isci.2019.08.045.
- **2016 (Bioinformatics):** *BioNetFit: a fitting tool compatible with BioNetGen, NFsim and distributed computing environments.* B.R. Thomas, L.A. Chylek, J. Colvin, S. Sirimulla, A.H.A. Clayton, W.S. Hlavacek. DOI: 10.1093/bioinformatics/btv655.
