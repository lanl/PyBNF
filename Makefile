# Run `make bootstrap` once after cloning. Idempotent — safe to re-run.

.DEFAULT_GOAL := bootstrap

.PHONY: bootstrap

bootstrap:
	uv run --no-sync pre-commit install --hook-type pre-push
	@echo
	@echo "Bootstrap complete. The pre-push hook will run the bngsim test"
	@echo "subset before every \`git push\`. See .pre-commit-config.yaml."
