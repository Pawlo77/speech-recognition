ifneq ("$(wildcard .env)","")
	include .env
	export
endif

export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=1

.PHONY: help install clean test pre-commit pre-commit-all datasets phase-1 phase-2 phase-3 phase-4 status full-pipeline full-pipeline-check eta-estimate estimate-ram mlflow mlflow-stop

RUN_ROOT ?= outputs/full-pipeline-check
BATCH_SIZES ?= 1 8 32
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5005
MLFLOW_WORKERS ?= 1

############################
# Repo Maintenance Targets #
############################

help:
	@echo "Available targets:"
	@echo "  make help                   - Show this help message"
	@echo "  make install                - Install dependencies and hooks"
	@echo "  make clean                  - Clean virtual environment and lockfile"
	@echo "  make test                   - Run tests"
	@echo "  make pre-commit             - Run pre-commit checks on changed files"
	@echo "  make pre-commit-all         - Run pre-commit checks on all files"
	@echo "  make datasets               - Download/build all dataset variants (default, small, extended)"
	@echo "  make phase-1                - Run phase 1 orchestration"
	@echo "  make phase-2                - Run phase 2 orchestration"
	@echo "  make phase-3                - Run phase 3 orchestration"
	@echo "  make phase-4                - Run phase 4 orchestration"
	@echo "  make status                 - Show current pipeline state"
	@echo "  make full-pipeline          - Run the full pipeline"
	@echo "  make full-pipeline-check    - Run all settings for one seed with few training steps"
	@echo "  make eta-estimate           - Estimate full-pipeline ETA from saved state files"
	@echo "  make estimate-ram           - Estimate RAM footprint by model family"
	@echo "  make mlflow                 - Launch MLflow UI for local runs"
	@echo "  make mlflow-stop            - Stop local MLflow UI processes"

# install dependencies and pre-commit hooks
install:
	uv sync --all-groups
	uv run pre-commit install

# clean up virtual environment and lockfile
clean:
	rm -rf .venv
	rm -rf uv.lock

# run tests with pytest
test::
	uv run pytest -v

# pre-commit checks on changed files only
pre-commit:
	uv run pre-commit run

# pre-commit checks (linting, formatting, type checking)
pre-commit-all:
	uv run pre-commit run --all-files

# Download/build all dataset variants used by the project.
datasets:
	uv run python -c "from speech_recognition import SpeechCommandsDataset; SpeechCommandsDataset(auto_download=True, only_1sec_samples=True, use_smaller_dataset=False, use_extended_dataset=False); SpeechCommandsDataset(auto_download=True, only_1sec_samples=True, use_smaller_dataset=True, use_extended_dataset=False); SpeechCommandsDataset(auto_download=True, only_1sec_samples=True, use_smaller_dataset=False, use_extended_dataset=True)"

#########################
# Orchestration Targets #
#########################

# Execute phase 1
phase-1:
	uv run speech-recognition phase-1 --output-dir outputs --run-name default

# Execute phase 2
phase-2:
	uv run speech-recognition phase-2 --output-dir outputs --run-name default

# Execute phase 3
phase-3:
	uv run speech-recognition phase-3 --output-dir outputs --run-name default

# Execute phase 4
phase-4:
	uv run speech-recognition phase-4 --output-dir outputs --run-name default

# Show pipeline state for the default run namespace
status:
	uv run speech-recognition status --output-dir outputs --run-name default

# Execute the full pipeline
full-pipeline:
	uv run speech-recognition run --output-dir outputs --run-name default

# Execute all orchestration phases with lightweight smoke-test limits.
full-pipeline-check:
	@set -eu; \
	set -o pipefail; \
	mkdir -p outputs report/build_latex; \
	touch outputs/.metadata_never_index report/build_latex/.metadata_never_index; \
	RUN_TS="$$(date +%Y%m%d_%H%M%S)"; \
	RUN_ROOT="outputs/full-pipeline-check"; \
	LOG_DIR="$$RUN_ROOT/logs/$$RUN_TS"; \
	mkdir -p "$$LOG_DIR"; \
	export TQDM_DISABLE=1; \
	echo "Full pipeline check logs: $$LOG_DIR"; \
	phase_1_start="$$(date +%s)"; \
	SPEECH_SWEEP_SEED=0 SPEECH_TRAIN_MAX_STEPS=3 uv run speech-recognition phase-1 --set training.batch_size=4 --set training.log_every_n_steps=1 --output-dir "$$RUN_ROOT" --run-name smoke 2>&1 | tee "$$LOG_DIR/phase-1.log"; \
	phase_1_end="$$(date +%s)"; \
	phase_1_secs="$$((phase_1_end - phase_1_start))"; \
	echo "phase-1 duration: $${phase_1_secs}s"; \
	phase_2_start="$$(date +%s)"; \
	SPEECH_SWEEP_SEED=0 SPEECH_TRAIN_MAX_STEPS=3 uv run speech-recognition phase-2 --set training.batch_size=4 --set training.log_every_n_steps=1 --output-dir "$$RUN_ROOT" --run-name smoke 2>&1 | tee "$$LOG_DIR/phase-2.log"; \
	phase_2_end="$$(date +%s)"; \
	phase_2_secs="$$((phase_2_end - phase_2_start))"; \
	echo "phase-2 duration: $${phase_2_secs}s"; \
	phase_3_start="$$(date +%s)"; \
	SPEECH_SWEEP_SEED=0 SPEECH_TRAIN_MAX_STEPS=3 uv run speech-recognition phase-3 --set training.batch_size=4 --set training.log_every_n_steps=1 --output-dir "$$RUN_ROOT" --run-name smoke 2>&1 | tee "$$LOG_DIR/phase-3.log"; \
	phase_3_end="$$(date +%s)"; \
	phase_3_secs="$$((phase_3_end - phase_3_start))"; \
	echo "phase-3 duration: $${phase_3_secs}s"; \
	phase_4_start="$$(date +%s)"; \
	SPEECH_SWEEP_SEED=0 SPEECH_TRAIN_MAX_STEPS=3 uv run speech-recognition phase-4 --set training.batch_size=4 --set training.log_every_n_steps=1 --output-dir "$$RUN_ROOT" --run-name smoke 2>&1 | tee "$$LOG_DIR/phase-4.log"; \
	phase_4_end="$$(date +%s)"; \
	phase_4_secs="$$((phase_4_end - phase_4_start))"; \
	echo "phase-4 duration: $${phase_4_secs}s"; \
	smoke_total_secs="$$((phase_1_secs + phase_2_secs + phase_3_secs + phase_4_secs))"; \
	estimated_full_secs="$$((phase_1_secs * 30 / 10 + phase_2_secs * 24 / 8 + phase_3_secs * 90 / 30 + phase_4_secs * 45 / 15))"; \
	echo "smoke_total_seconds=$$smoke_total_secs" | tee "$$LOG_DIR/summary.txt"; \
	echo "estimated_full_seconds_by_trial_scaling=$$estimated_full_secs" | tee -a "$$LOG_DIR/summary.txt"; \
	echo "estimated_full_hms=$$((estimated_full_secs / 3600))h $$(((estimated_full_secs % 3600) / 60))m $$((estimated_full_secs % 60))s" | tee -a "$$LOG_DIR/summary.txt"; \
	uv run python scripts/eta_estimate.py --run-root "$$RUN_ROOT" | tee -a "$$LOG_DIR/summary.txt"; \
	echo "Logs saved under $$LOG_DIR"; \
	echo "Note: compare trial-scaled ETA with historical P50/P90 ETA bands above."

#################
# Other Targets #
#################

# Launch MLflow UI for local runs
mlflow:
	@PORT="$(MLFLOW_PORT)"; \
	if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "Port $$PORT is already in use. Run 'make mlflow-stop' or use another port: make mlflow MLFLOW_PORT=5001"; \
		lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN; \
		exit 1; \
	fi; \
	uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --host "$(MLFLOW_HOST)" --port "$$PORT" --workers "$(MLFLOW_WORKERS)"

# Stop local MLflow UI processes
mlflow-stop:
	@PORT="$(MLFLOW_PORT)"; \
	PIDS="$$( ( \
		pgrep -f 'mlflow.server.fastapi_app' || true; \
		pgrep -f 'python -m mlflow' || true; \
		pgrep -f 'mlflow ui' || true; \
		pgrep -f 'mlflow server' || true \
	) | sort -u )"; \
	if [ -z "$$PIDS" ] && command -v lsof >/dev/null 2>&1; then \
		PORT_PIDS="$$(lsof -tiTCP:"$$PORT" -sTCP:LISTEN 2>/dev/null || true)"; \
		if [ -n "$$PORT_PIDS" ]; then \
			for PID in $$PORT_PIDS; do \
				ARGS="$$(ps -p $$PID -o args= 2>/dev/null || true)"; \
				case "$$ARGS" in \
					*mlflow*|*fastapi_app:app*) PIDS="$$PIDS $$PID" ;; \
				esac; \
			done; \
			PIDS="$$(printf '%s\n' $$PIDS | awk 'NF' | sort -u | tr '\n' ' ')"; \
		fi; \
	fi; \
	if [ -n "$$PIDS" ]; then \
		echo "Stopping MLflow UI processes: $$PIDS"; \
		kill $$PIDS; \
	else \
		echo "No MLflow UI process found."; \
		if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN >/dev/null 2>&1; then \
			echo "Note: port $$PORT is occupied by a non-MLflow process:"; \
			lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN; \
		fi; \
	fi

# Estimate full-pipeline ETA from saved phase state files.
eta-estimate:
	uv run python scripts/eta_estimate.py --run-root "$(RUN_ROOT)"

# Estimate rough RAM usage by model family for one or more batch sizes.
estimate-ram:
	uv run python scripts/estimate_ram.py --batch-sizes $(BATCH_SIZES)
