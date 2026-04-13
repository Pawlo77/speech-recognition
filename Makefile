ifneq ("$(wildcard .env)","")
	include .env
	export
endif

export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=1

.PHONY: help install clean test pre-commit pre-commit-all datasets phase-1 phase-2 phase-3 phase-4 full-pipeline mlflow

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
	@echo "  make full-pipeline          - Run the full pipeline"
	@echo "  make mlflow                 - Launch MLflow UI for local runs"

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

# Execute the full pipeline
full-pipeline:
	uv run speech-recognition run --output-dir outputs --run-name default

#################
# Other Targets #
#################

# Launch MLflow UI for local runs
mlflow:
	uv run mlflow ui --backend-store-uri mlruns
