# Speech Recognition (KWS)

Minimal research pipeline for keyword spotting on the Kaggle TensorFlow Speech Recognition Challenge.

## Scope

- Task: 32-class single-word classification over the Kaggle label space.
- Input: 1-second audio clips at 16 kHz, with documented outliers ranging from 0.37 to 95.18 seconds.
- Core focus: feature ablation, architecture comparison, non-command handling, resumable training, and reproducible evaluation.

## Methodology

The repository mirrors the four-phase experimental plan documented in [report/report.tex](report/report.tex):

1. Phase 1: feature strategy ablation on `train_small` and `valid_small`.
2. Phase 2: global hyperparameter tuning.
3. Phase 3: architecture comparison across AST, ConvNeXt, SSAMBA, xLSTM, and MLP-Mixer.
4. Phase 4: non-command handling and final held-out evaluation.

The codebase keeps the training loop explicit and resumable. The trainer targets Apple MPS first and falls back to CPU when MPS is unavailable. Mixed precision stays disabled by default and can be enabled by configuration when the local PyTorch build supports it.

## Dataset Notes

- The official `validation_list.txt` and `testing_list.txt` splits are required.
- `__unknown__` synthesis uses only command clips, blends 2 to 3 source clips, and tracks uniqueness by source-file tuple.
- Feature extraction is implemented as composable `nn.Module`s with mel, high-temporal mel, MFCC, PCEN, and SpecAugment variants.
- Model adapters validate `[B, 1, F, T]` inputs, pad or truncate temporal outliers, and expose profiling via `fvcore`.

## Reproducibility Metadata

Each run persists:

- `seed`
- `python_random_seeded`
- `numpy_seeded`
- `torch_seeded`
- `dataloader_worker_seeding`
- `deterministic_flags`
- `cudnn_or_mps_determinism_mode`
- `python_version`
- `dependency_lock_hash`
- `git_commit`

## Quick Start

Create `.env` with your Kaggle credentials:

```env
KAGGLE_USERNAME=your_username
KAGGLE_API_TOKEN=your_token
```

Then run the main checks and pipeline entry points:

```bash
uv sync
make test
make pre-commit-all
make phase-1
make phase-2
make phase-3
make phase-4
make full-pipeline
```

## MLflow

Local MLflow runs are stored under `.mlruns/` by default. Launch the UI with:

```bash
mlflow ui --backend-store-uri .mlruns
```

For the CLI wrapper, use:

```bash
speech-recognition mlflow-ui
```
