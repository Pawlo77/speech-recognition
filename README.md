# Speech Recognition (KWS)

Repository for a reproducible keyword-spotting (KWS) experimentation stack on the Kaggle TensorFlow Speech Recognition Challenge.

## Scope

- Problem framing: 12-class single-word classification (10 commands + `__unknown__` + `__silence__`) on 16 kHz, 1-second clips.
- What this repo provides: dataset preparation, feature extraction modules, model adapters, phase-based orchestration, tracking, and reproducible evaluation artifacts.
- Engineering emphasis: deterministic preprocessing, resumable sweeps, process-isolated trials, and machine-readable run state under `outputs/`.
- Intended use: run controlled comparisons across features, optimization settings, architectures, and non-command handling strategies.

Methodological rationale, statistical interpretation, and research discussion are documented in [report](report/build_latex/report.pdf). The README focuses on repository usage and implementation boundaries.

## Methodological Blueprint

The implementation follows this four-phase execution plan (methodological detail remains in [report](report/build_latex/report.pdf)):

1. Phase 1: feature strategy ablation on `train_small` and `valid_small`.
2. Phase 2: global hyperparameter tuning.
3. Phase 3: architecture comparison across AST, ConvNeXt, SSAMBA, xLSTM, and MLP-Mixer.
4. Phase 4: non-command handling and final held-out test evaluation.

The orchestration layer is intentionally isolated from the training loop. Each sweep trial is executed in a child Python process so that PyTorch and MPS memory are reclaimed by the operating system after every run. State is persisted as JSON under `outputs/phase_{1..4}/runs/`, and rerunning a phase resumes from the next uncompleted trial without redoing completed work.

The repository Makefile exports the child-process environment globally:

```makefile
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=1
```

This ensures all subprocesses inherit the same Python import path, MPS fallback behavior, and OpenMP thread cap.

## Dataset and Synthetic Class Construction

The Kaggle corpus is not perfectly uniform. While most clips are approximately one second long, the dataset also contains short and long anomalies. The project now uses a strict 12-class taxonomy:

- Target commands: `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`.
- `__silence__`: mapped from `_background_noise_`.
- `__unknown__`: every other spoken label merged into one class.

Before merging, preprocessing automatically writes `train/split_lists/unknown_origin_labels.csv`, which stores the mapping from merged unknown samples back to their original labels.

For duration handling, short clips are never dropped. In dataset preprocessing mode, clips shorter than 1 second are zero-padded to exactly 1 second; longer clips are truncated at feature-loading time.

`__unknown__` samples are generated only from command-class audio. The synthesis pipeline is deterministic at the source-file level and follows these steps:

- Select 2 or 3 source clips from distinct command classes.
- Mean-center each clip.
- Normalize each clip by RMS toward a shared target level.
- Apply a delayed component smoothing term to reduce abrupt transients.
- Blend the clips with dense short-frame overlap across the full 1-second window.
- Reject candidates that fail the smoothness gate and regenerate until the target quota is met.

This procedure avoids duplicating the same synthetic sample and keeps the non-command classes structurally separated from the target commands.

For extended splits, unknown candidates are deduplicated by audio-content hash first; if the unknown pool is still too small, synthetic unknowns are generated as a top-up until split targets are met (or generation attempts are exhausted).

## Feature Extraction and Model Adapters

Feature extraction is implemented as composable PyTorch `nn.Module`s backed by `torchaudio`.

- Mel spectrogram baseline: `n_fft=1024`, `hop_length=160`.
- High-temporal-resolution Mel: `n_fft=512`, `hop_length=80`.
- MFCC: 40 coefficients.
- PCEN: per-channel energy normalization.
- Mel + SpecAugment: time and frequency masking.

Model adapters accept `[B, 1, F, T]` tensors, normalize temporal length by padding or truncation, and expose profiling through `fvcore` using `FlopCountAnalysis` and `parameter_count`. The class count is fixed at 12.

Current defaults target a comparable ~25M-parameter regime across AST/SSAMBA/xLSTM while keeping ConvNeXt/MLP-Mixer in a similar band. Measured counts are exported to `outputs/model_parameter_counts.csv`.

## Phase Sweep Protocol

The experiment funnel is designed for idempotent reruns.

- Phase 1 sweeps 30 feature/proxy/seed combinations.
- Phase 2 sweeps 24 global optimization combinations.
- Phase 3 sweeps 90 architecture/seed combinations.
- Phase 4 evaluates 45 held-out configurations derived from the top three Phase 3 backbones across three fixed seeds.

Every phase persists a local state file and a best-selection artifact before the next trial begins. If a subprocess crashes or the machine reboots, rerunning the same Makefile target resumes from the most recent valid state file in the same `--output-dir` and `--run-name` namespace.

Phase 4 winner selection is seed-aggregated by method/backbone configuration, then ranked by mean non-command macro-F1 with leakage and false-trigger tie-breakers.

## Phase 4 Evaluation

Phase 4 freezes the three winning Phase 3 backbones and evaluates five non-command strategies across three fixed seeds:

1. Flat multiclass baseline.
2. Sampling control with target priors `p_unknown=0.25`, `p_silence=0.15`, and `p_cmd=0.06`.
3. Loss reweighting with weighted cross-entropy.
4. Two-stage detector with a binary gate and a command classifier.
5. Shared two-head model.

Selection is constrained by a strict gate: any configuration that reduces core-command macro-F1 by more than 1.0 percentage point relative to its Phase 3 baseline is rejected. The non-command objective is

$$
\mathrm{Macro-F1}_{\mathrm{NC}} = \frac{\mathrm{F1}_{\mathrm{unknown}} + \mathrm{F1}_{\mathrm{silence}}}{2}.
$$

Evaluation also tracks wall-clock inference latency after a 50-iteration warmup and records `inference_latency_ms_mean` alongside the F1 metrics. In Phase 4, data loading additionally applies explicit silence up-weighting and conditionally augments `__unknown__` via blending when unknown support falls below the mean target-command support.

Held-out test evaluation runs for all completed Phase 4 trials and writes per-trial test predictions; method-level ensemble metrics are computed from those prediction artifacts.

## Reproducibility Metadata

Each run persists a reproducibility snapshot and the active phase state. The stored metadata includes the random seed, runtime environment, git commit hash, git branch/dirty state, Python/platform details, installed package versions, and hardware context (RAM and disk capacity).

## Quick Start

Create `.env` with your Kaggle credentials:

```env
KAGGLE_USERNAME=your_username
KAGGLE_API_TOKEN=your_token
```

Run the main checks and pipeline entry points:

```bash
make install
make datasets
make test
make pre-commit-all

# one by one
make phase-1
make phase-2
make phase-3
make phase-4

# or all at once with automatic resumption
make full-pipeline
```

Useful Make targets:

```bash
make status
```

Utility scripts for planning and capacity checks:

```bash
# historical full-pipeline ETA from saved phase state files
make eta-estimate

# rough RAM estimate (params + activations approximation) by model family
make estimate-ram

# override defaults when needed
make eta-estimate RUN_ROOT=outputs/full-pipeline-check
make estimate-ram BATCH_SIZES="1 8 32"
```

## MLflow

Local MLflow runs are stored under `mlruns/` by default. Launch the UI with:

```bash
make mlflow

# use a different port if 5000 is occupied
make mlflow MLFLOW_PORT=5001

# equivalent direct command
mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns

# equivalent CLI entry point
speech-recognition mlflow-ui
```
