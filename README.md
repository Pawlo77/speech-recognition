# Speech Recognition (KWS)

Minimal research pipeline for keyword spotting on the Kaggle TensorFlow Speech Recognition Challenge (derived from Google Speech Commands v1).

## Scope

- Task: 35-class single-word classification from 1-second audio clips.
- Input: waveform shape `(B, 1, 16000)` at 16 kHz.
- Core focus: feature ablation, architecture comparison, LoRA vs full fine-tuning, and reproducible training.

## Fixed Project Rules

- Official splits are mandatory: `validation_list.txt` and `testing_list.txt`.
- If either split file is missing, fail fast unless `allow_missing_official_splits=true`.
- Mel baseline is locked to `n_mels=128` for mel-based features and shape checks.
- LoRA init is fixed globally: `A` = Kaiming uniform, `B` = zeros.
- Loss policy is config-driven: start with CE, switch one-way to Focal after warmup when configured macro-F1 gap criteria is met.

## Experiment Flow

1. Phase 1: feature extraction ablation.
2. Phase 2: model family comparison.
3. Phase 3: LoRA vs full fine-tuning on the best model.
4. Phase 4: footprint reporting (latency, FLOPs, size).

## Reproducibility Metadata

Each run should persist: `seed`, `python_random_seeded`, `numpy_seeded`, `torch_seeded`, `dataloader_worker_seeding`, `deterministic_flags`, `cudnn_or_mps_determinism_mode`, `python_version`, `dependency_lock_hash`, `git_commit`.

## Quick Start

Create `.env` with your Kaggle API credentials:

```env
KAGGLE_USERNAME=your_username
KAGGLE_API_TOKEN=your_token
```

Run the [data notebook](notebooks/train_split_eda.ipynb) for EDA and to verify data access. Then execute:

```bash
uv sync
make test
make train
make eval
```
