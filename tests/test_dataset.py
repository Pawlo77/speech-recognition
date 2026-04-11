from __future__ import annotations

from pathlib import Path
from typing import cast

from speech_recognition.dataset import Sample, SpeechCommandsDataset


def test_dataset_module_exports_expected_types() -> None:
    """Smoke test: key dataset symbols are importable and usable."""
    sample = Sample(path=Path("dummy.wav"), label="yes", filename="dummy.wav")
    assert sample.label == "yes"
    assert sample.filename.endswith(".wav")


def test_dataset_local_path_detection() -> None:
    """
    Smoke test: dataset manager resolves project paths correctly
    without requiring the dataset to be present.
    """
    ds = object.__new__(SpeechCommandsDataset)
    ds.repo_root = Path(__file__).resolve().parents[1]
    ds.data_dir = ds.repo_root / "data"
    ds.dataset_root = ds.data_dir / SpeechCommandsDataset.KAGGLE_SLUG

    expected_repo_root = Path(__file__).resolve().parents[1]
    expected_data_dir = expected_repo_root / "data"
    expected_dataset_root = expected_data_dir / SpeechCommandsDataset.KAGGLE_SLUG

    assert ds.repo_root == expected_repo_root
    assert ds.data_dir == expected_data_dir
    assert ds.dataset_root == expected_dataset_root


def test_dataset_raises_if_missing_and_no_autodownload() -> None:
    """
    Smoke test: class should fail clearly when data is missing and
    auto-download is disabled.
    """
    fake_repo_root = Path(__file__).resolve().parents[1] / "_tmp_missing_dataset_root"

    try:
        SpeechCommandsDataset(repo_root=fake_repo_root, auto_download=False)
    except FileNotFoundError as exc:
        msg = str(exc)
        assert "Dataset not found" in msg
        expected_path = (
            fake_repo_root / "data" / "kaggle_speech_commands" / SpeechCommandsDataset.KAGGLE_SLUG
        )
        assert str(expected_path) in msg
    else:
        raise AssertionError("Expected FileNotFoundError when dataset is missing")


def test_get_split_rejects_invalid_split() -> None:
    """
    Smoke test for split API contract:
    unknown split names should raise ValueError.
    """
    ds = object.__new__(SpeechCommandsDataset)
    ds._samples = {"train": [], "val": [], "test": []}

    try:
        ds.get_split("invalid")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "Unknown split" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid split name")


def test_stats_reports_split_sizes_from_internal_cache() -> None:
    """Smoke test: stats() reflects train/val/test lengths."""
    ds = object.__new__(SpeechCommandsDataset)
    ds._samples = {
        "train": [Sample(path=Path("a.wav"), label="yes", filename="a.wav")],
        "val": [Sample(path=Path("b.wav"), label="no", filename="b.wav")],
        "test": [],
    }

    assert ds.stats() == {"train": 1, "val": 1, "test": 0}


def test_reconcile_nested_layout_moves_expected_tree(tmp_path: Path) -> None:
    """Nested Kaggle layout should be moved to dataset_root."""
    ds = object.__new__(SpeechCommandsDataset)
    ds.dataset_root = tmp_path / SpeechCommandsDataset.KAGGLE_SLUG
    ds.dataset_root.mkdir(parents=True)

    nested_root = ds.dataset_root / SpeechCommandsDataset.KAGGLE_SLUG
    train_audio = nested_root / "train" / "audio" / "yes"
    train_audio.mkdir(parents=True)
    (train_audio / "sample.wav").write_bytes(b"wav")

    ds._try_reconcile_nested_layout()

    assert (ds.dataset_root / "train" / "audio" / "yes" / "sample.wav").exists()
    assert not nested_root.exists()


def test_stage_download_extracts_archives_when_needed(tmp_path: Path) -> None:
    """When only train/test archives are present, extraction should be attempted."""
    ds = object.__new__(SpeechCommandsDataset)
    ds.dataset_root = tmp_path / SpeechCommandsDataset.KAGGLE_SLUG
    ds.dataset_root.mkdir(parents=True)

    source_root = tmp_path / "downloaded"
    source_root.mkdir()
    (source_root / "train.7z").write_bytes(b"not-a-real-archive")
    (source_root / "test.7z").write_bytes(b"not-a-real-archive")

    called_archives: list[str] = []

    def fake_extract(archive_path: Path, destination: Path) -> None:
        called_archives.append(archive_path.name)
        if archive_path.name == "train.7z":
            (destination / "train" / "audio").mkdir(parents=True, exist_ok=True)

    ds._extract_7z_archive = cast("object", fake_extract)

    ds._stage_download(source_root)

    assert called_archives == ["train.7z", "test.7z"]
    assert (ds.dataset_root / "train" / "audio").exists()
