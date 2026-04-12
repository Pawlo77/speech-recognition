import random
import wave
from array import array
from pathlib import Path
from typing import cast

import speech_recognition.dataset as dataset_module
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


def test_dataset_raises_if_missing_and_no_autodownload(tmp_path: Path) -> None:
    """
    Smoke test: class should fail clearly when data is missing and
    auto-download is disabled.
    """
    fake_repo_root = tmp_path / "missing_repo_root"

    try:
        SpeechCommandsDataset(repo_root=fake_repo_root, auto_download=False)
    except FileNotFoundError as exc:
        msg = str(exc)
        assert "Dataset not found" in msg
        expected_path = (
            fake_repo_root / "../data/kaggle_speech_commands" / SpeechCommandsDataset.KAGGLE_SLUG
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


def test_create_unknown_label_samples_does_not_reuse_source_pair(
    tmp_path: Path, monkeypatch
) -> None:
    """Unknown sample generation should use multiple distinct source clips."""
    ds = object.__new__(SpeechCommandsDataset)
    ds.dataset_root = tmp_path / SpeechCommandsDataset.KAGGLE_SLUG
    ds.dataset_root.mkdir(parents=True)
    ds.seed = 123
    ds.unknown_label_samples_size = 1

    audio_root = ds.dataset_root / "train" / "audio"
    class_one = audio_root / "yes"
    class_two = audio_root / "no"
    class_three = audio_root / "up"
    class_one.mkdir(parents=True)
    class_two.mkdir(parents=True)
    class_three.mkdir(parents=True)

    def write_wav(path: Path) -> None:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            samples = array(
                "h",
                [int(1200 * ((index % 32) - 16) / 16) for index in range(16000)],
            )
            wav_file.writeframes(samples.tobytes())

    file_a = class_one / "a.wav"
    file_b = class_one / "b.wav"
    file_c = class_two / "c.wav"
    file_d = class_two / "d.wav"
    file_e = class_three / "e.wav"
    for path in (file_a, file_b, file_c, file_d, file_e):
        write_wav(path)

    label_plan = ["yes", "no", "up"]
    file_plan = {
        "yes": [file_a],
        "no": [file_c],
        "up": [file_e],
    }

    original_random_class = random.Random

    class FakeRandom:
        def __init__(self, seed: int) -> None:
            self._rng = original_random_class(seed)

        def sample(self, population, k: int):
            if population and isinstance(population[0], str):
                return label_plan[:k]
            if population and isinstance(population[0], Path):
                return [file_plan[population[0].parent.name][0]]
            return population[:k]

        def random(self) -> float:
            return 0.0

        def uniform(self, a: float, b: float) -> float:
            return self._rng.uniform(a, b)

        def randint(self, a: int, b: int) -> int:
            return self._rng.randint(a, b)

    opened_pairs: list[tuple[Path, str]] = []
    original_wave_open = dataset_module.wave.open

    def tracking_wave_open(file, mode="rb", *args, **kwargs):
        if mode == "rb":
            opened_pairs.append((Path(file), mode))
        return original_wave_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(dataset_module.random, "Random", FakeRandom)
    monkeypatch.setattr(dataset_module.wave, "open", tracking_wave_open)
    monkeypatch.setattr(
        ds,
        "_get_unknown_source_profile",
        lambda path: {
            file_a.as_posix(): ("female", 0.20),
            file_c.as_posix(): ("female", 0.52),
            file_e.as_posix(): ("female", 0.84),
        }[Path(path).as_posix()],
    )

    ds._create_unknown_label_samples()

    unknown_dir = audio_root / SpeechCommandsDataset.UNKNOWN_LABEL
    unknown_files = sorted(unknown_dir.glob("*.wav"))

    assert len(unknown_files) == 1
    assert {path.name for path, mode in opened_pairs if mode == "rb"} == {
        "a.wav",
        "c.wav",
        "e.wav",
    }
    assert {path.parent.name for path, mode in opened_pairs if mode == "rb"} == {
        "yes",
        "no",
        "up",
    }
