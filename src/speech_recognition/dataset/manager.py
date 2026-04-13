"""Dataset utilities for Kaggle speech recognition challenge."""

import csv
import hashlib
import importlib
import logging
import random
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import kagglehub
from dotenv import load_dotenv

from .unknown import UnknownSampleGenerationMixin

load_dotenv()

logger = logging.getLogger(__name__)

type Split = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class Sample:
    """Single audio sample descriptor."""

    path: Path
    label: str
    filename: str


class SpeechCommandsDataset(UnknownSampleGenerationMixin):
    """Manage local/downloaded Kaggle speech dataset and splits."""

    KAGGLE_SLUG: Final[str] = "tensorflow-speech-recognition-challenge"
    """Kaggle competition slug."""

    EXPECTED_MAIN_DIR: Final[str] = "train/audio"
    """Required relative directory for availability check."""

    VAL_LIST_FILE: Final[str] = "train/split_lists/validation_list.txt"
    """Relative path to official validation split list."""

    TEST_LIST_FILE: Final[str] = "train/split_lists/testing_list.txt"
    """Relative path to official testing split list."""

    TRAIN_LIST_FILE: Final[str] = "train/split_lists/training_list.txt"
    """Relative path to generated training split list."""

    SMALL_VAL_LIST_FILE: Final[str] = "train/split_lists/small_validation_list.txt"
    """Relative path to generated validation split list for smaller dataset mode."""

    SMALL_TEST_LIST_FILE: Final[str] = "train/split_lists/small_testing_list.txt"
    """Relative path to generated testing split list for smaller dataset mode."""

    SMALL_TRAIN_LIST_FILE: Final[str] = "train/split_lists/small_training_list.txt"
    """Relative path to generated training split list for smaller dataset mode."""

    EXTENDED_TRAIN_LIST_FILE: Final[str] = "train/split_lists/extended_training_list.txt"
    """Relative path to generated training split list for extended dataset mode."""

    EXTENDED_VAL_LIST_FILE: Final[str] = "train/split_lists/extended_validation_list.txt"
    """Relative path to generated validation split list for extended dataset mode."""

    EXTENDED_TEST_LIST_FILE: Final[str] = "train/split_lists/extended_testing_list.txt"
    """Relative path to generated testing split list for extended dataset mode."""

    TRAIN_LABELS_CSV: Final[str] = "train.csv"
    """Relative path to optional filename-label mapping CSV."""

    TEST_AUDIO_DIR: Final[str] = "test/audio"
    """Relative path to competition test audio."""

    UNKNOWN_LABEL: Final[str] = "__unknown__"
    """Fallback label for unlabeled test items."""

    BACKGROUND_NOISE_LABEL: Final[str] = "_background_noise_"
    """Label name for background noise directory."""

    SILENCE_LABEL: Final[str] = "__silence__"
    """Canonical silence label used by training/evaluation code."""

    PADDED_AUDIO_DIR: Final[str] = "__padded_1sec__"
    """Internal directory used for generated padded clips."""

    TARGET_COMMAND_LABELS: Final[tuple[str, ...]] = (
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    )
    """Command labels kept as dedicated classes in the 12-class setup."""

    UNKNOWN_ORIGIN_CSV: Final[str] = "train/split_lists/unknown_origin_labels.csv"
    """CSV mapping merged unknown samples back to their original labels."""

    SMALL_TRAIN_PER_CLASS: Final[int] = 1000
    SMALL_VAL_PER_CLASS: Final[int] = 250
    SMALL_TEST_PER_CLASS: Final[int] = 250
    MAX_EXTENDED_UNKNOWN_PER_SPLIT: Final[int] = 20000

    SMALLER_DATASET_LABELS: Final[set[str]] = set(TARGET_COMMAND_LABELS)

    def __init__(
        self,
        repo_root: str | Path | None = None,
        data_dir_name: str = "../data/kaggle_speech_commands",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        unknown_label_samples_size: int = 10000,
        seed: int = 42,
        auto_download: bool = True,
        only_1sec_samples: bool = True,
        use_smaller_dataset: bool = False,
        use_extended_dataset: bool = False,  # unknown and background noise
    ) -> None:
        """Initialize dataset manager."""
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

        if use_smaller_dataset and use_extended_dataset:
            raise ValueError("Cannot use both smaller and extended dataset modes simultaneously.")

        self.repo_root: Path = Path(repo_root).resolve() if repo_root else self._infer_repo_root()
        self.data_dir: Path = self.repo_root / data_dir_name
        self.dataset_root: Path = self.data_dir / self.KAGGLE_SLUG

        self.val_ratio: float = val_ratio
        self.test_ratio: float = test_ratio
        self.seed: int = seed
        self.only_1sec_samples: bool = only_1sec_samples
        self.unknown_label_samples_size: int = unknown_label_samples_size
        self.use_smaller_dataset: bool = use_smaller_dataset
        self.use_extended_dataset: bool = use_extended_dataset
        self._unknown_source_profile_cache: dict[str, tuple[str, float]] = {}

        logger.debug(
            (
                "Initializing SpeechCommandsDataset with repo_root=%s, data_dir=%s, "
                "dataset_root=%s, val_ratio=%s, test_ratio=%s, seed=%s, auto_download=%s, "
                "only_1sec_samples=%s, use_smaller_dataset=%s, use_extended_dataset=%s"
            ),
            self.repo_root,
            self.data_dir,
            self.dataset_root,
            self.val_ratio,
            self.test_ratio,
            self.seed,
            auto_download,
            self.only_1sec_samples,
            self.use_smaller_dataset,
            self.use_extended_dataset,
        )

        self.data_dir.mkdir(parents=True, exist_ok=True)

        if not self.is_available():
            if auto_download:
                self.download()
            else:
                raise FileNotFoundError(
                    f"Dataset not found at '{self.dataset_root}'. "
                    "Set auto_download=True or call download()."
                )

        if not (
            self.dataset_root / "train" / "audio" / "_background_noise_" / "_background_noise_long"
        ).exists():
            self._split_background_noise_samples()

        self._write_unknown_origin_map()

        if (
            not self.use_smaller_dataset and self.use_extended_dataset
            # and not (self.dataset_root / "train" / "audio" / self.UNKNOWN_LABEL).exists()
        ):
            self._create_unknown_label_samples()

        self._regenerate_split_lists()

        self._samples: dict[Split, list[Sample]] = self._build_splits()
        logger.info("Dataset initialized with split sizes: %s", self.stats())

    def is_available(self) -> bool:
        """Return whether expected dataset layout exists."""
        return (self.dataset_root / self.EXPECTED_MAIN_DIR).exists()

    def download(self, force: bool = False) -> Path:
        """Download and stage dataset locally."""
        logger.info("Preparing dataset download into '%s' (force=%s)", self.data_dir, force)
        if force and self.dataset_root.exists():
            logger.info(
                "Removing existing dataset root due to force=True: %s",
                self.dataset_root,
            )
            shutil.rmtree(self.dataset_root)

        if self.is_available():
            logger.info(
                "Dataset already available at '%s'. Skipping download.",
                self.dataset_root,
            )
            return self.dataset_root

        try:
            downloaded_root = Path(
                kagglehub.competition_download(self.KAGGLE_SLUG, output_dir=self.data_dir)
            ).resolve()
            logger.info("Kaggle download completed. Downloaded root: %s", downloaded_root)
        except FileExistsError:
            logger.info(
                "Dataset already downloaded by another process. Attempting to use existing files."
            )
            downloaded_root = self.data_dir

        self._stage_download(downloaded_root)
        logger.info("Dataset staged successfully at '%s'", self.dataset_root)
        return self.dataset_root

    def get_split(self, split: Split) -> list[Sample]:
        """Return split samples."""
        if split not in self._samples:
            raise ValueError(f"Unknown split '{split}'. Expected one of: train, val, test.")
        return self._samples[split]

    @property
    def train(self) -> list[Sample]:
        """Return train split."""
        return self.get_split("train")

    @property
    def val(self) -> list[Sample]:
        """Return validation split."""
        return self.get_split("val")

    @property
    def test(self) -> list[Sample]:
        """Return test split."""
        return self.get_split("test")

    def stats(self) -> dict[str, int]:
        """Return split sizes."""
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def _is_1sec_sample(self, sample: Sample) -> bool:
        """Check if sample is exactly 1 second (16,000 frames at 16 kHz)."""
        try:
            with wave.open(str(sample.path), "rb") as wav_file:
                n_frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                # 1 second at 16 kHz = 16,000 frames
                return n_frames == 16000 and sample_rate == 16000
        except (OSError, wave.Error) as exc:
            logger.warning("Failed to read WAV duration for '%s': %s", sample.path, exc)
            return False

    def _duration_in_frames(self, sample: Sample) -> int | None:
        """Return number of audio frames for a sample, or None on read failure."""

        try:
            with wave.open(str(sample.path), "rb") as wav_file:
                return wav_file.getnframes()
        except (OSError, wave.Error) as exc:
            logger.warning("Failed to read WAV frame count for '%s': %s", sample.path, exc)
            return None

    def _is_shorter_than_1sec(self, sample: Sample) -> bool:
        """Check whether a sample is shorter than one second at 16 kHz."""

        frame_count = self._duration_in_frames(sample)
        return frame_count is not None and frame_count < 16000

    def _pad_sample_to_1sec(self, sample: Sample) -> Sample:
        """Create (or reuse) a zero-padded 1-second copy for short clips."""

        train_audio_dir = self.dataset_root / "train" / "audio"
        rel_path = sample.path.relative_to(train_audio_dir)
        padded_dir = train_audio_dir / "__padded_1sec__" / rel_path.parent
        padded_dir.mkdir(parents=True, exist_ok=True)
        padded_path = padded_dir / f"{sample.path.stem}__pad1s.wav"

        if padded_path.exists():
            return Sample(path=padded_path, label=sample.label, filename=padded_path.name)

        with wave.open(str(sample.path), "rb") as source_wav:
            channels = source_wav.getnchannels()
            sample_width = source_wav.getsampwidth()
            sample_rate = source_wav.getframerate()
            frames = source_wav.readframes(source_wav.getnframes())

        if sample_rate != 16000:
            logger.warning(
                "Sample '%s' has sample rate %d (expected 16000). Skipping padding.",
                sample.path,
                sample_rate,
            )
            return sample

        frame_count = len(frames) // max(1, sample_width * channels)
        if frame_count >= 16000:
            return sample

        pad_frames = 16000 - frame_count
        frames += b"\x00" * (pad_frames * sample_width * channels)

        with wave.open(str(padded_path), "wb") as padded_wav:
            padded_wav.setnchannels(channels)
            padded_wav.setsampwidth(sample_width)
            padded_wav.setframerate(sample_rate)
            padded_wav.writeframes(frames)

        return Sample(path=padded_path, label=sample.label, filename=padded_path.name)

    def _canonicalize_label(self, label: str) -> str:
        """Map raw dataset labels into the 12-class taxonomy."""

        if label == self.BACKGROUND_NOISE_LABEL:
            return self.SILENCE_LABEL
        if label in self.TARGET_COMMAND_LABELS:
            return label
        return self.UNKNOWN_LABEL

    def _write_unknown_origin_map(self) -> None:
        """Persist mapping from merged unknown samples to original source labels."""

        train_audio_dir = self.dataset_root / "train" / "audio"
        if not train_audio_dir.exists():
            return

        rows: list[tuple[str, str, str]] = []
        for label_dir in sorted(train_audio_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            if label_dir.name == self.PADDED_AUDIO_DIR:
                continue

            original_label = label_dir.name
            if original_label in (*self.TARGET_COMMAND_LABELS, self.BACKGROUND_NOISE_LABEL):
                continue

            for wav_file in sorted(label_dir.glob("*.wav")):
                rel_path = wav_file.relative_to(train_audio_dir).as_posix()
                rows.append((rel_path, original_label, self.UNKNOWN_LABEL))

        csv_path = self.dataset_root / self.UNKNOWN_ORIGIN_CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
            writer = csv.writer(csv_handle)
            writer.writerow(["relative_path", "original_label", "merged_label"])
            writer.writerows(rows)

        logger.info("Saved unknown origin mapping CSV with %d rows: %s", len(rows), csv_path)

    def _infer_repo_root(self) -> Path:
        """Infer repository root from file location."""
        return Path(__file__).resolve().parents[2]

    def _stage_download(self, downloaded_root: Path) -> None:
        """Copy downloaded files into local dataset directory."""
        logger.info(
            "Staging dataset from '%s' into '%s'",
            downloaded_root,
            self.dataset_root,
        )
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        if downloaded_root.resolve() != self.dataset_root.resolve():
            for item in downloaded_root.iterdir():
                destination = self.dataset_root / item.name
                logger.debug("Staging item '%s' -> '%s'", item, destination)
                if item.is_dir():
                    if destination.exists():
                        shutil.rmtree(destination)
                    shutil.copytree(item, destination)
                else:
                    shutil.copy2(item, destination)
        else:
            logger.debug("Downloaded root equals dataset root; skipping copy step.")

        if not (self.dataset_root / self.EXPECTED_MAIN_DIR).exists():
            logger.info(
                "Expected layout missing after copy. Attempting nested layout reconciliation."
            )
            self._try_reconcile_nested_layout()

        if not (self.dataset_root / self.EXPECTED_MAIN_DIR).exists():
            logger.info("Expected layout still missing. Attempting archive extraction.")
            self._extract_archives_if_present()

        if not (self.dataset_root / self.EXPECTED_MAIN_DIR).exists():
            logger.info(
                "Expected layout still missing after extraction. Retrying nested reconciliation."
            )
            self._try_reconcile_nested_layout()

        if not (self.dataset_root / self.EXPECTED_MAIN_DIR).exists():
            raise FileNotFoundError(
                "Download completed but expected dataset layout was not found. "
                f"Missing '{self.EXPECTED_MAIN_DIR}' under '{self.dataset_root}'."
            )

    def _try_reconcile_nested_layout(self) -> None:
        """Move files from a nested layout root into dataset_root when found."""
        nested_root = self._find_dataset_layout_root(self.dataset_root)
        if not nested_root or nested_root.resolve() == self.dataset_root.resolve():
            logger.debug("No nested dataset layout found to reconcile.")
            return

        logger.info("Reconciling nested dataset layout from '%s'", nested_root)

        for item in nested_root.iterdir():
            destination = self.dataset_root / item.name
            logger.debug("Moving reconciled item '%s' -> '%s'", item, destination)
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), str(destination))

        if nested_root.exists() and nested_root != self.dataset_root:
            shutil.rmtree(nested_root, ignore_errors=True)
            logger.debug("Removed nested root after reconciliation: %s", nested_root)

    def _find_dataset_layout_root(self, search_root: Path) -> Path | None:
        """Locate the directory containing train/audio for this dataset."""
        expected = search_root / self.EXPECTED_MAIN_DIR
        if expected.exists():
            logger.debug("Expected layout already present under '%s'", search_root)
            return search_root

        matches = sorted(
            search_root.rglob(self.EXPECTED_MAIN_DIR),
            key=lambda path: len(path.parts),
        )
        if not matches:
            logger.debug("No '%s' match found under '%s'", self.EXPECTED_MAIN_DIR, search_root)
            return None

        logger.debug(
            "Found candidate nested layout roots: %s",
            [match.parent.parent for match in matches],
        )
        return matches[0].parent.parent

    def _extract_archives_if_present(self) -> None:
        """Extract known Kaggle archives when the download provides compressed files."""
        archive_names = ("train.7z", "test.7z")
        archives = [
            self.dataset_root / name
            for name in archive_names
            if (self.dataset_root / name).exists()
        ]

        if not archives:
            logger.debug("No known .7z archives found in '%s'", self.dataset_root)
            return

        logger.info("Found archives to extract: %s", [archive.name for archive in archives])

        for archive in archives:
            self._extract_7z_archive(archive, self.dataset_root)

    def _extract_7z_archive(self, archive_path: Path, destination: Path) -> None:
        """Extract a .7z archive using system 7z/7zz and then py7zr as fallback."""
        for executable in ("7zz", "7z"):
            try:
                logger.info("Extracting '%s' using '%s'", archive_path.name, executable)
                subprocess.run(  # noqa: S603
                    [executable, "x", "-y", str(archive_path), f"-o{destination}"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info("Extraction complete for '%s'", archive_path.name)
                return
            except FileNotFoundError:
                logger.debug("Extractor '%s' not found on PATH", executable)
                continue
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "Extractor '%s' failed for '%s' (exit=%s). Falling back if possible.",
                    executable,
                    archive_path.name,
                    exc.returncode,
                )
                continue

        try:
            py7zr_module = importlib.import_module("py7zr")
        except ImportError as exc:
            raise RuntimeError(
                "Found .7z archives but no extractor is available. "
                "Install py7zr or ensure 7z/7zz is installed and on PATH."
            ) from exc

        logger.info("Extracting '%s' using py7zr fallback", archive_path.name)
        with py7zr_module.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(path=destination)
        logger.info("Extraction complete for '%s'", archive_path.name)

    def _build_splits(self) -> dict[Split, list[Sample]]:
        """Build train/val/test splits."""
        train_audio = self.dataset_root / "train" / "audio"
        if not train_audio.exists():
            raise FileNotFoundError(f"Missing train audio directory: {train_audio}")

        all_labeled = self._collect_labeled_samples(train_audio)
        if self.use_extended_dataset:
            split_candidates = all_labeled
        else:
            split_candidates = [
                sample for sample in all_labeled if sample.label in self.SMALLER_DATASET_LABELS
            ]

        train_list_file, val_list_file, test_list_file = self._split_list_files()
        train_rel_paths = self._read_rel_paths(train_list_file)
        val_rel_paths = self._read_rel_paths(val_list_file)
        test_rel_paths = self._read_rel_paths(test_list_file)

        if val_rel_paths or test_rel_paths:
            logger.info(
                "Building splits using official lists (training=%d, validation=%d, testing=%d)",
                len(train_rel_paths),
                len(val_rel_paths),
                len(test_rel_paths),
            )
            train_samples, val_samples, test_samples = self._split_with_official_lists(
                all_labeled=split_candidates,
                train_audio_dir=train_audio,
                train_rel_paths=train_rel_paths,
                val_rel_paths=val_rel_paths,
                test_rel_paths=test_rel_paths,
            )
        else:
            logger.info(
                (
                    "Official split lists not found. "
                    "Creating random splits (val_ratio=%s, test_ratio=%s, seed=%s)."
                ),
                self.val_ratio,
                self.test_ratio,
                self.seed,
            )
            train_samples, val_samples, test_samples = self._random_train_val_test_split(
                split_candidates
            )

        if self.only_1sec_samples:
            logger.info(
                "Normalizing split durations for 1-second mode: short clips are zero-padded, "
                "long clips are excluded."
            )

            def _normalize(samples: list[Sample]) -> tuple[list[Sample], int, int]:
                normalized: list[Sample] = []
                padded = 0
                dropped_long = 0
                for sample in samples:
                    if self._is_1sec_sample(sample):
                        normalized.append(sample)
                        continue
                    if self._is_shorter_than_1sec(sample):
                        normalized.append(self._pad_sample_to_1sec(sample))
                        padded += 1
                        continue
                    dropped_long += 1
                return normalized, padded, dropped_long

            train_samples, train_padded, train_dropped = _normalize(train_samples)
            val_samples, val_padded, val_dropped = _normalize(val_samples)
            test_samples, test_padded, test_dropped = _normalize(test_samples)

            logger.info(
                "1-second normalization summary: train padded=%d dropped_long=%d, "
                "val padded=%d dropped_long=%d, test padded=%d dropped_long=%d",
                train_padded,
                train_dropped,
                val_padded,
                val_dropped,
                test_padded,
                test_dropped,
            )

        logger.info(
            "Built splits with counts: train=%d, val=%d, test=%d",
            len(train_samples),
            len(val_samples),
            len(test_samples),
        )
        return {"train": train_samples, "val": val_samples, "test": test_samples}

    def _split_with_official_lists(
        self,
        all_labeled: list[Sample],
        train_audio_dir: Path,
        train_rel_paths: list[str],
        val_rel_paths: list[str],
        test_rel_paths: list[str],
    ) -> tuple[list[Sample], list[Sample], list[Sample]]:
        """Split using official validation/testing lists."""
        train_set = set(train_rel_paths)
        val_set = set(val_rel_paths)
        test_set = set(test_rel_paths)

        train_samples: list[Sample] = []
        val_samples: list[Sample] = []
        test_samples: list[Sample] = []

        for sample in all_labeled:
            rel = sample.path.relative_to(train_audio_dir).as_posix()
            if rel in val_set:
                val_samples.append(sample)
            elif rel in test_set:
                test_samples.append(sample)
            elif rel in train_set:
                train_samples.append(sample)

        return train_samples, val_samples, test_samples

    def _collect_labeled_samples(self, train_audio: Path) -> list[Sample]:
        """Collect labeled samples from train/audio."""
        samples: list[Sample] = []

        for label_dir in sorted(train_audio.iterdir()):
            if not label_dir.is_dir():
                continue
            if label_dir.name == self.PADDED_AUDIO_DIR:
                continue

            label = self._canonicalize_label(label_dir.name)
            for wav_file in sorted(label_dir.glob("*.wav")):
                samples.append(Sample(path=wav_file, label=label, filename=wav_file.name))

        if not samples:
            raise RuntimeError(f"No .wav files found under: {train_audio}")

        logger.debug("Collected %d labeled samples from '%s'", len(samples), train_audio)

        return samples

    def _read_rel_paths(self, file_path: Path) -> list[str]:
        """Read non-empty lines from a split file."""
        if not file_path.exists():
            logger.debug("Split file not found: %s", file_path)
            return []

        with file_path.open(encoding="utf-8") as handle:
            rel_paths = [line.strip() for line in handle if line.strip()]

        logger.debug("Loaded %d entries from split file '%s'", len(rel_paths), file_path)
        return rel_paths

    def _random_train_val_test_split(
        self, samples: list[Sample]
    ) -> tuple[list[Sample], list[Sample], list[Sample]]:
        """Create deterministic random train/val/test split."""
        shuffled = samples[:]
        rng = random.Random(self.seed)  # noqa: S311 - deterministic split only
        rng.shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * self.val_ratio))
        n_test = max(1, int(len(shuffled) * self.test_ratio))
        val_samples = shuffled[:n_val]
        test_samples = shuffled[n_val : n_val + n_test]
        train_samples = shuffled[n_val + n_test :]
        logger.debug(
            "Random split produced train=%d, val=%d, and test=%d samples",
            len(train_samples),
            len(val_samples),
            len(test_samples),
        )

        train_audio_dir = self.dataset_root / "train" / "audio"
        train_rel_paths = sorted(
            sample.path.relative_to(train_audio_dir).as_posix() for sample in train_samples
        )
        val_rel_paths = sorted(
            sample.path.relative_to(train_audio_dir).as_posix() for sample in val_samples
        )
        test_rel_paths = sorted(
            sample.path.relative_to(train_audio_dir).as_posix() for sample in test_samples
        )

        train_list_file, val_list_file, test_list_file = self._split_list_files()
        self._write_rel_paths(train_list_file, train_rel_paths)
        self._write_rel_paths(val_list_file, val_rel_paths)
        self._write_rel_paths(test_list_file, test_rel_paths)

        logger.info(
            "Saved random split path lists: training=%d, validation=%d, testing=%d",
            len(train_rel_paths),
            len(val_rel_paths),
            len(test_rel_paths),
        )

        return train_samples, val_samples, test_samples

    def _write_rel_paths(self, file_path: Path, rel_paths: list[str]) -> None:
        """Write split relative paths to txt file, one path per line."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as handle:
            if rel_paths:
                handle.write("\n".join(rel_paths))
                handle.write("\n")

    def _split_list_files(self) -> tuple[Path, Path, Path]:
        """Return (training_list_path, validation_list_path, testing_list_path) for current mode."""
        if self.use_extended_dataset:
            return (
                self.dataset_root / self.EXTENDED_TRAIN_LIST_FILE,
                self.dataset_root / self.EXTENDED_VAL_LIST_FILE,
                self.dataset_root / self.EXTENDED_TEST_LIST_FILE,
            )
        if self.use_smaller_dataset:
            return (
                self.dataset_root / self.SMALL_TRAIN_LIST_FILE,
                self.dataset_root / self.SMALL_VAL_LIST_FILE,
                self.dataset_root / self.SMALL_TEST_LIST_FILE,
            )
        return (
            self.dataset_root / self.TRAIN_LIST_FILE,
            self.dataset_root / self.VAL_LIST_FILE,
            self.dataset_root / self.TEST_LIST_FILE,
        )

    def _regenerate_split_lists(self) -> None:
        """Generate all split-list variants according to the project split policy."""

        train_audio_dir = self.dataset_root / "train" / "audio"
        if not train_audio_dir.exists():
            raise FileNotFoundError(f"Missing train audio directory: {train_audio_dir}")

        all_rel_paths = sorted(
            wav_path.relative_to(train_audio_dir).as_posix()
            for label_dir in sorted(train_audio_dir.iterdir())
            if label_dir.is_dir() and label_dir.name != self.PADDED_AUDIO_DIR
            for wav_path in sorted(label_dir.glob("*.wav"))
        )

        official_val = set(self._read_rel_paths(self.dataset_root / self.VAL_LIST_FILE))
        official_test = set(self._read_rel_paths(self.dataset_root / self.TEST_LIST_FILE))
        official_train = [
            rel_path
            for rel_path in all_rel_paths
            if rel_path not in official_val and rel_path not in official_test
        ]

        self._create_full_command_lists(official_train, official_val, official_test)
        self._create_small_command_lists(all_rel_paths)
        self._create_extended_lists(official_train, official_val, official_test)

    def _is_target_rel_path(self, rel_path: str) -> bool:
        label = Path(rel_path).parts[0]
        return label in self.TARGET_COMMAND_LABELS

    def _create_full_command_lists(
        self,
        official_train: list[str],
        official_val: set[str],
        official_test: set[str],
    ) -> None:
        """Create full split lists over all available target-command samples only."""

        train_rel_paths = sorted([rel for rel in official_train if self._is_target_rel_path(rel)])
        val_rel_paths = sorted([rel for rel in official_val if self._is_target_rel_path(rel)])
        test_rel_paths = sorted([rel for rel in official_test if self._is_target_rel_path(rel)])

        self._write_rel_paths(self.dataset_root / self.TRAIN_LIST_FILE, train_rel_paths)
        self._write_rel_paths(self.dataset_root / self.VAL_LIST_FILE, val_rel_paths)
        self._write_rel_paths(self.dataset_root / self.TEST_LIST_FILE, test_rel_paths)

        logger.info(
            "Command-only lists saved: train=%d, val=%d, test=%d",
            len(train_rel_paths),
            len(val_rel_paths),
            len(test_rel_paths),
        )

    def _create_small_command_lists(self, all_rel_paths: list[str]) -> None:
        """Create deterministic small command-only splits with fixed per-class sizes."""

        rng = random.Random(self.seed)  # noqa: S311 - deterministic subset only
        train_rel_paths: list[str] = []
        val_rel_paths: list[str] = []
        test_rel_paths: list[str] = []

        required = self.SMALL_TRAIN_PER_CLASS + self.SMALL_VAL_PER_CLASS + self.SMALL_TEST_PER_CLASS
        per_label: dict[str, list[str]] = {label: [] for label in self.TARGET_COMMAND_LABELS}
        for rel_path in all_rel_paths:
            label = Path(rel_path).parts[0]
            if label in per_label:
                per_label[label].append(rel_path)

        for label in self.TARGET_COMMAND_LABELS:
            label_samples = per_label[label]
            if len(label_samples) < required:
                raise RuntimeError(
                    f"Not enough samples for label '{label}' to build small split "
                    f"(required={required}, available={len(label_samples)})."
                )
            shuffled = label_samples[:]
            rng.shuffle(shuffled)
            train_rel_paths.extend(shuffled[: self.SMALL_TRAIN_PER_CLASS])
            val_start = self.SMALL_TRAIN_PER_CLASS
            val_end = val_start + self.SMALL_VAL_PER_CLASS
            test_end = val_end + self.SMALL_TEST_PER_CLASS
            val_rel_paths.extend(shuffled[val_start:val_end])
            test_rel_paths.extend(shuffled[val_end:test_end])

        self._write_rel_paths(
            self.dataset_root / self.SMALL_TRAIN_LIST_FILE,
            sorted(train_rel_paths),
        )
        self._write_rel_paths(self.dataset_root / self.SMALL_VAL_LIST_FILE, sorted(val_rel_paths))
        self._write_rel_paths(self.dataset_root / self.SMALL_TEST_LIST_FILE, sorted(test_rel_paths))

        logger.info(
            "Small command-only lists saved with fixed per-class sizes: train=%d, val=%d, test=%d",
            len(train_rel_paths),
            len(val_rel_paths),
            len(test_rel_paths),
        )

    def _canonical_label_from_rel_path(self, rel_path: str) -> str:
        return self._canonicalize_label(Path(rel_path).parts[0])

    def _unknown_dedup_key(self, rel_path: str, train_audio_dir: Path) -> str:
        """Return dedup key for unknown sample using content hash."""

        path = train_audio_dir / rel_path
        return hashlib.sha1(path.read_bytes()).hexdigest()  # noqa: S324 - non-security hashing

    def _balanced_extended_split(
        self,
        rel_paths: list[str],
        train_audio_dir: Path,
        rng: random.Random,
    ) -> list[str]:
        """Balance one extended split: keep all commands/silence, cap and dedupe unknown."""

        command_paths = [
            rel
            for rel in rel_paths
            if self._canonical_label_from_rel_path(rel) in self.TARGET_COMMAND_LABELS
        ]
        silence_paths = [
            rel
            for rel in rel_paths
            if self._canonical_label_from_rel_path(rel) == self.SILENCE_LABEL
        ]
        unknown_paths = [
            rel
            for rel in rel_paths
            if self._canonical_label_from_rel_path(rel) == self.UNKNOWN_LABEL
        ]

        seen: set[str] = set()
        dedup_unknown: list[str] = []
        for rel_path in unknown_paths:
            key = self._unknown_dedup_key(rel_path, train_audio_dir)
            if key in seen:
                continue
            seen.add(key)
            dedup_unknown.append(rel_path)

        target_unknown = min(len(command_paths), self.MAX_EXTENDED_UNKNOWN_PER_SPLIT)
        if len(dedup_unknown) > target_unknown:
            shuffled = dedup_unknown[:]
            rng.shuffle(shuffled)
            dedup_unknown = shuffled[:target_unknown]

        return sorted(command_paths + dedup_unknown + silence_paths)

    def _create_extended_lists(
        self,
        official_train: list[str],
        official_val: set[str],
        official_test: set[str],
    ) -> None:
        """Create balanced extended split lists with all 12 classes present."""

        train_audio_dir = self.dataset_root / "train" / "audio"
        rng = random.Random(self.seed)  # noqa: S311 - deterministic balancing only

        command_train = sorted([rel for rel in official_train if self._is_target_rel_path(rel)])
        command_val = sorted([rel for rel in official_val if self._is_target_rel_path(rel)])
        command_test = sorted([rel for rel in official_test if self._is_target_rel_path(rel)])

        unknown_pool = [
            rel
            for rel in sorted(official_train + sorted(official_val) + sorted(official_test))
            if self._canonical_label_from_rel_path(rel) == self.UNKNOWN_LABEL
        ]
        seen_unknown: set[str] = set()
        dedup_unknown_pool: list[str] = []
        for rel in unknown_pool:
            key = self._unknown_dedup_key(rel, train_audio_dir)
            if key in seen_unknown:
                continue
            seen_unknown.add(key)
            dedup_unknown_pool.append(rel)
        rng.shuffle(dedup_unknown_pool)

        unknown_train_target = min(len(command_train), self.MAX_EXTENDED_UNKNOWN_PER_SPLIT)
        unknown_val_target = len(command_val)
        unknown_test_target = len(command_test)
        required_unknown = unknown_train_target + unknown_val_target + unknown_test_target
        if len(dedup_unknown_pool) < required_unknown:
            raise RuntimeError(
                "Not enough deduplicated unknown samples to build balanced extended splits "
                f"(required={required_unknown}, available={len(dedup_unknown_pool)})."
            )

        unknown_train = dedup_unknown_pool[:unknown_train_target]
        offset = unknown_train_target
        unknown_val = dedup_unknown_pool[offset : offset + unknown_val_target]
        offset += unknown_val_target
        unknown_test = dedup_unknown_pool[offset : offset + unknown_test_target]

        silence_pool = [
            rel
            for rel in sorted(official_train + sorted(official_val) + sorted(official_test))
            if self._canonical_label_from_rel_path(rel) == self.SILENCE_LABEL
        ]
        rng.shuffle(silence_pool)
        total_commands = max(1, len(command_train) + len(command_val) + len(command_test))
        silence_train_target = round(len(silence_pool) * len(command_train) / total_commands)
        silence_val_target = round(len(silence_pool) * len(command_val) / total_commands)
        silence_test_target = len(silence_pool) - silence_train_target - silence_val_target
        if silence_pool:
            silence_train_target = max(1, silence_train_target)
            silence_val_target = max(1, silence_val_target)
            silence_test_target = max(1, silence_test_target)
            overflow = (
                silence_train_target + silence_val_target + silence_test_target - len(silence_pool)
            )
            if overflow > 0:
                silence_train_target = max(1, silence_train_target - overflow)

        silence_train = silence_pool[:silence_train_target]
        silence_val = silence_pool[silence_train_target : silence_train_target + silence_val_target]
        silence_test = silence_pool[silence_train_target + silence_val_target :]

        train_rel_paths = sorted(command_train + unknown_train + silence_train)
        val_rel_paths = sorted(command_val + unknown_val + silence_val)
        test_rel_paths = sorted(command_test + unknown_test + silence_test)

        self._write_rel_paths(self.dataset_root / self.EXTENDED_TRAIN_LIST_FILE, train_rel_paths)
        self._write_rel_paths(self.dataset_root / self.EXTENDED_VAL_LIST_FILE, val_rel_paths)
        self._write_rel_paths(self.dataset_root / self.EXTENDED_TEST_LIST_FILE, test_rel_paths)

        logger.info(
            "Extended balanced lists saved: train=%d, val=%d, test=%d",
            len(train_rel_paths),
            len(val_rel_paths),
            len(test_rel_paths),
        )

    def _collect_competition_test_samples(self) -> list[Sample]:
        """Collect competition test samples."""
        test_audio = self.dataset_root / self.TEST_AUDIO_DIR
        labels_map = self._read_test_labels_map()

        if not test_audio.exists():
            logger.info("Competition test audio directory not found: %s", test_audio)
            return []

        samples: list[Sample] = []
        for wav_file in sorted(test_audio.glob("*.wav")):
            label = labels_map.get(wav_file.name, self.UNKNOWN_LABEL)
            samples.append(Sample(path=wav_file, label=label, filename=wav_file.name))

        logger.debug(
            "Collected %d competition test samples (%d labeled via CSV)",
            len(samples),
            len(labels_map),
        )

        return samples

    def _read_test_labels_map(self) -> dict[str, str]:
        """Read optional filename-label mapping from train.csv."""
        labels_csv = self.dataset_root / self.TRAIN_LABELS_CSV
        if not labels_csv.exists():
            logger.debug("Optional labels CSV not found: %s", labels_csv)
            return {}

        mapping: dict[str, str] = {}
        with labels_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)

            fieldnames = reader.fieldnames or []
            if "fname" not in fieldnames or "label" not in fieldnames:
                logger.warning(
                    "Labels CSV is missing expected columns 'fname' and 'label': %s",
                    labels_csv,
                )
                return {}

            for row in reader:
                fname = row.get("fname")
                label = row.get("label")
                if fname and label:
                    mapping[fname] = label

            logger.debug("Loaded %d label mappings from '%s'", len(mapping), labels_csv)

        return mapping

    def _split_background_noise_samples(self) -> list[Sample]:
        """Split background-noise files into 1-second segments.

        Original long files are moved to a separate directory.
        """

        logger.info(
            "Splitting background noise samples into 1-second segments and "
            "moving originals to a separate directory."
        )

        noise_dir = self.dataset_root / "train" / "audio" / "_background_noise_"
        long_noise_dir = noise_dir / "_background_noise_long"

        long_noise_dir.mkdir(parents=True, exist_ok=True)

        for noise_file in noise_dir.glob("*.wav"):
            with wave.open(str(noise_file), "rb") as wav:
                n_channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                frame_rate = wav.getframerate()
                n_frames = wav.getnframes()

                if frame_rate != 16000:
                    logger.warning(
                        "Background noise file '%s' has unexpected sample rate %d. Skipping.",
                        noise_file,
                        frame_rate,
                    )
                    continue

                if n_frames <= 16000:
                    logger.debug(
                        "Background noise file '%s' is already 1 second or shorter. "
                        "Skipping splitting.",
                        noise_file,
                    )
                    continue

                frames_per_segment = 16000  # 1 second segments
                n_segments = n_frames // frames_per_segment

                for i in range(n_segments):
                    segment_frames = wav.readframes(frames_per_segment)
                    segment_path = noise_file.parent / f"{noise_file.stem}_segment_{i}.wav"
                    with wave.open(str(segment_path), "wb") as segment_wav:
                        segment_wav.setnchannels(n_channels)
                        segment_wav.setsampwidth(sample_width)
                        segment_wav.setframerate(frame_rate)
                        segment_wav.writeframes(segment_frames)

            shutil.move(str(noise_file), str(long_noise_dir / noise_file.name))

    def _create_minimal_dataset(self) -> None:
        """Create minimal split lists without modifying dataset directories."""
        train_audio_dir = self.dataset_root / "train" / "audio"
        if not train_audio_dir.exists():
            raise FileNotFoundError(f"Missing train audio directory: {train_audio_dir}")

        logger.info(
            "Creating minimal dataset with labels: %s",
            sorted(self.SMALLER_DATASET_LABELS),
        )

        selected_samples = [
            sample
            for sample in self._collect_labeled_samples(train_audio_dir)
            if sample.label in self.SMALLER_DATASET_LABELS
        ]
        if not selected_samples:
            raise RuntimeError(
                "Minimal dataset is empty after filtering to SMALLER_DATASET_LABELS."
            )

        train_samples, val_samples, test_samples = self._stratified_split_samples(selected_samples)

        train_rel_paths = sorted(
            sample.path.relative_to(train_audio_dir).as_posix() for sample in train_samples
        )
        val_rel_paths = sorted(
            sample.path.relative_to(train_audio_dir).as_posix() for sample in val_samples
        )
        test_rel_paths = sorted(
            sample.path.relative_to(train_audio_dir).as_posix() for sample in test_samples
        )

        self._write_rel_paths(self.dataset_root / self.SMALL_TRAIN_LIST_FILE, train_rel_paths)
        self._write_rel_paths(self.dataset_root / self.SMALL_VAL_LIST_FILE, val_rel_paths)
        self._write_rel_paths(self.dataset_root / self.SMALL_TEST_LIST_FILE, test_rel_paths)

        logger.info(
            "Minimal dataset lists saved: train=%d, val=%d, test=%d",
            len(train_rel_paths),
            len(val_rel_paths),
            len(test_rel_paths),
        )

    def _create_extended_dataset(self) -> None:
        """Create extended split lists that include unknown and background noise."""
        train_audio_dir = self.dataset_root / "train" / "audio"
        if not train_audio_dir.exists():
            raise FileNotFoundError(f"Missing train audio directory: {train_audio_dir}")

        extended_samples = self._collect_labeled_samples(train_audio_dir)
        if not extended_samples:
            raise RuntimeError("Extended dataset is empty; no samples were found.")

        train_samples, val_samples, test_samples = self._stratified_split_samples(extended_samples)

        train_rel_paths = sorted(
            sample.path.relative_to(train_audio_dir).as_posix() for sample in train_samples
        )
        val_rel_paths = sorted(
            sample.path.relative_to(train_audio_dir).as_posix() for sample in val_samples
        )
        test_rel_paths = sorted(
            sample.path.relative_to(train_audio_dir).as_posix() for sample in test_samples
        )

        self._write_rel_paths(
            self.dataset_root / self.EXTENDED_TRAIN_LIST_FILE,
            train_rel_paths,
        )
        self._write_rel_paths(
            self.dataset_root / self.EXTENDED_VAL_LIST_FILE,
            val_rel_paths,
        )
        self._write_rel_paths(
            self.dataset_root / self.EXTENDED_TEST_LIST_FILE,
            test_rel_paths,
        )

        logger.info(
            "Extended dataset lists saved: train=%d, val=%d, test=%d",
            len(train_rel_paths),
            len(val_rel_paths),
            len(test_rel_paths),
        )

    def _stratified_split_samples(
        self,
        samples: list[Sample],
    ) -> tuple[list[Sample], list[Sample], list[Sample]]:
        """Split samples per label to preserve class distribution across splits."""

        grouped: dict[str, list[Sample]] = {}
        for sample in samples:
            grouped.setdefault(sample.label, []).append(sample)

        rng = random.Random(self.seed)  # noqa: S311 - deterministic split only
        train_samples: list[Sample] = []
        val_samples: list[Sample] = []
        test_samples: list[Sample] = []

        for label_samples in grouped.values():
            shuffled = label_samples[:]
            rng.shuffle(shuffled)

            total = len(shuffled)
            if total == 1:
                train_samples.extend(shuffled)
                continue

            n_val = int(total * self.val_ratio)
            n_test = int(total * self.test_ratio)
            if n_val + n_test >= total:
                overflow = (n_val + n_test) - (total - 1)
                reduce_val = min(n_val, overflow)
                n_val -= reduce_val
                overflow -= reduce_val
                n_test -= min(n_test, overflow)

            val_samples.extend(shuffled[:n_val])
            test_samples.extend(shuffled[n_val : n_val + n_test])
            train_samples.extend(shuffled[n_val + n_test :])

        if not train_samples:
            raise RuntimeError("Stratified split generation produced an empty training split.")

        return train_samples, val_samples, test_samples
