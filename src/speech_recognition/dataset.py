"""Dataset utilities for Kaggle speech recognition challenge."""

import csv
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

load_dotenv()

logger = logging.getLogger(__name__)

type Split = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class Sample:
    """Single audio sample descriptor."""

    path: Path
    label: str
    filename: str


class SpeechCommandsDataset:
    """Manage local/downloaded Kaggle speech dataset and splits."""

    KAGGLE_SLUG: Final[str] = "tensorflow-speech-recognition-challenge"
    """Kaggle competition slug."""

    EXPECTED_MAIN_DIR: Final[str] = "train/audio"
    """Required relative directory for availability check."""

    VAL_LIST_FILE: Final[str] = "train/validation_list.txt"
    """Relative path to official validation split list."""

    TEST_LIST_FILE: Final[str] = "train/testing_list.txt"
    """Relative path to official testing split list."""

    TRAIN_LABELS_CSV: Final[str] = "train.csv"
    """Relative path to optional filename-label mapping CSV."""

    TEST_AUDIO_DIR: Final[str] = "test/audio"
    """Relative path to competition test audio."""

    UNKNOWN_LABEL: Final[str] = "__unknown__"
    """Fallback label for unlabeled test items."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        data_dir_name: str = "data/kaggle_speech_commands",
        val_ratio: float = 0.1,
        seed: int = 42,
        auto_download: bool = True,
        only_1sec_samples: bool = True,
    ) -> None:
        """Initialize dataset manager."""
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

        self.repo_root: Path = Path(repo_root).resolve() if repo_root else self._infer_repo_root()
        self.data_dir: Path = self.repo_root / data_dir_name
        self.dataset_root: Path = self.data_dir / self.KAGGLE_SLUG

        self.val_ratio: float = val_ratio
        self.seed: int = seed
        self.only_1sec_samples: bool = only_1sec_samples

        logger.debug(
            (
                "Initializing SpeechCommandsDataset with repo_root=%s, data_dir=%s, "
                "dataset_root=%s, val_ratio=%s, seed=%s, auto_download=%s, only_1sec_samples=%s"
            ),
            self.repo_root,
            self.data_dir,
            self.dataset_root,
            self.val_ratio,
            self.seed,
            auto_download,
            self.only_1sec_samples,
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

        self._samples: dict[Split, list[Sample]] = self._build_splits()
        logger.info("Dataset initialized with split sizes: %s", self.stats())

    def is_available(self) -> bool:
        """Return whether expected dataset layout exists."""
        return (self.dataset_root / self.EXPECTED_MAIN_DIR).exists()

    def download(self, force: bool = False) -> Path:
        """Download and stage dataset locally."""
        logger.info("Preparing dataset download into '%s' (force=%s)", self.data_dir, force)
        if force and self.dataset_root.exists():
            logger.info("Removing existing dataset root due to force=True: %s", self.dataset_root)
            shutil.rmtree(self.dataset_root)

        if self.is_available():
            logger.info("Dataset already available at '%s'. Skipping download.", self.dataset_root)
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
            "Found candidate nested layout roots: %s", [match.parent.parent for match in matches]
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

        val_rel_paths = self._read_rel_paths(self.dataset_root / self.VAL_LIST_FILE)
        test_rel_paths = self._read_rel_paths(self.dataset_root / self.TEST_LIST_FILE)

        if val_rel_paths or test_rel_paths:
            logger.info(
                "Building splits using official lists (validation=%d, testing=%d)",
                len(val_rel_paths),
                len(test_rel_paths),
            )
            train_samples, val_samples = self._split_with_official_lists(
                all_labeled=all_labeled,
                train_audio_dir=train_audio,
                val_rel_paths=val_rel_paths,
                test_rel_paths=test_rel_paths,
            )
        else:
            logger.info(
                (
                    "Official split lists not found. "
                    "Falling back to random split (val_ratio=%s, seed=%s)."
                ),
                self.val_ratio,
                self.seed,
            )
            train_samples, val_samples = self._random_train_val_split(all_labeled)

        test_samples = self._collect_competition_test_samples()

        if self.only_1sec_samples:
            logger.info("Filtering splits to keep only 1-second samples (only_1sec_samples=True)")
            train_before = len(train_samples)
            val_before = len(val_samples)
            test_before = len(test_samples)

            train_samples = [s for s in train_samples if self._is_1sec_sample(s)]
            val_samples = [s for s in val_samples if self._is_1sec_sample(s)]
            test_samples = [s for s in test_samples if self._is_1sec_sample(s)]

            logger.info(
                "After 1-second filtering: train=%d->%d, val=%d->%d, test=%d->%d",
                train_before,
                len(train_samples),
                val_before,
                len(val_samples),
                test_before,
                len(test_samples),
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
        val_rel_paths: list[str],
        test_rel_paths: list[str],
    ) -> tuple[list[Sample], list[Sample]]:
        """Split using official validation/testing lists."""
        val_set = set(val_rel_paths)
        test_set = set(test_rel_paths)

        train_samples: list[Sample] = []
        val_samples: list[Sample] = []

        for sample in all_labeled:
            rel = sample.path.relative_to(train_audio_dir).as_posix()
            if rel in val_set:
                val_samples.append(sample)
            elif rel in test_set:
                continue
            else:
                train_samples.append(sample)

        return train_samples, val_samples

    def _collect_labeled_samples(self, train_audio: Path) -> list[Sample]:
        """Collect labeled samples from train/audio."""
        samples: list[Sample] = []

        for label_dir in sorted(train_audio.iterdir()):
            if not label_dir.is_dir():
                continue

            label = label_dir.name
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

    def _random_train_val_split(self, samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
        """Create deterministic random train/val split."""
        shuffled = samples[:]
        rng = random.Random(self.seed)  # noqa: S311 - deterministic split only
        rng.shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * self.val_ratio))
        val_samples = shuffled[:n_val]
        train_samples = shuffled[n_val:]
        logger.debug(
            "Random split produced train=%d and val=%d samples",
            len(train_samples),
            len(val_samples),
        )
        return train_samples, val_samples

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
