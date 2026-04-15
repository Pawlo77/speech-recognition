"""Dataset utilities for Kaggle speech recognition challenge."""

import csv
import logging
import random
import shutil
import wave
from pathlib import Path
from typing import Final

import kagglehub
from dotenv import load_dotenv

from .manager_split_mixin import _SpeechCommandsSplitMixin
from .manager_types import Sample, Split
from .unknown import UnknownSampleGenerationMixin

load_dotenv()

logger = logging.getLogger(__name__)


class SpeechCommandsDataset(_SpeechCommandsSplitMixin, UnknownSampleGenerationMixin):
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
    """Number of samples per class in the smaller dataset mode (before 1-second filtering)."""
    SMALL_VAL_PER_CLASS: Final[int] = 250
    """Number of samples per class in the smaller dataset mode (before 1-second filtering)."""
    SMALL_TEST_PER_CLASS: Final[int] = 250
    """Number of samples per class in the smaller dataset mode (before 1-second filtering)."""
    MAX_EXTENDED_UNKNOWN_PER_SPLIT: Final[int] = 20000
    """Maximum number of unknown samples per split in the extended dataset
    mode (after 1-second filtering)."""
    SMALLER_DATASET_LABELS: Final[set[str]] = set(TARGET_COMMAND_LABELS)
    """Labels included in the smaller dataset mode."""

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

        if (
            not self.use_smaller_dataset and self.use_extended_dataset
            # and not (self.dataset_root / "train" / "audio" / self.UNKNOWN_LABEL).exists()
        ):
            self._create_unknown_label_samples()

        self._write_unknown_origin_map()

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
