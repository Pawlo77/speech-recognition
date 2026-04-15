"""Split/list generation mixin for SpeechCommandsDataset."""

import csv
import hashlib
import importlib
import json
import logging
import random
import shutil
import subprocess
from pathlib import Path

from .manager_types import Sample, Split

logger = logging.getLogger(__name__)


class _SpeechCommandsSplitMixin:
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

        unknown_dir = train_audio_dir / self.UNKNOWN_LABEL
        if unknown_dir.exists():
            for metadata_file in sorted(unknown_dir.glob("*.sources.json")):
                try:
                    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    logger.warning("Skipping malformed synthetic metadata file: %s", metadata_file)
                    continue

                synthetic_rel_path = payload.get("synthetic_relative_path")
                source_labels = payload.get("source_labels")
                if not isinstance(synthetic_rel_path, str) or not isinstance(source_labels, list):
                    continue
                for source_label in source_labels:
                    if isinstance(source_label, str) and source_label:
                        rows.append((synthetic_rel_path, source_label, self.UNKNOWN_LABEL))

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
        """Return whether a relative path belongs to a target command label."""
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
        """Return the canonical label for a relative path, mapping all unknowns to UNKNOWN_LABEL."""
        return self._canonicalize_label(Path(rel_path).parts[0])

    def _raw_label_from_rel_path(self, rel_path: str) -> str:
        """Return the original top-level label directory name for a relative path."""
        return Path(rel_path).parts[0]

    def _is_original_unknown_rel_path(self, rel_path: str) -> bool:
        """Return whether a path belongs to an original non-target, non-silence unknown label."""
        raw_label = self._raw_label_from_rel_path(rel_path)
        return raw_label not in {
            *self.TARGET_COMMAND_LABELS,
            self.BACKGROUND_NOISE_LABEL,
            self.UNKNOWN_LABEL,
            self.PADDED_AUDIO_DIR,
        }

    def _collect_synthetic_unknown_rel_paths(self, train_audio_dir: Path) -> list[str]:
        """Collect generated unknown WAV paths from the synthetic unknown directory."""
        unknown_dir = train_audio_dir / self.UNKNOWN_LABEL
        if not unknown_dir.exists():
            return []
        return sorted(
            wav_file.relative_to(train_audio_dir).as_posix()
            for wav_file in unknown_dir.glob("*.wav")
        )

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

        unknown_train_target = min(len(command_train), self.MAX_EXTENDED_UNKNOWN_PER_SPLIT)
        unknown_val_target = len(command_val)
        unknown_test_target = len(command_test)
        required_unknown = unknown_train_target + unknown_val_target + unknown_test_target

        all_split_rel_paths = sorted(official_train + sorted(official_val) + sorted(official_test))
        original_unknown_pool = [
            rel for rel in all_split_rel_paths if self._is_original_unknown_rel_path(rel)
        ]
        seen_unknown: set[str] = set()
        dedup_unknown_pool: list[str] = []

        for rel in original_unknown_pool:
            key = self._unknown_dedup_key(rel, train_audio_dir)
            if key in seen_unknown:
                continue
            seen_unknown.add(key)
            dedup_unknown_pool.append(rel)

        if len(dedup_unknown_pool) < required_unknown:
            synth_attempts = 0
            while len(dedup_unknown_pool) < required_unknown and synth_attempts < 3:
                synth_attempts += 1
                missing = required_unknown - len(dedup_unknown_pool)
                current_synthetic = len(self._collect_synthetic_unknown_rel_paths(train_audio_dir))
                self._create_unknown_label_samples(minimum_total=current_synthetic + missing)

                for rel in self._collect_synthetic_unknown_rel_paths(train_audio_dir):
                    key = self._unknown_dedup_key(rel, train_audio_dir)
                    if key in seen_unknown:
                        continue
                    seen_unknown.add(key)
                    dedup_unknown_pool.append(rel)

        if len(dedup_unknown_pool) < required_unknown:
            raise RuntimeError(
                "Not enough unknown samples after dedup + synthetic top-up to build "
                f"balanced extended splits (required={required_unknown}, "
                f"available={len(dedup_unknown_pool)})."
            )

        rng.shuffle(dedup_unknown_pool)

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
