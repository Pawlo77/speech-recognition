"""Unknown-class sample generation helpers for the dataset package."""

import logging
import math
import random
import wave
from array import array
from contextlib import ExitStack
from itertools import pairwise
from pathlib import Path

logger = logging.getLogger(__name__)


def _require_pcm16(sample_width: int) -> None:
    """Validate that operations run on 16-bit PCM data."""

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM audio is supported.")


def _frames_to_int16_samples(frames: bytes, sample_width: int) -> list[int]:
    """Convert PCM16 frames into integer samples."""

    _require_pcm16(sample_width)
    samples = array("h")
    samples.frombytes(frames[: len(frames) - (len(frames) % 2)])
    return samples.tolist()


def _int16_samples_to_frames(samples: list[int]) -> bytes:
    """Convert integer samples into clipped PCM16 frames."""

    clipped = array("h", [max(-32768, min(32767, sample)) for sample in samples])
    return clipped.tobytes()


def _pcm_avg(frames: bytes, sample_width: int) -> int:
    """Return the integer mean sample value."""

    samples = _frames_to_int16_samples(frames, sample_width)
    if not samples:
        return 0
    return int(sum(samples) / len(samples))


def _pcm_rms(frames: bytes, sample_width: int) -> int:
    """Return RMS energy for PCM16 samples."""

    samples = _frames_to_int16_samples(frames, sample_width)
    if not samples:
        return 0
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    return round(math.sqrt(mean_square))


def _pcm_bias(frames: bytes, sample_width: int, bias: int) -> bytes:
    """Add a constant bias to all samples."""

    samples = _frames_to_int16_samples(frames, sample_width)
    return _int16_samples_to_frames([sample + bias for sample in samples])


def _pcm_mul(frames: bytes, sample_width: int, factor: float) -> bytes:
    """Scale PCM16 samples by a floating-point factor."""

    samples = _frames_to_int16_samples(frames, sample_width)
    return _int16_samples_to_frames([round(sample * factor) for sample in samples])


def _pcm_add(left: bytes, right: bytes, sample_width: int) -> bytes:
    """Add two PCM16 byte streams sample-wise."""

    left_samples = _frames_to_int16_samples(left, sample_width)
    right_samples = _frames_to_int16_samples(right, sample_width)
    sample_count = min(len(left_samples), len(right_samples))
    mixed = [left_samples[index] + right_samples[index] for index in range(sample_count)]
    return _int16_samples_to_frames(mixed)


class UnknownSampleGenerationMixin:
    """Mixin that encapsulates __unknown__ sample synthesis and filtering."""

    def _create_unknown_label_samples(self) -> None:
        """Create the __unknown__ label as interpolation of existing samples."""

        logger.info("Creating __unknown__ label samples as interpolation of existing samples.")

        unknown_dir = self.dataset_root / "train" / "audio" / self.UNKNOWN_LABEL
        unknown_dir.mkdir(parents=True, exist_ok=True)
        existing_unknown_samples = len(list(unknown_dir.glob("*.wav")))
        samples_to_create = self.unknown_label_samples_size - existing_unknown_samples

        samples_by_label: dict[str, list[Path]] = {}
        for label_dir in (self.dataset_root / "train" / "audio").iterdir():
            if label_dir.is_dir() and label_dir.name not in (
                "_background_noise_",
                self.UNKNOWN_LABEL,
            ):
                samples_by_label[label_dir.name] = sorted(label_dir.glob("*.wav"))

        available_labels = [label for label, samples in samples_by_label.items() if samples]
        if len(available_labels) < 2:
            logger.warning("Not enough labeled samples to create unknown-label interpolations.")
            return

        rng = random.Random(self.seed)  # noqa: S311 - deterministic sampling only
        used_source_sets: set[tuple[str, ...]] = set()
        max_attempts = max(100, len(available_labels) * 10)
        created_samples = 0
        generation_attempts = 0
        max_generation_attempts = max(samples_to_create * 25, 100)

        while created_samples < samples_to_create and generation_attempts < max_generation_attempts:
            generation_attempts += 1

            source_paths_and_key = self._pick_unknown_source_paths(
                samples_by_label=samples_by_label,
                available_labels=available_labels,
                rng=rng,
                used_source_sets=used_source_sets,
                max_attempts=max_attempts,
            )
            if source_paths_and_key is None:
                logger.warning(
                    "Could not find a new unique source combination after %d attempts. "
                    "Created %d of %d requested unknown samples.",
                    max_attempts,
                    created_samples,
                    samples_to_create,
                )
                break

            source_paths, source_key = source_paths_and_key

            blended_frames = self._blend_unknown_audio(source_paths=source_paths, rng=rng)
            if not blended_frames:
                logger.warning(
                    "Skipping interpolation of %s because no blended audio was produced.",
                    [str(path) for path in source_paths],
                )
                continue

            if not self._is_smooth_unknown_audio(
                frames=blended_frames,
                sample_width=2,
                sample_rate=16000,
                channels=1,
            ):
                logger.debug(
                    "Rejected blended sample from %s because it failed smoothness checks.",
                    [str(path) for path in source_paths],
                )
                continue

            unknown_sample_path = (
                unknown_dir / f"unknown_{existing_unknown_samples + created_samples}.wav"
            )
            with wave.open(str(unknown_sample_path), "wb") as unknown_wav:
                with wave.open(str(source_paths[0]), "rb") as source_wav:
                    unknown_wav.setnchannels(source_wav.getnchannels())
                    unknown_wav.setsampwidth(source_wav.getsampwidth())
                    unknown_wav.setframerate(source_wav.getframerate())
                unknown_wav.writeframes(blended_frames)

            used_source_sets.add(source_key)
            created_samples += 1

    def _pick_unknown_source_paths(
        self,
        samples_by_label: dict[str, list[Path]],
        available_labels: list[str],
        rng: random.Random,
        used_source_sets: set[tuple[str, ...]],
        max_attempts: int,
    ) -> tuple[list[Path], tuple[str, ...]] | None:
        """Pick 2 or 3 source files uniquely by sample IDs, with class diversity only."""

        blend_count = 3 if len(available_labels) >= 3 and rng.random() < 0.6 else 2
        blend_count = min(blend_count, len(available_labels))

        for _ in range(max_attempts):
            chosen_labels = rng.sample(available_labels, blend_count)
            if len(set(chosen_labels)) != blend_count:
                continue

            chosen_paths = [rng.sample(samples_by_label[label], 1)[0] for label in chosen_labels]

            # Unique combination is based on source sample IDs (file paths), not class tuples.
            source_key = tuple(sorted(path.as_posix() for path in chosen_paths))
            if source_key in used_source_sets:
                continue

            return chosen_paths, source_key

        return None

    def _blend_unknown_audio(
        self,
        source_paths: list[Path],
        rng: random.Random,
    ) -> bytes:
        """Blend two or three clips with smooth speech-like overlap."""

        if len(source_paths) < 2:
            return b""

        with ExitStack() as stack:
            wav_files = [stack.enter_context(wave.open(str(path), "rb")) for path in source_paths]

            frame_rate = wav_files[0].getframerate()
            sample_width = wav_files[0].getsampwidth()
            channels = wav_files[0].getnchannels()
            frame_size = sample_width * channels

            if any(
                wav_file.getframerate() != frame_rate
                or wav_file.getsampwidth() != sample_width
                or wav_file.getnchannels() != channels
                for wav_file in wav_files[1:]
            ):
                return b""

            target_frames = min(min(wav_file.getnframes() for wav_file in wav_files), frame_rate)
            if target_frames <= 0:
                return b""

            crops: list[bytes] = []
            rms_values: list[int] = []
            for wav_file in wav_files:
                start_frame = rng.randint(0, max(0, wav_file.getnframes() - target_frames))
                wav_file.setpos(start_frame)
                frames = wav_file.readframes(target_frames)
                frames = frames[: target_frames * frame_size]
                if len(frames) < frame_size:
                    return b""
                crops.append(frames)
                rms_values.append(_pcm_rms(frames, sample_width))

            if any(rms == 0 for rms in rms_values):
                return b""

            target_rms = sorted(rms_values)[len(rms_values) // 2]
            conditioned_crops = []
            for crop in crops:
                conditioned_crop = self._condition_unknown_source_audio(
                    frames=crop,
                    sample_width=sample_width,
                    sample_rate=frame_rate,
                    channels=channels,
                    target_rms=target_rms,
                    rng=rng,
                )
                if not conditioned_crop:
                    return b""
                conditioned_crops.append(conditioned_crop)

            source_count = len(conditioned_crops)
            source_phases = [rng.uniform(0.0, math.tau) for _ in range(source_count)]
            slice_frames = max(1, int(frame_rate * 0.005))
            output = bytearray(target_frames * frame_size)

            for slice_start in range(0, target_frames, slice_frames):
                slice_end = min(target_frames, slice_start + slice_frames)
                slice_length = slice_end - slice_start
                if slice_length <= 0:
                    continue

                time_point = (slice_start + (slice_length / 2)) / target_frames
                raw_weights = [
                    self._smooth_source_weight(time_point, source_count, phase)
                    for phase in source_phases
                ]
                weight_sum = sum(raw_weights)
                if weight_sum <= 0.0:
                    continue

                mixed_slice = bytes(slice_length * frame_size)
                for source_index, crop in enumerate(conditioned_crops):
                    source_slice = crop[slice_start * frame_size : slice_end * frame_size]
                    if not source_slice:
                        continue

                    normalized_weight = raw_weights[source_index] / weight_sum
                    scaled_slice = _pcm_mul(
                        source_slice,
                        sample_width,
                        0.98 * normalized_weight,
                    )
                    mixed_slice = _pcm_add(mixed_slice, scaled_slice, sample_width)

                dest_start_byte = slice_start * frame_size
                dest_end_byte = dest_start_byte + len(mixed_slice)
                output[dest_start_byte:dest_end_byte] = mixed_slice

            return bytes(output)

    def _get_unknown_source_profile(self, path: Path) -> tuple[str, float]:
        """Return a cached (gender_bucket, peak_fraction) profile for a source clip."""

        cache_key = path.as_posix()
        cached_profile = self._unknown_source_profile_cache.get(cache_key)
        if cached_profile is not None:
            return cached_profile

        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            frames = wav_file.readframes(wav_file.getnframes())

        gender = self._estimate_gender_bucket(frames, sample_rate, sample_width, channels)
        peak_fraction = self._estimate_peak_fraction(frames, sample_rate, sample_width, channels)
        profile = (gender, peak_fraction)
        self._unknown_source_profile_cache[cache_key] = profile
        return profile

    def _estimate_gender_bucket(
        self,
        frames: bytes,
        sample_rate: int,
        sample_width: int,
        channels: int,
    ) -> str:
        """Heuristically bucket a voice as male or female using zero-crossing pitch estimates."""

        if sample_width != 2 or sample_rate <= 0:
            return "unknown"

        mono_samples = self._frames_to_mono_samples(frames, sample_width, channels)
        if len(mono_samples) < sample_rate // 5:
            return "unknown"

        window_size = max(1, int(sample_rate * 0.04))
        min_rms = max(120, _pcm_rms(frames, sample_width) // 3)
        pitch_estimates: list[float] = []

        for start in range(0, len(mono_samples) - window_size + 1, window_size // 2 or 1):
            window = mono_samples[start : start + window_size]
            window_frames = self._samples_to_frames(window)
            if _pcm_rms(window_frames, sample_width) < min_rms:
                continue

            zero_crossings = 0
            previous = window[0]
            for current in window[1:]:
                if (previous <= 0 < current) or (previous >= 0 > current):
                    zero_crossings += 1
                previous = current

            estimated_pitch = zero_crossings * sample_rate / (2.0 * len(window))
            if 70.0 <= estimated_pitch <= 320.0:
                pitch_estimates.append(estimated_pitch)

        if not pitch_estimates:
            return "unknown"

        median_pitch = sorted(pitch_estimates)[len(pitch_estimates) // 2]
        return "female" if median_pitch >= 165.0 else "male"

    def _estimate_peak_fraction(
        self,
        frames: bytes,
        sample_rate: int,
        sample_width: int,
        channels: int,
    ) -> float:
        """Estimate where the loudest part of a clip occurs, as a fraction of its duration."""

        if sample_width != 2 or sample_rate <= 0:
            return 0.5

        mono_samples = self._frames_to_mono_samples(frames, sample_width, channels)
        if not mono_samples:
            return 0.5

        window_size = max(1, int(sample_rate * 0.05))
        step_size = max(1, window_size // 2)
        windows: list[int] = []
        window_rms: list[int] = []

        for start in range(0, max(1, len(mono_samples) - window_size + 1), step_size):
            window = mono_samples[start : start + window_size]
            if not window:
                continue
            windows.append(start)
            window_rms.append(_pcm_rms(self._samples_to_frames(window), sample_width))

        if not window_rms:
            return 0.5

        peak_index = window_rms.index(max(window_rms))
        peak_start = windows[peak_index]
        return min(0.99, max(0.01, (peak_start + (window_size / 2)) / len(mono_samples)))

    def _frames_to_mono_samples(
        self,
        frames: bytes,
        sample_width: int,
        channels: int,
    ) -> list[int]:
        """Convert PCM16 frames to mono integer samples."""

        if sample_width != 2:
            return []

        samples = array("h")
        samples.frombytes(frames[: len(frames) - (len(frames) % 2)])
        if channels <= 1:
            return samples.tolist()

        mono_samples: list[int] = []
        for index in range(0, len(samples) - (len(samples) % channels), channels):
            mono_samples.append(sum(samples[index : index + channels]) // channels)
        return mono_samples

    def _samples_to_frames(self, samples: list[int]) -> bytes:
        """Convert int16 samples to PCM16 frames."""

        clipped_samples = array(
            "h",
            [max(-32768, min(32767, sample)) for sample in samples],
        )
        return clipped_samples.tobytes()

    def _slot_weight(self, time_point: float, slot_index: int, source_count: int) -> float:
        """Return a broad smooth weight with a nonzero floor for one source."""

        center = (slot_index + 1) / (source_count + 1)
        half_width = 0.42 if source_count == 2 else 0.34
        floor_weight = 0.24 if source_count == 2 else 0.16

        distance = abs(time_point - center)
        if distance >= half_width:
            return floor_weight

        normalized_distance = distance / half_width
        smooth_shape = 0.5 * (1.0 + math.cos(math.pi * normalized_distance))
        return floor_weight + (1.0 - floor_weight) * smooth_shape

    def _condition_unknown_source_audio(
        self,
        frames: bytes,
        sample_width: int,
        sample_rate: int,
        channels: int,
        target_rms: int,
        rng: random.Random,
    ) -> bytes:
        """Normalize and lightly smear a source clip before blending."""

        if not frames:
            return b""

        centered_frames = _pcm_bias(frames, sample_width, -_pcm_avg(frames, sample_width))
        current_rms = _pcm_rms(centered_frames, sample_width)
        if current_rms == 0:
            return b""

        gain = target_rms / current_rms
        gain = min(1.15, max(0.85, gain))
        gain *= rng.uniform(0.96, 1.04)

        conditioned_frames = _pcm_mul(centered_frames, sample_width, gain)
        frame_size = sample_width * channels
        delay_frames = rng.randint(max(1, int(sample_rate * 0.02)), max(1, int(sample_rate * 0.05)))
        delay_bytes = delay_frames * frame_size
        if len(conditioned_frames) > delay_bytes:
            echo_gain = rng.uniform(0.10, 0.18)
            dry = conditioned_frames
            wet = _pcm_mul(dry[:-delay_bytes], sample_width, echo_gain)
            mixed_tail = _pcm_add(dry[delay_bytes:], wet, sample_width)
            conditioned_frames = dry[:delay_bytes] + mixed_tail

        return conditioned_frames

    def _is_smooth_unknown_audio(
        self,
        frames: bytes,
        sample_width: int,
        sample_rate: int,
        channels: int,
    ) -> bool:
        """Reject samples with obvious discontinuities or multiple separated loud peaks."""

        mono_samples = self._frames_to_mono_samples(frames, sample_width, channels)
        if len(mono_samples) < max(1, sample_rate // 4):
            return False

        window_size = max(1, int(sample_rate * 0.03))
        hop_size = max(1, int(window_size * 0.25))
        envelope: list[int] = []
        for start in range(0, max(1, len(mono_samples) - window_size + 1), hop_size):
            window = mono_samples[start : start + window_size]
            if not window:
                continue
            envelope.append(_pcm_rms(self._samples_to_frames(window), sample_width))

        if len(envelope) < 4:
            return False

        peak = max(envelope)
        if peak <= 0:
            return False

        deltas = [abs(second - first) / peak for first, second in pairwise(envelope)]
        if deltas and max(deltas) > 1.15:
            return False
        if deltas and (sum(deltas) / len(deltas)) > 0.45:
            return False

        dominant_indices = [index for index, value in enumerate(envelope) if value >= 0.82 * peak]
        if len(dominant_indices) >= 2:
            start_index = dominant_indices[0]
            end_index = dominant_indices[-1]
            valley = min(envelope[start_index : end_index + 1])
            if valley < 0.55 * peak:
                return False

        mean_energy = sum(envelope) / len(envelope)
        return peak / max(mean_energy, 1.0) <= 2.9

    def _smooth_source_weight(
        self,
        time_point: float,
        source_count: int,
        phase: float,
    ) -> float:
        """Return a slow, high-floor weight so all sources stay present throughout the clip."""

        base_weight = 0.72 / source_count
        modulation = 0.28 / source_count
        envelope = 0.5 * (1.0 + math.cos(math.tau * time_point + phase))
        return base_weight + modulation * envelope

    def _expected_peak_positions(self, source_count: int) -> list[float]:
        """Return ideal peak locations for the ordered source peaks."""

        if source_count == 2:
            return [0.32, 0.68]
        return [0.18, 0.50, 0.82]

    def _gaussian_weight(self, time_point: float, center: float, width: float) -> float:
        """Return a smooth window weight for a clip centered at time_point."""

        distance = (time_point - center) / width
        return math.exp(-0.5 * distance * distance)
