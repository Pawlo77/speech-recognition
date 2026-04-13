"""Composable torchaudio feature extraction modules."""

import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import torch
import torchaudio
import torchaudio.functional as audio_functional
from torch import Tensor, nn

from ..config import FeaturePipelineConfig

DEFAULT_SAMPLE_RATE: Final[int] = 16000
"""Default audio sample rate used by waveform loading and feature transforms."""

DEFAULT_TARGET_SECONDS: Final[float] = 1.0
"""Default clip duration used for padding/trimming waveforms."""


class DynamicWaveformPad(nn.Module):
    """Pad or trim waveforms to a fixed sample length."""

    def __init__(self, target_num_samples: int = DEFAULT_SAMPLE_RATE) -> None:
        super().__init__()
        if target_num_samples <= 0:
            raise ValueError("target_num_samples must be positive.")
        self.target_num_samples = target_num_samples

    def forward(self, waveform: Tensor) -> Tensor:
        """Return mono waveforms with shape [B, 1, target_num_samples]."""

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        elif waveform.dim() != 3:
            raise ValueError("waveform must have shape [T], [B, T], or [B, C, T].")

        if waveform.size(1) != 1:
            waveform = waveform.mean(dim=1, keepdim=True)

        sample_count = waveform.size(-1)
        if sample_count < self.target_num_samples:
            pad_width = self.target_num_samples - sample_count
            waveform = nn.functional.pad(waveform, (0, pad_width))
        elif sample_count > self.target_num_samples:
            waveform = waveform[..., : self.target_num_samples]
        return waveform


class WaveformLoader(nn.Module):
    """Load waveform files and normalize them to fixed-length mono tensors."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        target_seconds: float = DEFAULT_TARGET_SECONDS,
    ) -> None:
        super().__init__()
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if target_seconds <= 0:
            raise ValueError("target_seconds must be positive.")
        self.sample_rate = sample_rate
        self.target_num_samples = round(target_seconds * sample_rate)
        self.padder = DynamicWaveformPad(self.target_num_samples)

    def _load_one(self, path: str | Path, device: torch.device | str | None = None) -> Tensor:
        """Load one file, resample to configured rate, and normalize duration."""

        try:
            waveform, original_sample_rate = torchaudio.load(str(path))
        except ImportError as err:
            with wave.open(str(path), "rb") as wav_file:
                original_sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frames = wav_file.readframes(wav_file.getnframes())

            if sample_width != 2:
                raise ValueError("WaveformLoader fallback expects 16-bit PCM WAV files.") from err

            int16_waveform = torch.frombuffer(bytearray(frames), dtype=torch.int16).to(
                torch.float32
            )
            int16_waveform = int16_waveform.view(-1, channels).transpose(0, 1)
            waveform = int16_waveform / 32767.0

        waveform = waveform.to(torch.float32)
        if waveform.size(0) != 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sample_rate != self.sample_rate:
            waveform = audio_functional.resample(waveform, original_sample_rate, self.sample_rate)

        waveform = self.padder(waveform)
        if device is not None:
            waveform = waveform.to(device)
        return waveform

    def forward(
        self, paths: Sequence[str | Path], device: torch.device | str | None = None
    ) -> Tensor:
        """Load many audio files and stack them as a batch tensor."""

        if not paths:
            raise ValueError("paths must not be empty.")
        waveforms = [self._load_one(path, device=device) for path in paths]
        return torch.cat(waveforms, dim=0)


class MelSpectrogramFeatures(nn.Module):
    """Baseline mel-spectrogram extractor."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = 1024,
        hop_length: int = 160,
        n_mels: int = 128,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: Tensor) -> Tensor:
        """Compute log-mel features with output shape [B, n_mels, frames]."""

        mel = self.mel(waveform)
        mel = self.to_db(mel)
        return mel.squeeze(1)


class HighTemporalMelFeatures(MelSpectrogramFeatures):
    """High temporal-resolution mel extractor with 5ms stride."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, n_mels: int = 128) -> None:
        super().__init__(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=80,
            n_mels=n_mels,
        )


class MFCCFeatures(nn.Module):
    """MFCC extractor using torchaudio DCT implementation."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_mfcc: int = 40,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 160,
    ) -> None:
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "center": True,
                "power": 2.0,
            },
        )

    def forward(self, waveform: Tensor) -> Tensor:
        """Compute MFCC features with output shape [B, n_mfcc, frames]."""

        return self.mfcc(waveform).squeeze(1)


class PCENFeatures(nn.Module):
    """Mel spectrogram followed by Per-Channel Energy Normalization."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = 1024,
        hop_length: int = 160,
        n_mels: int = 128,
        smoothing: float = 0.1,
        alpha: float = 0.98,
        delta: float = 2.0,
        root: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.alpha = alpha
        self.delta = delta
        self.root = root
        self.eps = eps
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1.0,
        )

    def _pcen(self, mel: Tensor) -> Tensor:
        """Apply differentiable PCEN normalization over the time axis."""

        smoothed = torch.empty_like(mel)
        smoothed[:, :, 0] = mel[:, :, 0]
        for time_index in range(1, mel.size(-1)):
            smoothed[:, :, time_index] = (1.0 - self.smoothing) * smoothed[
                :, :, time_index - 1
            ] + self.smoothing * mel[:, :, time_index]

        normalized = mel / (self.eps + smoothed).pow(self.alpha)
        return (normalized + self.delta).pow(self.root) - self.delta**self.root

    def forward(self, waveform: Tensor) -> Tensor:
        """Compute PCEN-normalized mel features."""

        mel = self.mel(waveform).squeeze(1)
        return self._pcen(mel)


class SpecAugment(nn.Module):
    """Frequency and time masking on mel-like feature tensors."""

    def __init__(self, freq_mask_param: int = 24, time_mask_param: int = 20) -> None:
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, features: Tensor) -> Tensor:
        """Apply frequency then time masking to feature maps."""
        return self.time_mask(self.freq_mask(features))


class MelSpecAugmentFeatures(nn.Module):
    """Baseline mel extractor plus SpecAugment masking."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = 1024,
        hop_length: int = 160,
        n_mels: int = 128,
    ) -> None:
        super().__init__()
        self.mel = MelSpectrogramFeatures(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.augment = SpecAugment()

    def forward(self, waveform: Tensor) -> Tensor:
        """Compute mel features and apply SpecAugment masks."""

        mel = self.mel(waveform)
        return self.augment(mel)


class FeaturePipeline(nn.Module):
    """Compose waveform normalization and configured feature extraction."""

    def __init__(self, padder: DynamicWaveformPad, extractor: nn.Module) -> None:
        super().__init__()
        self.padder = padder
        self.extractor = extractor

    def forward(self, waveform: Tensor) -> Tensor:
        """Pad/trim waveform input and run the configured extractor."""

        waveform = self.padder(waveform)
        return self.extractor(waveform)


def build_feature_extractor(
    config: FeaturePipelineConfig,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Build a feature extractor module from feature config."""

    if seed is not None:
        torch.manual_seed(seed)

    if config.name == "mel_spectrogram":
        extractor: nn.Module = MelSpectrogramFeatures(
            sample_rate=sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
    elif config.name == "high_temporal_mel":
        extractor = HighTemporalMelFeatures(sample_rate=sample_rate, n_mels=config.n_mels)
    elif config.name == "mfcc":
        extractor = MFCCFeatures(
            sample_rate=sample_rate,
            n_mfcc=config.n_mfcc,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )
    elif config.name == "pcen":
        extractor = PCENFeatures(
            sample_rate=sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            smoothing=config.pcen_smoothing,
        )
    elif config.name == "mel_specaugment":
        extractor = MelSpecAugmentFeatures(
            sample_rate=sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
    else:
        raise ValueError(f"Unsupported feature pipeline '{config.name}'.")

    if device is not None:
        extractor = extractor.to(device)
    return extractor


def build_feature_pipeline(
    config: FeaturePipelineConfig,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    target_seconds: float = DEFAULT_TARGET_SECONDS,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> FeaturePipeline:
    """Build the full waveform-to-feature pipeline from config."""

    padder = DynamicWaveformPad(target_num_samples=round(target_seconds * sample_rate))
    extractor = build_feature_extractor(
        config=config,
        sample_rate=sample_rate,
        seed=seed,
        device=device,
    )
    pipeline = FeaturePipeline(padder=padder, extractor=extractor)
    if device is not None:
        pipeline = pipeline.to(device)
    return pipeline
