import wave
from pathlib import Path

import pytest

from speech_recognition.config import FeaturePipelineConfig
from speech_recognition.features import (
    DynamicWaveformPad,
    WaveformLoader,
    build_feature_extractor,
    build_feature_pipeline,
)

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")


def _waveform(batch_size: int = 2, seconds: float = 1.0, sample_rate: int = 16000) -> torch.Tensor:
    num_samples = round(seconds * sample_rate)
    return torch.randn(batch_size, 1, num_samples, dtype=torch.float32)


def _write_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    clamped = waveform.squeeze(0).clamp(-1.0, 1.0)
    pcm = (clamped * 32767.0).to(torch.int16).cpu().numpy().tobytes()
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


def test_baseline_mel_shape_and_size() -> None:
    extractor = build_feature_extractor(FeaturePipelineConfig(name="mel_spectrogram"), seed=123)
    features = extractor(_waveform(batch_size=2, seconds=1.0))

    assert features.shape == (2, 128, 101)
    assert features.dtype == torch.float32
    assert features[0].numel() == 128 * 101


def test_high_temporal_mel_shape() -> None:
    extractor = build_feature_extractor(
        FeaturePipelineConfig(name="high_temporal_mel", n_fft=512, hop_length=80),
        seed=7,
    )
    features = extractor(_waveform(batch_size=1, seconds=1.0))

    assert features.shape == (1, 128, 201)


def test_mfcc_shape_and_dtype() -> None:
    extractor = build_feature_extractor(FeaturePipelineConfig(name="mfcc", n_mfcc=40), seed=2)
    features = extractor(_waveform(batch_size=3, seconds=1.0))

    assert features.shape == (3, 40, 101)
    assert features.dtype == torch.float32


def test_pcen_shape() -> None:
    extractor = build_feature_extractor(FeaturePipelineConfig(name="pcen", pcen_smoothing=0.1))
    features = extractor(_waveform(batch_size=2, seconds=1.0))

    assert features.shape == (2, 128, 101)


def test_specaugment_reproducibility_with_manual_seed() -> None:
    config = FeaturePipelineConfig(name="mel_specaugment", specaugment=True)
    extractor = build_feature_extractor(config)
    waveform = _waveform(batch_size=1, seconds=1.0)

    torch.manual_seed(3407)
    first = extractor(waveform)
    torch.manual_seed(3407)
    second = extractor(waveform)

    assert torch.equal(first, second)


def test_dynamic_padding_handles_short_and_long_outliers() -> None:
    padder = DynamicWaveformPad(target_num_samples=16000)

    short_waveform = torch.randn(1, 1, round(0.37 * 16000), dtype=torch.float32)
    long_waveform = torch.randn(1, 1, round(95.18 * 16000), dtype=torch.float32)

    short_padded = padder(short_waveform)
    long_trimmed = padder(long_waveform)

    assert short_padded.shape[-1] == 16000
    assert long_trimmed.shape[-1] == 16000


def test_waveform_loader_loads_and_normalizes_to_one_second(tmp_path: Path) -> None:
    sample_rate = 16000
    loader = WaveformLoader(sample_rate=sample_rate, target_seconds=1.0)

    first = torch.randn(1, round(0.37 * sample_rate), dtype=torch.float32)
    second = torch.randn(1, round(2.0 * sample_rate), dtype=torch.float32)
    first_path = tmp_path / "short.wav"
    second_path = tmp_path / "long.wav"
    _write_wav(first_path, first, sample_rate)
    _write_wav(second_path, second, sample_rate)

    loaded = loader([first_path, second_path])

    assert loaded.shape == (2, 1, sample_rate)
    assert loaded.dtype == torch.float32


def test_factory_builds_full_pipeline_from_config() -> None:
    pipeline = build_feature_pipeline(FeaturePipelineConfig(name="mfcc", n_mfcc=40), seed=42)
    output = pipeline(_waveform(batch_size=1, seconds=1.0))

    assert output.shape == (1, 40, 101)


def test_mps_execution_when_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend is not available.")

    device = torch.device("mps")
    pipeline = build_feature_pipeline(
        FeaturePipelineConfig(name="mel_spectrogram"),
        device=device,
        seed=1,
    )
    waveform = _waveform(batch_size=1, seconds=1.0).to(device)
    features = pipeline(waveform)

    assert features.device.type == "mps"
    assert features.shape == (1, 128, 101)
