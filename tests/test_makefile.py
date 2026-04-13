from pathlib import Path


def test_makefile_exports_required_env_vars_and_targets() -> None:
    root = Path(__file__).resolve().parents[1]
    makefile_path = root / "Makefile"
    makefile = makefile_path.read_text(encoding="utf-8")

    assert "export PYTHONPATH=." in makefile
    assert "export PYTORCH_ENABLE_MPS_FALLBACK=1" in makefile
    assert "export OMP_NUM_THREADS=1" in makefile

    for target in ("phase-1", "phase-2", "phase-3", "phase-4", "full-pipeline"):
        assert f"{target}:" in makefile
