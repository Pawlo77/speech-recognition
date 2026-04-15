from pathlib import Path


def test_package_imports() -> None:
    """Smoke test: package can be imported."""
    import speech_recognition  # noqa: F401


def test_project_scaffold_layout() -> None:
    """Smoke test: expected top-level scaffold directories/files exist."""
    root = Path(__file__).resolve().parents[1]

    expected_paths = [
        root / "src" / "speech_recognition",
        root / "tests",
        root / "notebooks",
        root / "report",
        root / "pyproject.toml",
        root / "README.md",
    ]

    missing = [str(path.relative_to(root)) for path in expected_paths if not path.exists()]
    assert not missing, f"Missing scaffold paths: {missing}"
