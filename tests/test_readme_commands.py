import re
import shlex
from pathlib import Path

SHELL_FENCE_LANGUAGES = {"bash", "sh", "shell", "zsh", "env", "makefile"}
ALLOWED_COMMANDS = {
    "export",
    "make",
    "mlflow",
    "speech-recognition",
    "uv",
}


def _readme_text() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "README.md").read_text(encoding="utf-8")


def _shell_fences(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"```(?P<lang>[A-Za-z0-9_-]*)\n(?P<body>.*?)\n```", re.DOTALL)
    fences: list[tuple[str, str]] = []
    for match in pattern.finditer(text):
        language = match.group("lang").lower()
        if language in SHELL_FENCE_LANGUAGES:
            fences.append((language, match.group("body")))
    return fences


def _assert_shell_line_is_valid(line: str) -> None:
    if line.startswith("#"):
        return

    if line.startswith("export "):
        assignments = line.split()[1:]
        assert assignments, f"export without assignments: {line!r}"
        for assignment in assignments:
            assert re.fullmatch(r"[A-Z_][A-Z0-9_]*=.*", assignment), assignment
        return

    if re.fullmatch(r"[A-Z_][A-Z0-9_]*=.*", line):
        return

    parts = shlex.split(line)
    assert parts, f"empty shell line: {line!r}"
    assert parts[0] in ALLOWED_COMMANDS, f"unexpected shell command: {parts[0]!r}"


def test_readme_shell_code_blocks_use_valid_commands() -> None:
    text = _readme_text()

    expected_commands = [
        "make install",
        "make test",
        "make pre-commit-all",
        "make phase-1",
        "make phase-2",
        "make phase-3",
        "make phase-4",
        "make status",
        "make full-pipeline",
        "make eta-estimate",
        "make estimate-ram",
        "mlflow ui --backend-store-uri mlruns",
        "speech-recognition mlflow-ui",
    ]
    for command in expected_commands:
        assert command in text

    for _language, body in _shell_fences(text):
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            _assert_shell_line_is_valid(line)
