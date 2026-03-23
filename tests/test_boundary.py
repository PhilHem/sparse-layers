"""Verify sparse-layers has no infrastructure dependencies."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "sparse_layers"

FORBIDDEN = [
    "from ma_butterfly",
    "import ma_butterfly",
    "import mlflow",
    "from mlflow",
    "import typer",
    "from typer",
    "import rich",
    "from rich",
    "import pylatex",
    "from pylatex",
    "import matplotlib",
    "from matplotlib",
]


def test_no_forbidden_imports():
    """sparse-layers must only depend on torch and pydantic."""
    violations = []
    for py_file in PACKAGE_ROOT.rglob("*.py"):
        content = py_file.read_text()
        for pattern in FORBIDDEN:
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith(pattern):
                    violations.append(f"{py_file.relative_to(PACKAGE_ROOT)}:{i}: {stripped}")

    assert not violations, "Forbidden imports found in sparse-layers:\n" + "\n".join(violations)
