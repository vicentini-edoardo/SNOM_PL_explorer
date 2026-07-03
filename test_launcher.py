from __future__ import annotations

from pathlib import Path


def test_macos_launcher_script_exists_and_runs_app_from_repo_root() -> None:
    launcher = Path(__file__).with_name("launch_macos.sh")

    assert launcher.exists()

    contents = launcher.read_text()
    assert "cd \"$(dirname \"$0\")\"" in contents
    assert "app.py" in contents
    assert ".venv/bin/python" in contents or "python3" in contents
