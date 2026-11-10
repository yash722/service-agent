from __future__ import annotations

import subprocess
import sys

from .settings import Settings


def command(settings: Settings) -> list[str]:
    app_path = settings.project_root / "src" / "servicing_agent" / "streamlit_app.py"
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        settings.ui_host,
        "--server.port",
        str(settings.ui_port),
        "--server.headless",
        str(settings.ui_headless).lower(),
        "--browser.gatherUsageStats",
        str(settings.ui_gather_usage_stats).lower(),
    ]


def main() -> int:
    settings = Settings.from_environment()
    try:
        return subprocess.call(command(settings), cwd=settings.project_root)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
