from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def windows_hidden_subprocess_kwargs(extra_creationflags: int = 0) -> dict:
    """Return Windows-only subprocess options that do not spawn a visible console."""
    if not platform.system().lower().startswith("win"):
        return {}
    kwargs: dict = {}
    creationflags = int(extra_creationflags or 0) | int(getattr(subprocess, "CREATE_NO_WINDOW", 0) or 0)
    if creationflags:
        kwargs["creationflags"] = creationflags
    try:
        startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
    except Exception:
        pass
    return kwargs


def merge_hidden_subprocess_kwargs(kwargs: dict, *, extra_creationflags: int = 0) -> dict:
    """Merge hidden-window defaults without overwriting explicit caller options."""
    merged = dict(kwargs)
    hidden = windows_hidden_subprocess_kwargs(extra_creationflags)
    if "creationflags" in hidden:
        merged["creationflags"] = int(merged.get("creationflags", 0) or 0) | int(hidden["creationflags"] or 0)
    if "startupinfo" not in merged and "startupinfo" in hidden:
        merged["startupinfo"] = hidden["startupinfo"]
    return merged


def run_subprocess_hidden(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess while keeping Windows frozen/GUI builds console-free."""
    return subprocess.run(command, **merge_hidden_subprocess_kwargs(kwargs))


def popen_subprocess_hidden(command: list[str], **kwargs) -> subprocess.Popen:
    """Start a subprocess while keeping Windows frozen/GUI builds console-free."""
    extra = int(kwargs.pop("extra_creationflags", 0) or 0)
    return subprocess.Popen(command, **merge_hidden_subprocess_kwargs(kwargs, extra_creationflags=extra))


def _looks_like_cli_python(candidate: str | os.PathLike[str] | None) -> bool:
    if not candidate:
        return False
    name = Path(str(candidate)).name.lower()
    return name in {"python", "python.exe", "python3", "python3.exe", "py", "py.exe"} or name.startswith("python")


def _existing_file(candidate: str | os.PathLike[str] | None) -> str:
    if not candidate:
        return ""
    try:
        path = Path(os.path.expandvars(str(candidate))).expanduser()
        if path.exists() and path.is_file():
            return str(path)
    except Exception:
        return ""
    return ""


def safe_python_executable(selected: str | os.PathLike[str] | None = None) -> str:
    """Return a CLI Python command for generated scripts, never the frozen GUI host.

    In PyInstaller/GUI builds ``sys.executable`` points to the app executable.
    Timeline .cmd/.sh helpers that call it with ``-c`` would reopen the main app.
    Prefer an explicit selected interpreter, then a real Python host, then PATH.
    """
    selected_file = _existing_file(selected)
    if selected_file and _looks_like_cli_python(selected_file):
        return selected_file

    frozen = bool(getattr(sys, "frozen", False))
    host = _existing_file(getattr(sys, "_base_executable", ""))
    if host and _looks_like_cli_python(host):
        return host

    executable = _existing_file(getattr(sys, "executable", ""))
    if executable and not frozen and _looks_like_cli_python(executable):
        return executable

    for env_name in ("PYTHON_EXECUTABLE", "PYTHON", "PYTHON3"):
        env_file = _existing_file(os.environ.get(env_name))
        if env_file and _looks_like_cli_python(env_file):
            return env_file

    command_names = ["python.exe", "python3.exe", "py.exe", "python", "python3", "py"] if platform.system().lower().startswith("win") else ["python3", "python"]
    for command in command_names:
        found = shutil.which(command)
        if found:
            return found

    # Last resort keeps source-tree runs working. In frozen builds this is only
    # reached when no CLI Python exists on PATH, so validation must report that gap.
    return executable or "python"
