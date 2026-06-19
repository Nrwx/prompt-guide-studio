#!/usr/bin/env python3
"""Build Prompt Guide Studio as a Windows GUI executable with PyInstaller.

This script is intentionally small and reviewable. It validates the source tree,
uses PyInstaller's windowed mode so no foreground console is opened, defaults
to a one-file build so PyInstaller does not leave a sibling _internal folder,
and bundles the schema/locale/assets folders that the Tk application reads at
runtime.
"""
from __future__ import annotations

import argparse
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_FILES = ["prompt.py", "ai_json_generator_core.py", "ai_json_generator.py"]
DATA_DIRS = ["schema", "locale", "assets"]


def _data_separator() -> str:
    return ";" if os.name == "nt" else ":"


def _add_data_arg(source: Path, dest_name: str) -> str:
    return f"{source}{_data_separator()}{dest_name}"


def _validate_source(source: Path) -> None:
    missing = [name for name in ROOT_FILES if not (source / name).is_file()]
    missing += [name for name in DATA_DIRS if not (source / name).is_dir()]
    if missing:
        raise SystemExit("Missing required build inputs: " + ", ".join(missing))
    for name in ROOT_FILES:
        py_compile.compile(str(source / name), doraise=True)


def _clean_build_outputs(source: Path, app_name: str) -> None:
    for rel in ["build", "dist", f"{app_name}.spec"]:
        path = source / rel
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def _pyinstaller_command(source: Path, mode: str, app_name: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--windowed",
        "--name",
        app_name,
    ]
    if mode == "onefile":
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    icon = source / "assets" / "prompt-guide-studio-logo.ico"
    if icon.is_file():
        cmd.extend(["--icon", str(icon)])
    for dirname in DATA_DIRS:
        cmd.extend(["--add-data", _add_data_arg(source / dirname, dirname)])
    cmd.append(str(source / "prompt.py"))
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Prompt Guide Studio Windows executable")
    parser.add_argument("--source", required=True, type=Path, help="Source repository root")
    parser.add_argument(
        "--mode",
        choices=["onedir", "onefile"],
        default="onefile",
        help="Build layout. Defaults to onefile so no sibling _internal folder is produced.",
    )
    parser.add_argument("--name", default="PromptGuideStudio")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source repo does not exist or is not a folder: {source}")
    _validate_source(source)
    cmd = _pyinstaller_command(source, args.mode, args.name)
    print("[build] source:", source)
    print("[build] mode:", args.mode)
    if args.mode == "onedir":
        print("[build] note: onedir builds may create a sibling _internal folder; use --mode onefile for a single EXE.")
    print("[build] command:", subprocess.list2cmdline(cmd))
    if args.validate_only:
        print("[build] validate-only passed")
        return 0
    if shutil.which("pyinstaller") is None:
        try:
            import PyInstaller  # noqa: F401
        except Exception as exc:
            raise SystemExit("PyInstaller is not installed in this Python environment. Install it first with: python -m pip install pyinstaller") from exc
    if not args.no_clean:
        _clean_build_outputs(source, args.name)
    completed = subprocess.run(cmd, cwd=str(source))
    return int(completed.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
