#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ai_json_generator_core import SaveTarget, default_schema_dir, default_targets, default_profile_ids, generate_files, load_schema


def csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate modern hook-based multi-path AI rules.")
    parser.add_argument("--language", default="GERMAN", choices=["GERMAN", "ENGLISH"])
    parser.add_argument("--project-name", default="unity-texture-generator")
    parser.add_argument("--output", default=".", help="Project root / working tree. Generated files are written to the configured EXPORT folder by default; the working tree is read-only evidence.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--create-log", action="store_true")
    parser.add_argument("--date", default="", help="Optional YYYY-MM-DD date to embed in generated operator prompts.")
    parser.add_argument("--schema-dir", default=str(default_schema_dir()))
    parser.add_argument("--no-copy-schema", action="store_true")
    parser.add_argument("--target", action="append", help="path:path_type:ai_target:file_types_csv. Example ./frontend:frontend:Codex:js,vue,scss")
    parser.add_argument("--profiles", default="")
    parser.add_argument("--scope-path", action="append", default=[], help="Project-tree path to include in the scoped file reference/export. Repeat for multiple paths. Defaults to full project.")
    parser.add_argument("--export-as-zip", action="store_true", help="Create a scope-limited project clone ZIP under export/ after generation.")
    parser.add_argument("--compact-export", action="store_true", help="Write only USER_PROMPT, PROJECT_TREE, ROLE/OPERATOR/BOILERPLATE manifest, EXPORT_CONDITIONS and optional compact ZIP.")
    parser.add_argument("--export-dir", default="", help="Generated/export folder. Defaults to a sibling EXPORT folder next to the project root.")
    parser.add_argument("--prompt-file", default="", help="Optional prompt text file to write next to the ZIP, outside the ZIP. Kept for backward compatibility.")
    parser.add_argument("--custom-prompt", default="", help="Own task prompt to wrap with selected weights, references, operation roles and project-tree scope.")
    parser.add_argument("--custom-prompt-file", default="", help="File containing an own task prompt to wrap with selected weights, references, operation roles and project-tree scope.")
    parser.add_argument("--reference", action="append", default=[], help="Reference domain id to force active. Repeat for multiple references.")
    parser.add_argument("--operation-role", action="append", default=[], help="Operation role id to force active. Repeat for multiple roles.")
    parser.add_argument("--include-imports", action="store_true", help="Expand selected --scope-path files with recursively matched project-local imports/references.")
    parser.add_argument("--include-dependency-manifests", action="store_true", help="When exporting a ZIP, also copy package.json, requirements.txt and requirements.json. Dependency lists are not embedded in AI-RULES.")
    parser.add_argument("--changed-files-only", action="store_true", help="Tell generated AI prompts/rules to return only changed files plus concise validation notes.")
    return parser.parse_args()


def build_targets(args: argparse.Namespace, schema: dict) -> list[SaveTarget]:
    profiles = csv(args.profiles) if args.profiles else default_profile_ids(schema)
    if not args.target:
        return [SaveTarget(t.path, t.path_type, t.ai_target, profiles, t.file_types) for t in default_targets(schema)]

    targets: list[SaveTarget] = []
    for raw in args.target:
        parts = raw.rsplit(":", 3)
        if len(parts) != 4:
            raise ValueError(f"Invalid target format: {raw}. Expected path:path_type:ai_target:file_types_csv")
        path, path_type, ai_target, file_types = parts
        targets.append(SaveTarget(path.strip(), path_type.strip().lower(), ai_target.strip(), profiles, csv(file_types)))
    return targets


def main() -> int:
    args = parse_args()
    schema_dir = Path(args.schema_dir)
    schema = load_schema(schema_dir)
    targets = build_targets(args, schema)
    prompt_text = ""
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    custom_prompt_text = args.custom_prompt.strip()
    if args.custom_prompt_file:
        custom_prompt_text = Path(args.custom_prompt_file).read_text(encoding="utf-8")
    messages = generate_files(
        output_base=Path(args.output),
        ai_language=args.language,
        project_name=args.project_name,
        targets=targets,
        overwrite=args.overwrite,
        schema_dir=schema_dir,
        copy_schema=not args.no_copy_schema,
        create_log=args.create_log,
        role_date=args.date.strip() or None,
        scope_paths=args.scope_path or None,
        export_as_zip=args.export_as_zip,
        export_dir=Path(args.export_dir) if args.export_dir else None,
        export_prompt_text=prompt_text,
        selected_reference_ids=args.reference,
        selected_operation_role_ids=args.operation_role,
        include_imports=args.include_imports,
        custom_prompt_text=custom_prompt_text,
        include_dependency_manifests=args.include_dependency_manifests,
        changed_files_only=args.changed_files_only,
        compact_export=args.compact_export,
        compact_export_context={"source": "cli"} if args.compact_export else None,
    )
    for message in messages:
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
