"""Build a source distribution without live indexes, uploads or credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

DIRECTORIES = ("scripts", "tests", "docs", "data", ".codex/agents", ".streamlit")
EXCLUDED = {
    "__pycache__", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "qdrant", "uploads", "models", ".git", ".backup_indexes",
}
WEIGHTS = {".gguf", ".safetensors", ".bin", ".pt", ".pth", ".onnx"}


def is_reparse(path: Path) -> bool:
    return bool(getattr(path.lstat(), "st_file_attributes", 0) & 1024)


def release_files(root: Path) -> list[Path]:
    """Select source files; never follow symbolic links or Windows junctions."""
    candidates = list(root.glob("*.py")) + list(root.glob("requirements*.txt"))
    candidates += [root / name for name in ("pyproject.toml", "README.md", ".editorconfig")]
    for directory in DIRECTORIES:
        base = root / directory
        if not base.is_dir() or base.is_symlink() or is_reparse(base):
            continue
        for current, dirs, files in os.walk(base, followlinks=False):
            parent = Path(current)
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED
                             and not (parent / d).is_symlink()
                             and not is_reparse(parent / d))
            candidates.extend(parent / name for name in files)
    return sorted({p for p in candidates if p.is_file() and not p.is_symlink()
                   and not any(part in EXCLUDED for part in p.relative_to(root).parts)
                   and p.suffix.lower() not in WEIGHTS | {".pyc", ".log", ".zip"}
                   and not p.name.lower().startswith((".env", "secrets", "credentials"))},
                  key=lambda p: p.relative_to(root).as_posix())


def build_release(root: Path, output: Path) -> dict[str, str]:
    """Write atomically and embed hashes of the exact bytes archived."""
    root, output = root.resolve(), output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    handle, temporary = tempfile.mkstemp(suffix=".zip", dir=output.parent)
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in release_files(root):
                if path.resolve() == output:
                    continue
                name = path.relative_to(root).as_posix()
                content = path.read_bytes()
                manifest[name] = hashlib.sha256(content).hexdigest()
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, content)
            archive.writestr(zipfile.ZipInfo("RELEASE_MANIFEST.json"),
                             json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Мы обнаружили повреждение архива")
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Мы собираем чистый релиз RAG")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("release_final_rag.zip"))
    args = parser.parse_args()
    manifest = build_release(args.root, args.output)
    print(f"Мы собрали {args.output}: {len(manifest)} файлов")


if __name__ == "__main__":
    main()
