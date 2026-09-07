"""Check release isolation and presentation source validation."""

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.build_release import build_release
from scripts.generate_presentation import read_slides


def test_release_excludes_private_and_generated_files(tmp_path: Path) -> None:
    for name in ("app.py", "data/documents/public.txt", "data/documents/uploads/private.txt",
                 "data/qdrant/storage.json", ".env", "docs/model.gguf",
                 "tests/__pycache__/test.pyc", ".streamlit/secrets.toml",
                 ".codex/agents/review.toml"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    manifest = build_release(tmp_path, tmp_path / "release.zip")
    assert set(manifest) == {"app.py", "data/documents/public.txt", ".codex/agents/review.toml"}
    assert (tmp_path / "data/qdrant/storage.json").exists()
    with zipfile.ZipFile(tmp_path / "release.zip") as archive:
        assert archive.testzip() is None
        stored = json.loads(archive.read("RELEASE_MANIFEST.json"))
        assert stored == manifest
        for name, digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest


def test_release_is_reproducible(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('our release')", encoding="utf-8")
    build_release(tmp_path, tmp_path / "one.zip")
    build_release(tmp_path, tmp_path / "two.zip")
    assert (tmp_path / "one.zip").read_bytes() == (tmp_path / "two.zip").read_bytes()


def test_presentation_reads_report_appendix(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text("# Report\n<!-- SLIDES -->\n### Our slide\n- Our claim\n", encoding="utf-8")
    assert read_slides(report) == [("Our slide", ["Our claim"])]


def test_presentation_rejects_missing_appendix(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text("# Report", encoding="utf-8")
    with pytest.raises(ValueError):
        read_slides(report)
