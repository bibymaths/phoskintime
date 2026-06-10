from __future__ import annotations

from zipfile import ZipFile
from io import BytesIO

import pytest

from dashboard.file_utils import create_result_zip, filter_by_suffix, human_size, read_text_preview, resolve_directory


def test_create_result_zip_archives_files_without_writing_zip(tmp_path):
    root = tmp_path / "run"
    (root / "tables").mkdir(parents=True)
    (root / "tables" / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (root / "plots").mkdir()
    (root / "plots" / "fit.png").write_bytes(b"png")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "ignored.pyc").write_bytes(b"cache")

    archive_bytes = create_result_zip(root)

    with ZipFile(BytesIO(archive_bytes)) as archive:
        assert sorted(archive.namelist()) == ["plots/fit.png", "tables/a.csv"]
    assert not list(root.glob("*.zip"))


def test_read_text_preview_truncates(tmp_path):
    path = tmp_path / "console.log"
    path.write_text("abcdef", encoding="utf-8")

    text, truncated = read_text_preview(path, max_bytes=3)

    assert text == "abc"
    assert truncated is True


def test_filter_by_suffix_and_human_size(tmp_path):
    csv = tmp_path / "a.csv"
    png = tmp_path / "b.png"
    csv.write_text("x", encoding="utf-8")
    png.write_bytes(b"p")

    assert filter_by_suffix([png, csv], {".csv"}) == [csv]
    assert human_size(1024) == "1.0 KB"


def test_resolve_directory_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_directory(tmp_path / "missing")
