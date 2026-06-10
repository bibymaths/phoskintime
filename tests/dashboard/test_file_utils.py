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

from io import BytesIO


from dashboard.file_utils import (
    create_upload_dir,
    detect_duplicate_filenames,
    preview_table,
    sanitize_filename,
    save_uploaded_file,
    validate_existing_file,
    validate_upload_filename,
)


class DummyUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getbuffer(self):
        return memoryview(self._data)


def test_upload_path_creation_and_filename_sanitization(tmp_path):
    upload_dir = create_upload_dir(tmp_path, "../Bad Run!!")

    assert upload_dir == tmp_path / "dashboard_uploads" / "Bad-Run"
    assert upload_dir.is_dir()
    assert sanitize_filename("../bad file.csv") == "bad-file.csv"


def test_save_uploaded_file_rejects_empty_and_unsupported(tmp_path):
    upload_dir = create_upload_dir(tmp_path, "run")

    with pytest.raises(ValueError, match="empty"):
        save_uploaded_file(DummyUpload("empty.csv", b""), upload_dir)
    assert validate_upload_filename("bad.exe")


def test_duplicate_filename_detection_uses_sanitized_names():
    assert detect_duplicate_filenames(["a file.csv", "a-file.csv"]) == ["a-file.csv"]


def test_preview_csv_tsv_xlsx_helpers(tmp_path):
    pd = pytest.importorskip("pandas")
    csv = tmp_path / "a.csv"
    tsv = tmp_path / "a.tsv"
    xlsx = tmp_path / "a.xlsx"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    tsv.write_text("a\tb\n3\t4\n", encoding="utf-8")
    pd.DataFrame({"a": [5], "b": [6]}).to_excel(xlsx, index=False)

    assert preview_table(csv).iloc[0].to_dict() == {"a": 1, "b": 2}
    assert preview_table(tsv).iloc[0].to_dict() == {"a": 3, "b": 4}
    assert preview_table(xlsx).iloc[0].to_dict() == {"a": 5, "b": 6}


def test_invalid_existing_file_handling(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    unsupported = tmp_path / "x.exe"
    unsupported.write_bytes(b"x")

    assert "File is empty" in validate_existing_file(empty)
    assert any("Unsupported extension" in problem for problem in validate_existing_file(unsupported))
