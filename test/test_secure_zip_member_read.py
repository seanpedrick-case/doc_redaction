import zipfile

import pytest

from tools.secure_path_utils import secure_zip_member_read


def test_secure_zip_member_read_returns_bytes(tmp_path):
    payload = b'{"Blocks": []}'
    zip_path = tmp_path / "textract.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("report.json", payload)

    with zipfile.ZipFile(zip_path, "r") as zf:
        assert secure_zip_member_read(zf, "report.json", tmp_path) == payload


def test_secure_zip_member_read_rejects_traversal(tmp_path):
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../../../outside.json", b"{}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        with pytest.raises(PermissionError):
            secure_zip_member_read(zf, "../../../outside.json", tmp_path)


def test_secure_zip_member_read_rejects_missing_member(tmp_path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("report.json", b"{}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        with pytest.raises(FileNotFoundError):
            secure_zip_member_read(zf, "missing.json", tmp_path)
