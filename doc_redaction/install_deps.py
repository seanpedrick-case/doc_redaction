"""
Install helper for external system dependencies (Tesseract + Poppler).

This is intended to reduce setup friction, especially on Windows where users may
not have admin rights or package managers available.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DepStatus:
    tesseract_ok: bool
    poppler_ok: bool
    tesseract_version: str | None = None
    poppler_version: str | None = None


def _run_and_capture(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return (127, "")
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return (int(p.returncode), out.strip())


def _detect_status() -> DepStatus:
    t_code, t_out = _run_and_capture(["tesseract", "--version"])
    p_code, p_out = _run_and_capture(["pdftoppm", "-v"])

    t_ok = t_code == 0 and bool(t_out.strip())
    p_ok = p_code == 0 and bool(p_out.strip())

    t_ver = None
    if t_ok:
        # Usually: "tesseract v5.x.x ..."
        m = re.search(r"tesseract(?:\s+v)?\s*([0-9]+(?:\.[0-9]+){1,3})", t_out, re.I)
        t_ver = m.group(1) if m else None

    p_ver = None
    if p_ok:
        # Usually: "pdftoppm version 25.xx.x"
        m = re.search(r"pdftoppm\s+version\s+([0-9]+(?:\.[0-9]+){1,3})", p_out, re.I)
        p_ver = m.group(1) if m else None

    return DepStatus(
        tesseract_ok=t_ok,
        poppler_ok=p_ok,
        tesseract_version=t_ver,
        poppler_version=p_ver,
    )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "doc_redaction-install-deps/1.0 (+https://github.com/seanpedrick-case/doc_redaction)"
        },
    )
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _github_latest_release_zip(repo: str, asset_prefix: str = "Release-") -> str:
    """
    Return a browser-download URL for the first zip asset matching the prefix
    from the latest GitHub release for `owner/name`.
    """
    api = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(
        api,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "doc_redaction-install-deps/1.0",
        },
    )
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8", errors="replace")

    # Minimal JSON parsing to avoid adding dependencies.
    # We search for assets "name":"Release-....zip" and then find its browser_download_url.
    # This is intentionally simple and resilient to additional fields.
    asset_re = re.compile(
        r'"name"\s*:\s*"('
        + re.escape(asset_prefix)
        + r'[^"]+?\.zip)".+?"browser_download_url"\s*:\s*"([^"]+)"',
        re.S,
    )
    m = asset_re.search(body)
    if not m:
        raise RuntimeError(
            f"Could not locate a {asset_prefix}*.zip asset in latest release JSON for {repo}"
        )
    return m.group(2)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)


def _find_poppler_bin(root: Path) -> Path | None:
    """
    Poppler-windows zips usually contain a `Library/bin` folder somewhere.
    """
    # Common: <extract_root>/poppler-xx.xx.x/Library/bin or <extract_root>/Library/bin
    candidates = [
        root / "Library" / "bin",
    ]
    for c in candidates:
        if (c / "pdftoppm.exe").exists() or (c / "pdftoppm").exists():
            return c

    for c in root.rglob("Library/bin"):
        if (c / "pdftoppm.exe").exists() or (c / "pdftoppm").exists():
            return c
    return None


def _env_upsert(env_path: Path, key: str, value: str) -> None:
    """
    Update or append a KEY=VALUE entry in a dotenv-style file.
    Preserves existing lines and comments.
    """
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if env_path.exists():
        raw = env_path.read_text(encoding="utf-8", errors="replace").splitlines()
    else:
        raw = []

    key_re = re.compile(rf"^\s*{re.escape(key)}\s*=")
    out_lines: list[str] = []
    replaced = False
    for line in raw:
        if key_re.match(line):
            out_lines.append(f"{key}={value}")
            replaced = True
        else:
            out_lines.append(line)

    if not replaced:
        if out_lines and out_lines[-1].strip() != "":
            out_lines.append("")
        out_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")


def _install_windows_poppler(base_dir: Path, force: bool) -> Path:
    """
    Install Poppler into base_dir/poppler by downloading and extracting the latest release ZIP.
    Returns the path to the Poppler `bin` directory (for `POPPLER_FOLDER`).
    """
    install_root = base_dir / "poppler"
    if force and install_root.exists():
        shutil.rmtree(install_root, ignore_errors=True)

    install_root.mkdir(parents=True, exist_ok=True)
    url = _github_latest_release_zip(
        "oschwartz10612/poppler-windows", asset_prefix="Release-"
    )
    zip_path = install_root / "poppler.zip"
    _download(url, zip_path)

    extract_root = install_root / "extracted"
    if extract_root.exists() and force:
        shutil.rmtree(extract_root, ignore_errors=True)
    _extract_zip(zip_path, extract_root)

    poppler_bin = _find_poppler_bin(extract_root)
    if not poppler_bin:
        raise RuntimeError(
            f"Poppler extracted but could not locate Library/bin under {extract_root}"
        )
    return poppler_bin


def _tesseract_latest_installer_url() -> str:
    """
    Best-effort: scrape the UB Mannheim directory listing and pick the newest
    64-bit installer filename.
    """
    index_url = "https://digi.bib.uni-mannheim.de/tesseract/"
    req = urllib.request.Request(
        index_url,
        headers={"User-Agent": "doc_redaction-install-deps/1.0"},
    )
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    # Example: tesseract-ocr-w64-setup-5.5.0.20241111.exe
    names = re.findall(r"(tesseract-ocr-w64-setup-[0-9][^\"'>\s]*?\.exe)", html)
    if not names:
        raise RuntimeError(
            "Could not find a tesseract-ocr-w64-setup-*.exe in the listing"
        )
    # Sort by string: version/date suffix makes this generally safe for these filenames.
    newest = sorted(set(names))[-1]
    return index_url.rstrip("/") + "/" + newest


def _install_windows_tesseract(base_dir: Path, force: bool) -> Path:
    """
    Download the latest UB Mannheim installer and attempt a per-user install
    into base_dir/tesseract (no admin).

    Returns the folder that contains the tesseract executable (for `TESSERACT_FOLDER`).
    """
    install_root = base_dir / "tesseract"
    install_root.mkdir(parents=True, exist_ok=True)

    # We use a stable filename so subsequent runs can reuse it.
    exe_path = install_root / "tesseract-installer.exe"
    if force and exe_path.exists():
        exe_path.unlink(missing_ok=True)

    if not exe_path.exists():
        url = _tesseract_latest_installer_url()
        _download(url, exe_path)

    # NSIS-style silent install often supports /S and /D=path (must be last).
    # We do best-effort and verify afterwards.
    target_dir = install_root / "install"
    if force and target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    silent_args = [str(exe_path), "/S", f"/D={str(target_dir)}"]
    try:
        p = subprocess.run(silent_args, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"Installer returned {p.returncode}")
    except Exception:
        # Fall back to interactive installer (still pointing user at a non-admin directory).
        print(
            "\nTesseract silent install did not complete successfully.\n"
            "Launching the installer interactively.\n"
            f"When prompted, choose an install directory under:\n  {target_dir}\n"
        )
        subprocess.run([str(exe_path)], check=False)

    # Try to locate tesseract.exe under the chosen install folder.
    candidates = [target_dir, target_dir / "tesseract", target_dir / "Tesseract-OCR"]
    for c in candidates:
        if (c / "tesseract.exe").exists() or (c / "tesseract").exists():
            return c

    for c in target_dir.rglob("tesseract.exe"):
        return c.parent

    raise RuntimeError(
        f"Tesseract install completed but tesseract.exe not found under {target_dir}"
    )


def _prepend_path(folder: Path) -> None:
    folder_str = str(folder)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if folder_str not in parts:
        os.environ["PATH"] = folder_str + os.pathsep + current


def _validate_with_added_paths(
    tesseract_dir: Path | None, poppler_bin: Path | None
) -> DepStatus:
    if tesseract_dir:
        _prepend_path(tesseract_dir)
    if poppler_bin:
        _prepend_path(poppler_bin)
    return _detect_status()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="doc_redaction_install_deps",
        description="Install or verify external dependencies (Tesseract + Poppler).",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only check whether dependencies are already available.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download/reinstall into the local third_party folder.",
    )
    p.add_argument(
        "--install-dir",
        default="third_party",
        help="Local install root (relative to current working directory). Default: third_party",
    )
    p.add_argument(
        "--app-config-path",
        default=os.environ.get("APP_CONFIG_PATH", "config/app_config.env"),
        help="Where to write TESSERACT_FOLDER/POPPLER_FOLDER. Default: config/app_config.env (or APP_CONFIG_PATH if set).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cwd = Path.cwd()
    base_dir = (cwd / args.install_dir).resolve()
    app_config_path = (cwd / args.app_config_path).resolve()

    status = _detect_status()
    if status.tesseract_ok and status.poppler_ok:
        print(
            "Dependencies already available.\n"
            f"- Tesseract: ok{f' ({status.tesseract_version})' if status.tesseract_version else ''}\n"
            f"- Poppler:   ok{f' ({status.poppler_version})' if status.poppler_version else ''}"
        )
        return 0

    if args.verify_only:
        print(
            "Dependency check:\n"
            f"- Tesseract: {'ok' if status.tesseract_ok else 'missing'}\n"
            f"- Poppler:   {'ok' if status.poppler_ok else 'missing'}"
        )
        return 0 if (status.tesseract_ok and status.poppler_ok) else 2

    system = platform.system().lower()
    if system != "windows":
        print(
            "Automated installation is currently implemented for Windows only.\n"
            "On macOS/Linux, please install Tesseract and Poppler using your preferred method,\n"
            "then re-run with --verify-only."
        )
        return 2

    print(f"Installing dependencies under: {base_dir}")
    base_dir.mkdir(parents=True, exist_ok=True)

    poppler_bin: Path | None = None
    tesseract_dir: Path | None = None

    if not status.poppler_ok or args.force:
        poppler_bin = _install_windows_poppler(base_dir=base_dir, force=args.force)
        # Store relative path if possible (works best with tools.config security checks)
        poppler_value = str(poppler_bin.relative_to(cwd)).replace("\\", "/")
        _env_upsert(app_config_path, "POPPLER_FOLDER", poppler_value)

    if not status.tesseract_ok or args.force:
        tesseract_dir = _install_windows_tesseract(base_dir=base_dir, force=args.force)
        tesseract_value = str(tesseract_dir.relative_to(cwd)).replace("\\", "/")
        _env_upsert(app_config_path, "TESSERACT_FOLDER", tesseract_value)

    # Validate in-process by prepending PATH so the checks can find the binaries
    # without requiring the user to restart their shell.
    status2 = _validate_with_added_paths(
        tesseract_dir=tesseract_dir, poppler_bin=poppler_bin
    )

    print("\nPost-install verification:")
    print(f"- Tesseract: {'ok' if status2.tesseract_ok else 'missing'}")
    print(f"- Poppler:   {'ok' if status2.poppler_ok else 'missing'}")
    print(f"\nWrote configuration to: {app_config_path}")

    if not (status2.tesseract_ok and status2.poppler_ok):
        print(
            "\nOne or more dependencies still appear unavailable in this shell.\n"
            "If you installed Tesseract interactively, ensure it was installed under the shown directory.\n"
            "You can also run: python -m doc_redaction.install_deps --verify-only"
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
