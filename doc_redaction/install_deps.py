"""
Install helper for external system dependencies (Tesseract + Poppler).

This is intended to reduce setup friction, especially on Windows where users may
not have admin rights or package managers available.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
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

    def _attempt_download_curl() -> None:
        """
        Windows fallback: use curl.exe with browser-like UA.
        """
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.unlink(missing_ok=True)
        p = subprocess.run(
            [
                "curl.exe",
                "-L",
                "--fail",
                "--silent",
                "--show-error",
                "-A",
                "Mozilla/5.0",
                "-o",
                str(tmp),
                url,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode != 0:
            tmp.unlink(missing_ok=True)
            stderr = (p.stderr or p.stdout or "").strip()
            raise RuntimeError(
                "curl.exe download fallback failed "
                f"(exit {p.returncode}) for {url}: {stderr[:300]}"
            )
        if not tmp.exists() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"curl.exe download fallback produced no file for {url}")
        tmp.replace(dest)

    def _attempt_download_powershell() -> None:
        """
        Windows fallback: use Invoke-WebRequest, which can behave better than
        Python's urllib in environments with TLS inspection/proxies.
        """
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.unlink(missing_ok=True)

        # Single-quote escaping for PowerShell string literals.
        url_ps = url.replace("'", "''")
        tmp_ps = str(tmp).replace("'", "''")
        cmd = (
            "$ProgressPreference='SilentlyContinue'; "
            f"$headers = @{{ 'User-Agent'='Mozilla/5.0'; 'Accept'='*/*' }}; "
            f"Invoke-WebRequest -Headers $headers -Uri '{url_ps}' -OutFile '{tmp_ps}' -MaximumRedirection 10"
        )
        p = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                cmd,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode != 0:
            tmp.unlink(missing_ok=True)
            stderr = (p.stderr or p.stdout or "").strip()
            raise RuntimeError(
                "PowerShell download fallback failed "
                f"(exit {p.returncode}) for {url}: {stderr[:300]}"
            )
        if not tmp.exists() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"PowerShell download fallback produced no file for {url}"
            )
        tmp.replace(dest)

    # NOTE: We deliberately validate headers + final size because corporate proxies
    # sometimes allow the initial bytes through and then truncate the stream.
    # `zipfile.is_zipfile()` will fail in that case, but the first bytes may still be `PK`.
    def _attempt_download() -> tuple[int | None, str | None, int | None]:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "doc_redaction-install-deps/1.0 (+https://github.com/seanpedrick-case/doc_redaction)",
                "Accept": "*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = getattr(resp, "status", None)
            if status is None:
                status = resp.getcode()
            if status != 200:
                raise RuntimeError(f"Download failed (HTTP {status}) for {url}")

            content_type = resp.headers.get("Content-Type")
            content_length_raw = resp.headers.get("Content-Length")
            content_length: int | None = None
            if content_length_raw and str(content_length_raw).strip().isdigit():
                content_length = int(str(content_length_raw).strip())

            tmp = dest.with_suffix(dest.suffix + ".part")
            if tmp.exists():
                tmp.unlink(missing_ok=True)

            bytes_written = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1 MiB
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_written += len(chunk)

            if content_length is not None and bytes_written != content_length:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    "Download appears truncated "
                    f"(got {bytes_written} bytes, expected {content_length} bytes). "
                    "This is commonly caused by a corporate proxy/filter or interrupted connection. "
                    "Try another network/VPN, or install Poppler via conda/chocolatey and re-run with --verify-only."
                )

            tmp.replace(dest)
            return status, content_type, content_length

    last_err: Exception | None = None
    for _ in range(3):
        try:
            _attempt_download()
            return
        except Exception as e:
            last_err = e
            # Best-effort cleanup (avoid leaving partial artifacts).
            dest.unlink(missing_ok=True)
            part = dest.with_suffix(dest.suffix + ".part")
            part.unlink(missing_ok=True)

    if platform.system().lower() == "windows":
        curl_err: Exception | None = None
        try:
            _attempt_download_curl()
            return
        except Exception as e:
            curl_err = e
        try:
            _attempt_download_powershell()
            return
        except Exception as ps_err:
            raise RuntimeError(
                f"Download failed after urllib retries, curl fallback, and PowerShell fallback for {url}: "
                f"urllib error={last_err}; curl error={curl_err}; powershell error={ps_err}"
            ) from ps_err

    raise RuntimeError(
        f"Download failed after 3 attempts for {url}: {last_err}"
    ) from last_err


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
            "User-Agent": "doc_redaction-install-deps/1.0 (+https://github.com/seanpedrick-case/doc_redaction)",
        },
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Unexpected GitHub API response for {repo}/releases/latest (not JSON)"
        ) from e
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected GitHub API payload for {repo}: {type(data)}")
    if "message" in data and "assets" not in data:
        raise RuntimeError(f"GitHub API error for {repo}: {data.get('message', data)}")
    assets = data.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError(f"GitHub API response for {repo} has no assets list")
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = asset.get("name", "")
        if (
            isinstance(name, str)
            and name.startswith(asset_prefix)
            and name.lower().endswith(".zip")
        ):
            url = asset.get("browser_download_url")
            if isinstance(url, str) and url.startswith("http"):
                return url
    raise RuntimeError(
        f"Could not locate a {asset_prefix}*.zip asset in latest release for {repo}"
    )


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest_dir)
    except zipfile.BadZipFile as e:
        raise RuntimeError(
            f"Not a valid zip archive: {zip_path}. "
            "The download may have been blocked or replaced by a proxy (HTML error page). "
            "Try again with --force, use another network/VPN, or install Poppler manually."
        ) from e
    except PermissionError as e:
        raise RuntimeError(
            f"Permission denied while extracting {zip_path} into {dest_dir}. "
            "This usually means a file in that folder is locked by another process "
            "(Explorer preview, antivirus scanner, OneDrive sync, or a running app). "
            "Close apps that may be using that folder and re-run with --force."
        ) from e


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


def _config_path_value(path: Path, cwd: Path) -> str:
    """
    Prefer a path relative to cwd; fall back to absolute when outside cwd.
    """
    try:
        return str(path.relative_to(cwd)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


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
    if not zipfile.is_zipfile(zip_path):
        size = None
        try:
            size = zip_path.stat().st_size
        except Exception:
            size = None
        snippet = zip_path.read_bytes()[:400]
        try:
            text_preview = snippet.decode("utf-8", errors="replace")
        except Exception:
            text_preview = repr(snippet)
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded Poppler archive is not a valid zip file "
            f"(URL: {url}). Downloaded size: {size if size is not None else 'unknown'} bytes. "
            f"First bytes: {snippet[:80]!r}. "
            f"Preview: {text_preview[:200]!r}. "
            "This often means a corporate proxy or filter returned an HTML page instead of the file. "
            "Try another network, browser-download the zip from the release page and extract it under "
            f"{install_root} (ensure it contains a Library/bin folder), "
            "or install Poppler via conda/chocolatey and re-run with --verify-only."
        )

    # Use a fresh extraction directory each run so we never need to overwrite
    # potentially locked DLLs from previous attempts.
    extract_root = install_root / f"extracted_{time.time_ns()}"
    _extract_zip(zip_path, extract_root)

    poppler_bin = _find_poppler_bin(extract_root)
    if not poppler_bin:
        raise RuntimeError(
            f"Poppler extracted but could not locate Library/bin under {extract_root}"
        )
    return poppler_bin


def _tesseract_latest_installer_url() -> str:
    """
    Use the GitHub API to get the latest (5.50) tesseract installer URL for Windows.
    """
    # Use the GitHub API to get release data
    api_url = "https://api.github.com/repos/tesseract-ocr/tesseract/releases/tags/5.5.0"
    req = urllib.request.Request(api_url, headers={"User-Agent": "python-script"})

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())

    # Search through the 'assets' list for the .exe
    for asset in data.get("assets", []):
        name = asset.get("name", "")
        print("name: ", name)
        if re.match(r"tesseract-ocr-w64-setup-.*\.exe", name):
            print("found tesseract download url: ", asset.get("browser_download_url"))
            return asset.get("browser_download_url")
        else:
            print("Not a tesseract installer: ", name)

    raise RuntimeError("Could not find Windows installer in GitHub assets.")


def _find_tesseract_bin_from_prefix(prefix: Path) -> Path | None:
    """
    Conda-forge on Windows typically installs tesseract.exe under:
      <prefix>/Library/bin/tesseract.exe
    """
    candidates = [
        prefix / "Library" / "bin",
        prefix / "Scripts",
    ]
    for c in candidates:
        if (c / "tesseract.exe").exists() or (c / "tesseract").exists():
            return c
    return None


def _install_tesseract_with_conda_prefix() -> Path | None:
    """
    Try a non-admin install into the current conda environment prefix.
    Returns the folder containing tesseract if successful.
    """
    prefix = Path(sys.prefix).resolve()
    existing = _find_tesseract_bin_from_prefix(prefix)
    if existing:
        return existing

    conda_exe = os.environ.get("CONDA_EXE")
    if not conda_exe:
        inferred = Path(sys.executable).resolve().parents[2] / "Scripts" / "conda.exe"
        if inferred.exists():
            conda_exe = str(inferred)
    if not conda_exe:
        return None

    cmd = [
        str(conda_exe),
        "install",
        "-y",
        "-c",
        "conda-forge",
        "--prefix",
        str(prefix),
        "tesseract",
    ]
    p = subprocess.run(cmd, check=False)
    if p.returncode != 0:
        return None
    return _find_tesseract_bin_from_prefix(prefix)


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
    except OSError as e:
        if getattr(e, "winerror", None) == 740:
            print(
                "\nTesseract installer requires elevation on this machine (WinError 740).\n"
                "Trying non-admin conda install into the current environment..."
            )
            conda_tesseract = _install_tesseract_with_conda_prefix()
            if conda_tesseract:
                return conda_tesseract
            raise RuntimeError(
                "Tesseract installer requires elevation and conda fallback did not succeed. "
                "Please install tesseract via conda manually:\n"
                "  conda install -c conda-forge tesseract\n"
                "Then re-run with --verify-only."
            ) from e
        raise
    except Exception:
        # Fall back to interactive installer (still pointing user at a non-admin directory).
        print(
            "\nTesseract silent install did not complete successfully.\n"
            "Launching the installer interactively.\n"
            f"When prompted, choose an install directory under:\n  {target_dir}\n"
        )
        try:
            subprocess.run([str(exe_path)], check=False)
        except OSError as e:
            if getattr(e, "winerror", None) == 740:
                print(
                    "\nInteractive installer also requires elevation.\n"
                    "Trying non-admin conda install into the current environment..."
                )
                conda_tesseract = _install_tesseract_with_conda_prefix()
                if conda_tesseract:
                    return conda_tesseract
                raise RuntimeError(
                    "Tesseract installer requires elevation and conda fallback did not succeed. "
                    "Please install tesseract via conda manually:\n"
                    "  conda install -c conda-forge tesseract\n"
                    "Then re-run with --verify-only."
                ) from e
            raise

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
        help="Re-download/reinstall into the local redaction_deps folder.",
    )
    p.add_argument(
        "--install-dir",
        default="redaction_deps",
        help="Local install root (relative to current working directory). Default: redaction_deps",
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
        poppler_value = _config_path_value(poppler_bin, cwd)
        _env_upsert(app_config_path, "POPPLER_FOLDER", poppler_value)

    if not status.tesseract_ok or args.force:
        tesseract_dir = _install_windows_tesseract(base_dir=base_dir, force=args.force)
        tesseract_value = _config_path_value(tesseract_dir, cwd)
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
