from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from typing import Any

from mcp_doc_redaction.schemas import ArtifactBundle, ArtifactFile


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _safe_filename(path: str) -> str:
    base = os.path.basename(path.strip().rstrip("/"))
    return base or "artifact"


@dataclass(frozen=True)
class BundledResult:
    manifest: ArtifactBundle
    zip_bytes: bytes


def bundle_artifacts(
    *,
    produced_by: str,
    base_url: str,
    downloaded: dict[str, bytes],
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> BundledResult:
    """
    downloaded: mapping of server_path -> bytes
    """
    notes_out = list(notes or [])
    extra_out: dict[str, Any] = dict(extra or {})

    files: list[ArtifactFile] = []
    # De-dupe by content hash; keep first filename encountered.
    seen_hashes: set[str] = set()
    deduped: list[tuple[str, str, bytes]] = []
    for server_path, b in downloaded.items():
        digest = sha256_bytes(b)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        deduped.append((server_path, digest, b))

    # Build zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for server_path, digest, b in deduped:
            name = _safe_filename(server_path)
            # Avoid collisions inside zip
            if any(f.filename == name for f in files):
                root, ext = os.path.splitext(name)
                name = f"{root}_{digest[:8]}{ext}"
            zf.writestr(name, b)
            files.append(
                ArtifactFile(
                    filename=name,
                    sha256=digest,
                    size_bytes=len(b),
                    source=server_path,
                )
            )

        manifest = ArtifactBundle(
            produced_by=produced_by,
            base_url=base_url,
            files=files,
            notes=notes_out,
            extra=extra_out,
        )
        zf.writestr("manifest.json", manifest.model_dump_json(indent=2))

    return BundledResult(manifest=manifest, zip_bytes=buf.getvalue())


def zip_bytes_to_base64(zip_bytes: bytes) -> str:
    return base64.b64encode(zip_bytes).decode("ascii")

