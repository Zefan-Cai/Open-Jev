"""Validate display-only media that is referenced outside the source catalog."""

import hashlib
import json
from pathlib import Path


PHONE_DISPLAY_ID = "phone-extraction"
PHONE_DISPLAY_FILES = frozenset({
    "media/phone-extraction-en.mp4",
    "media/phone-extraction-en.jpg",
    "media/phone-extraction-en.vtt",
    "media/phone-extraction-en.txt",
})
MEDIA_FIELDS = ("video", "poster", "captions", "transcript")


def approved_media_paths(site, items, overview):
    """Return catalog paths plus verified derivatives used by selected items."""
    approved = {item[field] for item in items + [overview] for field in MEDIA_FIELDS}
    selected_ids = {item["id"] for item in items}
    if PHONE_DISPLAY_ID not in selected_ids:
        return approved

    site_root = site.resolve()
    manifest_path = site / "media/phone-extraction-en.json"
    if (manifest_path.is_symlink() or not manifest_path.is_file()
            or not manifest_path.resolve().is_relative_to(site_root)):
        raise ValueError("Missing or unsafe English phone display manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("inference_performed") is not False:
        raise ValueError("English phone display must not claim inference")
    if manifest.get("request_and_offsets_unchanged") is not True:
        raise ValueError("English phone display must preserve the original request and offsets")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != PHONE_DISPLAY_FILES:
        raise ValueError("Unexpected English phone display media set")

    for relative, expected_sha256 in files.items():
        path = site / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(site_root):
            raise ValueError(f"Missing or unsafe English phone display media: {relative}")
        data = path.read_bytes()
        if path.suffix in (".vtt", ".txt"):
            data = data.replace(b"\r\n", b"\n")
            if b"\r" in data:
                raise ValueError(f"Unexpected line ending in English phone display media: {relative}")
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"English phone display media hash mismatch: {relative}")

    approved.update(files)
    return approved
