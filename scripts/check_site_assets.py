"""Verify the published demo catalog, playable media, and caption artifacts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

if __package__:
    from .site_display_media import approved_media_paths
else:
    from site_display_media import approved_media_paths


def check(site):
    catalog = json.loads((site / "catalog.json").read_text())
    assert catalog.get("status") == "complete", "catalog is still being assembled"
    items = catalog["items"]
    assert items, "empty video catalog"
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids)), "duplicate demo IDs"
    kinds = {"model_replay": 0, "interface_walkthrough": 0}
    videos = []
    for item in items:
        assert re.fullmatch(r"[a-z0-9_-]+", item["id"]), item["id"]
        assert item["evidence_kind"] in kinds, item["id"]
        kinds[item["evidence_kind"]] += 1
        assert item["request"], f"missing request: {item['id']}"
        assert item["limitations"], f"missing limitations: {item['id']}"
        assert item["source_path"], f"missing provenance: {item['id']}"
        if item["evidence_kind"] == "interface_walkthrough":
            assert not item.get("answer"), f"interface clip has a model answer: {item['id']}"
        else:
            assert item.get("answer") or item.get("frames"), f"replay without output: {item['id']}"
        for key in ("video", "poster", "captions"):
            relative = Path(item[key])
            path = site / relative
            assert not relative.is_absolute() and ".." not in relative.parts
            assert path.is_file() and not path.is_symlink(), str(path)
        captions = (site / item["captions"]).read_text()
        assert captions.startswith("WEBVTT") and "-->" in captions, item["id"]
        assert (site / item["captions"]).with_suffix(".txt").is_file(), item["id"]
        videos.append(item["video"])
    if catalog.get("overview"):
        for key in ("video", "poster", "captions", "transcript"):
            path = site / catalog["overview"][key]
            assert path.is_file() and path.resolve().is_relative_to(site), str(path)
        assert (site / catalog["overview"]["captions"]).read_text().startswith("WEBVTT")
    if catalog.get("showcase_selection"):
        assert catalog["showcase_selection"]["policy"] == "reviewed_success_or_interface"
        for item in items:
            review = item["showcase_review"]
            assert review["status"] in ("success", "walkthrough"), item["id"]
            evidence = {key: item.get(key) for key in ("id", "request", "answer", "frames", "records", "pixel_predictions", "source_sha256", "selector")}
            digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            assert digest == review["evidence_sha256"], item["id"]
        approved = approved_media_paths(site, items, catalog["overview"])
        published = {path.relative_to(site).as_posix() for path in (site / "media").iterdir()
                     if path.suffix in (".mp4", ".jpg", ".vtt", ".txt")}
        assert published == approved, "Unreviewed or missing media in published directory"
        gameplay = json.loads((site / "media/gameplay-verification.json").read_text())
        assert all(item["episode_metrics"].get("success") is True for item in gameplay["items"])
        assert {item["id"] for item in gameplay["items"]} == {chapter["id"] for chapter in gameplay["overview"]["chapters"]}
    # Verify every media file that will be published.
    videos = [str(path.relative_to(site)) for path in sorted((site / "media").glob("*.mp4"))]
    media = []
    for relative in videos:
        path = site / relative
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ], capture_output=True, text=True, check=True)
        probe = json.loads(result.stdout)
        stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        assert stream["codec_name"] == "h264", relative
        assert stream["pix_fmt"] == "yuv420p", relative
        duration = float(probe["format"]["duration"])
        assert 2 <= duration <= 600, (relative, duration)
        data = path.read_bytes()
        assert 0 < data.find(b"moov") < data.find(b"mdat"), f"missing faststart: {relative}"
        media.append({"path": relative, "duration_seconds": duration,
                      "width": stream["width"], "height": stream["height"],
                      "codec": stream["codec_name"], "pixel_format": stream["pix_fmt"]})
    files = {}
    for path in sorted(site.rglob("*")):
        if path.is_file():
            assert not path.is_symlink(), str(path)
            data = path.read_bytes()
            files[str(path.relative_to(site))] = {
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "status": "passed",
            "domain_videos": len(items), "evidence_kinds": kinds,
            "total_bytes": sum(v["bytes"] for v in files.values()),
            "media": media, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path(__file__).resolve().parents[1] / "site")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.site)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "domain_videos", "evidence_kinds", "total_bytes")}))


if __name__ == "__main__":
    main()
