"""Publish reviewed successful examples and interface walkthroughs only."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

if __package__:
    from .site_display_media import approved_media_paths
else:
    from site_display_media import approved_media_paths

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_FIELDS = ("id", "request", "answer", "frames", "records",
                   "pixel_predictions", "source_sha256", "selector")


def evidence_sha(item):
    data = {key: item.get(key) for key in EVIDENCE_FIELDS}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def main():
    site = ROOT / "site"
    catalog = json.loads((site / "catalog.json").read_text())
    review_path = ROOT / "reports/site/demo-outcome-review.json"
    review = json.loads(review_path.read_text())
    decisions = {item["id"]: item for item in review["items"]}
    if len(decisions) != len(review["items"]):
        raise ValueError("Duplicate outcome review IDs")
    selected, excluded = [], []
    for item in catalog["items"]:
        decision = decisions[item["id"]]
        if evidence_sha(item) != decision["evidence_sha256"]:
            raise ValueError(f"Review does not match displayed evidence: {item['id']}")
        if decision["status"] not in ("success", "walkthrough"):
            excluded.append(item["id"])
            continue
        if decision["status"] == "walkthrough" and item["evidence_kind"] != "interface_walkthrough":
            raise ValueError("A model replay cannot be approved as a walkthrough")
        item.pop("decision_view", None)
        item["source_url"] = "https://github.com/Zefan-Cai/Open-Jev/blob/main/" + item["source_path"]
        item["showcase_review"] = {key: decision[key] for key in
                                   ("status", "reason", "evidence_sha256")}
        selected.append(item)
    if not selected:
        raise ValueError("No reviewed demos available")
    catalog["items"] = selected
    sources = sorted({source for item in selected for source in item.get("covered_source_ids", [])})
    mixture_sources = sorted(set(catalog["source_coverage"]["mixture_task_sources"]) & set(sources))
    catalog["counts"] = {
        "items": len(selected), **dict(Counter(item["evidence_kind"] for item in selected)),
        "mixture_task_sources": len(mixture_sources),
        "new_control_sources": len(set(sources) - set(mixture_sources)),
        "community_interfaces": sum(bool(item.get("community_interface")) for item in selected),
        "recipe_forms": len({recipe for item in selected for recipe in item.get("covers_recipes", [])}),
        "reasoning_subdomains": sum(item["category"] == "Reasoning" for item in selected),
        "painting_representations": sum(item["id"].startswith("painting-") for item in selected),
    }
    catalog["source_coverage"] = {"mixture_task_sources": mixture_sources, "all_source_ids": sources,
        "meaning": "Selected showcase coverage only. Interface walkthroughs have no model result; full evaluation metrics are reported separately."}
    catalog["selection_policy"] = "Reviewed successful examples and interface walkthroughs selected for the showcase. This outcome-selected gallery is not a representative evaluation or success-rate estimate. Original evaluation records remain unchanged."
    catalog["showcase_selection"] = {"policy": "reviewed_success_or_interface", "review": str(review_path.relative_to(ROOT)),
        "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "reviewed_at": datetime.now(timezone.utc).isoformat()}

    manifest_path = site / "media/gameplay-verification.json"
    manifest = json.loads(manifest_path.read_text())
    selected_ids = {item["id"] for item in selected}
    manifest["items"] = [item for item in manifest["items"]
                         if item["id"].removeprefix("gameplay-") in selected_ids]
    if not manifest["items"] or not all(item["episode_metrics"].get("success") is True for item in manifest["items"]):
        raise ValueError("Published game recordings must have achieved their goals")
    if {c["id"] for c in manifest["overview"]["chapters"]} != {i["id"] for i in manifest["items"]}:
        raise ValueError("Overview chapters differ from approved game episodes")
    manifest["count"] = len(manifest["items"])
    manifest["total_video_bytes"] = sum(item["bytes"] for item in manifest["items"])
    manifest["selection"] = "Successful complete seed-10001 episodes selected from the original saved 9B pilot; outcomes were used for showcase selection."
    keep = approved_media_paths(site, selected, catalog["overview"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    catalog["overview"]["source_sha256"] = manifest_sha
    catalog["overview"]["source_url"] = "https://github.com/Zefan-Cai/Open-Jev/blob/main/site/media/gameplay-verification.json"
    catalog["provenance"]["gameplay_attachment"]["manifest_sha256"] = manifest_sha

    # The original technical montage manifest describes the uncurated edition.
    # Its historical bytes remain in Git and the release audit reports.
    old_manifest = site / "media/verification.json"
    if old_manifest.exists():
        if old_manifest.is_symlink():
            raise ValueError("Unsafe technical manifest path")
        old_manifest.unlink()

    removed = []
    for path in sorted((site / "media").iterdir()):
        if path.suffix not in (".mp4", ".jpg", ".vtt", ".txt"):
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unsafe media path: {path}")
        relative = path.relative_to(site).as_posix()
        if relative not in keep:
            removed.append(relative)
            path.unlink()
    (site / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selected": len(selected), "excluded_ids": excluded, "removed_media_files": removed}))


if __name__ == "__main__":
    main()
