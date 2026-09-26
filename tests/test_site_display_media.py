import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import curate_demo_site
from scripts.site_display_media import (PHONE_DISPLAY_FILES,
                                        approved_media_paths)


class ApprovedSiteMediaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.site = Path(self.temporary.name) / "site"
        media = self.site / "media"
        media.mkdir(parents=True)
        self.payloads = {
            relative: ("asset:" + relative).encode("utf-8")
            for relative in PHONE_DISPLAY_FILES
        }
        for relative, data in self.payloads.items():
            path = self.site / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.manifest_path = media / "phone-extraction-en.json"
        self.manifest = {
            "inference_performed": False,
            "request_and_offsets_unchanged": True,
            "files": {
                relative: hashlib.sha256(data).hexdigest()
                for relative, data in self.payloads.items()
            },
        }
        self._write_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def _paths(self, items):
        overview = {
            "video": "media/overview.mp4",
            "poster": "media/overview.jpg",
            "captions": "media/overview.vtt",
            "transcript": "media/overview.txt",
        }
        return approved_media_paths(self.site, items, overview)

    def test_selected_phone_item_keeps_exact_manifest_backed_derivatives(self):
        items = [{
            "id": "phone-extraction",
            "video": "media/phone-extraction.mp4",
            "poster": "media/phone-extraction.jpg",
            "captions": "media/phone-extraction.vtt",
            "transcript": "media/phone-extraction.txt",
        }]

        approved = self._paths(items)

        self.assertTrue(PHONE_DISPLAY_FILES.issubset(approved))
        self.assertTrue(all((self.site / path).is_file() for path in PHONE_DISPLAY_FILES))

    def test_derivatives_are_not_kept_when_phone_item_is_not_selected(self):
        items = [{
            "id": "other-demo",
            "video": "media/other.mp4",
            "poster": "media/other.jpg",
            "captions": "media/other.vtt",
            "transcript": "media/other.txt",
        }]

        approved = self._paths(items)

        self.assertTrue(PHONE_DISPLAY_FILES.isdisjoint(approved))

    def test_windows_text_line_endings_match_manifest_hashes(self):
        for relative in PHONE_DISPLAY_FILES:
            if Path(relative).suffix in (".vtt", ".txt"):
                path = self.site / relative
                path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

        approved = self._paths([{"id": "phone-extraction", "video": "v", "poster": "p",
                                 "captions": "c", "transcript": "t"}])

        self.assertTrue(PHONE_DISPLAY_FILES.issubset(approved))

    def test_changed_derivative_is_rejected_before_curation(self):
        relative = next(iter(PHONE_DISPLAY_FILES))
        (self.site / relative).write_bytes(b"different content")

        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self._paths([{"id": "phone-extraction", "video": "v", "poster": "p",
                          "captions": "c", "transcript": "t"}])

    def test_manifest_cannot_approve_an_unexpected_path(self):
        self.manifest["files"]["media/unrelated.mp4"] = "0" * 64
        self._write_manifest()

        with self.assertRaisesRegex(ValueError, "Unexpected English phone display media set"):
            self._paths([{"id": "phone-extraction", "video": "v", "poster": "p",
                          "captions": "c", "transcript": "t"}])

    def test_curation_retains_phone_derivatives_in_temporary_site(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site = root / "site"
            media = site / "media"
            media.mkdir(parents=True)
            phone_files = {
                "video": "media/phone-extraction.mp4",
                "poster": "media/phone-extraction.jpg",
                "captions": "media/phone-extraction.vtt",
                "transcript": "media/phone-extraction.txt",
            }
            demo_files = {
                "video": "media/demo.mp4",
                "poster": "media/demo.jpg",
                "captions": "media/demo.vtt",
                "transcript": "media/demo.txt",
            }
            overview = {
                "video": "media/overview.mp4",
                "poster": "media/overview.jpg",
                "captions": "media/overview.vtt",
                "transcript": "media/overview.txt",
            }
            for relative in (*phone_files.values(), *demo_files.values(),
                             *overview.values(), *PHONE_DISPLAY_FILES,
                             "media/unrelated.mp4"):
                (site / relative).write_bytes(relative.encode("utf-8"))
            display_manifest = {
                "inference_performed": False,
                "request_and_offsets_unchanged": True,
                "files": {
                    relative: hashlib.sha256((site / relative).read_bytes()).hexdigest()
                    for relative in PHONE_DISPLAY_FILES
                },
            }
            (media / "phone-extraction-en.json").write_text(
                json.dumps(display_manifest), encoding="utf-8")

            def item(item_id, evidence_kind, files, answer):
                return {
                    "id": item_id,
                    "evidence_kind": evidence_kind,
                    "source_path": f"examples/{item_id}.json",
                    "category": "Reasoning",
                    "covered_source_ids": [],
                    "covers_recipes": [],
                    "request": "fixture request",
                    "answer": answer,
                    "frames": None,
                    "records": [],
                    "pixel_predictions": [],
                    "source_sha256": "0" * 64,
                    "selector": "fixture",
                    **files,
                }

            phone = item("phone-extraction", "interface_walkthrough", phone_files, None)
            demo = item("demo", "model_replay", demo_files, "fixture result")
            review_items = []
            for selected_item, status in ((phone, "walkthrough"), (demo, "success")):
                review_items.append({
                    "id": selected_item["id"],
                    "status": status,
                    "reason": "fixture approval",
                    "evidence_sha256": curate_demo_site.evidence_sha(selected_item),
                })
            catalog = {
                "items": [phone, demo],
                "overview": overview,
                "source_coverage": {"mixture_task_sources": []},
                "provenance": {"gameplay_attachment": {}},
            }
            (site / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
            reports = root / "reports/site"
            reports.mkdir(parents=True)
            (reports / "demo-outcome-review.json").write_text(
                json.dumps({"items": review_items}), encoding="utf-8")
            gameplay = {
                "items": [{"id": "gameplay-demo", "episode_metrics": {"success": True}, "bytes": 1}],
                "overview": {"chapters": [{"id": "gameplay-demo"}]},
            }
            (media / "gameplay-verification.json").write_text(
                json.dumps(gameplay), encoding="utf-8")

            with patch.object(curate_demo_site, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                curate_demo_site.main()

            self.assertTrue(all((site / path).is_file() for path in PHONE_DISPLAY_FILES))
            self.assertFalse((media / "unrelated.mp4").exists())


if __name__ == "__main__":
    unittest.main()
