"""Repository-owned guardrails for the Neo worker manifest candidate.

The complete JSON Schema stays in rhobear-product-architecture; this test keeps
Neo's local identity, stable handles, and evidence boundary from drifting.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / ".rhobear" / "surfaces" / "neo.worker.json"
VALID_STATES = {
    "AVAILABLE", "MORPHED", "MOVED", "INTENTIONAL_OMIT",
    "PLATFORM_UNAVAILABLE", "REGRESSION", "UNVERIFIED",
}
STABLE_ID = re.compile(r"^F-(?:00[1-9]|0[1-9][0-9]|[1-9][0-9]{2,})$")
SEMANTIC_ID = re.compile(r"^[a-z][a-z0-9.]*\.[a-z0-9.]+$")


class NeoSurfaceManifestTests(unittest.TestCase):
    def test_worker_manifest_candidate_contract(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "1.4.0")
        self.assertEqual(manifest["product_id"], "neo")
        self.assertEqual(manifest["repo_role"], "service")
        self.assertEqual(set(manifest["surfaces"]), {"neo.worker"})
        surface = manifest["surfaces"]["neo.worker"]
        self.assertIsNone(surface["entrypoint"])
        self._assert_not_future(manifest["last_verified"])
        self._assert_not_future(surface["live_verification"]["timestamp"])

        stable_ids: set[str] = set()
        for semantic_id, feature in manifest["features"].items():
            self.assertRegex(semantic_id, SEMANTIC_ID)
            stable_id = feature["stable_id"]
            self.assertRegex(stable_id, STABLE_ID)
            self.assertNotIn(stable_id, stable_ids)
            stable_ids.add(stable_id)
            self.assertTrue(feature["name"].strip())
            availability = feature["availability"]
            self.assertEqual(set(availability), {"neo.worker"})
            evidence = availability["neo.worker"]
            self.assertIn(evidence["state"], VALID_STATES)
            if evidence["state"] != "AVAILABLE":
                self.assertTrue(evidence.get("reason", "").strip())
            self._assert_not_future(evidence["last_checked"])

        self.assertEqual(stable_ids, {"F-040", "F-041", "F-042", "F-043", "F-044"})

    def _assert_not_future(self, timestamp: str) -> None:
        parsed = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        self.assertLessEqual(parsed, dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5))


if __name__ == "__main__":
    unittest.main()
