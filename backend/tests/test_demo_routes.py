"""Demo Lab service and routes: real engine output, labelled as demo."""

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from investigation.agent.demo import DEMO_KEYS
from routes.demo import router
from services import demo


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class DemoServiceTests(unittest.TestCase):
    def test_catalog_matches_engine_catalog(self):
        self.assertEqual([s["key"] for s in demo.list_scenarios()], list(DEMO_KEYS))

    def test_every_payload_is_labelled_demo_and_carries_engine_output(self):
        for key in DEMO_KEYS:
            with self.subTest(key=key):
                payload = demo.scenario_payload(key)
                self.assertEqual(payload["data_source"], "demo")
                self.assertTrue(payload["sessions"])
                self.assertTrue(payload["trace"])
                self.assertIn("conclusion", payload["report"])
                self.assertEqual(
                    payload["stop_reason"], payload["final_assessment"]["stop_reason"]
                )

    def test_payload_is_deterministic(self):
        self.assertEqual(
            demo.scenario_payload("persistent_change"),
            demo.scenario_payload("persistent_change"),
        )

    def test_unknown_scenario(self):
        with self.assertRaises(demo.UnknownScenario):
            demo.scenario_payload("nope")


class DemoRouteTests(unittest.TestCase):
    def test_list_and_run(self):
        client = _client()
        listing = client.get("/api/demo/scenarios")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(len(listing.json()), len(DEMO_KEYS))
        run = client.get("/api/demo/scenarios/persistent_change")
        self.assertEqual(run.status_code, 200)
        self.assertEqual(run.json()["report"]["conclusion"]["status"], "grounded")

    def test_unknown_is_404(self):
        self.assertEqual(_client().get("/api/demo/scenarios/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()


class ExportedDemoFilesTests(unittest.TestCase):
    """The static dashboard copies must match the engine (re-run the export)."""

    def test_exported_files_match_engine(self):
        import json
        from pathlib import Path

        out = Path(__file__).resolve().parents[2] / "dashboard" / "demo"
        if not out.exists():
            self.skipTest("dashboard/demo not exported")
        for key in DEMO_KEYS:
            with self.subTest(key=key):
                stored = json.loads((out / f"{key}.json").read_text())
                self.assertEqual(stored, demo.scenario_payload(key))
