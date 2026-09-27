"""Check-in API and the offline local store, end to end."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_store import LocalClient


class LocalClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.client = LocalClient(Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_insert_select_filter_limit_order(self):
        t = self.client.table("typing_sessions")
        a = t.insert({"user_id": "u1", "typing_speed": 1}).execute().data[0]
        self.client.table("typing_sessions").insert({"user_id": "u1", "typing_speed": 2}).execute()
        self.client.table("typing_sessions").insert({"user_id": "u2", "typing_speed": 3}).execute()
        self.assertTrue(a["id"])
        rows = self.client.table("typing_sessions").select("*").eq("user_id", "u1").execute().data
        self.assertEqual([r["typing_speed"] for r in rows], [1, 2])
        top = (
            self.client.table("typing_sessions").select("typing_speed")
            .eq("user_id", "u1").order("typing_speed", desc=True).limit(1).execute().data
        )
        self.assertEqual(top, [{"typing_speed": 2}])

    def test_update_and_delete(self):
        row = self.client.table("baselines").insert({"user_id": "u1", "sample_count": 3}).execute().data[0]
        self.client.table("baselines").update({"sample_count": 5}).eq("id", row["id"]).execute()
        got = self.client.table("baselines").select("*").eq("user_id", "u1").execute().data
        self.assertEqual(got[0]["sample_count"], 5)
        self.client.table("baselines").delete().eq("user_id", "u1").execute()
        self.assertEqual(self.client.table("baselines").select("*").execute().data, [])

    def test_unknown_table_is_rejected(self):
        with self.assertRaises(ValueError):
            self.client.table("users; drop table x")

    def test_persists_across_instances(self):
        self.client.table("checkins").insert({"user_id": "u1", "factor": "tired", "date": "2026-09-20"}).execute()
        again = LocalClient(self.client.path)
        self.assertEqual(len(again.table("checkins").select("*").execute().data), 1)


class CheckinRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = LocalClient(Path(self.tmp.name) / "t.db")
        fake_db = mock.MagicMock(supabase=self.store)
        self.patch = mock.patch.dict("sys.modules", {"database": fake_db})
        self.patch.start()
        from routes.checkins import router

        app = FastAPI()
        app.include_router(router)
        self.http = TestClient(app)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_create_and_list(self):
        r = self.http.post("/api/users/u1/checkins", json={"factor": "poor_sleep", "date": "2026-09-20", "note": " hi "})
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.json()["label"], "poor sleep")
        self.assertEqual(r.json()["note"], "hi")
        self.http.post("/api/users/u1/checkins", json={"factor": "tired", "date": "2026-09-19"})
        listed = self.http.get("/api/users/u1/checkins").json()
        self.assertEqual([c["factor"] for c in listed], ["tired", "poor_sleep"])
        self.assertEqual(self.http.get("/api/users/u2/checkins").json(), [])

    def test_validation(self):
        self.assertEqual(self.http.post("/api/users/u1/checkins", json={"factor": "hungover"}).status_code, 422)
        self.assertEqual(
            self.http.post("/api/users/u1/checkins", json={"factor": "tired", "note": "x" * 281}).status_code, 422
        )
        self.assertEqual(
            self.http.post("/api/users/u1/checkins", json={"factor": "tired", "date": "2999-01-01"}).status_code, 422
        )

    def test_date_defaults_to_today(self):
        from datetime import datetime, timezone

        r = self.http.post("/api/users/u1/checkins", json={"factor": "feeling_well"})
        self.assertEqual(r.json()["date"], datetime.now(timezone.utc).date().isoformat())


class SupabaseRepositoryCheckinTests(unittest.TestCase):
    def test_missing_table_means_no_store(self):
        from investigation.repository import SupabaseSessionRepository

        class Boom:
            def table(self, name):
                raise RuntimeError("relation checkins does not exist")

        self.assertIsNone(SupabaseSessionRepository(client=Boom()).list_checkins("u1"))

    def test_reads_rows_through_the_local_client(self):
        from investigation.repository import SupabaseSessionRepository

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalClient(Path(tmp) / "t.db")
            store.table("checkins").insert({"user_id": "u1", "date": "2026-09-20", "factor": "tired", "note": "x"}).execute()
            rows = SupabaseSessionRepository(client=store).list_checkins("u1")
        self.assertEqual(rows, [{"date": "2026-09-20", "factor": "tired"}])


if __name__ == "__main__":
    unittest.main()
