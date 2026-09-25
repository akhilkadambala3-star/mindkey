"""Unit tests for the Phase 4 read projections (``services.reads``).

The read layer is a projection: these tests pin the documented contract
(exact keys, ordering, units, defensive empties), the reuse of the frozen
kernel helpers (``is_valid_stored_session``, ``to_wpm``, ``to_milliseconds``),
the read-only guarantee, and the explicit ``RepositoryError`` failure mode.
"""

import unittest
from unittest import mock

from investigation import units
from investigation.repository import RepositoryError
from services import reads
from tests import fixtures
from tests.api_support import (
    BASELINE_KEYS,
    SESSION_KEYS,
    FakeBaselineClient,
    ReadOnlyRepository,
    baseline_row,
)

USER = fixtures.DEFAULT_USER


class ListSessionsTests(unittest.TestCase):
    def _repo(self, sessions):
        return fixtures.FakeRepository(sessions=sessions)

    def test_contract_keys_are_exact(self):
        repo = self._repo(fixtures.make_history(4))
        for session in reads.list_sessions(USER, repository=repo):
            self.assertEqual(frozenset(session), SESSION_KEYS)

    def test_no_status_and_no_consistency(self):
        # Phase 4 decision C4/C5: neither field may ever be emitted.
        repo = self._repo(fixtures.make_history(3))
        for session in reads.list_sessions(USER, repository=repo):
            self.assertNotIn("status", session)
            self.assertNotIn("consistency", session)

    def test_oldest_first_regardless_of_row_order(self):
        rows = [
            fixtures.make_session(session_id="MID", days_ago=3),
            fixtures.make_session(session_id="NEW", days_ago=1),
            fixtures.make_session(session_id="OLD", days_ago=9),
        ]
        sessions = reads.list_sessions(USER, repository=self._repo(rows))
        self.assertEqual(
            [s["session_id"] for s in sessions], ["OLD", "MID", "NEW"]
        )

    def test_ties_broken_by_id(self):
        rows = [
            fixtures.make_session(session_id="B", days_ago=2),
            fixtures.make_session(session_id="A", days_ago=2),
        ]
        sessions = reads.list_sessions(USER, repository=self._repo(rows))
        self.assertEqual([s["session_id"] for s in sessions], ["A", "B"])

    def test_invalid_rows_are_excluded(self):
        rows = fixtures.make_history(3)
        rows.append(fixtures.make_invalid_session(session_id="BAD1", days_ago=1))
        sessions = reads.list_sessions(USER, repository=self._repo(rows))
        self.assertEqual(len(sessions), 3)
        self.assertNotIn("BAD1", [s["session_id"] for s in sessions])

    def test_unknown_user_gets_empty_list(self):
        rows = fixtures.make_history(3)  # rows belong to USER
        sessions = reads.list_sessions("someone-else", repository=self._repo(rows))
        self.assertEqual(sessions, [])

    def test_empty_store_gets_empty_list(self):
        sessions = reads.list_sessions(USER, repository=self._repo([]))
        self.assertEqual(sessions, [])

    def test_wpm_reuses_units_conversion(self):
        rows = [fixtures.make_session(session_id="S1", days_ago=1, typing_speed=285.0)]
        session = reads.list_sessions(USER, repository=self._repo(rows))[0]
        self.assertEqual(session["wpm"], units.to_wpm(285.0))
        self.assertEqual(session["wpm"], 57.0)
        self.assertEqual(session["typing_speed"], 285.0)

    def test_dwell_and_flight_are_milliseconds_via_units(self):
        rows = [
            fixtures.make_session(
                session_id="S1", days_ago=1, dwell_mean=0.12, flight_mean=0.08
            )
        ]
        session = reads.list_sessions(USER, repository=self._repo(rows))[0]
        self.assertEqual(session["dwell_mean_ms"], units.to_milliseconds(0.12))
        self.assertEqual(session["dwell_mean_ms"], 120.0)
        self.assertEqual(session["flight_mean_ms"], 80.0)

    def test_date_is_the_first_ten_characters_of_session_start(self):
        rows = [fixtures.make_session(session_id="S1", days_ago=1)]
        session = reads.list_sessions(USER, repository=self._repo(rows))[0]
        self.assertEqual(session["date"], session["session_start"][:10])
        self.assertRegex(session["date"], r"^\d{4}-\d{2}-\d{2}$")

    def test_timestamps_are_echoed_not_recomputed(self):
        rows = [fixtures.make_session(session_id="S1", days_ago=1)]
        stored = rows[0]
        session = reads.list_sessions(USER, repository=self._repo(rows))[0]
        self.assertEqual(session["session_start"], stored["session_start"])
        self.assertEqual(session["session_end"], stored["session_end"])

    def test_session_id_is_a_string_even_for_numeric_ids(self):
        rows = [fixtures.make_session(session_id=12345, days_ago=1)]
        session = reads.list_sessions(USER, repository=self._repo(rows))[0]
        self.assertEqual(session["session_id"], "12345")

    def test_repository_error_propagates(self):
        with self.assertRaises(RepositoryError):
            reads.list_sessions(USER, repository=fixtures.FailingRepository())

    def test_only_protocol_reads_happen_and_rows_are_not_mutated(self):
        rows = fixtures.make_history(3)
        before = [dict(row) for row in rows]
        spy = ReadOnlyRepository(fixtures.FakeRepository(sessions=rows))
        reads.list_sessions(USER, repository=spy)
        self.assertEqual(spy.calls, [("list_sessions", USER)])
        self.assertEqual(rows, before)


class LatestSessionIdTests(unittest.TestCase):
    def _repo(self, sessions):
        return fixtures.FakeRepository(sessions=sessions)

    def test_picks_greatest_session_start(self):
        rows = [
            fixtures.make_session(session_id="OLD", days_ago=9),
            fixtures.make_session(session_id="NEW", days_ago=1),
            fixtures.make_session(session_id="MID", days_ago=3),
        ]
        self.assertEqual(
            reads.latest_session_id(USER, repository=self._repo(rows)), "NEW"
        )

    def test_ties_broken_by_id(self):
        rows = [
            fixtures.make_session(session_id="A", days_ago=2),
            fixtures.make_session(session_id="B", days_ago=2),
        ]
        self.assertEqual(
            reads.latest_session_id(USER, repository=self._repo(rows)), "B"
        )

    def test_empty_store_returns_none(self):
        self.assertIsNone(reads.latest_session_id(USER, repository=self._repo([])))

    def test_unknown_user_returns_none(self):
        rows = fixtures.make_history(3)
        self.assertIsNone(
            reads.latest_session_id("someone-else", repository=self._repo(rows))
        )

    def test_deterministic_regardless_of_row_order(self):
        rows = fixtures.make_history(6)
        forwards = reads.latest_session_id(USER, repository=self._repo(rows))
        backwards = reads.latest_session_id(
            USER, repository=self._repo(list(reversed(rows)))
        )
        self.assertEqual(forwards, backwards)

    def test_considers_stored_rows_even_when_invalid(self):
        # The investigation may target the true newest session and let the
        # kernel report any data-quality problem itself.
        rows = [
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=days)
            for i, days in ((1, 7), (2, 6), (3, 5))
        ]
        rows.append(fixtures.make_invalid_session(session_id="BAD1", days_ago=1))
        self.assertEqual(
            reads.latest_session_id(USER, repository=self._repo(rows)), "BAD1"
        )

    def test_rows_without_ids_are_not_targetable(self):
        self.assertIsNone(reads.latest_session_id_for([{"session_start": "2026-01-01"}]))

    def test_repository_error_propagates(self):
        with self.assertRaises(RepositoryError):
            reads.latest_session_id(USER, repository=fixtures.FailingRepository())


class UserBaselineTests(unittest.TestCase):
    def test_contract_keys_are_exact(self):
        client = FakeBaselineClient([baseline_row()])
        baseline = reads.user_baseline(USER, client=client)
        self.assertEqual(frozenset(baseline), BASELINE_KEYS)

    def test_stored_values_are_echoed(self):
        client = FakeBaselineClient([baseline_row(sample_count=17)])
        baseline = reads.user_baseline(USER, client=client)
        self.assertEqual(baseline["typing_speed"], 285.0)
        self.assertEqual(baseline["dwell_mean"], 0.12)
        self.assertEqual(baseline["flight_mean"], 0.08)
        self.assertEqual(baseline["correction_rate"], 0.06)
        self.assertEqual(baseline["rhythm_variability"], 0.19)
        self.assertEqual(baseline["pause_count"], 4.0)
        self.assertEqual(baseline["sample_count"], 17)
        self.assertEqual(baseline["updated_at"], "2026-09-20T08:00:00+00:00")

    def test_wpm_reuses_units_conversion(self):
        client = FakeBaselineClient([baseline_row(typing_speed=285.0)])
        baseline = reads.user_baseline(USER, client=client)
        self.assertEqual(baseline["wpm"], units.to_wpm(285.0))
        self.assertEqual(baseline["wpm"], 57.0)

    def test_missing_baseline_is_defensive_not_an_error(self):
        baseline = reads.user_baseline(USER, client=FakeBaselineClient([]))
        self.assertEqual(frozenset(baseline), BASELINE_KEYS)
        self.assertEqual(baseline["sample_count"], 0)
        self.assertIsNone(baseline["updated_at"])
        for key in BASELINE_KEYS - {"sample_count", "updated_at"}:
            self.assertIsNone(baseline[key], key)

    def test_reads_only_the_requested_users_row(self):
        client = FakeBaselineClient(
            [baseline_row(user_id="other-user"), baseline_row(user_id=USER)]
        )
        baseline = reads.user_baseline(USER, client=client)
        self.assertEqual(baseline["sample_count"], 12)

    def test_only_reads_are_performed_against_the_baselines_table(self):
        client = FakeBaselineClient([baseline_row()])
        reads.user_baseline(USER, client=client)
        self.assertEqual(client.tables, ["baselines"])
        self.assertTrue(set(client.ops) <= {"select", "eq", "limit", "execute"})

    def test_store_failure_raises_repository_error(self):
        client = FakeBaselineClient([baseline_row()], fail=True)
        with self.assertRaises(RepositoryError):
            reads.user_baseline(USER, client=client)

    def test_unavailable_default_client_raises_repository_error(self):
        with mock.patch.object(
            reads,
            "baseline_client",
            side_effect=RepositoryError("Supabase client is unavailable"),
        ):
            with self.assertRaises(RepositoryError):
                reads.user_baseline(USER)

    def test_sample_count_is_never_null_on_a_stored_row(self):
        client = FakeBaselineClient([baseline_row(sample_count=None)])
        baseline = reads.user_baseline(USER, client=client)
        self.assertEqual(baseline["sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
