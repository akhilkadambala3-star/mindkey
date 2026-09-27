"""Check-ins as investigation input: store semantics, evidence, H2, alternatives."""

import json
import unittest

from investigation.agent import build_grounded_report, run_investigation
from investigation.agent.demo import (
    DEFAULT_AS_OF,
    DEFAULT_USER,
    DEMO_DATASETS,
    InMemoryRepository,
)
from investigation.context import checkins_in_window, normalize_checkin


def _run(dataset, checkins):
    sessions, anomalies, session_id = DEMO_DATASETS[dataset]["build"](DEFAULT_AS_OF)
    repo = InMemoryRepository(sessions=sessions, anomalies=anomalies, checkins=checkins)
    result = run_investigation(
        DEFAULT_USER, session_id, repository=repo,
        clock=lambda: DEFAULT_AS_OF, as_of=DEFAULT_AS_OF,
    )
    return result, build_grounded_report(result, clock=lambda: DEFAULT_AS_OF)


def _hyp(report, hid):
    return next(h.status for h in report.hypothesis_assessment if h.id == hid)


def _alt(report, candidate):
    return next(a.status for a in report.alternatives if a.id == candidate)


def _kinds(result):
    return [e["kind"] for e in result.model_dump(mode="json")["state"]["evidence"]]


def _day(days_ago):
    from datetime import timedelta
    return (DEFAULT_AS_OF - timedelta(days=days_ago)).date().isoformat()


class StoreSemanticsTests(unittest.TestCase):
    def test_no_store_is_unchanged(self):
        result, report = _run("persistent_change", None)
        self.assertIn("context_absence", _kinds(result))
        self.assertEqual(_hyp(report, "H2"), "uncertain")
        self.assertEqual(_alt(report, "poor_sleep"), "unavailable")
        self.assertIn("unanswered:context_factors", report.uncertainty.reasons)

    def test_empty_store_is_not_absence_of_a_store(self):
        result, report = _run("persistent_change", [])
        kinds = _kinds(result)
        self.assertIn("context_not_reported", kinds)
        self.assertNotIn("context_absence", kinds)
        self.assertEqual(_hyp(report, "H2"), "uncertain")
        self.assertEqual(_alt(report, "poor_sleep"), "unavailable")

    def test_failing_store_reports_unavailable(self):
        class Broken(InMemoryRepository):
            def list_checkins(self, user_id):
                raise RuntimeError("table missing")

        sessions, anomalies, sid = DEMO_DATASETS["persistent_change"]["build"](DEFAULT_AS_OF)
        result = run_investigation(
            DEFAULT_USER, sid, repository=Broken(sessions, anomalies, checkins=[]),
            clock=lambda: DEFAULT_AS_OF, as_of=DEFAULT_AS_OF,
        )
        self.assertIn("context_absence", _kinds(result))


class ContextDecisionTests(unittest.TestCase):
    def test_reported_disruption_supports_h2_and_matching_alternative(self):
        _, report = _run(
            "persistent_change",
            [{"date": _day(2), "factor": "poor_sleep"}, {"date": _day(1), "factor": "tired"}],
        )
        self.assertEqual(_hyp(report, "H2"), "supported")
        self.assertEqual(_alt(report, "poor_sleep"), "supported")
        self.assertEqual(_alt(report, "fatigue"), "supported")
        self.assertEqual(_alt(report, "stress"), "partially_evaluated")
        self.assertEqual(report.rejected_claims, [])
        self.assertEqual(report.conclusion.basis, "grounded_persistent_change")

    def test_feeling_well_weakens_context(self):
        _, report = _run("persistent_change", [{"date": _day(2), "factor": "feeling_well"}])
        self.assertEqual(_hyp(report, "H2"), "weakened")
        for candidate in ("poor_sleep", "fatigue", "stress", "illness_or_mood", "distraction"):
            self.assertEqual(_alt(report, candidate), "weakened")
        self.assertNotIn("unanswered:context_factors", report.uncertainty.reasons)

    def test_no_deviation_means_context_explains_nothing(self):
        _, report = _run("consistent", [{"date": _day(2), "factor": "poor_sleep"}])
        self.assertEqual(_hyp(report, "H2"), "uncertain")
        self.assertEqual(_alt(report, "poor_sleep"), "partially_evaluated")

    def test_checkins_outside_the_window_are_ignored(self):
        result, report = _run("persistent_change", [{"date": _day(20), "factor": "poor_sleep"}])
        self.assertIn("context_not_reported", _kinds(result))
        self.assertEqual(_hyp(report, "H2"), "uncertain")

    def test_notes_never_reach_the_investigation(self):
        secret = "my private note about work"
        result, report = _run(
            "persistent_change",
            [{"date": _day(2), "factor": "stressed", "note": secret}],
        )
        blob = json.dumps(result.model_dump(mode="json")) + report.model_dump_json()
        self.assertNotIn(secret, blob)

    def test_every_supported_claim_cites_evidence(self):
        _, report = _run("recent_variation", [{"date": _day(3), "factor": "poor_sleep"}])
        for claim in report.hypothesis_assessment + report.alternatives:
            if claim.status == "supported":
                self.assertTrue(claim.evidence_ids, claim.id)


class NormalizationTests(unittest.TestCase):
    def test_unknown_factor_and_bad_date_are_dropped(self):
        self.assertIsNone(normalize_checkin({"date": "2026-09-20", "factor": "hungover"}))
        self.assertIsNone(normalize_checkin({"date": "not a date", "factor": "tired"}))

    def test_window_is_inclusive_and_deduplicated(self):
        rows = [
            {"date": _day(0), "factor": "tired"},
            {"date": _day(0), "factor": "tired"},
            {"date": _day(6), "factor": "stressed"},
            {"date": _day(7), "factor": "poor_sleep"},
        ]
        kept = checkins_in_window(rows, DEFAULT_AS_OF, 7)
        self.assertEqual([c["factor"] for c in kept], ["stressed", "tired"])


if __name__ == "__main__":
    unittest.main()
