"""Tests for the Phase 3.4 grounded report.

The report must stay grounded (every claim cites real evidence), non-diagnostic,
deterministic, and honest about insufficient evidence. These tests exercise each
conclusion status, the trace/evidence link, privacy, and safety.
"""

import copy
import json
import unittest
from datetime import datetime, timezone
from unittest import mock

from investigation.agent import (
    REPORT_SCHEMA_VERSION,
    InvestigationTrace,
    build_grounded_report,
    run_investigation,
    trace_digest,
)
from investigation.agent.critic import CLINICAL_TERMS, SAFETY_EXEMPTIONS
from investigation.agent.report import CONCLUSION_TEXT, REPORT_LIMITATIONS
from investigation.repository import RepositoryError
from tests import fixtures

USER = "user-1"
CLK = lambda: datetime(2026, 9, 24, 9, 42, tzinfo=timezone.utc)  # noqa: E731

CONTENT_KEYS = ("text", "content", "keystrokes", "message", "note_text", "typed_text")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sessions(tail=None):
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(7, 36)
    ]
    return sessions + (tail or [])


def _persistent_repo():
    tail = [
        fixtures.make_session(
            session_id=f"R{i:04d}",
            days_ago=i,
            typing_speed=200.0,
            dwell_mean=0.20,
            flight_mean=0.14,
            rhythm_variability=0.30,
            correction_rate=0.12,
            pause_count=9,
        )
        for i in range(1, 6)
    ]
    return fixtures.FakeRepository(
        sessions=_sessions(tail),
        anomalies={"R0001": {"anomaly_score": 0.62, "is_anomaly": True}},
    )


def _shifted_repo():
    tail = [
        fixtures.make_session(session_id=f"R{i:04d}", days_ago=i, typing_speed=233.0)
        for i in range(1, 6)
    ]
    return fixtures.FakeRepository(sessions=_sessions(tail))


def _stable_repo():
    return fixtures.FakeRepository(
        sessions=[
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
            for i in range(1, 35)
        ]
    )


def _thin_repo():
    return fixtures.FakeRepository(
        sessions=[
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i) for i in range(3)
        ]
    )


def _result(repository, session_id="R0001"):
    return run_investigation(
        USER,
        session_id,
        repository=repository,
        trace=InvestigationTrace(clock=CLK),
        as_of=datetime.now(timezone.utc),
    )


def _report(repository, session_id="R0001"):
    return build_grounded_report(_result(repository, session_id), clock=CLK)


def _string_values(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _string_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _string_values(value)


def _keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _keys(value)


def _no_repo_result():
    with mock.patch(
        "investigation.agent.engine.default_repository",
        side_effect=RepositoryError("no client"),
    ):
        return run_investigation(USER, "R0001", repository=None)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


class ShapeTests(unittest.TestCase):
    def test_report_is_json_serializable(self):
        json.dumps(_report(_persistent_repo()).model_dump(mode="json"))

    def test_ids_schema_and_disclaimer(self):
        report = _report(_persistent_repo())
        self.assertEqual(report.schema_version, REPORT_SCHEMA_VERSION)
        self.assertEqual(report.user_id, USER)
        self.assertEqual(report.session_id, "R0001")
        self.assertIn("no medical claim", report.disclaimer)

    def test_conclusion_status_is_in_the_closed_set(self):
        report = _report(_persistent_repo())
        self.assertIn(report.conclusion.status, set(CONCLUSION_TEXT))

    def test_engine_assessment_mirrors_the_result(self):
        result = _result(_persistent_repo())
        report = build_grounded_report(result, clock=CLK)
        self.assertEqual(report.engine_assessment, result.final_assessment)

    def test_next_action_mirrors_the_state(self):
        result = _result(_persistent_repo())
        report = build_grounded_report(result, clock=CLK)
        self.assertEqual(report.next_action, result.state.next_action)


# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------


class ConclusionTests(unittest.TestCase):
    def test_persistent_change_is_grounded(self):
        report = _report(_persistent_repo())
        self.assertEqual(report.conclusion.status, "grounded")
        self.assertEqual(report.conclusion.basis, "grounded_persistent_change")
        self.assertTrue(report.conclusion.evidence_ids)

    def test_stable_baseline_is_no_deviation(self):
        report = _report(_stable_repo(), "S0001")
        self.assertEqual(report.conclusion.status, "no_deviation")
        self.assertEqual(report.conclusion.basis, "stable_baseline")

    def test_recent_variation_is_preliminary(self):
        report = _report(_shifted_repo())
        self.assertEqual(report.conclusion.status, "preliminary")
        self.assertEqual(report.conclusion.basis, "deviation_not_persistent")

    def test_thin_history_is_inconclusive(self):
        report = _report(_thin_repo(), "S0001")
        self.assertEqual(report.conclusion.status, "inconclusive")
        self.assertEqual(report.conclusion.basis, "insufficient_history")

    def test_data_unavailable_is_inconclusive(self):
        report = build_grounded_report(_no_repo_result(), clock=CLK)
        self.assertEqual(report.conclusion.status, "inconclusive")
        self.assertEqual(report.conclusion.basis, "data_unavailable")

    def test_unsound_critique_forces_inconclusive(self):
        result = _result(_persistent_repo())
        result.state.evidence[0]["statement"] = "Evidence of dementia."
        report = build_grounded_report(result, clock=CLK)
        self.assertEqual(report.conclusion.status, "inconclusive")
        self.assertEqual(report.conclusion.basis, "critic_rejections_present")

    def test_insufficient_evidence_is_never_grounded(self):
        for report in (
            _report(_thin_repo(), "S0001"),
            _report(_stable_repo(), "S0001"),
            _report(_shifted_repo()),
        ):
            with self.subTest(status=report.conclusion.status):
                self.assertNotEqual(report.conclusion.status, "grounded")


# ---------------------------------------------------------------------------
# Summary, observations, claims
# ---------------------------------------------------------------------------


class SummaryTests(unittest.TestCase):
    def test_summary_lines_are_present_and_cite_evidence(self):
        report = _report(_persistent_repo())
        self.assertGreaterEqual(len(report.summary), 3)
        self.assertTrue(any("[" in line and "E" in line for line in report.summary))
        self.assertTrue(any("persistence" in line.lower() for line in report.summary))

    def test_summary_names_supported_hypotheses(self):
        report = _report(_persistent_repo())
        self.assertTrue(
            any("Persistent behavioral change" in line for line in report.summary)
        )


class ObservationTests(unittest.TestCase):
    def test_every_evidence_item_becomes_an_observation(self):
        result = _result(_persistent_repo())
        report = build_grounded_report(result, clock=CLK)
        self.assertEqual(len(report.observations), len(result.state.evidence))

    def test_observations_use_the_registry_text_verbatim(self):
        result = _result(_persistent_repo())
        report = build_grounded_report(result, clock=CLK)
        statements = {item["statement"] for item in result.state.evidence}
        for claim in report.observations:
            self.assertIn(claim.statement, statements)
            self.assertEqual(len(claim.evidence_ids), 1)

    def test_observations_link_to_trace_events(self):
        result = _result(_persistent_repo())
        report = build_grounded_report(result, clock=CLK)
        by_seq = {event["seq"]: event for event in result.state.trace}
        linked = [claim for claim in report.observations if claim.trace_seqs]
        self.assertTrue(linked)
        for claim in linked:
            for seq in claim.trace_seqs:
                self.assertIn(claim.evidence_ids[0], by_seq[seq]["evidence_ids"])

    def test_rejected_evidence_is_excluded_from_observations(self):
        result = _result(_persistent_repo())
        target = result.state.evidence[0]["id"]
        result.state.evidence[0]["statement"] = "Evidence of dementia."
        report = build_grounded_report(result, clock=CLK)
        self.assertNotIn(target, [c.evidence_ids[0] for c in report.observations])
        self.assertTrue(report.rejected_claims)


class ClaimTests(unittest.TestCase):
    def test_hypothesis_claims_match_the_catalog_and_result(self):
        result = _result(_persistent_repo())
        report = build_grounded_report(result, clock=CLK)
        self.assertEqual(
            [claim.id for claim in report.hypothesis_assessment],
            ["H1", "H2", "H3", "H4"],
        )
        statuses = {c.id: c.status for c in report.hypothesis_assessment}
        for spec, hypothesis in zip(report.hypothesis_assessment, result.hypotheses):
            self.assertEqual(spec.status, hypothesis.status)

    def test_rejected_hypothesis_is_excluded(self):
        result = _result(_persistent_repo())
        result.hypotheses[2].hypothesis = "Cognitive decline"
        report = build_grounded_report(result, clock=CLK)
        self.assertNotIn("H3", [c.id for c in report.hypothesis_assessment])
        self.assertIn("H3", [r.subject for r in report.rejected_claims])

    def test_alternative_claims_match_the_catalog_order(self):
        result = _result(_persistent_repo())
        report = build_grounded_report(result, clock=CLK)
        self.assertEqual(
            [claim.id for claim in report.alternatives],
            [a.candidate for a in result.alternatives],
        )
        for claim, alternative in zip(report.alternatives, result.alternatives):
            self.assertEqual(claim.status, alternative.status)

    def test_rejected_alternative_is_excluded(self):
        result = _result(_persistent_repo())
        result.alternatives[0].candidate = "made_up"
        report = build_grounded_report(result, clock=CLK)
        self.assertNotIn("made_up", [c.id for c in report.alternatives])
        self.assertIn("made_up", [r.subject for r in report.rejected_claims])


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


class UncertaintyTests(unittest.TestCase):
    def test_thin_history_is_high_uncertainty(self):
        report = _report(_thin_repo(), "S0001")
        self.assertEqual(report.uncertainty.level, "high")
        self.assertIn("insufficient_history", report.uncertainty.reasons)

    def test_data_unavailable_is_high_uncertainty(self):
        report = build_grounded_report(_no_repo_result(), clock=CLK)
        self.assertEqual(report.uncertainty.level, "high")
        self.assertIn("evidence_unavailable", report.uncertainty.reasons)

    def test_persistent_change_records_open_questions_and_no_cause(self):
        report = _report(_persistent_repo())
        self.assertEqual(report.uncertainty.level, "moderate")
        self.assertIn("cause_not_established", report.uncertainty.reasons)
        self.assertIn("unanswered:context_factors", report.uncertainty.reasons)

    def test_unassessed_robustness_is_recorded(self):
        report = _report(_shifted_repo())
        self.assertIn(
            "window_robustness_not_assessed", report.uncertainty.reasons
        )

    def test_disagreeing_robustness_is_recorded(self):
        result = _result(_persistent_repo())
        result.final_assessment["persistence"]["robustness"] = "disagrees"
        report = build_grounded_report(result, clock=CLK)
        self.assertIn("window_robustness_disagrees", report.uncertainty.reasons)

    def test_cause_is_always_unestablished(self):
        for report in (
            _report(_persistent_repo()),
            _report(_thin_repo(), "S0001"),
        ):
            with self.subTest(level=report.uncertainty.level):
                self.assertIn("cause_not_established", report.uncertainty.reasons)

    def test_reasons_are_deduplicated(self):
        reasons = _report(_persistent_repo()).uncertainty.reasons
        self.assertEqual(len(reasons), len(set(reasons)))


# ---------------------------------------------------------------------------
# Limitations, link, trace continuity
# ---------------------------------------------------------------------------


class LimitationTests(unittest.TestCase):
    def test_report_and_engine_limitations_are_included(self):
        result = _result(_persistent_repo())
        report = build_grounded_report(result, clock=CLK)
        for limitation in result.limitations:
            self.assertIn(limitation, report.limitations)
        for limitation in REPORT_LIMITATIONS:
            self.assertIn(limitation, report.limitations)

    def test_each_rejection_adds_a_limitation(self):
        result = _result(_persistent_repo())
        result.hypotheses[2].supporting_evidence = []
        report = build_grounded_report(result, clock=CLK)
        self.assertTrue(
            any("rejected by the critic" in line for line in report.limitations)
        )


class LinkTests(unittest.TestCase):
    def test_link_digests_and_counts(self):
        result = _result(_persistent_repo())
        report = build_grounded_report(result, clock=CLK)
        self.assertEqual(len(report.link.evidence_digest), 64)
        self.assertEqual(len(report.link.engine_trace_digest), 64)
        self.assertEqual(len(report.link.critic_trace_digest), 64)
        self.assertEqual(
            report.link.trace_event_count,
            len(result.state.trace) + len(report.critic_events),
        )

    def test_stop_event_seq_points_at_the_engine_stop(self):
        result = _result(_persistent_repo())
        report = build_grounded_report(result, clock=CLK)
        stops = [e for e in result.state.trace if e["event_type"] == "stop"]
        self.assertEqual(report.link.stop_event_seq, stops[-1]["seq"])

    def test_critic_events_continue_the_sequence(self):
        result = _result(_persistent_repo())
        report = build_grounded_report(result, clock=CLK)
        max_engine = max(e["seq"] for e in result.state.trace)
        self.assertTrue(report.critic_events)
        self.assertTrue(all(e["seq"] > max_engine for e in report.critic_events))
        self.assertEqual(
            [e["seq"] for e in report.critic_events],
            list(range(max_engine + 1, max_engine + 1 + len(report.critic_events))),
        )

    def test_engine_trace_is_not_modified(self):
        result = _result(_persistent_repo())
        before = copy.deepcopy(result.state.trace)
        build_grounded_report(result, clock=CLK)
        self.assertEqual(result.state.trace, before)


# ---------------------------------------------------------------------------
# Determinism, privacy, safety
# ---------------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    def test_repeated_builds_are_identical(self):
        result = _result(_persistent_repo())
        first = build_grounded_report(result, clock=CLK).model_dump(mode="json")
        second = build_grounded_report(result, clock=CLK).model_dump(mode="json")
        self.assertEqual(first, second)

    def test_engine_trace_digest_is_timestamp_independent(self):
        result = _result(_persistent_repo())
        report = build_grounded_report(result, clock=CLK)
        shifted = [dict(event, ts="2099-01-01T00:00:00+00:00") for event in result.state.trace]
        self.assertEqual(trace_digest(shifted), report.link.engine_trace_digest)


class PrivacyTests(unittest.TestCase):
    def test_no_content_keys_anywhere(self):
        payload = _report(_persistent_repo()).model_dump(mode="json")
        for key in _keys(payload):
            self.assertNotIn(key, CONTENT_KEYS)

    def test_no_keystroke_or_password_text(self):
        blob = json.dumps(_report(_persistent_repo()).model_dump(mode="json")).lower()
        for forbidden in ("keystroke", "password", "\"typed_text\"", "\"message\""):
            self.assertNotIn(forbidden, blob)


class SafetyTests(unittest.TestCase):
    def _text(self, report):
        values = [report.disclaimer, report.next_action]
        values += report.summary
        values += report.limitations
        values += [report.conclusion.statement]
        values += [claim.statement for claim in report.observations]
        values += [claim.statement for claim in report.hypothesis_assessment]
        values += [claim.statement for claim in report.alternatives]
        values += [claim.reason for claim in report.rejected_claims]
        values += list(report.uncertainty.reasons)
        return " ".join(values)

    def test_report_text_has_no_clinical_language(self):
        for report in (
            _report(_persistent_repo()),
            _report(_thin_repo(), "S0001"),
            _report(_stable_repo(), "S0001"),
        ):
            blob = self._text(report).lower()
            for phrase in SAFETY_EXEMPTIONS:
                blob = blob.replace(phrase, "")
            for term in CLINICAL_TERMS:
                self.assertNotIn(term, blob, term)

    def test_report_templates_are_non_diagnostic(self):
        templates = " ".join(CONCLUSION_TEXT.values()).lower()
        for term in CLINICAL_TERMS:
            self.assertNotIn(term, templates)


if __name__ == "__main__":
    unittest.main()
