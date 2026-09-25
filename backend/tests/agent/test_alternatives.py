"""Tests for the Phase 3.2 alternative explanation scan.

The point of these tests is honesty: candidates are only supported or weakened
when measured evidence decides them, and never presented as refuted when the
data source simply does not exist.
"""

import inspect
import unittest
from datetime import datetime, timezone

from investigation import tools
from investigation.adapter import build_behavioral_evidence
from investigation.agent import (
    ALTERNATIVE_CATALOG,
    CONTEXT_CANDIDATES,
    EvidenceRegistry,
    InvestigationTrace,
    assess_alternatives,
    assess_persistence,
    register_from_behavioral_evidence,
)
from tests import fixtures

USER = "user-1"


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


def _stable_repo():
    return fixtures.FakeRepository(
        sessions=[
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
            for i in range(1, 34)
        ]
    )


def _persistent_repo():
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(7, 36)
    ]
    sessions += [
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
        sessions=sessions,
        anomalies={"R0001": {"anomaly_score": 0.62, "is_anomaly": True}},
    )


def _shifted_repo():
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(7, 36)
    ]
    sessions += [
        fixtures.make_session(session_id=f"R{i:04d}", days_ago=i, typing_speed=233.0)
        for i in range(1, 6)
    ]
    return fixtures.FakeRepository(sessions=sessions)


def _thin_repo():
    return fixtures.FakeRepository(
        sessions=[
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i) for i in range(3)
        ]
    )


def _invalid_repo():
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(1, 20)
    ]
    sessions.append(fixtures.make_invalid_session(session_id="BAD1", days_ago=3))
    return fixtures.FakeRepository(sessions=sessions)


def _sparse_recent_repo():
    """A dense baseline window and only two recent sessions."""
    sessions = [
        fixtures.make_session(session_id=f"R{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(1, 3)
    ]
    sessions += [
        fixtures.make_session(session_id=f"B{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(7, 31)
    ]
    return fixtures.FakeRepository(sessions=sessions)


def _scan(repository, session_id="R0001", **kwargs):
    evidence = build_behavioral_evidence(USER, session_id, repository=repository)
    anomaly = tools.get_ml_evidence(USER, session_id, repository=repository)
    recent = tools.get_recent_sessions(USER, limit=500, window_days=7, repository=repository)
    registry = EvidenceRegistry()
    register_from_behavioral_evidence(registry, evidence, anomaly)
    finding = assess_persistence(
        registry, evidence, recent, as_of=datetime.now(timezone.utc), **kwargs
    )
    findings = assess_alternatives(registry, evidence, finding)
    return registry, evidence, finding, findings


def _by_candidate(findings):
    return {item.candidate: item for item in findings}


# ---------------------------------------------------------------------------
# Catalog shape
# ---------------------------------------------------------------------------


class CatalogTests(unittest.TestCase):
    def test_every_candidate_is_reported_in_fixed_order(self):
        _, _, _, findings = _scan(_shifted_repo())
        self.assertEqual(
            [item.candidate for item in findings],
            [candidate for candidate, _ in ALTERNATIVE_CATALOG],
        )
        self.assertEqual(
            [item.label for item in findings],
            [label for _, label in ALTERNATIVE_CATALOG],
        )

    def test_every_finding_has_a_reason_code(self):
        _, _, _, findings = _scan(_shifted_repo())
        for item in findings:
            with self.subTest(candidate=item.candidate):
                self.assertTrue(item.reason)
                self.assertNotIn(" ", item.reason)


# ---------------------------------------------------------------------------
# Evidence-backed candidates
# ---------------------------------------------------------------------------


class EvaluatedCandidateTests(unittest.TestCase):
    def test_temporary_variation_supported_when_not_eligible(self):
        _, _, finding, findings = _scan(_shifted_repo())
        item = _by_candidate(findings)["temporary_variation"]
        self.assertFalse(finding.eligible_for_persistence_claim)
        self.assertEqual(item.status, "supported")
        self.assertEqual(item.reason, "phase2_status_not_persistent")
        self.assertTrue(item.evidence_ids)

    def test_temporary_variation_weakened_when_eligible(self):
        _, _, finding, findings = _scan(_persistent_repo())
        item = _by_candidate(findings)["temporary_variation"]
        self.assertTrue(finding.eligible_for_persistence_claim)
        self.assertEqual(item.status, "weakened")
        self.assertEqual(item.reason, "eligible_for_persistence_claim")

    def test_insufficient_data_follows_the_model_minimum(self):
        registry, evidence, _, findings = _scan(_thin_repo(), session_id="S0001")
        item = _by_candidate(findings)["insufficient_data"]
        self.assertFalse(evidence.data_quality.meets_model_minimum)
        self.assertEqual(item.status, "supported")
        self.assertEqual(item.reason, "below_model_minimum")
        self.assertTrue(item.evidence_ids)

        _, _, _, ok_findings = _scan(_stable_repo(), session_id="S0001")
        ok_item = _by_candidate(ok_findings)["insufficient_data"]
        self.assertEqual(ok_item.status, "weakened")
        self.assertEqual(ok_item.reason, "meets_model_minimum")

    def test_data_quality_artifact_tracks_invalid_sessions(self):
        registry, evidence, _, findings = _scan(_invalid_repo(), session_id="S0001")
        item = _by_candidate(findings)["data_quality_artifact"]
        self.assertEqual(evidence.data_quality.invalid_sessions, 1)
        self.assertEqual(item.status, "supported")
        self.assertEqual(item.reason, "invalid_sessions_present")
        self.assertEqual(item.evidence_ids, [i.id for i in registry.filter(kind="quality_invalid")])

        _, _, _, clean_findings = _scan(_stable_repo(), session_id="S0001")
        self.assertEqual(
            _by_candidate(clean_findings)["data_quality_artifact"].status, "weakened"
        )

    def test_technical_failure_tracks_unavailable_evidence(self):
        registry, _, _, findings = _scan(_thin_repo(), session_id="S0001")
        unavailable = registry.filter(kind="anomaly_unavailable")
        item = _by_candidate(findings)["technical_failure"]
        self.assertTrue(unavailable)
        self.assertEqual(item.status, "supported")
        self.assertEqual(item.reason, "evidence_source_unavailable")
        self.assertEqual(item.evidence_ids, [i.id for i in unavailable])

        _, _, _, ok_findings = _scan(_persistent_repo())
        ok_item = _by_candidate(ok_findings)["technical_failure"]
        self.assertEqual(ok_item.status, "weakened")
        self.assertEqual(ok_item.reason, "all_required_sources_available")
        self.assertTrue(ok_item.evidence_ids)


# ---------------------------------------------------------------------------
# Workload / schedule proxy
# ---------------------------------------------------------------------------


class WorkloadProxyTests(unittest.TestCase):
    def test_sparse_recent_sampling_supports_the_proxy(self):
        registry, evidence, _, findings = _scan(_sparse_recent_repo(), session_id="R0001")
        item = _by_candidate(findings)["unusual_workload_or_schedule"]
        self.assertEqual(item.status, "partially_evaluated")
        self.assertEqual(item.reason, "recent_sampling_sparser_than_baseline")

        # Phase 3.1 registers the recent-window cadence item; this scan adds the
        # baseline-window one so the proxy is citable from both windows.
        cadence = registry.filter(kind="cadence")
        self.assertEqual(len(cadence), 2)
        baseline_window = evidence.temporal_analysis.baseline.window_days
        baseline_cadence = [c for c in cadence if c.window_days == baseline_window]
        self.assertEqual(len(baseline_cadence), 1)
        self.assertIn(baseline_cadence[0].id, item.evidence_ids)
        self.assertEqual(baseline_cadence[0].source_tool, "assess_alternatives")

    def test_comparable_sampling_weakens_the_proxy(self):
        _, _, _, findings = _scan(_stable_repo(), session_id="S0001")
        item = _by_candidate(findings)["unusual_workload_or_schedule"]
        self.assertEqual(item.status, "weakened")
        self.assertEqual(item.reason, "recent_sampling_comparable_to_baseline")

    def test_uncomparable_sampling_is_reported_unavailable(self):
        repository = fixtures.FakeRepository(
            sessions=[
                fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
                for i in range(1, 20)
            ]
        )
        registry, evidence, _, findings = _scan(repository, session_id="S0001")
        item = _by_candidate(findings)["unusual_workload_or_schedule"]
        self.assertIn(item.status, ("unavailable", "weakened"))
        if item.status == "unavailable":
            self.assertEqual(item.reason, "sampling_cadence_not_comparable")


# ---------------------------------------------------------------------------
# Honest unavailability
# ---------------------------------------------------------------------------


class UnavailabilityTests(unittest.TestCase):
    def test_contextual_candidates_are_unavailable_and_cite_the_absence(self):
        registry, _, _, findings = _scan(_persistent_repo())
        context_ids = [item.id for item in registry.filter(kind="context_absence")]
        by_candidate = _by_candidate(findings)
        for candidate in CONTEXT_CANDIDATES:
            with self.subTest(candidate=candidate):
                item = by_candidate[candidate]
                self.assertEqual(item.status, "unavailable")
                self.assertEqual(item.reason, "no_contextual_store")
                self.assertEqual(item.evidence_ids, context_ids)

    def test_keyboard_or_environment_change_is_unavailable(self):
        _, _, _, findings = _scan(_shifted_repo())
        item = _by_candidate(findings)["keyboard_or_environment_change"]
        self.assertEqual(item.status, "unavailable")
        self.assertEqual(item.reason, "no_capture_metadata_stored")

    def test_absence_is_never_reported_as_refutation(self):
        _, _, _, findings = _scan(_shifted_repo())
        for item in findings:
            if item.candidate in CONTEXT_CANDIDATES or item.candidate == "keyboard_or_environment_change":
                self.assertNotEqual(item.status, "weakened", item.candidate)


# ---------------------------------------------------------------------------
# Grounding, determinism, constraints
# ---------------------------------------------------------------------------


class GroundingTests(unittest.TestCase):
    def test_decided_candidates_always_cite_evidence(self):
        for repository, session_id in (
            (_persistent_repo(), "R0001"),
            (_shifted_repo(), "R0001"),
            (_thin_repo(), "S0001"),
            (_invalid_repo(), "S0001"),
            (_sparse_recent_repo(), "R0001"),
        ):
            with self.subTest(session=session_id):
                _, _, _, findings = _scan(repository, session_id)
                for item in findings:
                    if item.status in ("supported", "weakened", "partially_evaluated"):
                        self.assertTrue(item.evidence_ids, item.candidate)

    def test_cited_ids_resolve_in_the_registry(self):
        registry, _, _, findings = _scan(_persistent_repo())
        for item in findings:
            for item_id in item.evidence_ids:
                self.assertIsNotNone(registry.get(item_id), item_id)

    def test_no_repository_parameter(self):
        self.assertNotIn("repository", inspect.signature(assess_alternatives).parameters)

    def test_registry_is_required(self):
        evidence = build_behavioral_evidence(USER, "R0001", repository=_shifted_repo())
        anomaly = tools.get_ml_evidence(USER, "R0001", repository=_shifted_repo())
        registry = EvidenceRegistry()
        register_from_behavioral_evidence(registry, evidence, anomaly)
        with self.assertRaises(TypeError):
            assess_alternatives(object(), evidence, None)

    def test_scan_is_deterministic(self):
        repository = _persistent_repo()
        evidence = build_behavioral_evidence(USER, "R0001", repository=repository)
        anomaly = tools.get_ml_evidence(USER, "R0001", repository=repository)
        recent = tools.get_recent_sessions(USER, limit=500, window_days=7, repository=repository)
        as_of = datetime.now(timezone.utc)

        runs = []
        for _ in range(2):
            registry = EvidenceRegistry()
            register_from_behavioral_evidence(registry, evidence, anomaly)
            finding = assess_persistence(registry, evidence, recent, as_of=as_of)
            runs.append([item.model_dump() for item in assess_alternatives(registry, evidence, finding)])
        self.assertEqual(runs[0], runs[1])

    def test_trace_records_every_candidate(self):
        repository = _persistent_repo()
        evidence = build_behavioral_evidence(USER, "R0001", repository=repository)
        anomaly = tools.get_ml_evidence(USER, "R0001", repository=repository)
        recent = tools.get_recent_sessions(USER, limit=500, window_days=7, repository=repository)
        registry = EvidenceRegistry()
        register_from_behavioral_evidence(registry, evidence, anomaly)
        finding = assess_persistence(registry, evidence, recent, as_of=datetime.now(timezone.utc))

        trace = InvestigationTrace(clock=lambda: datetime(2026, 9, 24, 9, 42, tzinfo=timezone.utc))
        assess_alternatives(registry, evidence, finding, trace=trace)
        events = [e for e in trace if e.event_type == "state_change"]
        self.assertEqual(len(events), len(ALTERNATIVE_CATALOG))


if __name__ == "__main__":
    unittest.main()
