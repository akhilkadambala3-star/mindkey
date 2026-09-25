"""Tests for the Phase 3.2 per-session persistence analysis.

Phase 2's persistence verdict must remain authoritative: these tests cover the
added depth/onset/robustness measurements and the fact that eligibility can only
narrow Phase 2, never widen it.
"""

import inspect
import unittest
from datetime import datetime, timedelta, timezone

from investigation import tools
from investigation.adapter import build_behavioral_evidence
from investigation.agent import (
    CachingRepository,
    EvidenceRegistry,
    InvestigationTrace,
    register_from_behavioral_evidence,
)
from investigation.agent.persistence import (
    assess_persistence,
    assess_robustness,
    baseline_reference,
    deviating_keys,
    moved_signature_from_drift,
    moved_signature_from_stats,
)
from investigation.contracts import CanonicalSession
from investigation.tools import BehavioralDriftOutput, FeatureDrift
from investigation.units import CANONICAL_FEATURE_KEYS
from tests import fixtures

USER = "user-1"


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


def _stable_repo():
    return fixtures.FakeRepository(
        sessions=[
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
            for i in range(1, 35)
        ]
    )


def _shifted_repo():
    """Stable history, then the last five sessions drop typing speed ~18%."""
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(7, 36)
    ]
    sessions += [
        fixtures.make_session(session_id=f"R{i:04d}", days_ago=i, typing_speed=233.0)
        for i in range(1, 6)
    ]
    return fixtures.FakeRepository(sessions=sessions)


def _persistent_repo():
    """Five recent sessions with aligned movement across five signals."""
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


def _single_deviation_repo():
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(2, 36)
    ]
    sessions.append(
        fixtures.make_session(session_id="R0001", days_ago=1, typing_speed=200.0)
    )
    return fixtures.FakeRepository(sessions=sessions)


def _recent_run_repo():
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(4, 36)
    ]
    sessions += [
        fixtures.make_session(session_id=f"R{i:04d}", days_ago=i, typing_speed=200.0)
        for i in range(1, 4)
    ]
    return fixtures.FakeRepository(sessions=sessions)


def _gapped_repo():
    """Deviating on days 1, 2 and 4; normal on days 3, 5 and 6."""
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(5, 36)
    ]
    sessions += [
        fixtures.make_session(session_id="R0001", days_ago=1, typing_speed=200.0),
        fixtures.make_session(session_id="R0002", days_ago=2, typing_speed=200.0),
        fixtures.make_session(session_id="R0004", days_ago=4, typing_speed=200.0),
    ]
    return fixtures.FakeRepository(sessions=sessions)


def _thin_repo():
    return fixtures.FakeRepository(
        sessions=[
            fixtures.make_session(session_id=f"S{i:04d}", days_ago=i) for i in range(3)
        ]
    )


def _sparse_recent_repo():
    """Dense baseline window, sparse recent window (workload proxy case)."""
    sessions = [
        fixtures.make_session(session_id=f"R{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(1, 3)
    ]
    sessions += [
        fixtures.make_session(session_id=f"B{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(7, 31)
    ]
    return fixtures.FakeRepository(sessions=sessions)


def _context(repository, session_id="R0001", window_days=7):
    """A populated registry plus the Phase 2 objects persistence needs."""
    evidence = build_behavioral_evidence(USER, session_id, repository=repository)
    anomaly = tools.get_ml_evidence(USER, session_id, repository=repository)
    recent = tools.get_recent_sessions(
        USER, limit=500, window_days=window_days, repository=repository
    )
    return evidence, anomaly, recent


def _registry(evidence, anomaly):
    registry = EvidenceRegistry()
    register_from_behavioral_evidence(registry, evidence, anomaly)
    return registry


def _alt_drift(reverse=False, available=True, windows=14):
    features = []
    for key in CANONICAL_FEATURE_KEYS:
        relative = (-0.5 if reverse else 0.5) if key != "typing_speed_cpm" else (
            0.3 if reverse else -0.3
        )
        features.append(
            FeatureDrift(
                key=key,
                label=key,
                canonical_unit="unit",
                available=available,
                baseline_value=1.0,
                recent_value=1.0 + relative,
                absolute_change=relative,
                relative_change=relative,
                direction="increase" if relative > 0 else "decrease",
            )
        )
    return BehavioralDriftOutput(
        available=True,
        user_id=USER,
        recent_window_days=windows,
        baseline_window_days=windows,
        recent_sample_count=10,
        baseline_sample_count=10,
        sufficient=True,
        features=features,
        note="synthetic",
    )


def _finding_for(repository, session_id="R0001", as_of=None, **kwargs):
    evidence, anomaly, recent = _context(repository, session_id)
    registry = _registry(evidence, anomaly)
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    return registry, evidence, recent, assess_persistence(
        registry, evidence, recent, as_of=as_of, **kwargs
    )


# ---------------------------------------------------------------------------
# Deviation predicate
# ---------------------------------------------------------------------------


class DeviationPredicateTests(unittest.TestCase):
    def test_threshold_boundary(self):
        baseline = {"typing_speed_cpm": 285.0}
        exactly = CanonicalSession(typing_speed_cpm=285.0 * 1.10, is_valid=True)
        below = CanonicalSession(typing_speed_cpm=285.0 * 1.099, is_valid=True)
        self.assertEqual(deviating_keys(exactly, baseline), ["typing_speed_cpm"])
        self.assertEqual(deviating_keys(below, baseline), [])

    def test_zero_and_missing_baselines_are_skipped(self):
        session = CanonicalSession(correction_rate=0.5, typing_speed_cpm=100.0, is_valid=True)
        self.assertEqual(deviating_keys(session, {"correction_rate": 0.0}), [])
        self.assertEqual(deviating_keys(session, {"typing_speed_cpm": 0.0}), [])

    def test_non_canonical_reference_keys_are_ignored(self):
        session = CanonicalSession(typing_speed_cpm=100.0, is_valid=True)
        self.assertEqual(deviating_keys(session, {"not_a_feature": 1.0}), [])

    def test_baseline_reference_drops_unusable_values(self):
        evidence, _, _ = _context(_shifted_repo())
        reference = baseline_reference(evidence)
        self.assertTrue(reference)
        self.assertNotIn(0.0, reference.values())


# ---------------------------------------------------------------------------
# Depth and onset
# ---------------------------------------------------------------------------


class DepthTests(unittest.TestCase):
    def test_stable_history_has_no_current_deviation(self):
        _, _, _, finding = _finding_for(_stable_repo())
        self.assertEqual(finding.depth, 0)
        self.assertEqual(finding.classification, "no_current_deviation")
        self.assertIsNone(finding.onset)
        self.assertFalse(finding.eligible_for_persistence_claim)

    def test_single_deviating_session_is_not_persistence(self):
        _, _, _, finding = _finding_for(_single_deviation_repo())
        self.assertEqual(finding.depth, 1)
        self.assertEqual(finding.classification, "single_session")
        self.assertFalse(finding.eligible_for_persistence_claim)

    def test_three_contiguous_sessions_are_a_recent_run(self):
        _, _, _, finding = _finding_for(_recent_run_repo())
        self.assertEqual(finding.depth, 3)
        self.assertEqual(finding.classification, "recent_run")
        self.assertFalse(finding.eligible_for_persistence_claim)

    def test_five_contiguous_sessions_are_sustained(self):
        _, _, _, finding = _finding_for(_shifted_repo())
        self.assertEqual(finding.depth, 5)
        self.assertEqual(finding.classification, "sustained")

    def test_one_interior_gap_is_tolerated(self):
        _, _, _, finding = _finding_for(_gapped_repo())
        self.assertEqual(finding.depth, 3)
        self.assertEqual(finding.classification, "recent_run")
        self.assertEqual(finding.deviating_feature_keys, ["typing_speed_cpm"])

    def test_onset_is_the_oldest_counted_session(self):
        repository = _shifted_repo()
        _, _, recent, finding = _finding_for(repository)
        expected = next(s for s in recent.sessions if s.session_id == "R0005")
        self.assertEqual(finding.onset, expected.session_start)
        self.assertEqual(finding.valid_recent_sessions, 5)

    def test_thin_history_never_produces_depth(self):
        registry, _, _, finding = _finding_for(_thin_repo(), session_id="S0001")
        self.assertIsNone(finding.depth)
        self.assertEqual(finding.classification, "insufficient_data")
        item = registry.filter(kind="persistence_depth")[0]
        self.assertFalse(item.available)
        self.assertTrue(item.unavailable_reason)
        self.assertFalse(finding.eligible_for_persistence_claim)
        self.assertEqual(finding.downgrade_reason, "insufficient_history")

    def test_as_of_excludes_sessions_outside_the_window(self):
        _, _, _, finding = _finding_for(
            _shifted_repo(), as_of=datetime.now(timezone.utc) - timedelta(days=60)
        )
        self.assertIsNone(finding.depth)
        self.assertEqual(finding.valid_recent_sessions, 0)

    def test_missing_recent_sessions_are_handled(self):
        evidence, anomaly, _ = _context(_shifted_repo())
        registry = _registry(evidence, anomaly)
        finding = assess_persistence(registry, evidence, None)
        self.assertIsNone(finding.depth)
        self.assertEqual(finding.classification, "insufficient_data")


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class RobustnessTests(unittest.TestCase):
    def test_agreeing_alternative_window(self):
        _, _, _, finding = _finding_for(_persistent_repo(), alt_drifts=[_alt_drift()])
        self.assertEqual(finding.robustness, "agrees")
        self.assertEqual(finding.robustness_comparisons, 1)
        self.assertEqual(finding.robustness_conflicts, 0)
        self.assertTrue(finding.eligible_for_persistence_claim)

    def test_disagreeing_alternative_window_blocks_the_claim(self):
        _, _, _, finding = _finding_for(
            _persistent_repo(), alt_drifts=[_alt_drift(reverse=True)]
        )
        self.assertEqual(finding.robustness, "disagrees")
        self.assertGreater(finding.robustness_conflicts, 0)
        self.assertFalse(finding.eligible_for_persistence_claim)
        self.assertEqual(finding.downgrade_reason, "window_robustness_disagrees")

    def test_uncomparable_windows_are_not_assessed(self):
        _, _, _, finding = _finding_for(
            _persistent_repo(), alt_drifts=[_alt_drift(available=False)]
        )
        self.assertEqual(finding.robustness, "not_assessed")
        self.assertEqual(finding.robustness_comparisons, 0)
        self.assertTrue(finding.eligible_for_persistence_claim)

    def test_assess_robustness_ignores_windows_without_overlap(self):
        status, comparisons, conflicts = assess_robustness({"typing_speed_cpm": -1}, [])
        self.assertEqual((status, comparisons, conflicts), ("not_assessed", 0, 0))

    def test_signatures_are_built_from_stats_and_drifts(self):
        evidence, _, _ = _context(_persistent_repo())
        stats_signature = moved_signature_from_stats(
            evidence.temporal_analysis.recent_feature_stats,
            evidence.temporal_analysis.baseline_feature_stats,
        )
        drift_signature = moved_signature_from_drift(_alt_drift())
        self.assertEqual(len(stats_signature), len(CANONICAL_FEATURE_KEYS))
        self.assertEqual(set(drift_signature), set(CANONICAL_FEATURE_KEYS))
        self.assertEqual(stats_signature["typing_speed_cpm"], -1)
        self.assertEqual(stats_signature["dwell_mean_s"], 1)


# ---------------------------------------------------------------------------
# Authoritative Phase 2 verdict
# ---------------------------------------------------------------------------


class Phase2AuthorityTests(unittest.TestCase):
    def test_phase2_verdict_is_carried_verbatim(self):
        _, evidence, _, finding = _finding_for(_shifted_repo())
        self.assertEqual(
            finding.phase2_status, evidence.temporal_analysis.persistence.status
        )
        self.assertEqual(finding.phase2_status, "recent_variation")

    def test_depth_cannot_override_a_non_persistent_phase2_verdict(self):
        _, _, _, finding = _finding_for(_shifted_repo())
        self.assertEqual(finding.depth, 5)
        self.assertEqual(finding.classification, "sustained")
        self.assertFalse(finding.eligible_for_persistence_claim)
        self.assertEqual(finding.downgrade_reason, "phase2_status_not_persistent")

    def test_ml_anomaly_alone_cannot_produce_persistence(self):
        registry, _, _, finding = _finding_for(_thin_repo(), session_id="S0001")
        self.assertTrue(registry.filter(kind="anomaly_unavailable"))
        self.assertFalse(finding.eligible_for_persistence_claim)
        self.assertEqual(finding.downgrade_reason, "insufficient_history")

    def test_eligible_only_when_every_rule_holds(self):
        _, evidence, _, finding = _finding_for(_persistent_repo())
        self.assertEqual(finding.phase2_status, "persistent_change")
        self.assertTrue(evidence.data_quality.meets_model_minimum)
        self.assertEqual(finding.classification, "sustained")
        self.assertNotEqual(finding.robustness, "disagrees")
        self.assertTrue(finding.eligible_for_persistence_claim)
        self.assertIsNone(finding.downgrade_reason)

    def test_phase2_persistence_item_is_not_recomputed_or_duplicated(self):
        repository = _persistent_repo()
        evidence, anomaly, recent = _context(repository)
        registry = _registry(evidence, anomaly)
        before = registry.filter(kind="persistence")
        assess_persistence(registry, evidence, recent, as_of=datetime.now(timezone.utc))
        after = registry.filter(kind="persistence")
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0], before[0])


# ---------------------------------------------------------------------------
# Evidence, determinism and constraints
# ---------------------------------------------------------------------------


class EvidenceTests(unittest.TestCase):
    def test_registers_depth_and_robustness_items(self):
        registry, _, _, finding = _finding_for(_shifted_repo())
        depth = registry.filter(kind="persistence_depth")
        robustness = registry.filter(kind="persistence_robustness")
        self.assertEqual(len(depth), 1)
        self.assertEqual(len(robustness), 1)

        self.assertEqual(depth[0].value, 5.0)
        self.assertEqual(depth[0].session_count, 5)
        self.assertEqual(depth[0].window_days, 7)
        self.assertEqual(depth[0].threshold, 5)
        self.assertEqual(depth[0].status, "sustained")
        self.assertIsNotNone(depth[0].onset)
        self.assertIn("consecutive valid session(s)", depth[0].statement)
        self.assertIn(finding.evidence_ids[0], [depth[0].id])

        self.assertEqual(robustness[0].status, "not_assessed")
        self.assertEqual(finding.evidence_ids, [depth[0].id, robustness[0].id])

    def test_new_items_continue_the_registry_ids(self):
        repository = _shifted_repo()
        evidence, anomaly, recent = _context(repository)
        registry = _registry(evidence, anomaly)
        before = len(registry)
        assess_persistence(registry, evidence, recent, as_of=datetime.now(timezone.utc))
        self.assertEqual(registry.ids()[before], f"E{before + 1}")

    def test_findings_are_deterministic(self):
        repository = _persistent_repo()
        evidence, anomaly, recent = _context(repository)
        first = EvidenceRegistry()
        second = EvidenceRegistry()
        register_from_behavioral_evidence(first, evidence, anomaly)
        register_from_behavioral_evidence(second, evidence, anomaly)

        as_of = datetime.now(timezone.utc)
        one = assess_persistence(first, evidence, recent, as_of=as_of, alt_drifts=[_alt_drift()])
        two = assess_persistence(second, evidence, recent, as_of=as_of, alt_drifts=[_alt_drift()])
        self.assertEqual(one.model_dump(), two.model_dump())
        self.assertEqual(first.digest(), second.digest())

    def test_trace_records_the_registered_evidence(self):
        repository = _shifted_repo()
        evidence, anomaly, recent = _context(repository)
        registry = _registry(evidence, anomaly)
        trace = InvestigationTrace(clock=lambda: datetime(2026, 9, 24, 9, 42, tzinfo=timezone.utc))
        finding = assess_persistence(
            registry, evidence, recent, as_of=datetime.now(timezone.utc), trace=trace
        )
        events = [event for event in trace if event.event_type == "evidence_added"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].evidence_ids, finding.evidence_ids)

    def test_no_repository_parameter(self):
        self.assertNotIn("repository", inspect.signature(assess_persistence).parameters)

    def test_registry_is_required(self):
        evidence, anomaly, recent = _context(_shifted_repo())
        with self.assertRaises(TypeError):
            assess_persistence(object(), evidence, recent)

    def test_caching_repository_output_can_be_analysed(self):
        repository = CachingRepository(_persistent_repo())
        evidence, anomaly, recent = _context(repository)
        registry = _registry(evidence, anomaly)
        finding = assess_persistence(registry, evidence, recent, as_of=datetime.now(timezone.utc))
        self.assertTrue(finding.eligible_for_persistence_claim)
        self.assertEqual(repository.stats()["list_sessions"], 1)

    def test_finding_carries_no_free_text_content_fields(self):
        _, _, _, finding = _finding_for(_shifted_repo())
        for forbidden in ("text", "content", "keystrokes", "message"):
            self.assertNotIn(forbidden, finding.model_dump())


if __name__ == "__main__":
    unittest.main()
