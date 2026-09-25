"""Tests for the Phase 3.4 deterministic critic.

The critic must reject unsupported and contradictory claims while accepting a
sound investigation. Most checks are exercised both on a real engine result and
on deliberately malformed inputs, so the critic is proven to change behaviour
rather than merely return ``is_sound=True``.
"""

import copy
import inspect
import unittest
from datetime import datetime, timezone
from typing import get_args
from unittest import mock

from investigation.agent import (
    CriticReason,
    InvestigationTrace,
    critique,
    evidence_index,
    run_investigation,
    trace_digest,
)
from investigation.agent.critic import (
    CLINICAL_TERMS,
    PERSISTENCE_CLAIM_HYPOTHESES,
    SAFETY_EXEMPTIONS,
)
from investigation.agent.trace import TraceEvent
from investigation.repository import RepositoryError
from tests import fixtures

USER = "user-1"
CLK = lambda: datetime(2026, 9, 24, 9, 42, tzinfo=timezone.utc)  # noqa: E731


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sessions(tail=None, base=285.0):
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=base)
        for i in range(7, 36)
    ]
    sessions += tail or []
    return sessions


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


def _run(repository, session_id="R0001", **kwargs):
    kwargs.setdefault("as_of", datetime.now(timezone.utc))
    kwargs.setdefault("trace", InvestigationTrace(clock=CLK))
    if repository is None:
        kwargs.pop("trace")
    return run_investigation(USER, session_id, repository=repository, **kwargs)


def _unavailable_id(result):
    for item in result.state.evidence:
        if item.get("available") is False:
            return item["id"]
    raise AssertionError("fixture produced no unavailable evidence")



# ---------------------------------------------------------------------------
# Trace digest
# ---------------------------------------------------------------------------


class TraceDigestTests(unittest.TestCase):
    def test_digest_is_deterministic(self):
        events = _run(_persistent_repo()).state.trace
        self.assertEqual(trace_digest(events), trace_digest(events))

    def test_digest_ignores_timestamps(self):
        events = [
            {
                "seq": 1,
                "ts": "2026-01-01T00:00:00+00:00",
                "node": "n",
                "event_type": "decision",
                "evidence_ids": [],
                "detail": "d",
            },
            {
                "seq": 2,
                "ts": "2030-06-06T06:06:06+00:00",
                "node": "n",
                "event_type": "decision",
                "evidence_ids": [],
                "detail": "d",
            },
        ]
        later = [dict(event, ts="2099-12-31T23:59:59+00:00") for event in events]
        self.assertEqual(trace_digest(events), trace_digest(later))

    def test_digest_is_sensitive_to_evidence_ids(self):
        first = [{"seq": 1, "event_type": "evidence_added", "evidence_ids": ["E1"]}]
        second = [{"seq": 1, "event_type": "evidence_added", "evidence_ids": ["E2"]}]
        self.assertNotEqual(trace_digest(first), trace_digest(second))

    def test_digest_accepts_models_and_dicts(self):
        event = TraceEvent(seq=1, ts=datetime.now(timezone.utc), node="n", event_type="stop")
        self.assertEqual(trace_digest([event]), trace_digest([event.model_dump(mode="json")]))

    def test_digest_of_nothing_is_stable(self):
        self.assertEqual(trace_digest([]), trace_digest(None))


# ---------------------------------------------------------------------------
# Evidence index
# ---------------------------------------------------------------------------


class EvidenceIndexTests(unittest.TestCase):
    def test_index_is_keyed_by_id_and_validates_items(self):
        result = _run(_persistent_repo())
        index = evidence_index(result.state.evidence)
        self.assertEqual(len(index), len(result.state.evidence))
        for item_id, item in index.items():
            self.assertEqual(item.id, item_id)

    def test_malformed_records_are_skipped(self):
        index = evidence_index([{"id": "E1"}, {"kind": "quality"}])
        self.assertEqual(index, {})

    def test_empty_input(self):
        self.assertEqual(evidence_index(None), {})


# ---------------------------------------------------------------------------
# Sound investigations are accepted
# ---------------------------------------------------------------------------


class SoundnessTests(unittest.TestCase):
    def test_persistent_scenario_is_sound(self):
        verdict = critique(_run(_persistent_repo()))
        self.assertTrue(verdict.is_sound)
        self.assertEqual(verdict.rejections, [])
        self.assertEqual(verdict.contradictions, [])

    def test_shifted_stable_and_thin_scenarios_are_sound(self):
        for repository, session_id in (
            (_shifted_repo(), "R0001"),
            (_stable_repo(), "S0001"),
            (_thin_repo(), "S0001"),
        ):
            with self.subTest(session=session_id):
                self.assertTrue(critique(_run(repository, session_id)).is_sound)

    def test_data_unavailable_scenario_is_sound(self):
        with mock.patch(
            "investigation.agent.engine.default_repository",
            side_effect=RepositoryError("no client"),
        ):
            result = run_investigation(USER, "R0001", repository=None)
        self.assertTrue(critique(result).is_sound)

    def test_accepted_sets_cover_the_catalogs(self):
        verdict = critique(_run(_persistent_repo()))
        self.assertEqual(
            verdict.accepted_hypothesis_ids, ["H1", "H2", "H3", "H4"]
        )
        self.assertEqual(len(verdict.accepted_alternative_candidates), 11)
        self.assertTrue(verdict.accepted_evidence_ids)

    def test_denied_count_matches_rejections(self):
        result = _run(_persistent_repo())
        result.hypotheses[2].supporting_evidence = []
        verdict = critique(result)
        self.assertEqual(verdict.denied_claim_count, len(verdict.rejections))
        self.assertFalse(verdict.is_sound)


# ---------------------------------------------------------------------------
# Grounding rejections
# ---------------------------------------------------------------------------


class GroundingTests(unittest.TestCase):
    def test_unresolved_evidence_id_is_rejected(self):
        result = _run(_persistent_repo())
        result.hypotheses[2].supporting_evidence.append("E999")
        verdict = critique(result)
        self.assertIn(
            "evidence_id_unresolved", [r.reason for r in verdict.rejections]
        )
        self.assertIn("H3", verdict.rejected_subjects)

    def test_supported_without_evidence_is_rejected(self):
        result = _run(_persistent_repo())
        result.hypotheses[2].supporting_evidence = []
        verdict = critique(result)
        self.assertIn("support_without_evidence", [r.reason for r in verdict.rejections])

    def test_weakened_without_contradiction_is_rejected(self):
        result = _run(_persistent_repo())
        result.hypotheses[0].contradicting_evidence = []
        verdict = critique(result)
        self.assertIn("support_without_evidence", [r.reason for r in verdict.rejections])

    def test_uncertain_with_evidence_is_rejected(self):
        result = _run(_persistent_repo())
        first = result.state.evidence[0]["id"]
        result.hypotheses[1].supporting_evidence = [first]
        verdict = critique(result)
        self.assertIn("support_without_evidence", [r.reason for r in verdict.rejections])

    def test_overlapping_support_and_contradiction_is_rejected(self):
        result = _run(_persistent_repo())
        first = result.state.evidence[0]["id"]
        result.hypotheses[2].supporting_evidence = [first]
        result.hypotheses[2].contradicting_evidence = [first]
        verdict = critique(result)
        self.assertIn(
            "support_contradiction_overlap", [r.reason for r in verdict.rejections]
        )

    def test_persistent_change_without_eligibility_is_rejected(self):
        result = _run(_thin_repo(), "S0001")
        self.assertFalse(result.final_assessment["persistence"]["eligible"])
        result.hypotheses[2].status = "supported"
        result.hypotheses[2].supporting_evidence = [result.state.evidence[0]["id"]]
        verdict = critique(result)
        self.assertIn(
            "persistent_change_without_eligibility",
            [r.reason for r in verdict.rejections],
        )

    def test_foreign_hypothesis_is_rejected(self):
        result = _run(_persistent_repo())
        result.hypotheses[2].hypothesis = "Cognitive decline"
        verdict = critique(result)
        self.assertIn("foreign_hypothesis", [r.reason for r in verdict.rejections])

    def test_foreign_alternative_is_rejected(self):
        result = _run(_persistent_repo())
        result.alternatives[0].candidate = "made_up_candidate"
        verdict = critique(result)
        self.assertIn("foreign_alternative", [r.reason for r in verdict.rejections])

    def test_decided_alternative_without_evidence_is_rejected(self):
        result = _run(_persistent_repo())
        result.alternatives[4].status = "supported"
        result.alternatives[4].evidence_ids = []
        verdict = critique(result)
        self.assertIn(
            "alternative_without_evidence", [r.reason for r in verdict.rejections]
        )

    def test_persistent_claim_cannot_cite_unavailable_evidence(self):
        result = _run(_thin_repo(), "S0001")
        unavailable = _unavailable_id(result)
        result.hypotheses[2].status = "supported"
        result.hypotheses[2].supporting_evidence = [unavailable]
        verdict = critique(result)
        rejections = [r for r in verdict.rejections if r.subject == "H3"]
        self.assertIn("unavailable_evidence_asserted", [r.reason for r in rejections])

    def test_conservative_and_data_quality_claims_may_cite_unavailable(self):
        # Phase 3.2 supports H1 and H4 with the uncomputable persistence depth
        # and unavailable tool results; those are legitimate for non-persistent
        # claims, so the sound verdict must keep them.
        result = _run(_thin_repo(), "S0001")
        unavailable = {
            item["id"] for item in result.state.evidence if item.get("available") is False
        }
        self.assertTrue(unavailable)
        self.assertTrue(set(result.hypotheses[3].supporting_evidence) & unavailable)
        verdict = critique(result)
        self.assertTrue(verdict.is_sound)
        self.assertNotIn("H1", verdict.rejected_subjects)
        self.assertNotIn("H4", verdict.rejected_subjects)
        self.assertNotIn("technical_failure", verdict.rejected_subjects)

    def test_policy_constants_are_explicit(self):
        self.assertEqual(PERSISTENCE_CLAIM_HYPOTHESES, frozenset({"H3"}))


# ---------------------------------------------------------------------------
# Contradictions
# ---------------------------------------------------------------------------


class ContradictionTests(unittest.TestCase):
    def test_mutually_exclusive_hypotheses_are_detected(self):
        result = _run(_thin_repo(), "S0001")
        result.hypotheses[2].status = "supported"
        result.hypotheses[2].supporting_evidence = [result.state.evidence[0]["id"]]
        result.final_assessment["persistence"]["eligible"] = True
        verdict = critique(result)
        self.assertIn(
            "mutually_exclusive_hypotheses", [c.code for c in verdict.contradictions]
        )

    def test_assessment_mismatches_are_detected(self):
        mutations = {
            "stop_reason": ("final_assessment", "stop_reason", "not_a_reason"),
            "supported_hypotheses": ("final_assessment", "supported_hypotheses", []),
            "missing_evidence": ("final_assessment", "open_questions", []),
        }
        for label, (_, key, value) in mutations.items():
            with self.subTest(check=label):
                result = _run(_persistent_repo())
                result.final_assessment[key] = value
                verdict = critique(result)
                self.assertIn(
                    "assessment_mismatch", [c.code for c in verdict.contradictions]
                )

    def test_hypothesis_status_mismatch_is_detected(self):
        result = _run(_persistent_repo())
        result.final_assessment["hypotheses"][2]["status"] = "uncertain"
        verdict = critique(result)
        self.assertIn(
            "assessment_mismatch", [c.code for c in verdict.contradictions]
        )

    def test_evidence_digest_mismatch_is_detected(self):
        result = _run(_persistent_repo())
        result.final_assessment["evidence_digest"] = "0" * 64
        verdict = critique(result)
        self.assertIn(
            "evidence_digest_mismatch", [c.code for c in verdict.contradictions]
        )

    def test_temporary_variation_conflict_is_detected(self):
        result = _run(_persistent_repo())
        temporary = next(a for a in result.alternatives if a.candidate == "temporary_variation")
        temporary.status = "supported"
        verdict = critique(result)
        self.assertIn(
            "hypothesis_alternative_conflict", [c.code for c in verdict.contradictions]
        )

    def test_insufficient_data_conflict_is_detected(self):
        result = _run(_persistent_repo())
        insufficient = next(
            a for a in result.alternatives if a.candidate == "insufficient_data"
        )
        insufficient.status = "supported"
        verdict = critique(result)
        self.assertIn(
            "hypothesis_alternative_conflict", [c.code for c in verdict.contradictions]
        )

    def test_contradiction_codes_are_in_the_vocabulary(self):
        result = _run(_persistent_repo())
        result.final_assessment["stop_reason"] = "not_a_reason"
        allowed = set(get_args(CriticReason))
        for contradiction in critique(result).contradictions:
            self.assertIn(contradiction.code, allowed)


# ---------------------------------------------------------------------------
# Clinical language
# ---------------------------------------------------------------------------


class ClinicalLanguageTests(unittest.TestCase):
    def test_fixed_phase2_negations_are_not_rejected(self):
        verdict = critique(_run(_persistent_repo()))
        self.assertEqual(
            [r for r in verdict.rejections if r.reason == "clinical_language"], []
        )
        self.assertIn("diagnos", CLINICAL_TERMS)
        self.assertTrue(SAFETY_EXEMPTIONS)

    def test_clinical_evidence_statement_is_rejected(self):
        result = _run(_persistent_repo())
        target = result.state.evidence[0]["id"]
        result.state.evidence[0]["statement"] = "Signs of dementia are present."
        verdict = critique(result)
        reasons = {r.subject: r.reason for r in verdict.rejections}
        self.assertEqual(reasons.get(target), "clinical_language")
        self.assertNotIn(target, verdict.accepted_evidence_ids)

    def test_clinical_next_action_is_rejected(self):
        result = _run(_persistent_repo())
        result.state.next_action = "Screen for early dementia."
        verdict = critique(result)
        reasons = {r.subject: r.reason for r in verdict.rejections}
        self.assertEqual(reasons.get("next_action"), "clinical_language")

    def test_clinical_limitation_is_rejected(self):
        result = _run(_persistent_repo())
        result.limitations.append("This may indicate a clinical disorder.")
        verdict = critique(result)
        self.assertIn("clinical_language", [r.reason for r in verdict.rejections])


# ---------------------------------------------------------------------------
# Trace recording, purity, constraints
# ---------------------------------------------------------------------------


class TraceAndPurityTests(unittest.TestCase):
    def test_from_events_preserves_and_continues_the_sequence(self):
        result = _run(_persistent_repo())
        trace = InvestigationTrace.from_events(result.state.trace, clock=CLK)
        self.assertEqual(len(trace), len(result.state.trace))
        self.assertEqual([e.seq for e in trace], [e["seq"] for e in result.state.trace])
        event = trace.record_state_change("critic", detail="probe")
        self.assertEqual(event.seq, len(result.state.trace) + 1)

    def test_critic_rejections_are_recorded_with_continuing_seqs(self):
        result = _run(_persistent_repo())
        result.hypotheses[2].supporting_evidence = []
        trace = InvestigationTrace.from_events(result.state.trace, clock=CLK)
        verdict = critique(result, trace=trace)
        rejections = [e for e in trace if e.event_type == "critic_rejection"]
        self.assertEqual(len(rejections), len(verdict.rejections))
        self.assertEqual(rejections[0].seq, len(result.state.trace) + 1)

    def test_sound_verdict_records_a_decision(self):
        trace = InvestigationTrace(clock=CLK)
        critique(_run(_persistent_repo()), trace=trace)
        decisions = [e for e in trace if e.event_type == "decision"]
        self.assertEqual(len(decisions), 1)
        self.assertIn("is_sound=True", decisions[0].detail)

    def test_no_trace_means_no_side_effects(self):
        result = _run(_persistent_repo())
        before = copy.deepcopy(result.state.trace)
        critique(result)
        self.assertEqual(result.state.trace, before)

    def test_result_is_not_mutated(self):
        result = _run(_persistent_repo())
        before = result.model_dump(mode="json")
        critique(result)
        self.assertEqual(result.model_dump(mode="json"), before)

    def test_no_repository_parameter(self):
        self.assertNotIn("repository", inspect.signature(critique).parameters)


if __name__ == "__main__":
    unittest.main()
