"""Tests for the Phase 3.1 evidence infrastructure.

Covers the claim templates, the EvidenceItem registry, the pure builders that
translate a real Phase 2 ``BehavioralEvidence`` package into evidence items, the
read-only caching repository wrapper, and the two additive ``AgentState`` fields.
"""

import json
import unittest

from investigation.adapter import build_behavioral_evidence
from investigation.agent import (
    CLAIM_TEMPLATES,
    CachingRepository,
    EvidenceItem,
    EvidenceRegistry,
    format_count,
    format_percent,
    format_value,
    register_from_behavioral_evidence,
    render_claim,
)
from investigation import tools
from investigation.repository import RepositoryError
from investigation.state import AgentState
from tests import fixtures


def _shifted_repo(anomalies=None):
    """A history that is stable, then shifts only typing speed recently."""
    sessions = [
        fixtures.make_session(session_id=f"S{i:04d}", days_ago=i, typing_speed=285.0)
        for i in range(7, 36)
    ]
    sessions += [
        fixtures.make_session(session_id=f"R{i:04d}", days_ago=i, typing_speed=233.0)
        for i in range(1, 6)
    ]
    return fixtures.FakeRepository(sessions=sessions, anomalies=anomalies or {})


def _anomaly_output(repository):
    return tools.get_ml_evidence("user-1", "R0001", repository=repository)


class FormattingTests(unittest.TestCase):
    def test_missing_values_are_explicit(self):
        self.assertEqual(format_value(None), "unavailable")
        self.assertEqual(format_count(None), "unavailable")
        self.assertIsNone(format_percent(None))

    def test_counts_render_without_decimals(self):
        self.assertEqual(format_value(4.0, "count"), "4")
        self.assertEqual(format_count(4.0), "4")

    def test_percentages_are_unsigned_and_rounded(self):
        self.assertEqual(format_percent(-0.182456), "18.2%")
        self.assertEqual(format_percent(0.05), "5.0%")


class ClaimTemplateTests(unittest.TestCase):
    def test_signal_change_renders_exactly(self):
        statement = render_claim(
            "signal_change",
            label="Typing speed",
            unit="characters per minute",
            direction="decrease",
            value=233.0,
            baseline_value=285.0,
            relative_change=-0.182456,
            window_days=30,
        )
        self.assertEqual(
            statement,
            "Typing speed decreased 18.2% from the 30-day baseline mean "
            "(285.00 \u2192 233.00 characters per minute).",
        )

    def test_persistence_statement_names_its_status(self):
        statement = render_claim(
            "persistence", status="recent_variation", value=1.0, session_count=5
        )
        self.assertEqual(
            statement,
            'The deterministic persistence assessment is "recent_variation" from '
            "1 moved signal(s) over 5 recent valid session(s).",
        )

    def test_anomaly_statement_labels_the_score_as_not_a_diagnosis(self):
        statement = render_claim("anomaly", value=0.62, status="true")
        self.assertIn("0.6200", statement)
        self.assertIn("true", statement)
        self.assertIn("not a diagnosis", statement.lower())

    def test_session_count_statement_includes_the_threshold(self):
        statement = render_claim("session_count", session_count=3, threshold=10)
        self.assertEqual(
            statement,
            "3 valid session(s) are available; at least 10 are required for "
            "model-based evidence.",
        )

    def test_context_absence_is_stated_honestly(self):
        statement = render_claim("context_absence")
        self.assertIn("No server-side contextual factors", statement)
        self.assertIn("cannot be confirmed or ruled out", statement)

    def test_templates_are_exact_and_repeatable(self):
        fields = {
            "label": "Dwell time",
            "unit": "seconds",
            "direction": "increase",
            "value": 0.2,
            "baseline_value": 0.12,
            "relative_change": 0.6667,
            "window_days": 30,
        }
        first = render_claim("signal_change", **fields)
        second = render_claim("signal_change", **fields)
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            "Dwell time increased 66.7% from the 30-day baseline mean "
            "(0.12 \u2192 0.20 seconds).",
        )

    def test_every_template_is_total_with_no_fields(self):
        for name, renderer in CLAIM_TEMPLATES.items():
            with self.subTest(template=name):
                self.assertTrue(renderer({}))

    def test_unknown_template_is_rejected(self):
        with self.assertRaises(KeyError):
            render_claim("invent_something", value=1.0)

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValueError):
            render_claim("quality", value=1, nonsense=2)


class EvidenceRegistryTests(unittest.TestCase):
    def test_ids_are_sequential_and_stable(self):
        registry = EvidenceRegistry()
        first = registry.register("quality", kind="quality", source_tool="t", value=1, session_count=2)
        second = registry.register("session_count", kind="session_count", source_tool="t", session_count=1, threshold=10)
        self.assertEqual([first.id, second.id], ["E1", "E2"])
        self.assertEqual(registry.ids(), ["E1", "E2"])
        self.assertIs(registry.get("E1"), first)
        self.assertIsNone(registry.get("E99"))
        self.assertEqual(len(registry), 2)

    def test_statements_come_from_the_templates(self):
        registry = EvidenceRegistry()
        fields = {"value": 1, "session_count": 2}
        item = registry.register("quality", kind="quality", source_tool="t", **fields)
        self.assertEqual(item.statement, render_claim("quality", value=1, session_count=2))

    def test_callers_cannot_supply_generated_fields(self):
        registry = EvidenceRegistry()
        with self.assertRaises(ValueError):
            registry.register(
                "quality", kind="quality", source_tool="t", value=1,
                session_count=2, statement="hand written",
            )
        with self.assertRaises(ValueError):
            registry.register("quality", kind="quality", source_tool="t", id="E1")

    def test_unavailable_items_record_a_reason(self):
        registry = EvidenceRegistry()
        item = registry.register_unavailable(
            "tool_unavailable", kind="tool_unavailable",
            source_tool="get_ml_evidence", reason="no_anomaly_record",
        )
        self.assertFalse(item.available)
        self.assertEqual(item.unavailable_reason, "no_anomaly_record")
        self.assertIn("no_anomaly_record", item.statement)

    def test_filter_by_kind_and_key(self):
        registry = EvidenceRegistry()
        registry.register("quality", kind="quality", source_tool="t", value=1, session_count=2)
        registry.register(
            "window_stat", kind="window_stat", source_tool="t",
            key="typing_speed_cpm", label="Typing speed",
            unit="characters per minute", value=10.0,
            session_count=3, window_days=7,
        )
        self.assertEqual(len(registry.filter(kind="quality")), 1)
        self.assertEqual(len(registry.filter(key="typing_speed_cpm")), 1)
        self.assertEqual(registry.filter(kind="window_stat", key="dwell_mean_s"), [])

    def test_digest_is_stable_and_sensitive_to_values(self):
        def build(value):
            registry = EvidenceRegistry()
            registry.register("quality", kind="quality", source_tool="t", value=value, session_count=2)
            return registry.digest()

        self.assertEqual(build(1), build(1))
        self.assertNotEqual(build(1), build(2))

    def test_items_round_trip_through_dicts(self):
        registry = EvidenceRegistry()
        registry.register("quality", kind="quality", source_tool="t", value=1, session_count=2)
        dumped = registry.to_dicts()
        json.dumps(dumped)
        self.assertEqual(EvidenceItem(**dumped[0]), registry.all()[0])


class BuilderTests(unittest.TestCase):
    def test_builder_registers_canonical_signal_evidence(self):
        repository = _shifted_repo()
        evidence = build_behavioral_evidence("user-1", "R0001", repository=repository)
        registry = EvidenceRegistry()

        items = register_from_behavioral_evidence(
            registry, evidence, _anomaly_output(repository)
        )

        changes = registry.filter(kind="signal_change")
        self.assertEqual(len(changes), len(evidence.signals))
        typing = [i for i in changes if i.key == "typing_speed_cpm"]
        self.assertEqual(len(typing), 1)
        self.assertEqual(typing[0].unit, "characters per minute")
        self.assertEqual(typing[0].direction, "decrease")
        self.assertEqual(typing[0].window_days, 30)
        self.assertEqual(
            typing[0].statement,
            "Typing speed decreased 18.2% from the 30-day baseline mean "
            "(285.00 \u2192 233.00 characters per minute).",
        )
        self.assertEqual(items, registry.all())

    def test_builder_registers_quality_persistence_and_window_evidence(self):
        repository = _shifted_repo()
        evidence = build_behavioral_evidence("user-1", "R0001", repository=repository)
        registry = EvidenceRegistry()
        register_from_behavioral_evidence(registry, evidence, _anomaly_output(repository))

        self.assertTrue(registry.filter(kind="quality"))
        self.assertTrue(registry.filter(kind="session_count"))
        self.assertTrue(registry.filter(kind="cadence"))
        self.assertTrue(registry.filter(kind="window_stat"))

        persistence = registry.filter(kind="persistence")
        self.assertEqual(len(persistence), 1)
        self.assertEqual(persistence[0].status, "recent_variation")
        self.assertEqual(
            persistence[0].status, evidence.temporal_analysis.persistence.status
        )

    def test_builder_registers_the_ml_evidence_as_a_number(self):
        repository = _shifted_repo(
            {"R0001": {"anomaly_score": 0.62, "is_anomaly": True}}
        )
        evidence = build_behavioral_evidence("user-1", "R0001", repository=repository)
        registry = EvidenceRegistry()
        register_from_behavioral_evidence(registry, evidence, _anomaly_output(repository))

        anomaly = registry.filter(kind="anomaly")
        self.assertEqual(len(anomaly), 1)
        self.assertEqual(anomaly[0].value, 0.62)
        self.assertEqual(anomaly[0].status, "true")
        self.assertEqual(anomaly[0].source_tool, "get_ml_evidence")
        self.assertFalse(registry.filter(kind="anomaly_unavailable"))

    def test_missing_ml_evidence_is_registered_as_missing(self):
        repository = _shifted_repo()
        evidence = build_behavioral_evidence("user-1", "R0001", repository=repository)

        # (a) the ML tool was never called for this investigation.
        uncalled = EvidenceRegistry()
        register_from_behavioral_evidence(uncalled, evidence)
        missing = uncalled.filter(kind="anomaly_unavailable")
        self.assertEqual(len(missing), 1)
        self.assertFalse(missing[0].available)
        self.assertIn("not called", missing[0].statement)

        # (b) the tool ran, but the store holds no anomaly row for the session.
        called = EvidenceRegistry()
        register_from_behavioral_evidence(called, evidence, _anomaly_output(repository))
        missing = called.filter(kind="anomaly_unavailable")
        self.assertEqual(len(missing), 1)
        self.assertFalse(missing[0].available)
        self.assertIn("no_anomaly_record", missing[0].statement)

    def test_absent_context_is_registered_as_absent_not_invented(self):
        repository = _shifted_repo()
        evidence = build_behavioral_evidence("user-1", "R0001", repository=repository)
        registry = EvidenceRegistry()
        register_from_behavioral_evidence(registry, evidence)

        self.assertEqual(len(registry.filter(kind="context_absence")), 1)
        self.assertFalse(registry.filter(kind="context_present"))

    def test_unknown_session_registers_unavailable_signals(self):
        repository = _shifted_repo()
        evidence = build_behavioral_evidence("user-1", "MISSING", repository=repository)
        registry = EvidenceRegistry()
        register_from_behavioral_evidence(registry, evidence)

        unavailable = registry.filter(kind="signal_unavailable")
        self.assertEqual(len(unavailable), len(evidence.signals))
        for item in unavailable:
            self.assertFalse(item.available)
            self.assertTrue(item.unavailable_reason)

    def test_insufficient_history_cannot_produce_persistent_change_evidence(self):
        repository = fixtures.FakeRepository(
            sessions=[
                fixtures.make_session(session_id=f"S{i:04d}", days_ago=i)
                for i in range(3)
            ]
        )
        evidence = build_behavioral_evidence("user-1", "S0000", repository=repository)
        registry = EvidenceRegistry()
        register_from_behavioral_evidence(registry, evidence)

        persistence = registry.filter(kind="persistence")[0]
        self.assertEqual(persistence.status, "insufficient_data")
        for item in registry.all():
            self.assertNotIn("persistent_change", item.statement)

    def test_builder_is_deterministic(self):
        repository = _shifted_repo(
            {"R0001": {"anomaly_score": 0.62, "is_anomaly": True}}
        )
        evidence = build_behavioral_evidence("user-1", "R0001", repository=repository)
        anomaly = _anomaly_output(repository)

        first, second = EvidenceRegistry(), EvidenceRegistry()
        register_from_behavioral_evidence(first, evidence, anomaly)
        register_from_behavioral_evidence(second, evidence, anomaly)

        self.assertEqual(first.to_dicts(), second.to_dicts())
        self.assertEqual(first.digest(), second.digest())


class WriteSpyRepository(fixtures.FakeRepository):
    """Read repository that records any attempt to write."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writes = []

    def insert(self, *args, **kwargs):
        self.writes.append("insert")
        raise AssertionError("the agent layer must never write")

    def update(self, *args, **kwargs):
        self.writes.append("update")
        raise AssertionError("the agent layer must never write")

    def delete(self, *args, **kwargs):
        self.writes.append("delete")
        raise AssertionError("the agent layer must never write")

    def upsert(self, *args, **kwargs):
        self.writes.append("upsert")
        raise AssertionError("the agent layer must never write")


class CachingRepositoryTests(unittest.TestCase):
    def test_repeated_window_tools_collapse_to_one_read(self):
        upstream = _shifted_repo()
        caching = CachingRepository(upstream)

        build_behavioral_evidence("user-1", "R0001", repository=caching)
        stats = caching.stats()

        self.assertEqual(stats["list_sessions"], 1)
        self.assertEqual(stats["get_anomaly_result"], 1)
        self.assertGreaterEqual(stats["cache_hits"], 1)

    def test_cache_hits_are_served_without_touching_the_upstream(self):
        upstream = WriteSpyRepository(sessions=[fixtures.make_session(session_id="A")])
        caching = CachingRepository(upstream)

        first = caching.list_sessions("user-1")
        second = caching.list_sessions("user-1")

        self.assertEqual(first, second)
        self.assertEqual(caching.stats()["list_sessions"], 1)
        self.assertEqual(caching.stats()["cache_hits"], 1)

    def test_cached_rows_cannot_be_mutated_by_a_caller(self):
        caching = CachingRepository(
            fixtures.FakeRepository(sessions=[fixtures.make_session(session_id="A")])
        )
        rows = caching.list_sessions("user-1")
        rows[0]["id"] = "MUTATED"
        self.assertEqual(caching.list_sessions("user-1")[0]["id"], "A")

    def test_write_methods_are_unreachable(self):
        upstream = WriteSpyRepository(sessions=[fixtures.make_session()])
        caching = CachingRepository(upstream)

        for name in ("insert", "update", "delete", "upsert", "table"):
            self.assertFalse(hasattr(caching, name), name)

        evidence = build_behavioral_evidence("user-1", "S0001", repository=caching)
        self.assertTrue(evidence.signals)
        self.assertEqual(upstream.writes, [])

    def test_upstream_returns_the_wrapped_repository(self):
        upstream = _shifted_repo()
        self.assertIs(CachingRepository(upstream).upstream(), upstream)

    def test_failures_are_not_cached_and_propagate(self):
        caching = CachingRepository(fixtures.FailingRepository())

        for _ in range(2):
            with self.assertRaises(RepositoryError):
                caching.list_sessions("user-1")

        stats = caching.stats()
        self.assertEqual(stats["list_sessions"], 0)
        self.assertEqual(stats["cache_hits"], 0)

    def test_caching_repository_preserves_evidence_results(self):
        repository = _shifted_repo({"R0001": {"anomaly_score": 0.62, "is_anomaly": True}})
        direct = build_behavioral_evidence("user-1", "R0001", repository=repository)
        cached = build_behavioral_evidence(
            "user-1", "R0001", repository=CachingRepository(repository)
        )

        # Phase 2 derives window bounds and generated_at from the wall clock, so
        # those timestamps differ between any two builds. Everything measured or
        # computed must be identical through the cache.
        self.assertEqual(direct.signals, cached.signals)
        self.assertEqual(direct.data_quality, cached.data_quality)
        self.assertEqual(direct.context, cached.context)
        self.assertEqual(direct.units, cached.units)
        self.assertEqual(direct.limitations, cached.limitations)
        self.assertEqual(
            direct.temporal_analysis.persistence,
            cached.temporal_analysis.persistence,
        )


class AgentStateFieldTests(unittest.TestCase):
    def test_new_fields_default_to_empty(self):
        state = AgentState()
        self.assertEqual(state.evidence, [])
        self.assertEqual(state.trace, [])

    def test_new_fields_are_independent_per_instance(self):
        first, second = AgentState(), AgentState()
        first.evidence.append({"id": "E1"})
        first.trace.append({"seq": 1})
        self.assertEqual(second.evidence, [])
        self.assertEqual(second.trace, [])

    def test_registry_dicts_fit_the_state_field(self):
        registry = EvidenceRegistry()
        registry.register("quality", kind="quality", source_tool="t", value=1, session_count=2)
        state = AgentState()
        state.evidence = registry.to_dicts()
        self.assertEqual(state.evidence[0]["id"], "E1")
        self.assertEqual(EvidenceItem(**state.evidence[0]).statement, registry.all()[0].statement)


class PrivacyTests(unittest.TestCase):
    def test_evidence_contains_no_free_text_fields(self):
        repository = _shifted_repo()
        evidence = build_behavioral_evidence("user-1", "R0001", repository=repository)
        registry = EvidenceRegistry()
        register_from_behavioral_evidence(registry, evidence)

        keys = set()
        for item in registry.to_dicts():
            keys |= set(item)

        for forbidden in ("text", "content", "keystrokes", "message", "note_text"):
            self.assertNotIn(forbidden, keys)

    def test_evidence_only_reads_canonical_behavioral_keys(self):
        repository = _shifted_repo()
        evidence = build_behavioral_evidence("user-1", "R0001", repository=repository)
        registry = EvidenceRegistry()
        register_from_behavioral_evidence(registry, evidence)

        signal_keys = {item.key for item in registry.filter(kind="signal_change")}
        self.assertTrue(signal_keys <= set(evidence.units))


if __name__ == "__main__":
    unittest.main()
