"""Tests for the investigation trace infrastructure (Phase 3.1)."""

import json
import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from investigation.agent import (
    EVENT_LABELS,
    InvestigationTrace,
    args_fingerprint,
)

FIXED_TS = datetime(2026, 9, 24, 9, 42, 0, tzinfo=timezone.utc)

#: The only keys a trace event may contain.
ALLOWED_EVENT_KEYS = {
    "seq",
    "ts",
    "node",
    "event_type",
    "tool",
    "args_fingerprint",
    "evidence_ids",
    "detail",
}


def _fixed_clock():
    return FIXED_TS


def _sample_trace(clock=None):
    trace = InvestigationTrace(clock=clock)
    trace.record_enter("ingest_signal")
    trace.record_tool_call(
        "ingest_signal",
        "get_ml_evidence",
        {"user_id": "user-1", "session_id": "S0001"},
    )
    trace.record_tool_result("ingest_signal", "get_ml_evidence")
    trace.record_evidence("ingest_signal", ["E1", "E2"], detail="6 signals registered")
    trace.record_decision("decide_next", "collect")
    trace.record_stop("build_report", "stop_insufficient_data")
    return trace


class TraceEventTests(unittest.TestCase):
    def test_sequence_numbers_start_at_one_and_increase(self):
        trace = InvestigationTrace()
        first = trace.record_enter("ingest_signal")
        second = trace.record_exit("ingest_signal")
        self.assertEqual([first.seq, second.seq], [1, 2])
        self.assertEqual(len(trace), 2)

    def test_event_type_vocabulary_is_closed(self):
        trace = InvestigationTrace()
        with self.assertRaises(ValidationError):
            trace.record("ingest_signal", "not_a_real_event_type")

    def test_every_event_type_has_a_label(self):
        for label in EVENT_LABELS.values():
            self.assertTrue(label)
        trace = _sample_trace()
        for event in trace:
            self.assertIn(event.event_type, EVENT_LABELS)

    def test_clock_injection_makes_the_trace_deterministic(self):
        first = _sample_trace(clock=_fixed_clock).to_dicts()
        second = _sample_trace(clock=_fixed_clock).to_dicts()
        self.assertEqual(first, second)
        self.assertEqual(first[0]["ts"].startswith("2026-09-24T09:42:00"), True)

    def test_timeline_renders_fixed_labels(self):
        lines = _sample_trace(clock=_fixed_clock).timeline()
        self.assertEqual(lines[0], "09:42:00 \u2014 Entered ingest_signal")
        self.assertIn("Called tool get_ml_evidence", lines[1])
        self.assertTrue(any("Stopped" in line for line in lines))

    def test_tool_call_records_a_fingerprint_and_not_the_raw_arguments(self):
        trace = _sample_trace(clock=_fixed_clock)
        call = trace.events[1]
        self.assertEqual(call.tool, "get_ml_evidence")
        self.assertEqual(len(call.args_fingerprint), 16)

        blob = json.dumps(trace.to_dicts())
        self.assertNotIn("S0001", blob)
        self.assertNotIn("user-1", blob)
        self.assertNotIn("session_id", blob)

    def test_evidence_ids_and_details_are_recorded(self):
        trace = _sample_trace(clock=_fixed_clock)
        event = trace.events[3]
        self.assertEqual(event.event_type, "evidence_added")
        self.assertEqual(event.evidence_ids, ["E1", "E2"])
        self.assertEqual(event.detail, "6 signals registered")

    def test_stop_event_carries_the_stop_reason(self):
        event = _sample_trace(clock=_fixed_clock).events[-1]
        self.assertEqual(event.event_type, "stop")
        self.assertEqual(event.detail, "stop_insufficient_data")

    def test_to_dicts_is_json_serializable_and_key_closed(self):
        trace = _sample_trace(clock=_fixed_clock)
        dumped = trace.to_dicts()
        json.dumps(dumped)
        for event in dumped:
            self.assertEqual(set(event), ALLOWED_EVENT_KEYS)

    def test_events_property_returns_a_copy(self):
        trace = InvestigationTrace()
        trace.record_enter("ingest_signal")
        self.assertIsNot(trace.events, trace.events)
        trace.events.clear()
        self.assertEqual(len(trace), 1)

    def test_trace_never_records_stored_row_data(self):
        trace = _sample_trace(clock=_fixed_clock)
        blob = json.dumps(trace.to_dicts())
        for forbidden in (
            "typing_speed",
            "dwell_mean",
            "flight_mean",
            "correction_rate",
            "rhythm_variability",
            "pause_count",
            "session_start",
        ):
            self.assertNotIn(forbidden, blob)


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_is_stable_and_order_independent(self):
        self.assertEqual(
            args_fingerprint("tool", a=1, b=2),
            args_fingerprint("tool", b=2, a=1),
        )

    def test_fingerprint_changes_with_arguments_and_tool(self):
        self.assertNotEqual(
            args_fingerprint("tool", a=1), args_fingerprint("tool", a=2)
        )
        self.assertNotEqual(
            args_fingerprint("one", a=1), args_fingerprint("two", a=1)
        )

    def test_fingerprint_does_not_reveal_its_inputs(self):
        fingerprint = args_fingerprint("tool", secret="hunter2")
        self.assertNotIn("hunter2", fingerprint)
        self.assertEqual(len(fingerprint), 16)


if __name__ == "__main__":
    unittest.main()
