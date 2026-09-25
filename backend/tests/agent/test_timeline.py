"""Tests for the Phase 3.5 plain-text renderers (``investigation/agent/timeline``).

The renderers are presentation only: they must reproduce existing, already
grounded fields, add nothing, and stay deterministic, private and free of
medical language.
"""

import re
import unittest
from datetime import datetime, timezone

from investigation.agent import (
    RejectedClaim,
    InvestigationTrace,
    render,
    render_report,
    render_report_lines,
    render_timeline,
    render_timeline_for,
)
from investigation.agent.report import REPORT_DISCLAIMER
from investigation.agent.timeline import TIMELINE_SCHEMA_VERSION
from tests.agent import scenarios
from tests.agent.scenarios import (
    clinical_terms_in,
    get_scenario,
    raw_field_names_in,
    run_scenario,
)

TIMELINE_RE = re.compile(r"^\d{2}:\d{2}:\d{2} \u2014 .+$")

#: Heading lines, in the fixed order the renderer must emit them.
SECTION_ORDER = (
    "Summary:",
    "Observations:",
    "Hypotheses:",
    "Alternatives:",
    "Limitations:",
)


def _bundle(key="persistent_multi_signal"):
    outcome = run_scenario(get_scenario(key))
    return outcome.result, outcome.report


class TimelineRenderTests(unittest.TestCase):
    def test_render_timeline_matches_the_trace_timeline(self):
        _, report = _bundle()
        result, _ = _bundle()
        events = result.state.trace

        expected = InvestigationTrace.from_events(
            events, clock=scenarios.frozen_clock
        ).timeline()
        self.assertEqual(render_timeline(events, clock=scenarios.frozen_clock), expected)

    def test_render_timeline_accepts_engine_and_critic_events_together(self):
        result, report = _bundle()
        combined = list(result.state.trace) + list(report.critic_events)
        self.assertEqual(
            render_timeline(combined, clock=scenarios.frozen_clock),
            render_timeline(result.state.trace, clock=scenarios.frozen_clock)
            + render_timeline(report.critic_events, clock=scenarios.frozen_clock),
        )

    def test_render_timeline_is_empty_without_events(self):
        self.assertEqual(render_timeline([]), [])
        self.assertEqual(render_timeline(None), [])

    def test_render_timeline_for_appends_critic_events(self):
        result, report = _bundle()
        engine_lines = render_timeline(result.state.trace, clock=scenarios.frozen_clock)
        critic_lines = render_timeline(report.critic_events, clock=scenarios.frozen_clock)

        combined = render_timeline_for(report, result, clock=scenarios.frozen_clock)
        self.assertEqual(combined, engine_lines + critic_lines)
        self.assertEqual(len(combined), len(engine_lines) + len(critic_lines))

    def test_render_timeline_uses_recorded_timestamps(self):
        # A trace recorded with an injected clock renders that clock's time ...
        later = datetime(2027, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
        trace = InvestigationTrace(clock=lambda: later)
        trace.record_enter("ingest_signal", detail="test")
        self.assertEqual(
            render_timeline(trace.to_dicts()),
            ["05:06:07 \u2014 Entered ingest_signal: test"],
        )

        # ... and events that were already recorded keep their own timestamps,
        # because rendering never adds events.
        result, _ = _bundle()
        self.assertEqual(
            render_timeline(result.state.trace, clock=lambda: later),
            render_timeline(result.state.trace, clock=scenarios.frozen_clock),
        )

    def test_render_timeline_lines_use_the_fixed_format(self):
        _, report = _bundle()
        result, _ = _bundle()
        for line in render_timeline_for(report, result, clock=scenarios.frozen_clock):
            self.assertRegex(line, TIMELINE_RE)

    def test_render_timeline_is_deterministic(self):
        result, _ = _bundle()
        first = render_timeline(result.state.trace, clock=scenarios.frozen_clock)
        second = render_timeline(result.state.trace, clock=scenarios.frozen_clock)
        self.assertEqual(first, second)

    def test_schema_version_is_published(self):
        self.assertEqual(TIMELINE_SCHEMA_VERSION, "1.0")


class ReportRenderTests(unittest.TestCase):
    def test_report_lines_start_with_the_conclusion(self):
        _, report = _bundle()
        lines = render_report_lines(report)
        self.assertEqual(
            lines[0],
            f"Conclusion: {report.conclusion.status} ({report.conclusion.basis})",
        )
        self.assertEqual(lines[1], report.conclusion.statement)

    def test_report_lines_use_a_fixed_section_order(self):
        _, report = _bundle()
        lines = render_report_lines(report)
        positions = [lines.index(heading) for heading in SECTION_ORDER]
        self.assertEqual(positions, sorted(positions))
        uncertainty = [i for i, line in enumerate(lines) if line.startswith("Uncertainty:")][0]
        self.assertTrue(positions[3] < uncertainty < positions[4])

    def test_report_lines_render_every_section_body(self):
        _, report = _bundle()
        lines = render_report_lines(report)
        for claim in report.observations:
            self.assertIn(f"- [{claim.id}] {claim.statement} ({', '.join(claim.evidence_ids)})", lines)
        for claim in report.hypothesis_assessment:
            self.assertIn(f"- {claim.id} [{claim.status}] {claim.statement}", lines)
        for claim in report.alternatives:
            self.assertIn(f"- {claim.id} [{claim.status}] {claim.statement}", lines)

    def test_report_lines_include_the_trace_link(self):
        _, report = _bundle()
        lines = render_report_lines(report)
        self.assertIn(f"Evidence digest: {report.link.evidence_digest}", lines)
        self.assertIn(f"Engine trace digest: {report.link.engine_trace_digest}", lines)
        self.assertIn(f"Critic trace digest: {report.link.critic_trace_digest}", lines)
        self.assertIn(f"Trace events: {report.link.trace_event_count}", lines)
        if report.link.stop_event_seq is not None:
            self.assertIn(f"Stop event seq: {report.link.stop_event_seq}", lines)

    def test_report_lines_end_with_the_disclaimer(self):
        _, report = _bundle()
        lines = render_report_lines(report)
        self.assertEqual(lines[-1], REPORT_DISCLAIMER)
        self.assertEqual(report.disclaimer, REPORT_DISCLAIMER)

    def test_rejected_claims_section_is_omitted_when_empty(self):
        _, report = _bundle()
        self.assertEqual(report.rejected_claims, [])
        self.assertNotIn("Rejected claims:", render_report_lines(report))

    def test_rejected_claims_section_lists_rejected_subjects(self):
        _, report = _bundle()
        rejection = RejectedClaim(
            reason="support_without_evidence",
            subject="H3",
            statement="rejected for the test",
            evidence_ids=[],
        )
        patched = report.model_copy(update={"rejected_claims": [rejection]})
        lines = render_report_lines(patched)
        self.assertIn("Rejected claims:", lines)
        self.assertIn("- H3 (support_without_evidence)", lines)

    def test_render_report_joins_the_lines(self):
        _, report = _bundle()
        self.assertEqual(render_report(report), "\n".join(render_report_lines(report)))

    def test_render_appends_exactly_one_timeline_heading(self):
        result, report = _bundle()
        text = render(report, result)
        self.assertEqual(text.count("Investigation timeline:"), 1)
        self.assertTrue(text.startswith(render_report(report)))
        self.assertTrue(
            text.endswith("\n".join(render_timeline_for(report, result, clock=scenarios.frozen_clock)))
        )

    def test_render_is_deterministic(self):
        result, report = _bundle()
        self.assertEqual(render(report, result), render(report, result))

    def test_render_works_for_an_inconclusive_report(self):
        result, report = _bundle("repository_unavailable")
        lines = render_report_lines(report)
        self.assertIn("Hypotheses:", lines)
        self.assertEqual(lines[-1], REPORT_DISCLAIMER)
        self.assertTrue(render(report, result))


class RenderingSafetyTests(unittest.TestCase):
    def test_rendering_contains_no_html_markup(self):
        for scenario in scenarios.EVALUATION_SCENARIOS:
            outcome = run_scenario(scenario)
            with self.subTest(scenario=scenario.key):
                text = outcome.render_text()
                for marker in ("<html", "<div", "<span", "<table", "<script"):
                    self.assertNotIn(marker, text)

    def test_rendering_contains_no_raw_stored_field_names(self):
        for scenario in scenarios.EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                self.assertEqual(raw_field_names_in(run_scenario(scenario).render_text()), ())

    def test_rendering_contains_no_clinical_language(self):
        for scenario in scenarios.EVALUATION_SCENARIOS:
            with self.subTest(scenario=scenario.key):
                self.assertEqual(clinical_terms_in(run_scenario(scenario).render_text()), ())

    def test_only_text_lines_are_rendered(self):
        _, report = _bundle()
        for line in render_report_lines(report):
            self.assertIsInstance(line, str)


if __name__ == "__main__":
    unittest.main()
