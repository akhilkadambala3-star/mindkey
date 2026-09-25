"""Tests for the Phase 3.5 reproducible demo (``investigation/agent/demo``).

The demo must be reproducible and fully offline: it builds synthetic data in
memory, runs the committed pipeline, and renders plain text. These tests pin
that contract, plus the privacy and medical-language guarantees of its output.
"""

import io
import json
import re
import unittest
from contextlib import redirect_stdout
from unittest import mock

from investigation.agent import (
    DEFAULT_AS_OF,
    DEMO_DATASETS,
    DEMO_KEYS,
    DemoRun,
    InMemoryRepository,
    demo_payload,
    format_run,
    main,
    render_demo,
    run_demo,
)
from investigation.agent import engine as engine_module
from investigation.agent.report import REPORT_DISCLAIMER
from tests.agent.scenarios import clinical_terms_in, raw_field_names_in

TIMELINE_RE = re.compile(r"^\d{2}:\d{2}:\d{2} \u2014 .+$")


def _capture(argv):
    """Run the CLI, returning (exit_code, stdout)."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(argv)
    return code, buffer.getvalue()


class DemoCatalogTests(unittest.TestCase):
    def test_demo_keys_are_fixed(self):
        self.assertEqual(
            DEMO_KEYS,
            (
                "consistent",
                "recent_variation",
                "persistent_change",
                "insufficient_history",
                "invalid_data",
            ),
        )
        self.assertEqual(tuple(DEMO_DATASETS), DEMO_KEYS)

    def test_every_dataset_describes_itself(self):
        for key in DEMO_KEYS:
            with self.subTest(scenario=key):
                entry = DEMO_DATASETS[key]
                self.assertTrue(entry["title"])
                self.assertTrue(entry["description"])
                self.assertTrue(callable(entry["build"]))

    def test_default_reference_time_is_fixed(self):
        self.assertEqual(DEFAULT_AS_OF.tzinfo.utcoffset(None).total_seconds(), 0)
        self.assertEqual(DEFAULT_AS_OF.year, 2026)


class DemoRunTests(unittest.TestCase):
    def test_every_scenario_runs_and_renders(self):
        for key in DEMO_KEYS:
            with self.subTest(scenario=key):
                run = run_demo(key)
                self.assertIsInstance(run, DemoRun)
                self.assertEqual((run.key, run.title), (key, DEMO_DATASETS[key]["title"]))
                self.assertTrue(run.result.stop_reason)
                self.assertTrue(run.timeline)
                self.assertTrue(run.report_lines)
                self.assertEqual(run.report.session_id, run.result.session_id)

    def test_default_scenario_is_persistent_change(self):
        self.assertEqual(run_demo().key, "persistent_change")

    def test_unknown_scenario_raises(self):
        with self.assertRaises(KeyError):
            run_demo("no-such-scenario")

    def test_persistent_change_scenario_is_grounded(self):
        run = run_demo("persistent_change")
        self.assertEqual(run.result.stop_reason, "evidence_sufficient")
        self.assertEqual(run.report.conclusion.status, "grounded")
        self.assertEqual(
            run.report.conclusion.basis, "grounded_persistent_change"
        )

    def test_consistent_scenario_reports_no_deviation(self):
        run = run_demo("consistent")
        self.assertEqual(run.report.conclusion.status, "no_deviation")

    def test_insufficient_history_is_inconclusive(self):
        run = run_demo("insufficient_history")
        self.assertEqual(run.report.conclusion.status, "inconclusive")

    def test_runs_are_reproducible(self):
        for key in DEMO_KEYS:
            with self.subTest(scenario=key):
                first = run_demo(key)
                second = run_demo(key)
                self.assertEqual(
                    first.result.evidence_digest, second.result.evidence_digest
                )
                self.assertEqual(
                    first.report.model_dump(mode="json"),
                    second.report.model_dump(mode="json"),
                )
                self.assertEqual(first.timeline, second.timeline)
                self.assertEqual(first.report_lines, second.report_lines)

    def test_render_demo_matches_the_report_renderer(self):
        run = run_demo("recent_variation")
        text = render_demo(run)
        self.assertIn("Investigation timeline:", text)
        self.assertIn(f"Conclusion: {run.report.conclusion.status}", text)

    def test_format_run_can_render_only_the_timeline(self):
        run = run_demo("consistent")
        self.assertEqual(format_run(run, timeline=True), "\n".join(run.timeline))
        for line in format_run(run, timeline=True).splitlines():
            self.assertRegex(line, TIMELINE_RE)

    def test_demo_payload_is_serializable(self):
        run = run_demo("persistent_change")
        payload = demo_payload(run)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["key"], "persistent_change")
        self.assertEqual(payload["stop_reason"], "evidence_sufficient")
        self.assertEqual(payload["timeline"], list(run.timeline))
        self.assertEqual(payload["report_lines"], list(run.report_lines))
        self.assertEqual(json.loads(json.dumps(payload)), payload)


class DemoOfflineTests(unittest.TestCase):
    def test_demo_never_uses_the_production_repository(self):
        with mock.patch.object(
            engine_module,
            "default_repository",
            side_effect=AssertionError("the demo must never build a Supabase client"),
        ):
            for key in DEMO_KEYS:
                with self.subTest(scenario=key):
                    self.assertTrue(run_demo(key).timeline)

    def test_demo_module_does_not_import_the_database_layer(self):
        source = engine_module.__file__.replace("engine.py", "demo.py")
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        for forbidden in (
            "import database",
            "from database",
            "import supabase",
            "from supabase",
            "dotenv",
            "create_client",
        ):
            self.assertNotIn(forbidden, text.lower())

    def test_in_memory_repository_copies_rows_and_anomalies(self):
        repository = InMemoryRepository(
            sessions=[{"id": "X1", "user_id": "u", "typing_speed": 1.0}],
            anomalies={"X1": {"anomaly_score": 0.5}},
        )
        first = repository.list_sessions("u")
        first[0]["typing_speed"] = 99.0
        self.assertEqual(repository.list_sessions("u")[0]["typing_speed"], 1.0)
        anomaly = repository.get_anomaly_result("X1")
        anomaly["anomaly_score"] = 99.0
        self.assertEqual(repository.get_anomaly_result("X1")["anomaly_score"], 0.5)
        self.assertIsNone(repository.get_anomaly_result("missing"))


class DemoCliTests(unittest.TestCase):
    def test_list_prints_every_scenario(self):
        code, output = _capture(["--list"])
        self.assertEqual(code, 0)
        for key in DEMO_KEYS:
            self.assertIn(key, output)
            self.assertIn(DEMO_DATASETS[key]["title"], output)
        self.assertNotIn("Investigation timeline:", output)

    def test_default_run_covers_every_scenario(self):
        code, output = _capture([])
        self.assertEqual(code, 0)
        for key in DEMO_KEYS:
            self.assertIn(f"=== {key}:", output)
        self.assertEqual(output.count("Investigation timeline:"), len(DEMO_KEYS))

    def test_scenario_flag_prints_a_plain_text_report_and_timeline(self):
        code, output = _capture(["--scenario", "persistent_change"])
        self.assertEqual(code, 0)
        self.assertIn("=== persistent_change:", output)
        self.assertIn("Conclusion: grounded (grounded_persistent_change)", output)
        self.assertIn("Investigation timeline:", output)
        self.assertIn("Stopped check_stop_conditions: evidence_sufficient", output)

    def test_timeline_flag_omits_the_report(self):
        code, output = _capture(["--scenario", "consistent", "--timeline"])
        self.assertEqual(code, 0)
        self.assertIn("=== consistent:", output)
        self.assertNotIn("Conclusion:", output)
        self.assertNotIn("Investigation timeline:", output)
        for line in output.splitlines():
            if line.startswith("==="):
                continue
            self.assertRegex(line, TIMELINE_RE)

    def test_json_flag_prints_machine_readable_output(self):
        code, output = _capture(["--scenario", "consistent", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(len(payload["runs"]), 1)
        run = payload["runs"][0]
        self.assertEqual(run["key"], "consistent")
        self.assertEqual(run["report"]["conclusion"]["status"], "no_deviation")
        self.assertTrue(run["timeline"])

    def test_unknown_scenario_exits_with_an_error(self):
        with self.assertRaises(SystemExit):
            main(["--scenario", "no-such-scenario"])

    def test_cli_output_has_no_raw_stored_field_names(self):
        _, output = _capture([])
        self.assertEqual(raw_field_names_in(output), ())

    def test_cli_output_has_no_clinical_language(self):
        _, output = _capture([])
        self.assertEqual(clinical_terms_in(output), ())

    def test_cli_output_has_no_html_markup(self):
        _, output = _capture([])
        for marker in ("<html", "<div", "<span", "<table", "<script"):
            self.assertNotIn(marker, output)

    def test_cli_report_includes_the_fixed_disclaimer(self):
        _, output = _capture(["--scenario", "consistent"])
        self.assertIn(REPORT_DISCLAIMER, output)
        # The report states the disclaimer before the timeline appendix.
        self.assertLess(
            output.index(REPORT_DISCLAIMER), output.index("Investigation timeline:")
        )


if __name__ == "__main__":
    unittest.main()
