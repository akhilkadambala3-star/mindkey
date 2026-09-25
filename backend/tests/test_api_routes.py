"""Integration tests for the Phase 4 routes (``routes.api``).

The suite calls the route functions directly (no HTTP client, no new
dependencies): FastAPI wires ``Depends`` only through the ASGI app, so each
handler receives its repository/client/clock explicitly — which is exactly
the dependency override the tests need. App-level wiring (CORS middleware,
router inclusion) is verified from ``main.py``'s source with ``ast`` because
importing the app requires Supabase environment variables.
"""

import ast
import json
import unittest
from pathlib import Path
from typing import get_args, get_origin
from unittest import mock

from fastapi import HTTPException

from investigation.agent import REPORT_DISCLAIMER
from investigation.repository import RepositoryError
from routes import api
from schemas import BaselineReadModel, InvestigationReadModel, SessionReadModel
from tests import fixtures
from tests.agent.scenarios import clinical_terms_in
from tests.api_support import (
    BASELINE_KEYS,
    INVESTIGATION_KEYS,
    SESSION_KEYS,
    FakeBaselineClient,
    ReadOnlyRepository,
    baseline_row,
    frozen_clock,
)

USER = fixtures.DEFAULT_USER

SESSIONS_PATH = "/api/users/{user_id}/sessions"
BASELINE_PATH = "/api/users/{user_id}/baseline"
INVESTIGATION_PATH = "/api/users/{user_id}/investigation"

SESSIONS_503 = "Failed to read typing sessions"
BASELINE_503 = "Failed to read baseline"
INVESTIGATION_503 = "Failed to read investigation data"

MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


def _main_tree():
    return ast.parse(MAIN_PATH.read_text(encoding="utf-8"))


def _calls_on(tree, attribute):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]


class RouterShapeTests(unittest.TestCase):
    def test_three_get_routes_with_the_documented_paths(self):
        routes = list(api.router.routes)
        self.assertEqual(
            [route.path for route in routes],
            [SESSIONS_PATH, BASELINE_PATH, INVESTIGATION_PATH],
        )
        for route in routes:
            self.assertEqual(route.methods, {"GET"})

    def test_each_route_declares_its_response_model(self):
        sessions, baseline, investigation = list(api.router.routes)
        self.assertEqual(get_origin(sessions.response_model), list)
        self.assertIs(get_args(sessions.response_model)[0], SessionReadModel)
        self.assertIs(baseline.response_model, BaselineReadModel)
        self.assertIs(investigation.response_model, InvestigationReadModel)


class SessionsEndpointTests(unittest.TestCase):
    def test_200_contract_shape(self):
        repo = fixtures.FakeRepository(sessions=fixtures.make_history(4))
        sessions = api.list_user_sessions(USER, repository=repo)
        self.assertEqual(len(sessions), 4)
        for session in sessions:
            self.assertEqual(frozenset(session), SESSION_KEYS)

    def test_oldest_first_order(self):
        rows = [
            fixtures.make_session(session_id="NEW", days_ago=1),
            fixtures.make_session(session_id="OLD", days_ago=9),
        ]
        sessions = api.list_user_sessions(
            USER, repository=fixtures.FakeRepository(sessions=rows)
        )
        self.assertEqual([s["session_id"] for s in sessions], ["OLD", "NEW"])

    def test_no_status_and_no_consistency(self):
        # Phase 4 decision C4/C5: neither field may ever be emitted.
        repo = fixtures.FakeRepository(sessions=fixtures.make_history(3))
        for session in api.list_user_sessions(USER, repository=repo):
            self.assertNotIn("status", session)
            self.assertNotIn("consistency", session)

    def test_unknown_user_returns_empty_array(self):
        repo = fixtures.FakeRepository(sessions=fixtures.make_history(3))
        self.assertEqual(api.list_user_sessions("someone-else", repository=repo), [])

    def test_empty_store_returns_empty_array(self):
        repo = fixtures.FakeRepository(sessions=[])
        self.assertEqual(api.list_user_sessions(USER, repository=repo), [])

    def test_503_on_store_read_failure(self):
        with self.assertRaises(HTTPException) as ctx:
            api.list_user_sessions(USER, repository=fixtures.FailingRepository())
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, SESSIONS_503)

    def test_503_when_the_production_store_cannot_be_constructed(self):
        with mock.patch.object(
            api,
            "default_repository",
            side_effect=RepositoryError("store down"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                api.list_user_sessions(USER, repository=None)
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, SESSIONS_503)

    def test_422_on_empty_user_id_before_any_store_access(self):
        with mock.patch.object(
            api, "default_repository", side_effect=AssertionError("store touched")
        ):
            with self.assertRaises(HTTPException) as ctx:
                api.list_user_sessions("", repository=None)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_422_on_whitespace_only_user_id(self):
        with self.assertRaises(HTTPException) as ctx:
            api.list_user_sessions("   ", repository=fixtures.FakeRepository())
        self.assertEqual(ctx.exception.status_code, 422)

    def test_response_model_conformance(self):
        repo = fixtures.FakeRepository(sessions=fixtures.make_history(3))
        for session in api.list_user_sessions(USER, repository=repo):
            model = SessionReadModel(**session)
            dumped = model.model_dump()
            self.assertNotIn("status", dumped)
            self.assertNotIn("consistency", dumped)

    def test_read_only_access_to_the_store(self):
        spy = ReadOnlyRepository(
            fixtures.FakeRepository(sessions=fixtures.make_history(3))
        )
        api.list_user_sessions(USER, repository=spy)
        self.assertEqual(spy.calls, [("list_sessions", USER)])

    def test_no_clinical_language_in_the_response(self):
        repo = fixtures.FakeRepository(sessions=fixtures.make_history(3))
        blob = json.dumps(api.list_user_sessions(USER, repository=repo), default=str)
        self.assertEqual(clinical_terms_in(blob), ())


class BaselineEndpointTests(unittest.TestCase):
    def test_200_contract_shape(self):
        client = FakeBaselineClient([baseline_row(sample_count=17)])
        baseline = api.get_user_baseline(USER, client=client)
        self.assertEqual(frozenset(baseline), BASELINE_KEYS)
        self.assertEqual(baseline["sample_count"], 17)
        self.assertEqual(baseline["wpm"], 57.0)

    def test_200_missing_baseline_is_defensive(self):
        baseline = api.get_user_baseline(USER, client=FakeBaselineClient([]))
        self.assertEqual(frozenset(baseline), BASELINE_KEYS)
        self.assertEqual(baseline["sample_count"], 0)
        for key in BASELINE_KEYS - {"sample_count", "updated_at"}:
            self.assertIsNone(baseline[key], key)

    def test_503_on_store_read_failure(self):
        client = FakeBaselineClient([baseline_row()], fail=True)
        with self.assertRaises(HTTPException) as ctx:
            api.get_user_baseline(USER, client=client)
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, BASELINE_503)

    def test_503_when_the_client_cannot_be_constructed(self):
        with mock.patch.object(
            api.reads,
            "baseline_client",
            side_effect=RepositoryError("client down"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                api.get_user_baseline(USER, client=None)
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, BASELINE_503)

    def test_422_on_empty_user_id_before_any_store_access(self):
        with mock.patch.object(
            api.reads, "baseline_client", side_effect=AssertionError("store touched")
        ):
            with self.assertRaises(HTTPException) as ctx:
                api.get_user_baseline("", client=None)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_only_reads_from_the_baselines_table(self):
        client = FakeBaselineClient([baseline_row()])
        api.get_user_baseline(USER, client=client)
        self.assertEqual(client.tables, ["baselines"])
        self.assertTrue(set(client.ops) <= {"select", "eq", "limit", "execute"})

    def test_response_model_conformance(self):
        client = FakeBaselineClient([baseline_row()])
        model = BaselineReadModel(**api.get_user_baseline(USER, client=client))
        self.assertEqual(model.sample_count, 12)

    def test_dependency_yields_none_when_the_store_is_unavailable(self):
        with mock.patch.object(
            api, "default_repository", side_effect=RepositoryError("down")
        ):
            self.assertIsNone(api.get_repository())

    def test_dependency_yields_the_store_when_available(self):
        sentinel = object()
        with mock.patch.object(api, "default_repository", return_value=sentinel):
            self.assertIs(api.get_repository(), sentinel)

    def test_baseline_dependency_yields_none_when_the_client_is_unavailable(self):
        with mock.patch.object(
            api.reads, "baseline_client", side_effect=RepositoryError("down")
        ):
            self.assertIsNone(api.get_baseline_client())

    def test_baseline_dependency_yields_the_client_when_available(self):
        sentinel = object()
        with mock.patch.object(api.reads, "baseline_client", return_value=sentinel):
            self.assertIs(api.get_baseline_client(), sentinel)

    def test_no_clinical_language_in_the_response(self):
        client = FakeBaselineClient([baseline_row()])
        blob = json.dumps(api.get_user_baseline(USER, client=client), default=str)
        self.assertEqual(clinical_terms_in(blob), ())


class InvestigationEndpointTests(unittest.TestCase):
    def _repo(self, count=6):
        return fixtures.FakeRepository(sessions=fixtures.make_history(count))

    def test_200_contract_shape(self):
        payload = api.run_user_investigation(
            USER, session_id="S0001", repository=self._repo(), clock=frozen_clock
        )
        self.assertEqual(frozenset(payload), INVESTIGATION_KEYS)
        self.assertEqual(payload["session_id"], "S0001")
        self.assertTrue(payload["report"])
        self.assertTrue(payload["timeline"])

    def test_defaults_to_the_latest_stored_session(self):
        payload = api.run_user_investigation(
            USER, repository=self._repo(), clock=frozen_clock
        )
        self.assertEqual(payload["session_id"], "S0006")

    def test_explicit_session_is_honored(self):
        payload = api.run_user_investigation(
            USER, session_id="S0003", repository=self._repo(), clock=frozen_clock
        )
        self.assertEqual(payload["session_id"], "S0003")

    def test_404_on_unknown_explicit_session(self):
        with self.assertRaises(HTTPException) as ctx:
            api.run_user_investigation(
                USER, session_id="GHOST", repository=self._repo(), clock=frozen_clock
            )
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail, "Unknown session")

    def test_200_insufficient_history_is_honest_not_an_error(self):
        payload = api.run_user_investigation(
            USER,
            session_id="S0001",
            repository=fixtures.FakeRepository(sessions=fixtures.make_history(3)),
            clock=frozen_clock,
        )
        conclusion = payload["report"]["conclusion"]
        self.assertEqual(payload["stop_reason"], "evidence_insufficient")
        self.assertEqual(conclusion["status"], "inconclusive")
        self.assertEqual(conclusion["basis"], "insufficient_history")

    def test_503_on_store_read_failure(self):
        with self.assertRaises(HTTPException) as ctx:
            api.run_user_investigation(
                USER, repository=fixtures.FailingRepository(), clock=frozen_clock
            )
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, INVESTIGATION_503)

    def test_503_when_the_production_store_cannot_be_constructed(self):
        with mock.patch.object(
            api,
            "default_repository",
            side_effect=RepositoryError("store down"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                api.run_user_investigation(USER, repository=None, clock=frozen_clock)
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, INVESTIGATION_503)

    def test_422_on_bad_as_of_before_any_store_access(self):
        with mock.patch.object(
            api, "default_repository", side_effect=AssertionError("store touched")
        ):
            with self.assertRaises(HTTPException) as ctx:
                api.run_user_investigation(
                    USER, as_of="not-a-timestamp", repository=None
                )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_422_on_empty_session_id(self):
        with self.assertRaises(HTTPException) as ctx:
            api.run_user_investigation(
                USER, session_id="   ", repository=self._repo()
            )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_422_on_empty_user_id(self):
        with self.assertRaises(HTTPException) as ctx:
            api.run_user_investigation("", repository=self._repo())
        self.assertEqual(ctx.exception.status_code, 422)

    def test_identical_requests_produce_identical_payloads(self):
        repo = self._repo()
        kwargs = dict(
            repository=repo,
            clock=frozen_clock,
            as_of="2026-09-24T09:42:00+00:00",
        )
        first = api.run_user_investigation(USER, **kwargs)
        second = api.run_user_investigation(USER, **kwargs)
        self.assertEqual(first, second)

    def test_naive_as_of_is_read_as_utc(self):
        repo = self._repo()
        naive = api.run_user_investigation(
            USER,
            repository=repo,
            clock=frozen_clock,
            as_of="2026-09-24T09:42:00",
        )
        aware = api.run_user_investigation(
            USER,
            repository=repo,
            clock=frozen_clock,
            as_of="2026-09-24T09:42:00+00:00",
        )
        self.assertEqual(naive, aware)

    def test_disclaimer_is_present(self):
        payload = api.run_user_investigation(
            USER, session_id="S0001", repository=self._repo(), clock=frozen_clock
        )
        self.assertEqual(payload["disclaimer"], REPORT_DISCLAIMER)

    def test_response_model_conformance(self):
        payload = api.run_user_investigation(
            USER, session_id="S0001", repository=self._repo(), clock=frozen_clock
        )
        model = InvestigationReadModel(**payload)
        self.assertIsNotNone(model.report)
        self.assertEqual(
            model.report.conclusion.status, payload["report"]["conclusion"]["status"]
        )
        self.assertEqual(model.disclaimer, REPORT_DISCLAIMER)

    def test_reportless_payload_passes_the_response_model(self):
        payload = api.run_user_investigation(
            USER,
            repository=fixtures.FakeRepository(sessions=[]),
            clock=frozen_clock,
        )
        model = InvestigationReadModel(**payload)
        self.assertIsNone(model.report)
        self.assertIsNone(model.session_id)
        self.assertEqual(model.stop_reason, "evidence_insufficient")

    def test_no_clinical_language_in_the_response(self):
        payload = api.run_user_investigation(
            USER, session_id="S0001", repository=self._repo(), clock=frozen_clock
        )
        self.assertEqual(clinical_terms_in(json.dumps(payload, default=str)), ())


class AppWiringTests(unittest.TestCase):
    """Verify main.py's approved changes from source (importing it needs env)."""

    def test_main_adds_cors_middleware_with_explicit_origins(self):
        calls = _calls_on(_main_tree(), "add_middleware")
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIsInstance(call.args[0], ast.Name)
        self.assertEqual(call.args[0].id, "CORSMiddleware")
        keywords = {kw.arg: kw.value for kw in call.keywords}
        origins = [elt.value for elt in keywords["allow_origins"].elts]
        self.assertIn("http://localhost:5500", origins)
        self.assertNotIn("*", origins)
        methods = [elt.value for elt in keywords["allow_methods"].elts]
        self.assertEqual(methods, ["GET"])
        self.assertFalse(keywords.get("allow_credentials"))

    def test_main_includes_the_api_router(self):
        tree = _main_tree()
        included = [
            call.args[0].id
            for call in _calls_on(tree, "include_router")
            if isinstance(call.args[0], ast.Name)
        ]
        self.assertIn("api_router", included)
        api_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "routes.api"
        ]
        self.assertTrue(api_imports)


if __name__ == "__main__":
    unittest.main()
