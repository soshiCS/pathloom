"""Provider failures at the planner boundary: transient ones are retried a bounded number of times, permanent ones
and exhausted ones end discovery with a project-owned error. No provider is ever contacted: every client is fake."""
import json
from types import SimpleNamespace

import pytest

from src.cua import planner as planner_module
from src.cua.__main__ import cmd_discover
from src.cua.escalation import Escalator, NoOperator, SessionControl
from src.cua.evidence import RunLog
from src.cua.planner import (MAX_PROVIDER_ATTEMPTS, ClaudePlanner, OpenAIPlanner, PlannerUnavailable,
                             failure_category)
from src.cua.policy import Policy
from src.cua.reuse import run_discovery
from tests.context import HOSTS, PARAMS, Observation
from tests.fake_surface import ENTRY, FakeSurface


class ApiStatusError(Exception):
    """Shaped like both SDKs' status errors: a status_code attribute, a message that is never inspected."""

    def __init__(self, status: int, message: str = "provider said no"):
        super().__init__(message)
        self.status_code = status


class APIConnectionError(Exception):
    """Named like the SDKs' connection errors; carries no status."""


class FakeResponses:
    """An OpenAI-style responses.create that follows a script of outcomes: an exception or a tool call."""

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        call = SimpleNamespace(type="function_call", name="choose_action", arguments=json.dumps(outcome))
        return SimpleNamespace(output=[call], output_text="")


def openai_planner(outcomes) -> OpenAIPlanner:
    return OpenAIPlanner(model="m", client=SimpleNamespace(responses=FakeResponses(outcomes)))


DONE = {"kind": "done", "element_index": None, "element_indexes": None, "value": None, "output_name": None,
        "pattern": None, "optional": False, "output_mode": None, "expect": None, "reason": "r", "outcomes": [],
        "stuck_cause": None, "candidate_id": None}
SCREEN = Observation(url=ENTRY, elements=[])


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(planner_module.time, "sleep", waits.append)
    return waits


def test_failure_categories_come_from_status_or_class_never_from_text():
    assert failure_category(ApiStatusError(429, "Our servers are currently overloaded")) == "rate_limited"
    assert failure_category(ApiStatusError(503, "fine")) == "server_error"
    assert failure_category(ApiStatusError(500)) == "server_error"
    assert failure_category(ApiStatusError(401, "please try again later")) == "permanent"
    assert failure_category(ApiStatusError(400)) == "permanent"
    assert failure_category(APIConnectionError("timed out")) == "connection"
    assert failure_category(TimeoutError()) == "connection"
    assert failure_category(RuntimeError("Error code: 503 - overloaded")) == "permanent"


def test_transient_failures_are_retried_with_backoff_and_reported_as_structured_records(no_sleep):
    planner = openai_planner([ApiStatusError(503), ApiStatusError(429), DONE])
    action = planner.decide("goal", {}, SCREEN, [])
    assert action.kind == "done" and planner.client.responses.calls == 3
    assert no_sleep == [1.0, 4.0]
    assert planner_module.drain_retries(planner) == [
        {"provider": "openai", "attempt": 1, "category": "server_error", "error_type": "ApiStatusError", "status": 503},
        {"provider": "openai", "attempt": 2, "category": "rate_limited", "error_type": "ApiStatusError", "status": 429}]
    assert planner_module.drain_retries(planner) == []     # drained once, logged once


def test_exhausted_transient_failures_raise_the_project_error_without_provider_text(no_sleep):
    planner = openai_planner([ApiStatusError(503, "Our servers are currently overloaded")] * MAX_PROVIDER_ATTEMPTS)
    with pytest.raises(PlannerUnavailable) as failed:
        planner.decide("goal", {}, SCREEN, [])
    error = failed.value
    assert (error.provider, error.category, error.attempts, error.status) == ("openai", "server_error", 3, 503)
    assert str(error) == "openai planner unavailable: server_error (HTTP 503) after 3 attempt(s)"
    assert "overloaded" not in str(error) and planner.client.responses.calls == MAX_PROVIDER_ATTEMPTS
    assert len(no_sleep) == MAX_PROVIDER_ATTEMPTS - 1


def test_permanent_failures_are_never_retried(no_sleep):
    planner = openai_planner([ApiStatusError(401, "invalid api key"), DONE])
    with pytest.raises(PlannerUnavailable, match=r"openai planner unavailable: permanent \(HTTP 401\) after 1 attempt"):
        planner.decide("goal", {}, SCREEN, [])
    assert planner.client.responses.calls == 1 and no_sleep == []


def test_connection_errors_are_retried_by_class_and_reported_without_a_status(no_sleep):
    planner = openai_planner([APIConnectionError("Connection error."), DONE])
    assert planner.decide("goal", {}, SCREEN, []).kind == "done"
    assert planner_module.drain_retries(planner) == [
        {"provider": "openai", "attempt": 1, "category": "connection", "error_type": "APIConnectionError", "status": None}]


def test_the_anthropic_adapter_shares_the_same_retry_boundary(no_sleep):
    class Messages:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ApiStatusError(529, "overloaded_error")
            block = SimpleNamespace(type="tool_use", input=DONE)
            return SimpleNamespace(content=[block], stop_reason="tool_use")

    planner = ClaudePlanner.__new__(ClaudePlanner)
    planner.client, planner.model, planner.vision_model, planner.name = SimpleNamespace(messages=Messages()), "m", "m", "c"
    assert planner.decide("goal", {}, SCREEN, []).kind == "done"
    assert [r["provider"] for r in planner_module.drain_retries(planner)] == ["anthropic"]


def discovery_run(planner, surface, log):
    escalator = Escalator(NoOperator(), SessionControl(), log)
    run_discovery(surface, goal="log in", name="login", params=dict(PARAMS), planner=planner,
                  policy=Policy(allowed_hosts=HOSTS), escalator=escalator, log=log, entry_url=ENTRY,
                  sensitive={"password"})


def test_direct_discovery_ends_with_a_concise_failure_and_evidence(no_sleep):
    planner = openai_planner([ApiStatusError(503, "Our servers are currently overloaded")] * MAX_PROVIDER_ATTEMPTS)
    log = RunLog("discovery", secrets=("secret_sauce",))
    with pytest.raises(PlannerUnavailable) as failed:
        discovery_run(planner, FakeSurface(), log)
    lines = [json.loads(line) for line in log.path.read_text().splitlines()]
    events = [line["event"] for line in lines]
    assert events[-2:] == ["discovery_failed", "screenshot"]
    assert {k: lines[-2][k] for k in ("event", "error", "kind")} == {
        "event": "discovery_failed", "error": str(failed.value), "kind": "PlannerUnavailable"}
    assert "planner_retry" not in events            # the retries never produced a decision to log them with
    assert "overloaded" not in log.path.read_text()


def test_retries_that_precede_a_decision_are_logged_with_that_decision(no_sleep):
    planner = openai_planner([ApiStatusError(503), DONE])
    log = RunLog("discovery", secrets=("secret_sauce",))
    discovery_run(planner, FakeSurface(), log)
    lines = [json.loads(line) for line in log.path.read_text().splitlines()]
    retry = next(line for line in lines if line["event"] == "planner_retry")
    assert {k: retry[k] for k in ("turn", "provider", "attempt", "category", "status")} == {
        "turn": 1, "provider": "openai", "attempt": 1, "category": "server_error", "status": 503}
    assert lines.index(retry) < next(i for i, line in enumerate(lines) if line["event"] == "planner_decided")


def test_the_cli_reports_the_failure_without_a_traceback(monkeypatch, capsys):
    from src.cua import __main__ as cli, evidence

    def failing_discovery(*args, **kwargs):
        kwargs["log"].event("discovery_started")
        raise PlannerUnavailable("openai", "server_error", 3, ApiStatusError(503, "overloaded"))

    monkeypatch.setattr(cli, "discover_with_reuse", failing_discovery)
    monkeypatch.setattr(cli, "make_planner", lambda args: openai_planner([]))
    monkeypatch.setattr(cli, "PlaywrightSurface", lambda **kwargs: FakeSurface())
    args = SimpleNamespace(param=["username=standard_user"], sensitive=[], quiet=True, url=ENTRY, allow_host=[],
                           outcome=[], missing_outcome=[], reuse_capability=None, reuse_artifact=None, operator="none",
                           headed=False, goal="g", name="login", max_steps=5, vision_fallback=False,
                           max_vision_attempts=2, no_auto_reuse=True, max_auto_reuses=5, library_dir=None)
    assert cmd_discover(args) == 2
    out = capsys.readouterr().out
    assert "Discovery failed: openai planner unavailable: server_error (HTTP 503) after 3 attempt(s)" in out
    assert "Traceback" not in out and "overloaded" not in out
    assert any(path.is_dir() for path in evidence.EVIDENCE_DIR.iterdir())


def test_a_campaign_records_the_unavailable_planner_and_finishes_its_cleanup(no_sleep):
    from src.cua import artifact as artifact_module, evidence as evidence_module
    from src.cua.campaign import CampaignFailed
    from tests.test_campaign import ClosableSurface, failing_summary, run, scripted

    surfaces = []

    def surface_factory(secrets):
        surfaces.append(ClosableSurface())
        return surfaces[-1]

    def planner_factory(scenario):
        if scenario.name == "url_phone":
            return openai_planner([ApiStatusError(503, "Our servers are currently overloaded")] * MAX_PROVIDER_ATTEMPTS)
        return scripted(scenario)

    with pytest.raises(CampaignFailed, match="scenario 'url_phone' failed: unexpected PlannerUnavailable: openai "
                                             r"planner unavailable: server_error \(HTTP 503\) after 3 attempt") as failed:
        run(planner_factory=planner_factory, surface_factory=surface_factory)
    assert all(s.closed for s in surfaces) and not artifact_module.ARTIFACTS_DIR.exists()
    summary = failing_summary(failed)
    assert [s["status"] for s in summary["scenarios"]] == ["succeeded", "succeeded", "failed"]
    log = (evidence_module.EVIDENCE_DIR / summary["scenarios"][2]["run_id"] / "run.jsonl").read_text()
    assert '"discovery_failed"' in log and '"kind": "PlannerUnavailable"' in log and "overloaded" not in log
