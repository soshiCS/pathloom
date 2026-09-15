"""Stability verification: N fresh unattended runs, metrics, eligibility, safety, evidence."""
import json
import subprocess
import sys

import pytest

from src.cua import __main__ as main_module
from src.cua import evidence as evidence_module
from src.cua import stability as stability_module
from src.cua.__main__ import main
from src.cua.artifact import save
from src.cua.graph import from_linear, nodes_by_id, save_graph
from src.cua.lifecycle import sha256_of
from src.cua.stability import run_stability
from tests.context import HOSTS, PARAMS, Step, checkout_artifact, ladder
from tests.fake_surface import FakeSurface

SENSITIVE = ["password"]
ARGS = [f"--param={k}={v}" for k, v in PARAMS.items()] + ["--sensitive", "password", "--quiet"]


class Session(FakeSurface):
    """A fake session that must be closed, like the browser."""

    def __init__(self, **options):
        super().__init__(**options)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class Sessions:
    """Creates one fresh session per call; `options_for(n)` picks the n-th session's faults."""

    def __init__(self, options_for=None, fail_on: int | None = None):
        self.created: list[Session] = []
        self.calls = 0
        self.options_for = options_for or (lambda n: {})
        self.fail_on = fail_on

    def __call__(self, secrets) -> Session:
        self.calls += 1
        index = self.calls
        if index == self.fail_on:
            raise OSError("chromium failed to launch")
        self.created.append(Session(**self.options_for(index)))
        return self.created[-1]


def finish_step() -> Step:
    return Step(id="s16", action="click", target=ladder("button", "Finish"),
                checkpoint={"text_contains": "Thank you"}, risk="risky")


def draft_v1(**kw):
    return save(checkout_artifact(**kw), secrets=("secret_sauce",))


def draft_v2(tmp_path, graph=None):
    return save_graph(graph or from_linear(checkout_artifact()), secrets=("secret_sauce",),
                      path=tmp_path / "checkout_review.graph.json")


def report_for(path, sessions=None, runs=3, params=None):
    sessions = sessions or Sessions()
    report_path = run_stability(path, params or dict(PARAMS), SENSITIVE, runs, sessions, HOSTS)
    return json.loads(report_path.read_text()), report_path, sessions


# ---------- both schemas, fresh sessions, metrics ----------

@pytest.mark.parametrize("schema", ["1.0", "2.0"])
def test_three_clean_runs_are_eligible_on_both_schemas(tmp_path, schema):
    path = draft_v1() if schema == "1.0" else draft_v2(tmp_path)
    report, report_path, sessions = report_for(path)
    assert len(sessions.created) == 3 and all(s.closed for s in sessions.created)
    assert all(s.screen == "overview" for s in sessions.created)              # each session ran the whole flow
    assert report["report_schema_version"] == "1.0" and report["stability_id"].startswith("stability-")
    assert report["artifact"] == {"name": "checkout_review", "version": 1, "schema_version": schema,
                                  "status": "draft", "path": str(path), "sha256": sha256_of(path)}
    assert (report["runs_requested"], report["runs_completed"]) == (3, 3)
    assert report["param_names"] == sorted(PARAMS) and report["sensitive_params"] == ["password"]
    assert report["selector_assignment"] is None and report["irreversible_policy"] == "deny"
    assert report["status_counts"] == {"success": 3} and report["outcome_counts"] == {"success": 3}
    assert (report["success_rate"], report["clean_run_rate"]) == (1.0, 1.0)
    assert (report["total_recoveries"], report["total_interventions"], report["total_drift_signals"]) == (0, 0, 0)
    assert report["eligible_for_approval"] is True and report["ineligible_reasons"] == []
    assert [r["run"] for r in report["runs"]] == [1, 2, 3]
    assert len({r["run_id"] for r in report["runs"]}) == 3
    for record in report["runs"]:
        assert record["artifact_sha256"] == sha256_of(path) and record["clean"] is True
        evidence_dir = evidence_module.EVIDENCE_DIR / record["run_id"]
        assert record["evidence"] == str(evidence_dir)
        assert json.loads((evidence_dir / "result.json").read_text())["status"] == "success"
        assert json.loads((evidence_dir / "manifest.json").read_text())["sha256"] == sha256_of(path)
        assert (evidence_dir / path.name).read_bytes() == path.read_bytes()
    assert report_path.parent.name == report["stability_id"] and (report_path.parent / "run.jsonl").exists()


def test_runs_continue_after_a_failure_and_an_intervention_disqualifies():
    sessions = Sessions(lambda n: {"faults": ["verification"]} if n == 2 else {})
    report, _, _ = report_for(draft_v1(), sessions)
    assert len(sessions.created) == 3 and all(s.closed for s in sessions.created)
    assert report["runs_completed"] == 3
    assert report["status_counts"] == {"failure": 1, "success": 2}
    assert report["outcome_counts"] == {"success": 2, "unknown_dialog": 1}
    assert report["success_rate"] == pytest.approx(2 / 3) and report["clean_run_rate"] == pytest.approx(2 / 3)
    assert report["total_interventions"] == 1 and report["runs"][1]["interventions"] == 1
    assert report["eligible_for_approval"] is False
    assert report["ineligible_reasons"] == ["not every run returned success", "a run required human intervention"]


def test_recoveries_and_drift_are_reported_but_do_not_disqualify(tmp_path):
    report, _, _ = report_for(draft_v1(), Sessions(lambda n: {"show_notice": True}))
    assert report["total_recoveries"] == 3 and [r["recoveries"] for r in report["runs"]] == [1, 1, 1]
    assert report["eligible_for_approval"] is True and report["clean_run_rate"] == 0.0

    drifted = from_linear(checkout_artifact())
    login = nodes_by_id(drifted)["s4"].action
    login.target = ladder("button", "Sign in")
    login.target.strategies[1] = {"kind": "css", "selector": "button:Login:"}
    report, _, _ = report_for(draft_v2(tmp_path, drifted))
    assert report["total_drift_signals"] == 3 and [r["drift_signals"] for r in report["runs"]] == [1, 1, 1]
    assert report["success_rate"] == 1.0 and report["eligible_for_approval"] is True
    assert report["clean_run_rate"] == 0.0


def test_fewer_than_three_completed_runs_is_ineligible():
    report, _, sessions = report_for(draft_v1(), runs=2)
    assert len(sessions.created) == 2 and report["runs_completed"] == 2
    assert report["success_rate"] == 1.0 and report["eligible_for_approval"] is False
    assert report["ineligible_reasons"] == ["only 2 of the required 3 runs completed"]


def test_a_crashing_run_is_recorded_and_the_others_still_run():
    sessions = Sessions(fail_on=2)
    report, _, _ = report_for(draft_v1(), sessions)
    assert len(sessions.created) == 2 and all(s.closed for s in sessions.created)
    assert report["runs_requested"] == 3 and report["runs_completed"] == 2
    assert report["runs"][1]["status"] == "error" and report["runs"][1]["error"] == "OSError: chromium failed to launch"
    assert report["runs"][1]["evidence"] and report["status_counts"] == {"error": 1, "success": 2}
    assert report["eligible_for_approval"] is False
    assert report["ineligible_reasons"][0] == "only 2 of the required 3 runs completed"


# ---------- safety ----------

def test_irreversible_actions_are_denied_and_never_repeated(tmp_path):
    finish = from_linear(checkout_artifact(extra_step=finish_step()))
    finish.success = {"text_contains": "Thank you"}
    report, _, sessions = report_for(draft_v2(tmp_path, finish))
    assert report["outcome_counts"] == {"irreversible_denied": 3} and report["total_interventions"] == 0
    assert all(("click", "Finish", "") not in s.actions and s.screen == "overview" for s in sessions.created)
    assert report["eligible_for_approval"] is False

    unknown = from_linear(checkout_artifact(extra_step=finish_step()))
    nodes_by_id(unknown)["s16"].effect = "unknown"
    report, _, sessions = report_for(save_graph(unknown, secrets=("secret_sauce",), path=tmp_path / "unknown.json"))
    assert report["outcome_counts"] == {"unknown_effect_denied": 3}
    assert all(("click", "Finish", "") not in s.actions for s in sessions.created)

    report, _, sessions = report_for(draft_v1(extra_step=finish_step()))         # version 1: a risky step
    assert report["outcome_counts"] == {"risky_step_not_confirmed": 3} and report["total_interventions"] == 3
    assert all(("click", "Finish", "") not in s.actions for s in sessions.created)
    assert report["eligible_for_approval"] is False


def test_sensitive_values_never_reach_the_report_or_evidence(tmp_path):
    report, report_path, _ = report_for(draft_v1())
    assert "secret_sauce" not in report_path.read_text()
    assert "standard_user" not in json.dumps(report)                            # no parameter values at all
    for path in tmp_path.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl"):
            assert "secret_sauce" not in path.read_text(), path


def test_stability_never_imports_the_planner_or_the_llm_sdk():
    probe = ("import sys, src.cua.stability, src.cua.approval, src.cua.lifecycle; "
             "print(sorted(m for m in sys.modules "
             "if m in ('src.cua.planner', 'src.cua.agent', 'anthropic', 'openai')))")
    output = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "[]"
    assert not any(hasattr(stability_module, name) for name in ("Planner", "ClaudePlanner", "OpenAIPlanner"))


def test_selector_assignment_is_reported_for_campaign_graphs(tmp_path):
    from src.cua.merge import merge_traces
    from tests.test_merge import LOGIN, link_trace, url_trace
    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], "campaign-test")
    path = save_graph(graph, secrets=("secret_sauce",), path=tmp_path / "paths.json")
    params = {**LOGIN, "product_name": "Sauce Labs Backpack", "cart_route": "cart_url", "cart_url": HOSTS[0]}
    params["cart_url"] = "https://www.saucedemo.com/cart.html"
    report, _, _ = report_for(path, params=params)
    assert report["selector_assignment"] == {"cart_route": "cart_url"} and report["eligible_for_approval"] is True


# ---------- CLI ----------

def test_cli_stability_command_writes_a_report(monkeypatch, tmp_path, capsys):
    sessions = Sessions()
    monkeypatch.setattr(main_module, "PlaywrightSurface", lambda **_: sessions(()))
    path = draft_v1()
    assert main(["stability", "--artifact", str(path), "--runs", "3", *ARGS]) == 0
    out = capsys.readouterr().out
    assert "3/3 runs completed" in out and "Eligible for approval: True" in out and "report.json" in out
    assert len(sessions.created) == 3 and all(s.closed for s in sessions.created)
    reports = list(tmp_path.glob("evidence/stability-*/report.json"))
    assert len(reports) == 1 and json.loads(reports[0].read_text())["eligible_for_approval"] is True

    assert main(["stability", "--artifact", str(path), "--runs", "1", *ARGS]) == 2   # completed, not eligible
    assert "only 1 of the required 3" in capsys.readouterr().out
    assert main(["stability", "--artifact", str(tmp_path / "missing.json"), *ARGS]) == 2
    assert "Cannot load artifact" in capsys.readouterr().out
