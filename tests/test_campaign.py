"""Campaigns: spec validation, one discovery per scenario, evidence, failure handling, merged replay, CLI."""
import json

import pytest

from src.cua import __main__ as main_module
from src.cua import agent as agent_module
from src.cua import artifact as artifact_module
from src.cua import campaign as campaign_module
from src.cua import evidence as evidence_module
from src.cua.__main__ import main
from src.cua.campaign import CampaignError, CampaignFailed, run_campaign, spec_from_dict
from src.cua.artifact import linear_path, load_artifact
from src.cua.escalation import NoOperator
from src.cua.models import Action
from src.cua.replay import replay as replay_engine
from tests.context import ENTRY, HOSTS, Escalator, Policy, RunLog, SessionControl
from tests.fake_surface import FakeSurface
from tests.scripted_planner import MONEY, ScriptedPlanner, ScriptedStep

BASE = {"username": "standard_user", "password": "secret_sauce", "product_name": "Sauce Labs Backpack",
        "first_name": "Test", "last_name": "User"}
PHONE = "+15551234567"
SCENARIOS = [
    {"name": "link_postal", "sensitive": ["password"],
     "params": {**BASE, "cart_route": "cart_link", "zip_source": "postal_code", "postal_code": "10001"}},
    {"name": "url_postal", "sensitive": ["password"],
     "params": {**BASE, "cart_route": "cart_url", "zip_source": "postal_code", "postal_code": "10001",
                "cart_url": ENTRY + "cart.html"}},
    {"name": "url_phone", "sensitive": ["password", "phone"],
     "params": {**BASE, "cart_route": "cart_url", "zip_source": "phone", "phone": PHONE,
                "cart_url": ENTRY + "cart.html"}},
]


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


TOTAL_CONTRACT = {"total": {"type": "number", "required": True, "pattern": r"Total: \$\s*([\d.]+)"}}


def spec_data(scenarios=None, outputs=None) -> dict:
    data = {"name": "checkout_paths", "goal": "add {{product_name}} to the cart and read the checkout overview",
            "url": ENTRY, "selectors": ["cart_route", "zip_source"], "scenarios": scenarios or SCENARIOS}
    if outputs is not None:
        data["outputs"] = outputs
    return data


def campaign_script(cart_route: str, zip_source: str) -> list[ScriptedStep]:
    """What the model decides for one scenario: the route to the cart and the value typed as the zip differ."""
    route = (ScriptedStep("click", "link", "cart", expect="Your Cart") if cart_route == "cart_link"
             else ScriptedStep("navigate", value="{{cart_url}}", expect="Your Cart"))
    zip_value = "{{postal_code}}" if zip_source == "postal_code" else "{{phone}}"
    return [
        ScriptedStep("type", "textbox", "Username", value="{{username}}"),
        ScriptedStep("type", "textbox", "Password", value="{{password}}"),
        ScriptedStep("click", "button", "Login", expect="Products"),
        ScriptedStep("click", "button", "Add to cart", context="{{product_name}}", expect="Remove"),
        route,
        ScriptedStep("click", "button", "Checkout", expect="Checkout: Your Information"),
        ScriptedStep("type", "textbox", "First Name", value="{{first_name}}"),
        ScriptedStep("type", "textbox", "Last Name", value="{{last_name}}"),
        ScriptedStep("type", "textbox", "Zip/Postal Code", value=zip_value),
        ScriptedStep("click", "button", "Continue", expect="Checkout: Overview"),
        ScriptedStep("extract", "text", text="Total:", output_name="total", pattern=MONEY),
        ScriptedStep("done", expect="Checkout: Overview",
                     outcomes=[{"code": "user_locked_out", "text_contains": "locked out"}]),
    ]


def scripted(scenario) -> ScriptedPlanner:
    return ScriptedPlanner(campaign_script(scenario.params["cart_route"], scenario.params["zip_source"]))


class ClosableSurface(FakeSurface):
    """A session that must be closed, like the browser; optionally one whose close itself fails."""

    def __init__(self, close_fails: bool = False):
        super().__init__()
        self.closed, self.close_fails = False, close_fails

    def close(self) -> None:
        self.closed = True
        if self.close_fails:
            raise OSError("browser already gone")


def run(data=None, planner_factory=scripted, operator=None, surface_factory=None):
    surfaces: list[FakeSurface] = []

    def default_surface_factory(secrets):
        surfaces.append(ClosableSurface())
        return surfaces[-1]

    surface_factory = surface_factory or default_surface_factory

    result = run_campaign(spec_from_dict(data or spec_data()), surface_factory, planner_factory,
                          operator or NoOperator(), HOSTS, spec_path="scenarios/checkout_paths.json")
    return result, surfaces


def replay(graph, params):
    surface = FakeSurface()
    log = RunLog("replay", secrets=("secret_sauce", PHONE))
    result = replay_engine(graph, params, surface, Policy(allowed_hosts=HOSTS),
                           Escalator(NoOperator(), SessionControl(), log), log)
    return result, surface


# ---------- specification ----------

def test_spec_is_parsed_and_checked():
    spec = spec_from_dict(spec_data())
    assert spec.name == "checkout_paths" and spec.selectors == ["cart_route", "zip_source"]
    assert [s.name for s in spec.scenarios] == ["link_postal", "url_postal", "url_phone"]
    assert spec.scenarios[2].sensitive == ["password", "phone"]

    def rejects(mutate, message):
        data = json.loads(json.dumps(spec_data()))
        mutate(data)
        with pytest.raises(CampaignError, match=message):
            spec_from_dict(data)

    rejects(lambda d: d.pop("selectors"), "missing keys: \\['selectors'\\]")
    rejects(lambda d: d.update(name="Checkout Paths"), "snake_case")
    rejects(lambda d: d.update(scenarios=[]), "non-empty list")
    rejects(lambda d: d.update(extra=1), "unknown keys: \\['extra'\\]")
    rejects(lambda d: d.update(selectors=["cart_route", "cart_route"]), "selectors must be unique")
    rejects(lambda d: d["scenarios"][0]["params"].pop("zip_source"),
            "scenario 'link_postal': missing a value for selector")
    rejects(lambda d: d["scenarios"][1].update(name="link_postal"), "scenario names must be unique")
    rejects(lambda d: d["scenarios"][1]["params"].update(cart_route="cart_link"),
            "scenarios 'link_postal' and 'url_postal' have the same selector values")
    rejects(lambda d: d["scenarios"][0].update(sensitive=["cart_route"]),
            "selectors \\['cart_route'\\] cannot be sensitive")
    rejects(lambda d: d["scenarios"][0].update(sensitive=["nope"]), "sensitive names \\['nope'\\] are not params")
    rejects(lambda d: d["scenarios"][0]["params"].update(postal_code=10001), "every param value must be a string")
    rejects(lambda d: d["scenarios"][0].update(mode="x"), "scenario 'link_postal' has unknown keys")


def test_spec_output_contract_is_checked():
    spec = spec_from_dict(spec_data(outputs=TOTAL_CONTRACT))
    assert spec.outputs == TOTAL_CONTRACT
    assert spec_from_dict(spec_data()).outputs == {}

    def rejects(outputs, message):
        with pytest.raises(CampaignError, match=message):
            spec_from_dict(spec_data(outputs=outputs))

    rejects([], "outputs must be an object")
    rejects({"Total": {"pattern": "x"}}, "outputs must be an object keyed by snake_case")
    rejects({"total": {"pattern": "x", "kind": "money"}}, "output 'total' has unknown keys: \\['kind'\\]")
    rejects({"total": {"type": "money", "pattern": "x"}}, "type must be one of")
    rejects({"total": {"required": "yes", "pattern": "x"}}, "required must be true or false")
    rejects({"total": {"type": "number"}}, "a regex pattern is required")
    rejects({"total": {"pattern": "("}}, "pattern is not a valid regex")


def test_the_declared_output_contract_wins_over_the_planner_and_is_enforced():
    def loose_pattern(scenario):
        script = campaign_script(scenario.params["cart_route"], scenario.params["zip_source"])
        script[-2] = ScriptedStep("extract", "text", text="Total:", output_name="total", pattern=r"([\d.]+)$")
        return ScriptedPlanner(script)

    result, _ = run(spec_data(outputs=TOTAL_CONTRACT), planner_factory=loose_pattern)
    assert result.graph.outputs["total"] == {**result.graph.outputs["total"], "type": "number", "required": True,
                                             "pattern": r"Total: \$\s*([\d.]+)", "example": "32.39"}
    for record in result.scenarios:
        trace = json.loads(open(record["trace"]).read())
        assert trace["outputs"]["total"]["pattern"] == r"Total: \$\s*([\d.]+)"
    assert replay(result.graph, dict(SCENARIOS[0]["params"]))[0].outputs == {"total": 32.39}

    def other_output(scenario):
        script = campaign_script(scenario.params["cart_route"], scenario.params["zip_source"])
        if scenario.name == "url_phone":
            script[-2] = ScriptedStep("extract", "text", text="Tax:", output_name="tax", pattern=MONEY)
        return ScriptedPlanner(script)

    # The declared total was never read: done is refused (the history names the missing output), the script has
    # nothing left, and the scenario ends through the ordinary stuck handoff rather than with a wrong artifact.
    with pytest.raises(CampaignFailed, match="scenario 'url_phone' failed: human aborted discovery") as failed:
        run(spec_data(outputs=TOTAL_CONTRACT), planner_factory=other_output)
    summary = failing_summary(failed)
    log = (evidence_module.EVIDENCE_DIR / summary["scenarios"][2]["run_id"] / "run.jsonl").read_text()
    assert '"event": "done_rejected", "outputs": {"total": "missing"}' in log

    def extra_output(scenario):
        script = campaign_script(scenario.params["cart_route"], scenario.params["zip_source"])
        if scenario.name == "url_phone":
            script.insert(-1, ScriptedStep("extract", "text", text="Tax:", output_name="tax", pattern=MONEY))
        return ScriptedPlanner(script)

    # An undeclared name is refused before it is read: the scenario records only the declared output and succeeds.
    result, _ = run(spec_data(outputs=TOTAL_CONTRACT), planner_factory=extra_output)
    assert sorted(result.graph.outputs) == ["total"]
    log = (evidence_module.EVIDENCE_DIR / result.scenarios[2]["run_id"] / "run.jsonl").read_text()
    assert "output 'tax' is not declared in the output contract; the only valid output names are ['total']" in log


def test_spec_file_errors_are_reported(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(CampaignError, match="cannot read campaign spec"):
        campaign_module.load_spec(bad)


# ---------- orchestration ----------

def test_each_scenario_is_discovered_on_its_own_session_and_merged_into_one_draft_graph(tmp_path):
    result, surfaces = run()
    assert len(surfaces) == 3 and all(s.screen == "overview" for s in surfaces)     # a fresh session each time
    assert result.artifact_path.endswith("checkout_paths.v1.json")
    assert sorted(p.name for p in artifact_module.ARTIFACTS_DIR.iterdir()) == ["checkout_paths.v1.json"]

    graph = load_artifact(result.artifact_path)
    assert graph.status == "draft" and graph.schema_version == "2.0"
    assert json.loads(open(result.artifact_path).read())["schema_version"] == "2.0"
    assert [n.id for n in graph.nodes if n.kind == "decision"] == ["d1", "d2", "d3"]
    assert graph.entry_node == "d1" and all(s.closed for s in surfaces)
    prov = graph.provenance
    assert prov["campaign_id"] == result.campaign_id and prov["merge_strategy"] == "prefix_tree"
    assert (prov["scenario_count"], prov["successful_scenario_count"], prov["planners"]) == (3, 3, ["scripted"])
    assert [s["name"] for s in prov["scenarios"]] == ["link_postal", "url_postal", "url_phone"]
    assert [s["run_id"].startswith("discovery-") for s in prov["scenarios"]] == [True] * 3
    assert len({s["run_id"] for s in prov["scenarios"]}) == 3
    assert all(s["verified"] and s["node_path"][:2] == ["d1", "s1"] and s["node_path"][-1] == "success"
               for s in prov["scenarios"])
    assert prov["scenarios"][2]["selectors"] == {"cart_route": "cart_url", "zip_source": "phone"}

    summary = json.loads(open(result.summary_path).read())
    assert summary["status"] == "succeeded" and summary["artifact"] == result.artifact_path
    assert summary["spec"] == "scenarios/checkout_paths.json" and summary["successful_scenario_count"] == 3
    for record in summary["scenarios"]:
        assert record["status"] == "succeeded" and record["node_path"] == \
            next(s["node_path"] for s in prov["scenarios"] if s["name"] == record["name"])
        assert (evidence_module.EVIDENCE_DIR / record["run_id"] / "run.jsonl").exists()
        assert record["trace"].endswith(f"{record['name']}.trace.json")
        trace = json.loads(open(record["trace"]).read())
        assert trace["schema_version"] == "2.0" and trace["name"] == "checkout_paths"
        assert '"risk"' not in json.dumps(trace)
        recorded = load_artifact(record["trace"])                 # every trace is a valid linear graph ...
        assert recorded.entry_node == "s1" and linear_path(recorded)[-1].action.action == "extract"
        assert all(n.effect in ("none", "reversible") for n in linear_path(recorded))
        assert {(n.action.action, n.effect, n.retry_safety) for n in linear_path(recorded)} >= {
            ("navigate", "none", "safe"), ("type", "reversible", "never_retry"),
            ("click", "reversible", "verify_before_retry"), ("extract", "none", "safe")}
    # ... and the merged graph keeps every node's classification exactly.
    for node in graph.nodes:
        if node.kind == "action":
            assert node.action.action != "type" or (node.effect, node.retry_safety) == ("reversible", "never_retry")
            assert node.action.action != "extract" or (node.effect, node.retry_safety) == ("none", "safe")


def test_merged_graph_replays_every_scenario_and_only_declared_combinations():
    result, _ = run()
    graph = result.graph
    for scenario in SCENARIOS:
        replayed, surface = replay(graph, dict(scenario["params"]))
        assert replayed.status == "success" and replayed.outputs == {"total": 32.39}, scenario["name"]
        assert surface.screen == "overview"
        if scenario["params"]["cart_route"] == "cart_url":
            assert ("navigate", ENTRY + "cart.html") in surface.actions and ("click", "cart", "") not in surface.actions
        if scenario["params"]["zip_source"] == "phone":
            assert ("type", "Zip/Postal Code", PHONE) in surface.actions

    undeclared = {**BASE, "cart_route": "cart_link", "zip_source": "phone", "phone": PHONE}
    replayed, surface = replay(graph, undeclared)
    assert replayed.status == "failure" and replayed.outcome_code == "no_matching_edge"
    assert replayed.step_id == "d1" and surface.actions == []                     # stopped at the entry gate


def test_missing_conditional_input_fails_before_any_action():
    result, _ = run()
    replayed, surface = replay(result.graph, {**BASE, "cart_route": "cart_url", "zip_source": "postal_code",
                                              "postal_code": "10001"})
    assert replayed.status == "failure" and replayed.outcome_code == "missing_inputs"
    assert "cart_url" in replayed.observed and surface.actions == []
    replayed, _ = replay(result.graph, {**BASE, "cart_route": "cart_link", "zip_source": "postal_code",
                                        "postal_code": "10001"})
    assert replayed.status == "success"                                  # cart_url is not needed on this path


def test_path_specific_sensitive_values_never_reach_disk(tmp_path):
    result, _ = run()
    graph = result.graph
    assert graph.inputs["phone"] == {**graph.inputs["phone"], "sensitive": True, "required": False,
                                     "required_when": [{"cart_route": "cart_url", "zip_source": "phone"}]}
    assert graph.inputs["password"]["sensitive"] is True and graph.inputs["password"]["required"] is True
    written = [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl")]
    assert len(written) >= 6                                             # artifact, summary, traces, logs
    for path in written:
        text = path.read_text()
        assert PHONE not in text and "secret_sauce" not in text, path


def test_a_failed_scenario_keeps_its_evidence_and_saves_nothing(tmp_path):
    run()                                                                 # an earlier version already exists
    before = {p.name: p.read_bytes() for p in artifact_module.ARTIFACTS_DIR.iterdir()}
    assert list(before) == ["checkout_paths.v1.json"]

    def planner_factory(scenario):
        if scenario.name == "url_phone":
            return ScriptedPlanner([ScriptedStep("click", "button", "Pay now")])   # not on screen: the planner is stuck
        return scripted(scenario)

    with pytest.raises(CampaignFailed, match="scenario 'url_phone' failed") as failed:
        run(planner_factory=planner_factory)
    assert failed.value.scenario == "url_phone"
    assert {p.name: p.read_bytes() for p in artifact_module.ARTIFACTS_DIR.iterdir()} == before

    summary = json.loads(failed.value.summary_path.read_text())
    assert summary["status"] == "failed" and summary["artifact"] is None
    assert [s["status"] for s in summary["scenarios"]] == ["succeeded", "succeeded", "failed"]
    assert summary["successful_scenario_count"] == 2
    failed_record = summary["scenarios"][2]
    assert "aborted" in failed_record["error"] and failed_record["node_path"] is None
    assert (evidence_module.EVIDENCE_DIR / failed_record["run_id"] / "run.jsonl").exists()
    assert all(json.loads(open(s["trace"]).read())["name"] == "checkout_paths" for s in summary["scenarios"][:2])
    campaign_log = (failed.value.summary_path.parent / "run.jsonl").read_text()
    assert '"scenario_failed"' in campaign_log and '"campaign_finished"' in campaign_log


def test_incompatible_scenarios_fail_the_campaign_without_saving():
    def planner_factory(scenario):
        script = campaign_script(scenario.params["cart_route"], scenario.params["zip_source"])
        if scenario.name == "url_phone":
            script[-2] = ScriptedStep("extract", "text", text="Tax:", output_name="tax", pattern=MONEY)   # other output
        return ScriptedPlanner(script)

    with pytest.raises(CampaignFailed, match="could not be merged: scenarios 'link_postal' and 'url_phone' disagree "
                                             "on outputs") as failed:
        run(planner_factory=planner_factory)
    assert failed.value.scenario is None and not artifact_module.ARTIFACTS_DIR.exists()
    assert json.loads(failed.value.summary_path.read_text())["status"] == "merge_failed"


def test_versions_increment_and_never_overwrite():
    first, _ = run()
    second, _ = run()
    assert first.artifact_path.endswith(".v1.json") and second.artifact_path.endswith(".v2.json")
    assert load_artifact(first.artifact_path).version == 1 and load_artifact(second.artifact_path).version == 2


def test_two_scenarios_with_identical_traces_share_one_path():
    twins = [dict(SCENARIOS[0]), {**SCENARIOS[0], "name": "link_postal_again",
                                  "params": {**SCENARIOS[0]["params"], "zip_source": "postal_code_again"}}]
    twins[1]["params"]["postal_code"] = "10001"

    def planner_factory(scenario):
        return ScriptedPlanner(campaign_script("cart_link", "postal_code"))

    result, _ = run(spec_data(twins), planner_factory=planner_factory)
    graph = result.graph
    assert [n.id for n in graph.nodes if n.kind == "decision"] == ["d1"]            # the entry gate only
    paths = [s["node_path"] for s in graph.provenance["scenarios"]]
    assert paths[0] == paths[1] and paths[0][:2] == ["d1", "s1"]


def test_a_campaign_without_selectors_is_a_linear_graph():
    spec = {**spec_data([SCENARIOS[0]]), "selectors": []}
    result, _ = run(spec)
    graph = result.graph
    assert graph.entry_node == "s1" and all(n.kind != "decision" for n in graph.nodes)
    assert replay(graph, dict(SCENARIOS[0]["params"]))[0].status == "success"


# ---------- unexpected failures ----------

class ExplodingPlanner(ScriptedPlanner):
    """The provider blows up mid-discovery, as an SDK or network error would."""

    def __init__(self, message: str):
        super().__init__([])
        self.message = message

    def decide(self, goal, params, observation, history, candidates=()) -> Action:
        raise ConnectionError(self.message)


def failing_summary(failed) -> dict:
    return json.loads(failed.value.summary_path.read_text())


def test_planner_factory_failure_ends_the_campaign_cleanly():
    def planner_factory(scenario):
        if scenario.name == "url_postal":
            raise RuntimeError("provider is not configured")
        return scripted(scenario)

    with pytest.raises(CampaignFailed, match="scenario 'url_postal' failed: unexpected RuntimeError: provider is "
                                             "not configured") as failed:
        run(planner_factory=planner_factory)
    assert failed.value.scenario == "url_postal" and not artifact_module.ARTIFACTS_DIR.exists()
    summary = failing_summary(failed)
    assert summary["status"] == "failed" and [s["status"] for s in summary["scenarios"]] == ["succeeded", "failed"]
    assert summary["scenarios"][1]["error"] == "unexpected RuntimeError: provider is not configured"
    assert (evidence_module.EVIDENCE_DIR / summary["scenarios"][1]["run_id"] / "run.jsonl").exists()


def test_surface_factory_failure_ends_the_campaign_cleanly():
    def surface_factory(secrets):
        raise OSError("chromium failed to launch")

    with pytest.raises(CampaignFailed, match="unexpected OSError: chromium failed to launch") as failed:
        run(surface_factory=surface_factory)
    summary = failing_summary(failed)
    assert summary["scenarios"][0]["status"] == "failed" and summary["artifact"] is None
    log = (evidence_module.EVIDENCE_DIR / summary["scenarios"][0]["run_id"] / "run.jsonl").read_text()
    assert '"discovery_failed"' in log and '"kind": "OSError"' in log and '"screenshot"' not in log


def test_provider_failure_during_discovery_closes_the_session_and_saves_nothing():
    surfaces = []

    def surface_factory(secrets):
        surfaces.append(ClosableSurface())
        return surfaces[-1]

    def planner_factory(scenario):
        return ExplodingPlanner("502 from the model API") if scenario.name == "url_phone" else scripted(scenario)

    with pytest.raises(CampaignFailed, match="scenario 'url_phone' failed: unexpected ConnectionError: 502") as failed:
        run(planner_factory=planner_factory, surface_factory=surface_factory)
    assert len(surfaces) == 3 and all(s.closed for s in surfaces)
    assert not artifact_module.ARTIFACTS_DIR.exists()
    summary = failing_summary(failed)
    assert [s["status"] for s in summary["scenarios"]] == ["succeeded", "succeeded", "failed"]
    evidence_dir = evidence_module.EVIDENCE_DIR / summary["scenarios"][2]["run_id"]
    assert (evidence_dir / "run.jsonl").exists() and '"discovery_failed"' in (evidence_dir / "run.jsonl").read_text()


def test_a_failing_close_does_not_hide_the_real_failure():
    def surface_factory(secrets):
        return ClosableSurface(close_fails=True)

    def planner_factory(scenario):
        return ExplodingPlanner("model timed out")

    with pytest.raises(CampaignFailed, match="unexpected ConnectionError: model timed out") as failed:
        run(planner_factory=planner_factory, surface_factory=surface_factory)
    summary = failing_summary(failed)
    log = (evidence_module.EVIDENCE_DIR / summary["scenarios"][0]["run_id"] / "run.jsonl").read_text()
    assert '"surface_close_failed"' in log and "browser already gone" in log


def test_secrets_in_error_messages_never_reach_logs_or_summaries(tmp_path):
    def planner_factory(scenario):
        if scenario.name == "url_phone":
            return ExplodingPlanner(f"rejected value {PHONE} for field phone (password secret_sauce)")
        return scripted(scenario)

    with pytest.raises(CampaignFailed) as failed:
        run(planner_factory=planner_factory)
    assert PHONE not in str(failed.value) and "secret_sauce" not in str(failed.value)
    assert "[REDACTED]" in str(failed.value)
    for path in tmp_path.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl"):
            assert PHONE not in path.read_text() and "secret_sauce" not in path.read_text(), path


# ---------- CLI ----------

def test_cli_runs_a_campaign_from_a_spec_file(monkeypatch, tmp_path, capsys):
    surfaces: list[ClosableSurface] = []
    monkeypatch.setattr(main_module, "PlaywrightSurface",
                        lambda **_: surfaces.append(ClosableSurface()) or surfaces[-1])
    real = main_module.run_campaign
    monkeypatch.setattr(main_module, "run_campaign",                    # only the LLM planner is replaced
                        lambda spec, surface_factory, planner_factory, **kw:
                        real(spec, surface_factory, scripted, **kw))
    spec_path = tmp_path / "checkout_paths.json"
    spec_path.write_text(json.dumps(spec_data()))

    assert main(["discover-campaign", "--spec", str(spec_path), "--operator", "none", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "Campaign succeeded: 3 scenario(s) merged into" in out and "checkout_paths.v1.json" in out
    assert len(surfaces) == 3 and all(s.closed for s in surfaces)

    spec_path.write_text(json.dumps({**spec_data(), "selectors": ["nope"]}))
    assert main(["discover-campaign", "--spec", str(spec_path), "--operator", "none", "--quiet"]) == 2
    assert "Cannot load campaign spec" in capsys.readouterr().out


# ---------- reviewer-declared outcomes (generic contract, proven on the fake shop) ----------

LOCKED = {"code": "user_locked_out", "kind": "business", "detect": {"text_contains": "locked out"}}
NOTICE = {"code": "dismiss_cookie_notice", "kind": "recoverable", "detect": {"dialog_contains": "We use cookies"},
          "recover": {"action": "click",
                      "target": {"strategies": [{"kind": "role", "role": "button", "name": "Accept"}]}}}


def test_declared_outcomes_reach_every_trace_the_graph_and_replay_classification():
    def quiet_planner(scenario):                       # a planner that declares nothing itself
        script = campaign_script(scenario.params["cart_route"], scenario.params["zip_source"])
        script[-1] = ScriptedStep("done", expect="Checkout: Overview")
        return ScriptedPlanner(script)

    result, _ = run(spec_data(outputs=TOTAL_CONTRACT) | {"outcomes": [LOCKED, NOTICE]}, planner_factory=quiet_planner)
    graph = result.graph
    assert [(o["code"], o["source"]) for o in graph.outcomes] == [("user_locked_out", "reviewer"),
                                                                  ("dismiss_cookie_notice", "reviewer")]
    for record in result.scenarios:
        trace = json.loads(open(record["trace"]).read())
        assert [o["code"] for o in trace["outcomes"]] == ["user_locked_out", "dismiss_cookie_notice"]

    locked, surface = replay(graph, {**SCENARIOS[0]["params"], "username": "locked_out_user"})
    assert locked.status == "business_outcome" and locked.outcome_code == "user_locked_out"
    surface = FakeSurface(show_notice=True)
    log = RunLog("replay", secrets=("secret_sauce", PHONE))
    recovered = replay_engine(graph, dict(SCENARIOS[0]["params"]), surface, Policy(allowed_hosts=HOSTS),
                              Escalator(NoOperator(), SessionControl(), log), log)
    assert recovered.status == "success" and recovered.recoveries == ["s1: dismiss_cookie_notice"]


def test_planner_agreeing_with_the_reviewer_is_deduplicated_silently():
    result, _ = run(spec_data(outputs=TOTAL_CONTRACT) | {"outcomes": [LOCKED]})     # the script declares it too
    assert [o for o in result.graph.outcomes if o["code"] == "user_locked_out"] == [{**LOCKED, "source": "reviewer"}]
    for record in result.scenarios:
        log = (evidence_module.EVIDENCE_DIR / record["run_id"] / "run.jsonl").read_text()
        assert "reviewer_outcome_overrode_planner" not in log


def test_reviewer_definition_overrides_the_planners_with_a_logged_warning():
    # The script declares invalid_credentials as "do not match"; the reviewer declares the fuller site text.
    reviewer = {"code": "invalid_credentials", "kind": "business",
                "detect": {"text_contains": "Username and password do not match any user"}}

    def guessing_planner(scenario):            # the planner declares the same code from a shorter guess
        script = campaign_script(scenario.params["cart_route"], scenario.params["zip_source"])
        script[-1] = ScriptedStep("done", expect="Checkout: Overview", outcomes=[
            {"code": "user_locked_out", "text_contains": "locked out"},
            {"code": "invalid_credentials", "text_contains": "do not match"}])
        return ScriptedPlanner(script)

    result, _ = run(spec_data(outputs=TOTAL_CONTRACT) | {"outcomes": [reviewer]}, planner_factory=guessing_planner)
    graph = result.graph
    assert [o for o in graph.outcomes if o["code"] == "invalid_credentials"] == [{**reviewer, "source": "reviewer"}]
    assert "user_locked_out" in {o["code"] for o in graph.outcomes}          # planner-only outcomes stay
    for record in result.scenarios:
        log = (evidence_module.EVIDENCE_DIR / record["run_id"] / "run.jsonl").read_text()
        [event] = [json.loads(l) for l in log.splitlines() if "reviewer_outcome_overrode_planner" in l]
        assert event["code"] == "invalid_credentials"
        assert event["planner"]["detect"] == {"text_contains": "do not match"}
        assert event["reviewer"]["detect"] == reviewer["detect"]
    replayed, _ = replay(graph, {**SCENARIOS[0]["params"], "password": "wrong"})
    assert (replayed.status, replayed.outcome_code) == ("business_outcome", "invalid_credentials")
