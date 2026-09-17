"""Verified prefix reuse: an approved capability opens a new discovery, deterministically, on a live session."""
import json
from dataclasses import replace

import pytest

from src.cua import agent as agent_module
from src.cua import artifact as artifact_module
from src.cua import reuse as reuse_module
from src.cua.__main__ import build_parser, main
from src.cua.agent import DiscoveryFailed
from src.cua.artifact import linear_path, nodes_by_id, save_artifact, to_dict, validate
from src.cua.campaign import CampaignError, CampaignFailed, spec_from_dict
from src.cua.escalation import NoOperator
from src.cua.lifecycle import load_artifact, run_replay, sha256_of
from src.cua.merge import merge_traces
from src.cua.models import Observation
from src.cua.replay import replay
from src.cua.reuse import ReuseError, discover_with_reuse, resolve_reuse
from tests.context import (ENTRY, HOSTS, PARAMS, Escalator, Policy, RecordingOperator, RunLog, SessionControl,
                           build_linear, checkout_artifact, checkout_nodes, click_node, finish_node)
from tests.fake_surface import FakeSurface
from tests.scripted_planner import ScriptedPlanner, ScriptedStep, checkout_script
from tests.test_merge import LOGIN, link_trace, url_trace


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


class Session(FakeSurface):
    def __init__(self, **options):
        super().__init__(**options)
        self.closes = 0

    @property
    def closed(self) -> bool:
        return self.closes == 1

    def close(self) -> None:
        self.closes += 1


class Sessions:
    """Counts the sessions a discovery opened and remembers each one."""

    def __init__(self, **options):
        self.created: list[Session] = []
        self.options = options

    def __call__(self) -> Session:
        self.created.append(Session(**self.options))
        return self.created[-1]


def login_prefix_artifact(name: str = "shop_login", nodes=None, **overrides):
    """sign in -> product list: the first four actions of the checkout flow, as its own capability."""
    base = checkout_artifact()
    fields = {"name": name, "goal": "sign in", "surface_meta": dict(base.surface), "params": {
        k: v for k, v in PARAMS.items() if k in ("username", "password")}, "sensitive": {"password"},
        "nodes": nodes if nodes is not None else checkout_nodes()[:4], "outputs": {},
        "outcomes": [dict(o) for o in base.outcomes], "success": {"text_contains": "Products"}, "run_id": "test-run"}
    fields.update(overrides)
    return build_linear(**fields)


def approved(artifact, version=1, path=None):
    """Tests approve by data: the lifecycle's approve() is exercised elsewhere."""
    return save_artifact(replace(artifact, status="approved", version=version), secrets=("secret_sauce",), path=path)


def draft(artifact, version=1):
    return save_artifact(replace(artifact, version=version), secrets=("secret_sauce",))


class SpyPlanner(ScriptedPlanner):
    def __init__(self, script):
        super().__init__(script)
        self.observations: list[Observation] = []

    def decide(self, goal, params, observation, history):
        self.observations.append(observation)
        return super().decide(goal, params, observation, history)


def after_login_script():
    return checkout_script()[4:]


def run(reuse, script=None, sessions=None, params=None, operator=None, sensitive=("password",), max_steps=25,
        allowed=HOSTS, entry=ENTRY):
    sessions = sessions or Sessions()
    planner = SpyPlanner(script if script is not None else after_login_script())
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover_with_reuse(sessions, reuse, goal="reach the checkout overview", name="checkout_review",
                                   params=dict(params or PARAMS), planner=planner,
                                   policy=Policy(allowed_hosts=list(allowed)),
                                   escalator=Escalator(operator or NoOperator(), SessionControl(), log), log=log,
                                   entry_url=entry, sensitive=set(sensitive), max_steps=max_steps)
    return artifact, planner, sessions, log


def events(log, name):
    return [json.loads(l) for l in log.path.read_text().splitlines() if f'"event": "{name}"' in l]


def replays(artifact, params=None):
    surface = FakeSurface()
    log = RunLog("replay", secrets=("secret_sauce",))
    return replay(artifact, dict(params or PARAMS), surface, Policy(allowed_hosts=HOSTS),
                  Escalator(NoOperator(), SessionControl(), log), log), surface


# ---------- the happy path ----------

def test_approved_prefix_runs_without_a_planner_and_discovery_continues_on_that_screen():
    path = approved(login_prefix_artifact())
    reuse = resolve_reuse("shop_login", None)
    artifact, planner, sessions, log = run(reuse)

    assert len(sessions.created) == 1 and sessions.created[0].closes == 1            # one session, kept, closed once
    assert planner.observations[0].url == ENTRY + "inventory.html"                    # the planner starts after login
    assert any(e.text == "Products" for e in planner.observations[0].elements)
    path_nodes = linear_path(artifact)
    assert [n.action.action for n in path_nodes] == ["navigate", "type", "type", "click"] + \
        ["click", "click", "click", "type", "type", "type", "click", "extract", "extract", "extract", "extract"]
    assert [n.id for n in path_nodes] == [f"s{i}" for i in range(1, 16)]              # renumbered, contiguous
    assert path_nodes[3].action.checkpoint == {"text_contains": "Products"} and path_nodes[1].action.value == "{{username}}"
    assert [o["code"] for o in artifact.outcomes if o["kind"] == "recoverable"] == ["dismiss_cookie_notice"]
    prov = artifact.provenance["reuse"]
    assert prov == {"source_name": "shop_login", "source_version": 1, "source_schema_version": "2.0",
                    "source_digest": sha256_of(path), "source_path": str(path),
                    "executed_path": ["s1", "s2", "s3", "s4"], "imported_nodes": 4, "reuse_run_id": log.run_id,
                    "planner_decisions": 12}
    text = log.path.read_text()
    for event in ("reuse_preflight_passed", "reuse_started", "reuse_step_executed", "reuse_succeeded",
                  "discovery_resumed_after_reuse"):
        assert f'"{event}"' in text, event
    assert len(events(log, "reuse_step_executed")) == 4 and "secret_sauce" not in text
    assert events(log, "reuse_succeeded")[0] == {**events(log, "reuse_succeeded")[0], "executed_path": 4,
                                                 "performed_attempts": 4, "imported_nodes": 4}
    assert "standard_user" not in json.dumps(prov)
    validate(artifact)
    assert '"risk"' not in json.dumps(to_dict(artifact))

    path.unlink()                                                                      # the source is gone ...
    result, surface = replays(artifact)
    assert result.status == "success" and result.outputs["total"] == 32.39            # ... and it still replays
    assert surface.actions[:4] == [("navigate", ENTRY), ("type", "Username", "standard_user"),
                                   ("type", "Password", "secret_sauce"), ("click", "Login", "")]


def test_reused_nodes_keep_their_effect_and_retry_safety_exactly():
    nodes = checkout_nodes()[:4]
    nodes[3].effect, nodes[3].retry_safety = "reversible", "never_retry"     # a reviewed click: never repeated
    nodes[1].effect, nodes[1].retry_safety = "reversible", "safe"
    nodes[0].effect, nodes[0].retry_safety = "none", "safe"
    approved(login_prefix_artifact(nodes=nodes))
    artifact, _, _, log = run(resolve_reuse("shop_login", None))
    imported = nodes_by_id(artifact)
    assert (imported["s4"].effect, imported["s4"].retry_safety) == ("reversible", "never_retry")
    assert (imported["s2"].effect, imported["s2"].retry_safety) == ("reversible", "safe")
    assert (imported["s1"].effect, imported["s1"].retry_safety) == ("none", "safe")
    assert imported["s4"].action == nodes[3].action and imported["s4"].action is not nodes[3].action
    assert [(e["effect"], e["retry_safety"]) for e in events(log, "reuse_step_executed")][3] == \
        ("reversible", "never_retry")
    assert (imported["s5"].effect, imported["s5"].retry_safety) == ("reversible", "verify_before_retry")   # new
    saved = json.loads(save_artifact(artifact, secrets=("secret_sauce",)).read_text())
    s4 = next(n for n in saved["nodes"] if n["id"] == "s4")
    assert (s4["effect"], s4["retry_safety"]) == ("reversible", "never_retry") and saved["schema_version"] == "2.0"


def test_reused_prefix_replays_no_planner_and_the_planner_is_never_asked_during_it():
    approved(login_prefix_artifact())
    reuse = resolve_reuse("shop_login", None)
    artifact, planner, sessions, log = run(reuse)
    lines = log.path.read_text().splitlines()
    first_decision = next(i for i, l in enumerate(lines) if '"planner_decided"' in l)
    last_reuse = max(i for i, l in enumerate(lines) if '"reuse_step_executed"' in l)
    assert last_reuse < first_decision and len(planner.observations) == 12


# ---------- a branching prefix imports only the branch that ran ----------

def test_branching_prefix_imports_only_the_selected_branch():
    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], "campaign-test")
    prefix_path = save_artifact(replace(graph, status="approved"), secrets=("secret_sauce",))
    reuse = resolve_reuse(None, str(prefix_path))
    params = {**PARAMS, "cart_route": "cart_url", "cart_url": ENTRY + "cart.html"}
    script = checkout_script()[6:]                                                     # from Checkout onward
    artifact, planner, sessions, log = run(reuse, script=script, params=params)
    assert planner.observations[0].url == ENTRY + "cart.html"
    path = linear_path(artifact)
    assert [n.action.action for n in path[:6]] == ["navigate", "type", "type", "click", "click", "navigate"]
    assert path[5].action.value == "{{cart_url}}"                                      # the url branch ...
    assert not any(n.action.action == "click" and n.action.target
                   and n.action.target.strategies[0].get("name") == "cart" for n in path)   # ... not the link branch
    assert artifact.provenance["reuse"]["executed_path"] == ["s1", "s2", "s3", "s4", "s5", "s7"]
    assert [n.id for n in path] == [f"s{i}" for i in range(1, len(path) + 1)]         # renumbered on import
    assert all(n.kind == "action" for n in artifact.nodes[:-1]) and artifact.nodes[-1].kind == "terminal"
    result, _ = replays(artifact, params)
    assert result.status == "success"


# ---------- preflight rejections happen before any surface exists ----------

def rejected(reuse, message, **overrides):
    sessions = Sessions()
    with pytest.raises(ReuseError, match=message):
        run(reuse, sessions=sessions, **overrides)
    assert sessions.created == []


def test_unsafe_or_incompatible_prefixes_are_rejected_before_any_surface_action():
    with pytest.raises(ReuseError, match="status is 'draft'"):
        resolve_reuse(None, str(draft(login_prefix_artifact())))
    assert resolve_reuse("shop_login", None) is None                                  # drafts do not count

    reuse = resolve_reuse(None, str(approved(login_prefix_artifact(), version=2)))
    rejected(reuse, "needs inputs \\['password'\\]", params={"username": "standard_user"})
    rejected(reuse, "sensitive in shop_login v2 but not marked sensitive", sensitive=())
    rejected(reuse, "starts at 'https://www.saucedemo.com/'", entry="https://other.example/")
    rejected(reuse, "may visit hosts \\['www.saucedemo.com'\\]", allowed=["other.example"])

    risky = login_prefix_artifact(name="risky_prefix", nodes=checkout_nodes()[:4] + [finish_node("s5")])
    rejected(resolve_reuse(None, str(approved(risky))),
             "actions that must never run unattended: \\['s5 \\(irreversible\\)'\\]")
    unknown = login_prefix_artifact(name="unknown_prefix")
    nodes_by_id(unknown)["s2"].effect, nodes_by_id(unknown)["s2"].retry_safety = "unknown", "never_retry"
    rejected(resolve_reuse(None, str(approved(unknown))), "\\['s2 \\(unknown\\)'\\]")


def test_resolution_is_deterministic_and_checks_file_names():
    with pytest.raises(ReuseError, match="not both"):
        resolve_reuse("shop_login", "artifacts/x.json")
    with pytest.raises(ReuseError, match="cannot reuse"):
        resolve_reuse(None, str(artifact_module.ARTIFACTS_DIR / "missing.v1.json"))
    log = RunLog("discovery")
    assert resolve_reuse("nothing_here", None, log) is None
    assert events(log, "reuse_not_found")[0]["capability"] == "nothing_here"
    assert resolve_reuse(None, None) is None

    approved(login_prefix_artifact(), version=1)
    approved(login_prefix_artifact(), version=3)
    draft(login_prefix_artifact(), version=4)                                          # newer, but a draft
    mislabeled = approved(login_prefix_artifact(), version=2,
                          path=artifact_module.ARTIFACTS_DIR / "shop_login.v9.json")  # says v9, holds v2
    foreign = approved(login_prefix_artifact(name="other_login"), version=1,
                       path=artifact_module.ARTIFACTS_DIR / "shop_login.v8.json")     # says shop_login, holds another
    plan = resolve_reuse("shop_login", None, log)
    assert plan.version == 3 and plan.path.endswith("shop_login.v3.json") and plan.digest == sha256_of(plan.path)
    assert events(log, "reuse_resolved")[0]["version"] == 3
    assert resolve_reuse("shop_login", None).path == plan.path                         # the same answer every time
    with pytest.raises(ReuseError, match="holds shop_login v2, not shop_login v9 as its name says"):
        resolve_reuse(None, str(mislabeled))
    with pytest.raises(ReuseError, match="holds other_login v1, not shop_login v8"):
        resolve_reuse(None, str(foreign))

    parser = build_parser()
    args = parser.parse_args(["discover", "--goal", "g", "--url", "u", "--name", "n",
                              "--reuse-capability", "shop_login"])
    assert args.reuse_capability == "shop_login" and args.reuse_artifact is None
    with pytest.raises(SystemExit):
        parser.parse_args(["discover", "--goal", "g", "--url", "u", "--name", "n", "--reuse-capability", "a",
                           "--reuse-artifact", "b"])


def test_cli_reports_a_misconfigured_reuse_without_opening_a_browser(monkeypatch, capsys):
    from src.cua import __main__ as main_module
    def no_browser(**_):
        raise AssertionError("no browser")
    monkeypatch.setattr(main_module, "PlaywrightSurface", no_browser)
    exit_code = main(["discover", "--goal", "g", "--url", ENTRY, "--name", "n", "--reuse-artifact",
                      str(artifact_module.ARTIFACTS_DIR / "nope.v1.json"), "--operator", "none", "--quiet"])
    assert exit_code == 2 and "Cannot reuse" in capsys.readouterr().out


# ---------- clean fallback ----------

def test_a_failing_prefix_closes_the_dirty_session_and_discovery_restarts_on_a_fresh_one():
    stale = login_prefix_artifact(name="stale_login")
    nodes_by_id(stale)["s4"].action.checkpoint = {"text_contains": "Welcome back"}    # the site never shows this
    reuse = resolve_reuse(None, str(approved(stale)))
    artifact, planner, sessions, log = run(reuse, script=checkout_script())
    first, second = sessions.created
    assert first.closes == 1 and second.closes == 1 and first is not second
    assert first.actions[:4] == [("navigate", ENTRY), ("type", "Username", "standard_user"),
                                 ("type", "Password", "secret_sauce"), ("click", "Login", "")]  # dirty: 4 actions ran
    assert planner.observations[0].url == ENTRY and second.screen == "overview"        # discovery began afresh
    assert "reuse" not in artifact.provenance and len(linear_path(artifact)) == 15
    failed = events(log, "reuse_failed")[0]
    assert failed == {**failed, "status": "failure", "outcome_code": "checkpoint_not_met", "step_id": "s4",
                      "executed_path": ["s1", "s2", "s3"], "performed_attempts": 4}
    assert events(log, "reuse_fallback_started") and "secret_sauce" not in log.path.read_text()


def test_a_business_outcome_from_the_prefix_also_falls_back():
    reuse = resolve_reuse(None, str(approved(login_prefix_artifact())))
    script = [ScriptedStep("type", "textbox", "Username", value="{{username}}"),
              ScriptedStep("type", "textbox", "Password", value="{{password}}"),
              ScriptedStep("click", "button", "Login", expect="locked out"),
              ScriptedStep("done", expect="locked out")]
    artifact, planner, sessions, log = run(reuse, script=script, params={**PARAMS, "username": "locked_out_user"})
    assert [s.closes for s in sessions.created] == [1, 1]
    assert events(log, "reuse_failed")[0]["status"] == "business_outcome"
    assert events(log, "reuse_failed")[0]["outcome_code"] == "user_locked_out"
    assert "reuse" not in artifact.provenance
    assert linear_path(artifact)[3].action.checkpoint == {"text_contains": "locked out"}


def test_a_prefix_that_cannot_be_imported_closes_the_first_session_and_falls_back(monkeypatch):
    real = reuse_module.execute_reuse

    def corrupted(*args, **kwargs):                       # the replay succeeds, but its trace is unusable
        result = real(*args, **kwargs)
        result.executed_path[2].pop("effect")
        return result

    monkeypatch.setattr(reuse_module, "execute_reuse", corrupted)
    reuse = resolve_reuse(None, str(approved(login_prefix_artifact())))
    artifact, planner, sessions, log = run(reuse, script=checkout_script())
    first, second = sessions.created
    assert first.closes == 1 and second.closes == 1 and first.logged_in and not second.cart == []
    assert events(log, "reuse_import_failed")[0]["error"].startswith("KeyError")
    assert events(log, "reuse_succeeded") == [] and events(log, "reuse_fallback_started")
    assert planner.observations[0].url == ENTRY and "reuse" not in artifact.provenance
    assert len(linear_path(artifact)) == 15 and replays(artifact)[0].status == "success"


def test_discovery_failure_after_a_successful_prefix_still_closes_the_session_once():
    reuse = resolve_reuse(None, str(approved(login_prefix_artifact())))
    sessions = Sessions()
    with pytest.raises(DiscoveryFailed, match="aborted"):
        run(reuse, script=[ScriptedStep("stuck")], sessions=sessions, operator=RecordingOperator("abort"))
    assert [s.closes for s in sessions.created] == [1]


def test_the_lifecycle_purpose_forces_deny_and_needs_approval():
    prefix = draft(login_prefix_artifact())
    log = RunLog("replay")
    result = run_replay(load_artifact(prefix), dict(PARAMS), FakeSurface(), Policy(allowed_hosts=HOSTS),
                        Escalator(NoOperator(), SessionControl(), log), log, purpose="discovery_reuse")
    assert result.outcome_code == "artifact_not_approved"
    finish = checkout_artifact(extra_node=finish_node())
    finish.status = "approved"
    surface = FakeSurface()
    result = run_replay(finish, dict(PARAMS), surface, Policy(allowed_hosts=HOSTS),
                        Escalator(NoOperator(), SessionControl(), log), log, purpose="discovery_reuse",
                        irreversible_policy="allow")
    assert result.outcome_code == "irreversible_denied" and ("click", "Finish", "") not in surface.actions


# ---------- only a successful path is ever imported ----------

def test_prefix_from_refuses_anything_but_a_successful_replay():
    reuse = resolve_reuse(None, str(approved(login_prefix_artifact())))
    locked, _ = replays(reuse.artifact, {**PARAMS, "username": "locked_out_user"})
    assert locked.status == "business_outcome" and locked.performed_attempts == 4
    with pytest.raises(ReuseError, match="only a successful replay can be imported, got 'business_outcome'"):
        reuse_module.prefix_from(locked, reuse, None, "run")
    good, _ = replays(reuse.artifact)
    prefix = reuse_module.prefix_from(good, reuse, None, "run")
    assert [n.id for n in prefix.nodes] == ["s1", "s2", "s3", "s4"] and all(n.kind == "action" for n in prefix.nodes)
    assert "performed_attempts" not in json.dumps(prefix.provenance)


# ---------- campaigns ----------

def test_campaign_scenarios_reuse_the_prefix_independently_and_merge():
    from tests.test_campaign import SCENARIOS, TOTAL_CONTRACT, campaign_script, spec_data
    from src.cua.campaign import run_campaign
    approved(login_prefix_artifact())
    spec = spec_from_dict(spec_data(outputs=TOTAL_CONTRACT) | {"reuse_capability": "shop_login"})
    sessions: list[Session] = []

    def surface_factory(secrets):
        sessions.append(Session())
        return sessions[-1]

    def planner_factory(scenario):
        return ScriptedPlanner(campaign_script(scenario.params["cart_route"], scenario.params["zip_source"])[3:])

    result = run_campaign(spec, surface_factory, planner_factory, NoOperator(), HOSTS, spec_path="spec.json")
    assert [s.closes for s in sessions] == [1, 1, 1] and len({id(s) for s in sessions}) == 3
    graph = result.graph
    assert [s["node_path"][:5] for s in graph.provenance["scenarios"]] == [["d1", "s1", "s2", "s3", "s4"]] * 3
    for record in result.scenarios:
        trace = json.loads(open(record["trace"]).read())
        assert trace["provenance"]["reuse"]["source_name"] == "shop_login"
        assert trace["provenance"]["reuse"]["imported_nodes"] == 4 and trace["schema_version"] == "2.0"
        assert "secret_sauce" not in open(record["trace"]).read()
        assert "secret_sauce" not in open(record["evidence"] + "/run.jsonl").read()
    assert "secret_sauce" not in open(result.artifact_path).read()

    with pytest.raises(CampaignError, match="not both"):
        spec_from_dict(spec_data() | {"reuse_capability": "a", "reuse_artifact": "b"})
    with pytest.raises(CampaignFailed, match="reuse is misconfigured"):
        run_campaign(spec_from_dict(spec_data() | {"reuse_artifact": "artifacts/none.v1.json"}), surface_factory,
                     planner_factory, NoOperator(), HOSTS)
    assert all(s.closes == 1 for s in sessions)
