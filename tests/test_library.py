"""Automatic verified composition: the artifact library, state-matched atomic candidates (one verified action
each), the planner's choice between every action, deterministic execution, inlining, loop protection, failure
handling and clean restarts. Offline."""
import json
from dataclasses import replace

import pytest

from src.cua import agent as agent_module
from src.cua import artifact as artifact_module
from src.cua import library as library_module
from src.cua import reuse as reuse_module
from src.cua.__main__ import build_parser
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import copy_node, linear_path, nodes_by_id, save_artifact
from src.cua.campaign import run_campaign, spec_from_dict
from src.cua.escalation import NoOperator
from src.cua.library import CatalogEntry, build_library, execute_candidate, find_candidates
from src.cua.merge import merge_traces
from src.cua.models import Observation
from src.cua.lifecycle import sha256_of
from src.cua.replay import replay
from src.cua.replay import replay as replay_engine
from src.cua.reuse import discover_with_reuse, resolve_reuse
from tests.context import (ENTRY, HOSTS, PARAMS, Escalator, GraphAction, Policy, RecordingOperator, RunLog,
                           SessionControl, build_linear, checkout_artifact, checkout_nodes, finish_node, linear_node,
                           navigate_node, text_ladder)
from tests.fake_surface import FakeSurface
from tests.scripted_planner import ScriptedPlanner, ScriptedStep, checkout_script
from tests.test_merge import link_trace, url_trace

MONEY = r"\$\s*([\d.]+)"


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


class Session(FakeSurface):
    def __init__(self, **options):
        super().__init__(**options)
        self.closes = 0

    def close(self) -> None:
        self.closes += 1


class Sessions:
    def __init__(self, **options):
        self.created: list[Session] = []
        self.options = options

    def __call__(self) -> Session:
        self.created.append(Session(**self.options))
        return self.created[-1]


class SpyPlanner(ScriptedPlanner):
    def __init__(self, script):
        super().__init__(script)
        self.observations: list[Observation] = []
        self.offered: list[list[str]] = []

    def decide(self, goal, params, observation, history, candidates=()):
        self.observations.append(observation)
        self.offered.append([c.candidate_id for c in candidates])
        return super().decide(goal, params, observation, history, candidates)


# ---------- approved artifacts for the library ----------

def capability(name: str, nodes, success: dict, params=None, outputs=None, **overrides):
    base = checkout_artifact()
    fields = {"name": name, "goal": f"{name} on the shop", "surface_meta": dict(base.surface),
              "params": params or {k: v for k, v in PARAMS.items() if k in ("username", "password", "product_name")},
              "sensitive": {"password"}, "nodes": nodes, "outputs": outputs or {},
              "outcomes": [dict(o) for o in base.outcomes], "success": success, "run_id": "test-run"}
    fields.update(overrides)
    return build_linear(**fields)


def login_artifact(name="shop_login", nodes=None):
    return capability(name, nodes if nodes is not None else checkout_nodes()[:4], {"text_contains": "Products"},
                      params={"username": "standard_user", "password": "secret_sauce"})


def cart_artifact(name="shop_cart", nodes=None):
    return capability(name, nodes if nodes is not None else checkout_nodes()[:6], {"text_contains": "Your Cart"})


def approved(artifact, version=1, path=None):
    return save_artifact(replace(artifact, status="approved", version=version), secrets=("secret_sauce",), path=path)


def library_for(params=None, sensitive=("password",), hosts=HOSTS, entry=ENTRY, max_reuses=5):
    log = RunLog("discovery", secrets=("secret_sauce",))
    return build_library(None, entry, list(hosts), dict(params or PARAMS), set(sensitive), log, max_reuses), log


def logged_in_surface() -> FakeSurface:
    surface = FakeSurface()
    surface.navigate(ENTRY)
    elements = {e.name: e for e in surface.observe().elements}
    surface.type(elements["Username"], "standard_user")
    surface.type(elements["Password"], "secret_sauce")
    surface.click(elements["Login"])
    surface.actions.clear()
    return surface


def reuse(name, **kw):
    return ScriptedStep("reuse", name=name, **kw)


def reuse_id(candidate_id):
    return ScriptedStep("reuse", value=candidate_id)


LOGIN_REUSE = [reuse("shop_login")] * 3            # the login prefix, one verified action at a time: s2, s3, s4


def run(script, sessions=None, params=None, operator=None, auto_reuse=True, max_auto_reuses=5, prefix=None,
        output_contract=None, extra_outcomes=None):
    sessions = sessions or Sessions()
    planner = SpyPlanner(script)
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover_with_reuse(sessions, prefix, goal="reach the checkout overview and read the totals",
                                   name="checkout_review", params=dict(params or PARAMS), planner=planner,
                                   policy=Policy(allowed_hosts=HOSTS),
                                   escalator=Escalator(operator or NoOperator(), SessionControl(), log), log=log,
                                   entry_url=ENTRY, sensitive={"password"}, max_steps=25, auto_reuse=auto_reuse,
                                   max_auto_reuses=max_auto_reuses, output_contract=output_contract,
                                   extra_outcomes=extra_outcomes)
    return artifact, planner, sessions, log


def events(log, name):
    return [json.loads(l) for l in log.path.read_text().splitlines() if f'"event": "{name}"' in l]


def replays(artifact, params=None):
    surface = FakeSurface()
    log = RunLog("replay", secrets=("secret_sauce",))
    return replay(artifact, dict(params or PARAMS), surface, Policy(allowed_hosts=HOSTS),
                  Escalator(NoOperator(), SessionControl(), log), log), surface


# ---------- 1-3: the catalog ----------

def test_only_approved_valid_safe_artifacts_enter_the_catalog(tmp_path):
    approved(login_artifact())
    save_artifact(login_artifact("draft_login"), secrets=("secret_sauce",))                     # a draft
    (artifact_module.ARTIFACTS_DIR / "broken.v1.json").write_text("{not json")
    approved(replace(login_artifact("elsewhere"), surface={**login_artifact().surface, "entry_url": "https://other/"}))
    approved(replace(login_artifact("foreign"), surface={**login_artifact().surface,
                                                         "allowed_hosts": ["www.saucedemo.com", "cdn.example"]}))
    library, log = library_for()
    assert [(e.name, e.version) for e in library.entries] == [("shop_login", 1)]
    reasons = {r["path"].rsplit("/", 1)[-1]: r["reason"] for r in library.rejected}
    assert reasons["draft_login.v1.json"].startswith("not approved")
    assert reasons["broken.v1.json"].startswith("invalid")
    assert reasons["elsewhere.v1.json"].startswith("different entry url")
    assert "cdn.example" in reasons["foreign.v1.json"]
    built = events(log, "artifact_catalog_built")[0]
    assert [c["name"] for c in built["cataloged"]] == ["shop_login"] and len(built["rejected"]) == 4
    assert "secret_sauce" not in log.path.read_text() and "standard_user" not in log.path.read_text()

    library, _ = library_for(sensitive=())                 # the password is sensitive there, not here
    assert library.entries == [] and any("sensitive" in r["reason"] for r in library.rejected)


def test_file_name_mismatch_and_digest_mutation_are_rejected():
    approved(login_artifact(), version=1, path=artifact_module.ARTIFACTS_DIR / "shop_login.v9.json")
    library, _ = library_for()
    assert library.entries == [] and "holds shop_login v1, not shop_login v9" in library.rejected[0]["reason"]

    path = approved(login_artifact())
    library, log = library_for()
    [entry] = library.entries
    data = json.loads(path.read_text())
    data["provenance"]["edited"] = True                     # still valid, but not what was cataloged
    path.write_text(json.dumps(data))
    surface = FakeSurface()
    surface.navigate(ENTRY)
    surface.actions.clear()
    candidate = find_candidates(library, surface.observe(), dict(PARAMS), surface, log, 1)[0]
    execution = execute_candidate(library, candidate, dict(PARAMS), surface, Policy(allowed_hosts=HOSTS), log, 1)
    assert execution.rejected == "source artifact changed since the catalog was built" and surface.actions == []


def test_wrong_entry_state_or_missing_input_prevents_candidacy():
    approved(login_artifact())
    approved(cart_artifact())
    library, log = library_for()
    surface = FakeSurface()
    surface.navigate(ENTRY)
    on_login = find_candidates(library, surface.observe(), dict(PARAMS), surface, log, 1)
    # Every action applicable on the sign-in screen, each on its own; the entry navigation is pointless here.
    assert sorted(c.candidate_id for c in on_login) == [f"shop_cart.v1@s{i}" for i in (2, 3, 4)] + \
        [f"shop_login.v1@s{i}" for i in (2, 3, 4)]
    assert all(c.entry_condition == {"text_contains": "Swag Labs"} and c.action_count == 1 for c in on_login)
    surface = logged_in_surface()
    inventory = find_candidates(library, surface.observe(), dict(PARAMS), surface, log, 2)
    assert [c.candidate_id for c in inventory] == ["shop_cart.v1@s5"]              # login is never offered again
    without_product = {k: v for k, v in PARAMS.items() if k != "product_name"}
    assert find_candidates(library, surface.observe(), without_product, surface, log, 3) == []
    rejected = events(log, "reuse_candidate_rejected")[-1]
    assert rejected["candidate_id"] == "shop_cart.v1@s5" and rejected["reason"] == "missing inputs ['product_name']"


# ---------- 4-5: atomic candidates from graph structure ----------

def test_a_candidate_is_exactly_one_verified_action_never_a_suffix():
    approved(cart_artifact())
    library, log = library_for()
    surface = logged_in_surface()
    [candidate] = find_candidates(library, surface.observe(), dict(PARAMS), surface, log, 1)
    assert candidate.node_ids == ["s5"] and candidate.action_count == 1
    assert candidate.entry_condition == {"text_contains": "Products"} and candidate.ending_condition == {"text_contains": "Remove"}
    assert candidate.required_inputs == ["product_name"] and candidate.effects == ["reversible"]
    assert candidate.outline == ["click button 'Add to cart'"]
    execution = execute_candidate(library, candidate, dict(PARAMS), surface, Policy(allowed_hosts=HOSTS), log, 1)
    assert execution.result.status == "success"
    assert surface.actions == [("click", "Add to cart", "Sauce Labs Backpack")]     # s6 was not run along with it
    assert [n.kind for n in execution.segment.nodes] == ["action", "terminal"]
    following = find_candidates(library, surface.observe(), dict(PARAMS), surface, log, 2)
    assert "shop_cart.v1@s6" in [c.candidate_id for c in following]                  # the next action, offered next
    assert "standard_user" not in json.dumps(library_module.candidate_summary(candidate))


def test_decision_guards_choose_only_the_applicable_branch():
    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], "campaign-test")
    approved(graph)
    surface = logged_in_surface()
    for _ in range(1):
        pass
    base = {**PARAMS, "cart_route": "cart_url", "cart_url": ENTRY + "cart.html"}
    library, log = library_for(params=base)
    [by_url] = find_candidates(library, surface.observe(), base, surface, log, 1)
    assert by_url.candidate_id == "checkout_paths.v1@s5" and by_url.node_ids == ["s5"]
    surface.click(next(e for e in surface.observe().elements if e.name == "Add to cart"))   # now s5's checkpoint holds
    after_add = [c.candidate_id for c in find_candidates(library, surface.observe(), base, surface, log, 2)]
    assert after_add == ["checkout_paths.v1@s5", "checkout_paths.v1@s7"]              # the cart_url branch of d2
    library, log = library_for(params={**PARAMS, "cart_route": "cart_link"})
    by_link = [c.candidate_id for c in find_candidates(library, surface.observe(), {**PARAMS, "cart_route": "cart_link"},
                                                       surface, log, 3)]
    assert by_link == ["checkout_paths.v1@s5", "checkout_paths.v1@s6"]                # the cart_link branch of d2
    library, log = library_for(params={**PARAMS, "cart_route": "teleport"})
    undeclared = find_candidates(library, surface.observe(), {**PARAMS, "cart_route": "teleport"}, surface, log, 4)
    assert undeclared == []                                                          # the entry gate admits nothing


# ---------- 6-8: the planner's choice ----------

def test_candidates_are_computed_before_every_planner_decision():
    approved(login_artifact())
    artifact, planner, _, log = run(checkout_script())
    assert len(events(log, "reuse_candidates_found")) == len(events(log, "planner_decided")) == len(planner.offered)
    assert planner.offered[0] == planner.offered[2] == [f"shop_login.v1@s{i}" for i in (2, 3, 4)]   # the sign-in screen
    assert planner.offered[3] == [] and planner.offered[-1] == []                  # the product list, the overview
    assert events(log, "reuse_started") == [] and len(artifact.nodes) == 16       # declined every time


def test_the_planner_is_consulted_between_every_reused_action():
    approved(login_artifact())
    artifact, planner, sessions, log = run(LOGIN_REUSE + checkout_script()[4:])
    assert [s.closes for s in sessions.created] == [1]
    assert [n.action.action for n in linear_path(artifact)][:5] == ["navigate", "type", "type", "click", "click"]
    assert [n.id for n in linear_path(artifact)] == [f"s{i}" for i in range(1, 16)]
    assert sessions.created[0].actions[:4] == [("navigate", ENTRY), ("type", "Username", "standard_user"),
                                               ("type", "Password", "secret_sauce"), ("click", "Login", "")]
    records = artifact.provenance["reuses"]
    assert [(r["candidate_id"], r["executed_path"], r["imported_as"], r["planner_turn"]) for r in records] == [
        ("shop_login.v1@s2", ["s2"], ["s2"], 1), ("shop_login.v1@s3", ["s3"], ["s3"], 2),
        ("shop_login.v1@s4", ["s4"], ["s4"], 3)]
    assert records[2] == {**records[2], "mode": "automatic", "source_name": "shop_login", "source_version": 1,
                          "start_node": "s4", "imported_nodes": 1, "entry_condition": {"text_contains": "Swag Labs"},
                          "ending_checkpoint": {"text_contains": "Products"}, "reuse_run_id": log.run_id,
                          "result": "succeeded"}
    lines = [json.loads(l) for l in log.path.read_text().splitlines()]
    decisions = [i for i, e in enumerate(lines) if e["event"] == "planner_decided"]
    starts = [i for i, e in enumerate(lines) if e["event"] == "reuse_started"]
    assert len(starts) == 3 and all(any(d < start < (decisions + [len(lines)])[k + 1] for k, d in enumerate(decisions))
                                    for start in starts)
    assert all(sum(d < start for d in decisions) == n + 1 for n, start in enumerate(starts))   # one decision per action
    assert artifact.provenance["planner_decisions"] == 15                        # 3 selections + 12 ordinary decisions
    for event in ("artifact_catalog_built", "reuse_candidates_found", "reuse_candidate_selected", "reuse_started",
                  "reuse_step_executed", "reuse_succeeded"):
        assert events(log, event), event
    assert "secret_sauce" not in log.path.read_text()
    assert replays(artifact)[0].status == "success"


def test_an_invented_candidate_id_is_rejected():
    approved(login_artifact())
    artifact, _, sessions, log = run([reuse_id("shop_login.v7@s9")] + checkout_script())
    [rejected] = events(log, "reuse_candidate_rejected")
    assert rejected["candidate_id"] == "shop_login.v7@s9" and "not among the candidates" in rejected["reason"]
    assert events(log, "reuse_started") == [] and "reuses" not in artifact.provenance
    assert replays(artifact)[0].status == "success" and sessions.created[0].closes == 1


# ---------- 9-12: composition, metadata, outputs, safety ----------

def test_shared_actions_are_reused_across_goals_with_different_final_extractions():
    # An approved artifact reaches the product list and reads the product's name; the new goal wants its price.
    source = capability("shop_product_name", checkout_nodes()[:4] + [
        checkout_nodes()[11]], {"text_contains": "Products"},
        outputs={"product_name": checkout_artifact().outputs["product_name"]})
    nodes_by_id(source)["s12"].id = "s5"
    source.edges[3].target, source.edges[4].source = "s5", "s5"
    source_path = approved(source)
    new_extraction = ScriptedStep("extract", "link", "View details for {{product_name}}", output_name="title",
                                  pattern=r"(.+)")
    # The three shared actions are reused one by one; the source's own extraction (s5) is offered on the product list
    # and declined (the script generates a different extraction instead), so it is never run.
    artifact, planner, sessions, log = run([reuse("shop_product_name")] * 3
                                           + [new_extraction, ScriptedStep("done", expect="Products")])
    assert planner.offered[3] == ["shop_product_name.v1@s5"]                       # offered, declined, not run
    assert [r["candidate_id"] for r in artifact.provenance["reuses"]] == [f"shop_product_name.v1@s{i}" for i in (2, 3, 4)]
    assert [n.action.action for n in linear_path(artifact)] == ["navigate", "type", "type", "click", "extract"]
    assert linear_path(artifact)[-1].action.value == "title" and sorted(artifact.outputs) == ["title"]
    assert "product_name" not in artifact.outputs and len(events(log, "reuse_started")) == 3
    kinds = [json.loads(l)["kind"] for l in log.path.read_text().splitlines() if '"planner_decided"' in l]
    assert kinds == ["reuse_candidate"] * 3 + ["extract", "done"]                    # asked again after every action
    source_path.unlink()
    result, _ = replays(artifact)
    assert result.status == "success" and result.outputs == {"title": "Sauce Labs Backpack"}


def test_multiple_artifacts_are_reused_in_one_discovery_and_the_result_is_self_contained():
    login_path, cart_path = approved(login_artifact()), approved(cart_artifact())
    artifact, planner, sessions, log = run(LOGIN_REUSE + [reuse("shop_cart")] * 2 + checkout_script()[6:])
    assert sorted(planner.offered[0]) == [f"shop_cart.v1@s{i}" for i in (2, 3, 4)] + [f"shop_login.v1@s{i}" for i in (2, 3, 4)]
    assert planner.offered[3] == ["shop_cart.v1@s5"] and "shop_cart.v1@s6" in planner.offered[4]
    records = artifact.provenance["reuses"]
    assert [(r["source_name"], r["start_node"], r["imported_as"]) for r in records] == [
        ("shop_login", "s2", ["s2"]), ("shop_login", "s3", ["s3"]), ("shop_login", "s4", ["s4"]),
        ("shop_cart", "s5", ["s5"]), ("shop_cart", "s6", ["s6"])]
    assert [n.action.action for n in linear_path(artifact)] == ["navigate", "type", "type", "click", "click", "click",
                                                                "click", "type", "type", "type", "click", "extract",
                                                                "extract", "extract", "extract"]
    assert len(events(log, "reuse_succeeded")) == 5 and sessions.created[0].screen == "overview"
    login_path.unlink()
    cart_path.unlink()
    result, surface = replays(artifact)
    assert result.status == "success" and result.outputs["total"] == 32.39
    assert "reuse" not in json.dumps(artifact_module.to_dict(artifact)["nodes"])


def test_reused_nodes_keep_effect_and_retry_safety_exactly():
    nodes = checkout_nodes()[:4]
    nodes[3].effect, nodes[3].retry_safety = "reversible", "never_retry"
    nodes[1].effect, nodes[1].retry_safety = "reversible", "safe"
    approved(login_artifact(nodes=nodes))
    artifact, _, _, log = run(LOGIN_REUSE + checkout_script()[4:])
    imported = nodes_by_id(artifact)
    assert (imported["s4"].effect, imported["s4"].retry_safety) == ("reversible", "never_retry")
    assert (imported["s2"].effect, imported["s2"].retry_safety) == ("reversible", "safe")
    assert imported["s4"].action == nodes[3].action and imported["s4"].action is not nodes[3].action
    assert (imported["s5"].effect, imported["s5"].retry_safety) == ("reversible", "verify_before_retry")   # new
    assert [(e["effect"], e["retry_safety"]) for e in events(log, "reuse_step_executed")][2] == ("reversible", "never_retry")


def test_compatible_extraction_outputs_are_imported_and_incompatible_ones_are_not():
    approved(checkout_artifact())                          # its s12..s15 read the totals on the overview
    overview = checkout_script()[:11]                      # login .. Continue: the overview is reached
    contract = {"total": {"type": "number", "required": True, "pattern": MONEY}}
    artifact, planner, _, log = run(overview + [reuse_id("checkout_review.v1@s15"),
                                                ScriptedStep("done", expect="Checkout: Overview")],
                                    output_contract=contract)
    [record] = artifact.provenance["reuses"]
    assert record["candidate_id"] == "checkout_review.v1@s15" and record["executed_path"] == ["s15"]
    assert record["imported_nodes"] == 1 and record["imported_as"] == ["s12"]     # renumbered into this artifact
    assert artifact.outputs == {"total": {**contract["total"], "example": "32.39"}}   # refreshed from this replay
    assert linear_path(artifact)[-1].action.value == "total" and replays(artifact)[0].outputs == {"total": 32.39}

    incompatible = {"total": {"type": "string", "required": True, "pattern": r"Total: (.+)"}}   # not the source's
    artifact, _, _, log = run(overview + [reuse_id("checkout_review.v1@s15"),
                                          ScriptedStep("done", expect="Checkout: Overview"),      # refused: no total yet
                                          ScriptedStep("extract", "text", text="Total:", output_name="total"),
                                          ScriptedStep("done", expect="Checkout: Overview")],
                              output_contract=incompatible)
    reasons = [e["reason"] for e in events(log, "reuse_candidate_rejected")]
    assert any(r.startswith("no progress") for r in reasons)
    assert "reuses" not in artifact.provenance                                   # nothing adoptable: still rejected
    assert [e["outputs"] for e in events(log, "done_rejected")] == [{"total": "missing"}]   # the contract is enforced
    assert artifact.outputs["total"] == {**incompatible["total"], "example": "$ 32.39",
                                         "description": artifact.outputs["total"]["description"]}


def test_no_irreversible_or_unknown_action_is_ever_offered():
    approved(capability("finishing", checkout_nodes() + [finish_node()], {"text_contains": "Thank you"},
                        params=dict(PARAMS), outputs=checkout_artifact().outputs))
    unknown = login_artifact("unknown_login")
    nodes_by_id(unknown)["s2"].effect, nodes_by_id(unknown)["s2"].retry_safety = "unknown", "never_retry"
    approved(unknown)
    library, log = library_for()
    overview = FakeSurface()
    replay(checkout_artifact(), dict(PARAMS), overview, Policy(allowed_hosts=HOSTS),
           Escalator(NoOperator(), SessionControl(), RunLog("replay")), RunLog("replay"))
    assert overview.screen == "overview"
    on_overview = find_candidates(library, overview.observe(), dict(PARAMS), overview, log, 1)
    assert {c.candidate_id for c in on_overview} >= {f"finishing.v1@s{i}" for i in (12, 13, 14, 15)}
    login = FakeSurface()
    login.navigate(ENTRY)
    offered = find_candidates(library, login.observe(), dict(PARAMS), login, log, 2)
    assert "unknown_login.v1@s2" not in [c.candidate_id for c in offered]          # the unknown action itself
    for candidate in offered + on_overview:
        assert "s16" not in candidate.node_ids and not ({"irreversible", "unknown"} & set(candidate.effects))
        assert all("Finish" not in line for line in candidate.outline) and candidate.action_count == 1


# ---------- 13: loop protection ----------

def test_no_progress_and_repeated_screens_are_bounded():
    # Clicking the product's details link does nothing in the fake shop: a verified action that changes nothing.
    stay = capability("shop_stay", checkout_nodes()[:4] + [
        checkout_nodes()[4].__class__(id="s5", kind="action", action=GraphAction(
            action="click", target=text_ladder("Sauce Labs Backpack"), checkpoint={"text_contains": "Products"}),
            effect="reversible", retry_safety="verify_before_retry")], {"text_contains": "Products"})
    approved(stay)
    artifact, planner, _, log = run(checkout_script()[:4] + [reuse("shop_stay"), reuse("shop_stay", skip_if_absent=True)]
                                    + checkout_script()[4:])
    assert events(log, "reuse_succeeded") == [] and len(events(log, "reuse_started")) == 1
    rejected = [e["reason"] for e in events(log, "reuse_candidate_rejected")]
    assert rejected[0].startswith("no progress") and rejected[1] == "already attempted on this screen"
    assert "reuses" not in artifact.provenance and replays(artifact)[0].status == "success"


def test_each_reused_action_consumes_one_unit_of_the_budget():
    approved(login_artifact())
    approved(cart_artifact())
    artifact, planner, _, log = run([reuse("shop_login"), reuse("shop_login", skip_if_absent=True)]
                                    + checkout_script()[2:], max_auto_reuses=1)
    assert [r["candidate_id"] for r in artifact.provenance["reuses"]] == ["shop_login.v1@s2"] and planner.offered[1] == []
    [exhausted] = events(log, "reuse_exhausted")
    assert exhausted == {**exhausted, "reuses": 1, "max_auto_reuses": 1}
    assert replays(artifact)[0].status == "success"


# ---------- 14-16: failure behaviour ----------

def test_failure_before_acting_falls_back_on_the_same_session():
    nodes = checkout_nodes()[:4]
    nodes[1].action.target.strategies[:] = [{"kind": "role", "role": "textbox", "name": "Usernam3"}]
    approved(login_artifact(nodes=nodes))
    artifact, _, sessions, log = run([reuse("shop_login")] + checkout_script())
    [failed] = events(log, "reuse_failed")
    assert failed == {**failed, "classification": "before_acting", "outcome_code": "target_not_found",
                      "performed_attempts": 0, "executed_path": []}
    assert [s.closes for s in sessions.created] == [1] and "reuses" not in artifact.provenance
    assert replays(artifact)[0].status == "success"


def test_a_failing_action_does_not_lose_the_actions_reused_before_it():
    nodes = checkout_nodes()[:6]
    nodes[5].action.target.strategies[:] = [{"kind": "role", "role": "link", "name": "kart"}]
    approved(cart_artifact(nodes=nodes))
    artifact, _, sessions, log = run(checkout_script()[:4] + [reuse("shop_cart"), reuse("shop_cart")] + checkout_script()[5:])
    [failed] = events(log, "reuse_failed")
    assert failed == {**failed, "candidate_id": "shop_cart.v1@s6", "classification": "before_acting", "executed_path": []}
    [record] = artifact.provenance["reuses"]
    assert record["candidate_id"] == "shop_cart.v1@s5" and record["imported_as"] == ["s5"]
    assert [s.closes for s in sessions.created] == [1]
    assert nodes_by_id(artifact)["s5"].action.checkpoint == {"text_contains": "Remove"}
    assert replays(artifact)[0].status == "success"


def uncertain_login(name="shop_login", at=3):
    nodes = checkout_nodes()[:4]
    nodes[at].action.checkpoint = {"text_contains": "Welcome back"}          # the action happens, the proof never comes
    return login_artifact(name, nodes=nodes)


def test_uncertain_failure_without_an_operator_restarts_cleanly_with_the_source_excluded():
    approved(uncertain_login())
    artifact, planner, sessions, log = run(LOGIN_REUSE + checkout_script())
    assert [s.closes for s in sessions.created] == [1, 1] and sessions.created[0].logged_in
    [failed] = events(log, "reuse_failed")
    assert failed == {**failed, "candidate_id": "shop_login.v1@s4", "classification": "uncertain",
                      "outcome_code": "checkpoint_not_met", "step_id": "s4", "executed_path": [], "performed_attempts": 1}
    assert '"action_repeated"' not in log.path.read_text()
    [restart] = events(log, "discovery_restarted")
    assert restart["attempt"] == 1 and restart["excluded"] == ["shop_login.v1"]
    assert events(log, "reuse_candidate_rejected")[-1]["reason"].startswith("excluded after an uncertain failure")
    assert planner.offered[3] == [] and "reuses" not in artifact.provenance          # the whole source is out
    assert sessions.created[1].screen == "overview" and replays(artifact)[0].status == "success"


def test_uncertain_failure_with_an_operator_follows_the_disposition():
    approved(uncertain_login())
    vouched = RecordingOperator(disposition="resume")
    artifact, _, sessions, log = run(LOGIN_REUSE + checkout_script()[4:], operator=vouched)
    assert [s.closes for s in sessions.created] == [1] and vouched.requests[0].kind == "stuck"
    assert "unverified" in vouched.requests[0].reason and events(log, "reuse_resumed_by_operator")
    assert [r["candidate_id"] for r in artifact.provenance["reuses"]] == ["shop_login.v1@s2", "shop_login.v1@s3"]
    assert sessions.created[0].actions.count(("click", "Login", "")) == 1

    aborted = RecordingOperator(disposition="abort")
    with pytest.raises(DiscoveryFailed, match="aborted discovery after an uncertain reuse"):
        run(LOGIN_REUSE + checkout_script()[4:], operator=aborted)

    restarting = RecordingOperator(disposition="restart")
    artifact, _, sessions, _ = run(LOGIN_REUSE + checkout_script(), operator=restarting)
    assert [s.closes for s in sessions.created] == [1, 1] and replays(artifact)[0].status == "success"


def test_restarts_are_bounded():
    approved(uncertain_login())
    approved(uncertain_login("shop_login_2"))
    approved(uncertain_login("shop_login_3"))
    script = [reuse("shop_login")] * 3 + [reuse("shop_login_2")] * 3 + [reuse("shop_login_3")] * 3 + checkout_script()
    sessions = Sessions()
    with pytest.raises(DiscoveryFailed, match="restarted 2 times"):
        run(script, sessions=sessions, max_auto_reuses=20)                  # nine actions: the budget is not the limit
    assert [s.closes for s in sessions.created] == [1, 1, 1]


def test_a_business_outcome_from_a_reused_action_is_never_a_success():
    approved(login_artifact())
    locked = {**PARAMS, "username": "locked_out_user"}
    declared = [{"code": "user_locked_out", "kind": "business", "source": "reviewer",
                 "detect": {"text_contains": "locked out"}}]
    artifact, _, sessions, log = run(LOGIN_REUSE + [ScriptedStep("done", expect="locked out")], params=locked,
                                     extra_outcomes=declared)
    [failed] = events(log, "reuse_failed")
    assert failed == {**failed, "candidate_id": "shop_login.v1@s4", "classification": "business_outcome",
                      "outcome_code": "user_locked_out", "declared": True}
    assert len(events(log, "reuse_succeeded")) == 2 and [s.closes for s in sessions.created] == [1]
    assert [r["candidate_id"] for r in artifact.provenance["reuses"]] == ["shop_login.v1@s2", "shop_login.v1@s3"]

    sessions = Sessions()
    _, _, sessions, log = run(LOGIN_REUSE + [ScriptedStep("type", "textbox", "Username", value="{{username}}"),
                                             ScriptedStep("type", "textbox", "Password", value="{{password}}"),
                                             ScriptedStep("click", "button", "Login", expect="locked out"),
                                             ScriptedStep("done", expect="locked out")],
                              sessions=sessions, params=locked)
    assert events(log, "reuse_failed")[0]["declared"] is False and [s.closes for s in sessions.created] == [1, 1]


# ---------- 18-20: campaigns, the switch, the forced prefix ----------

def test_campaign_scenarios_do_not_share_reuse_state():
    from tests.test_campaign import SCENARIOS, TOTAL_CONTRACT, campaign_script, spec_data
    approved(login_artifact())
    spec = spec_from_dict(spec_data(SCENARIOS[:2], outputs=TOTAL_CONTRACT))

    def planner_factory(scenario):
        return ScriptedPlanner(LOGIN_REUSE + campaign_script(scenario.params["cart_route"],
                                                             scenario.params["zip_source"])[3:])

    result = run_campaign(spec, lambda secrets: Session(), planner_factory, NoOperator(), HOSTS, spec_path="s.json")
    for record in result.scenarios:
        text = open(record["evidence"] + "/run.jsonl").read()
        assert text.count('"artifact_catalog_built"') == 1 and text.count('"reuse_succeeded"') == 3
        assert "secret_sauce" not in text
    assert [s["node_path"][:5] for s in result.graph.provenance["scenarios"]] == [["d1", "s1", "s2", "s3", "s4"]] * 2


def test_no_auto_reuse_restores_ordinary_discovery():
    approved(login_artifact())
    artifact, planner, _, log = run([reuse("shop_login", skip_if_absent=True)] + checkout_script(), auto_reuse=False)
    assert events(log, "artifact_catalog_built") == [] and events(log, "reuse_candidates_found") == []
    assert planner.offered == [[]] * len(planner.offered) and len(artifact.nodes) == 16
    parser = build_parser()
    args = parser.parse_args(["discover", "--goal", "g", "--url", "u", "--name", "n"])
    assert args.no_auto_reuse is False and args.max_auto_reuses == 5 and args.library_dir is None
    args = parser.parse_args(["discover-campaign", "--spec", "s.json", "--no-auto-reuse", "--max-auto-reuses", "2",
                              "--library-dir", "lib"])
    assert args.no_auto_reuse is True and args.max_auto_reuses == 2 and args.library_dir == "lib"


def test_a_forced_prefix_still_works_and_automatic_reuse_continues_after_it():
    approved(login_artifact())
    approved(cart_artifact())
    plan = resolve_reuse("shop_login", None)
    artifact, planner, sessions, log = run([reuse("shop_cart")] * 2 + checkout_script()[6:], prefix=plan)
    assert planner.observations[0].url == ENTRY + "inventory.html" and planner.offered[0] == ["shop_cart.v1@s5"]
    modes = [(r["mode"], r["source_name"]) for r in artifact.provenance["reuses"]]
    assert modes == [("forced_prefix", "shop_login"), ("automatic", "shop_cart"), ("automatic", "shop_cart")]
    assert [s.closes for s in sessions.created] == [1] and replays(artifact)[0].status == "success"
    artifact, planner, _, log = run([reuse("shop_cart", skip_if_absent=True)] + checkout_script()[4:], prefix=plan,
                                    auto_reuse=False)
    assert [r["mode"] for r in artifact.provenance["reuses"]] == ["forced_prefix"] and planner.offered[0] == []


# ---------- the automatic budget spans every session of one discovery ----------

def capture_library(monkeypatch) -> list:
    """The Library object discover_with_reuse builds, so its bookkeeping can be inspected afterwards."""
    captured = []
    real = reuse_module.build_library

    def keep(*args, **kwargs):
        captured.append(real(*args, **kwargs))
        return captured[-1]

    monkeypatch.setattr(reuse_module, "build_library", keep)
    return captured


def uncertain_checkout(name="cart_then_uncertain"):
    """Add to cart and open the cart, then a Checkout click whose proof never comes: uncertain on the cart screen."""
    nodes = checkout_nodes()[:7]
    nodes[6].action.checkpoint = {"text_contains": "Welcome to checkout, friend"}
    return capability(name, nodes, {"text_contains": "Checkout: Your Information"})


def test_the_automatic_budget_survives_a_clean_restart_and_attempts_do_not(monkeypatch):
    approved(cart_artifact())
    approved(uncertain_checkout())
    captured = capture_library(monkeypatch)
    login = checkout_script()[:4]
    script = (login + [reuse("shop_cart")] * 2 + [reuse_id("cart_then_uncertain.v1@s7")]   # session 1: three, restart
              + login + [reuse_id("shop_cart.v1@s5"), reuse("shop_cart", skip_if_absent=True)]   # session 2: s5 again,
              + checkout_script()[5:])                                                     #   then exhausted: ordinary
    artifact, planner, sessions, log = run(script, max_auto_reuses=4)
    [library] = captured
    assert [s.closes for s in sessions.created] == [1, 1]
    started = [e["candidate_id"] for e in events(log, "reuse_started") if e.get("candidate_id")]
    assert started == ["shop_cart.v1@s5", "shop_cart.v1@s6", "cart_then_uncertain.v1@s7", "shop_cart.v1@s5"]
    assert library.count == 4 == library.max_reuses                                 # 3 before + 1 after the restart
    assert library.excluded == {next(e.digest for e in library.entries if e.name == "cart_then_uncertain")}
    assert events(log, "discovery_restarted")[0]["excluded"] == ["cart_then_uncertain.v1"]
    # The cart action ran on the same screen key in both sessions: attempts were cleared, exclusions kept.
    rejected = [(e["candidate_id"], e["reason"]) for e in events(log, "reuse_candidate_rejected")]
    assert any(c.startswith("cart_then_uncertain.v1@") and r.startswith("excluded after an uncertain") for c, r in rejected)
    assert not any(r.startswith("already attempted") and c == "shop_cart.v1@s5" for c, r in rejected)
    [exhausted] = events(log, "reuse_exhausted")
    assert exhausted == {**exhausted, "reuses": 4, "max_auto_reuses": 4} and library.exhausted_logged
    assert [r["candidate_id"] for r in artifact.provenance["reuses"]] == ["shop_cart.v1@s5"]   # the second session's
    assert replays(artifact)[0].status == "success"


def test_an_exhausted_budget_offers_nothing_after_the_restart_and_the_planner_finishes(monkeypatch):
    approved(uncertain_login(at=1))                        # typing the username "must" show text that never appears
    approved(cart_artifact())
    captured = capture_library(monkeypatch)
    artifact, planner, sessions, log = run([reuse("shop_login")] + checkout_script(), max_auto_reuses=1)
    [library] = captured
    assert [s.closes for s in sessions.created] == [1, 1] and library.count == 1 == library.max_reuses
    started = [e for e in events(log, "reuse_started") if e.get("candidate_id")]
    assert [e["candidate_id"] for e in started] == ["shop_login.v1@s2"]            # the budget was spent before the restart
    session_two = planner.offered[1:]
    assert session_two and all(offer == [] for offer in session_two)              # nothing offered after the restart
    assert len(events(log, "reuse_exhausted")) == 1 and library.attempted == set()
    assert "reuses" not in artifact.provenance and sessions.created[1].screen == "overview"
    assert replays(artifact)[0].status == "success"


def test_a_forced_prefix_does_not_consume_the_automatic_budget(monkeypatch):
    approved(login_artifact())
    approved(cart_artifact())
    captured = capture_library(monkeypatch)
    plan = resolve_reuse("shop_login", None)
    artifact, planner, sessions, log = run([reuse("shop_cart"), reuse("shop_cart", skip_if_absent=True)]
                                           + checkout_script()[5:], prefix=plan, max_auto_reuses=1)
    [library] = captured
    assert library.count == 1 and [r["mode"] for r in artifact.provenance["reuses"]] == ["forced_prefix", "automatic"]
    automatic = [e for e in events(log, "reuse_started") if e.get("candidate_id")]
    assert len(automatic) == 1 and len(events(log, "reuse_started")) == 2           # the prefix's own start event
    assert len(events(log, "reuse_exhausted")) == 1 and replays(artifact)[0].status == "success"


# ---------- read-only actions: successful extraction is progress ----------

OVERVIEW = checkout_script()[:11]                                  # login .. Continue: the overview is reached
DONE = ScriptedStep("done", expect="Checkout: Overview")
TOTALS_REUSE = [reuse_id(f"checkout_review.v1@s{i}") for i in (12, 13, 14, 15)]


class HistorySpy(SpyPlanner):
    def __init__(self, script):
        super().__init__(script)
        self.histories: list[list[str]] = []

    def decide(self, goal, params, observation, history, candidates=()):
        self.histories.append([f"{a.kind}:{a.result}" for a in history])
        return super().decide(goal, params, observation, history, candidates)


def secrets_absent(log, artifact) -> bool:
    return "secret_sauce" not in log.path.read_text() + json.dumps(artifact_module.to_dict(artifact))


def read_only_source(name, nodes, outputs):
    return build_linear(name=name, goal="read the totals", surface_meta=dict(checkout_artifact().surface), params=PARAMS,
                        sensitive={"password"}, nodes=checkout_nodes()[:11] + nodes, outputs=outputs,
                        outcomes=[dict(o) for o in checkout_artifact().outcomes],
                        success={"text_contains": "Checkout: Overview"}, run_id="test-run")


def test_extraction_actions_on_an_unchanged_screen_are_progress_and_their_outputs_are_adopted():
    source_path = approved(checkout_artifact())                      # s12..s15 read the totals; no contract declared
    planner = HistorySpy(OVERVIEW + TOTALS_REUSE + [DONE])
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover_with_reuse(Sessions(), None, goal="read the totals", name="checkout_review", params=dict(PARAMS),
                                   planner=planner, policy=Policy(allowed_hosts=HOSTS),
                                   escalator=Escalator(NoOperator(), SessionControl(), log), log=log, entry_url=ENTRY,
                                   sensitive={"password"}, max_steps=25)
    records = artifact.provenance["reuses"]
    assert [(r["executed_path"], r["imported_as"], r["outputs"]) for r in records] == [
        (["s12"], ["s12"], ["product_name"]), (["s13"], ["s13"], ["subtotal"]), (["s14"], ["s14"], ["tax"]),
        (["s15"], ["s15"], ["total"])]
    assert len(events(log, "reuse_succeeded")) == 4
    assert {e["reason"] for e in events(log, "reuse_candidate_rejected")} <= {"already attempted on this screen"}
    assert [n.action.action for n in linear_path(artifact)[-4:]] == ["extract"] * 4 and len(artifact.nodes) == 16
    assert artifact.outputs["total"] == {**checkout_artifact().outputs["total"], "example": "32.39"}
    assert artifact.outputs["product_name"]["example"] == "{{product_name}}"       # parameterized like any sample
    assert artifact.outputs["subtotal"]["type"] == "number" and artifact.outputs["tax"]["required"] is True
    kinds = [json.loads(l)["kind"] for l in log.path.read_text().splitlines() if '"planner_decided"' in l]
    assert kinds[-5:] == ["reuse_candidate"] * 4 + ["done"] and "extract" not in kinds   # nothing was read twice
    assert planner.histories[-1][-1] == ("reuse_candidate:reused segment checkout_review.v1@s15 completed: 1 verified "
                                         "action(s) recorded; outputs recorded: ['total']")
    assert secrets_absent(log, artifact)
    source_path.unlink()
    result, _ = replays(artifact)
    assert result.status == "success" and result.outputs == {"product_name": "Sauce Labs Backpack", "subtotal": 29.99,
                                                              "tax": 2.4, "total": 32.39}


def test_the_real_shape_two_scalars_and_a_list_on_an_unchanged_screen():
    money_list = {"type": "list", "required": True, "items": {"type": "number", "pattern": MONEY},
                  "description": "the three money lines"}
    source = read_only_source("checkout_totals", [
        copy_node(checkout_nodes()[12], "s12"), copy_node(checkout_nodes()[13], "s13"),            # subtotal, tax
        linear_node("s14", GraphAction(action="extract_many", target=None, value="totals",
                                       targets=[text_ladder("Item total:"), text_ladder("Tax:"), text_ladder("Total:")])),
    ], {"subtotal": checkout_artifact().outputs["subtotal"], "tax": checkout_artifact().outputs["tax"],
        "totals": money_list})
    approved(source)
    artifact, _, _, log = run(OVERVIEW + [reuse_id(f"checkout_totals.v1@s{i}") for i in (12, 13, 14)] + [DONE])
    records = artifact.provenance["reuses"]
    assert [(r["outputs"], r["imported_as"]) for r in records] == [(["subtotal"], ["s12"]), (["tax"], ["s13"]),
                                                                    (["totals"], ["s14"])]
    assert artifact.outputs["totals"] == {**money_list, "example": ["29.99", "2.4", "32.39"]}
    assert linear_path(artifact)[-1].action.action == "extract_many"
    assert len(events(log, "reuse_succeeded")) == 3 and secrets_absent(log, artifact)
    assert replays(artifact)[0].outputs == {"subtotal": 29.99, "tax": 2.4, "totals": [29.99, 2.4, 32.39]}


def test_an_absent_optional_output_is_never_imported():
    source = read_only_source("checkout_fees", [
        linear_node("s12", GraphAction(action="extract", target=text_ladder("Shipping:"), value="shipping")),   # absent
        copy_node(checkout_nodes()[14], "s13"),                                                                # total
        linear_node("s14", GraphAction(action="extract_many", target=None, value="fees",
                                       targets=[text_ladder("Tax:"), text_ladder("Handling:")])),               # absent
    ], {"shipping": {"type": "number", "required": False, "pattern": MONEY},
        "total": checkout_artifact().outputs["total"],
        "fees": {"type": "list", "required": False, "items": {"type": "number", "pattern": MONEY}}})
    approved(source)
    artifact, _, _, log = run(OVERVIEW + [reuse_id(f"checkout_fees.v1@s{i}") for i in (12, 13, 14)] + [DONE])
    [record] = artifact.provenance["reuses"]
    assert record["candidate_id"] == "checkout_fees.v1@s13" and record["outputs"] == ["total"]
    assert record["imported_as"] == ["s12"]                                          # the only produced output
    reasons = [e["reason"] for e in events(log, "reuse_candidate_rejected")]
    assert sum(r.startswith("no progress") for r in reasons) == 2                   # the absent ones changed nothing
    assert sorted(artifact.outputs) == ["total"] and linear_path(artifact)[-1].action.value == "total"
    assert replays(artifact)[0].outputs == {"total": 32.39}


def test_an_explicit_incompatible_contract_is_not_adopted_and_a_compatible_one_still_imports():
    approved(checkout_artifact())
    compatible = {"total": dict(checkout_artifact().outputs["total"])}
    artifact, _, _, _ = run(OVERVIEW + [reuse_id("checkout_review.v1@s15"), DONE], output_contract=compatible)
    assert artifact.provenance["reuses"][0]["outputs"] == ["total"] and sorted(artifact.outputs) == ["total"]
    incompatible = {"total": {**compatible["total"], "type": "string"}}
    artifact, _, _, log = run(OVERVIEW + [reuse_id("checkout_review.v1@s15"), DONE,
                                          ScriptedStep("extract", "text", text="Total:", output_name="total"), DONE],
                              output_contract=incompatible)
    assert "reuses" not in artifact.provenance and sorted(artifact.outputs) == ["total"]
    assert any(e["reason"].startswith("no progress") for e in events(log, "reuse_candidate_rejected"))
    assert [e["outputs"] for e in events(log, "done_rejected")] == [{"total": "missing"}]


# ---------- an atomic reuse verifies its own action, never the whole flow ----------

def test_an_intermediate_action_reuses_without_the_sources_final_success_condition():
    """The live shape: a mid-flow type node with no checkpoint, in an artifact whose success needs the whole flow."""
    source = capability("shop_login_flow", checkout_nodes()[:4], {"text_contains": "Products"},
                        params={"username": "standard_user", "password": "secret_sauce"})
    approved(source)
    artifact, planner, sessions, log = run([reuse_id("shop_login_flow.v1@s2")] + checkout_script()[1:])
    [record] = artifact.provenance["reuses"]
    assert record["candidate_id"] == "shop_login_flow.v1@s2" and record["result"] == "succeeded"
    # the source's global success ("Products") could not hold after typing a username; the segment never asked for it
    started = [e for e in events(log, "replay_started") if e["capability"] == "shop_login_flow"]
    assert started and all(e["nodes"] == 2 for e in started)
    assert events(log, "reuse_candidate_rejected") == [] or all(
        "success_condition_not_met" not in e["reason"] for e in events(log, "reuse_candidate_rejected"))
    assert [n.action.action for n in linear_path(artifact)][:3] == ["navigate", "type", "type"]


def test_the_planner_is_consulted_immediately_after_the_single_reused_action():
    source = capability("shop_login_flow", checkout_nodes()[:4], {"text_contains": "Products"},
                        params={"username": "standard_user", "password": "secret_sauce"})
    approved(source)
    artifact, planner, sessions, log = run([reuse_id("shop_login_flow.v1@s2")] + checkout_script()[1:])
    # exactly one action came from the segment, and the next decision was the planner's own
    assert [e["actions"] for e in events(log, "reuse_started")] == [1]
    assert len(events(log, "reuse_step_executed")) == 1
    assert len(planner.offered) > 1                          # a fresh decision followed the single action


def test_the_artifact_replays_after_the_source_is_deleted():
    import os

    source = capability("shop_login_flow", checkout_nodes()[:4], {"text_contains": "Products"},
                        params={"username": "standard_user", "password": "secret_sauce"})
    path = approved(source)
    artifact, _, _, _ = run([reuse_id("shop_login_flow.v1@s2")] + checkout_script()[1:])
    os.remove(path)
    result = replays(artifact)[0]
    assert result.status == "success" and result.outputs["total"] == 32.39


def test_a_reused_actions_own_failed_checkpoint_still_rejects_the_reuse():
    # the login click's own checkpoint cannot hold when the password was never typed
    from src.cua.library import build_segment, candidate_from

    source = login_artifact()
    path = approved(source)
    entry = CatalogEntry(path=str(path), digest=sha256_of(path), artifact=replace(source, status="approved", version=1))
    login = next(node for node in source.nodes if node.id == "s4")
    candidate = candidate_from(entry, login, {"text_contains": "Swag Labs"})
    segment = build_segment(entry, candidate)
    assert segment.success == {"text_contains": "Products"}          # the action's own checkpoint, still enforced
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay_engine(segment, dict(PARAMS), FakeSurface(), Policy(allowed_hosts=HOSTS),
                           Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "failure"                                 # no credentials were typed, so it cannot hold


def test_unrelated_source_outputs_are_not_required_by_a_reused_action():
    source = capability("shop_totals", checkout_nodes()[:15], {"text_contains": "Checkout: Overview"},
                        params=dict(PARAMS), outputs=dict(checkout_artifact().outputs))
    approved(source)
    artifact, planner, sessions, log = run([reuse_id("shop_totals.v1@s2")] + checkout_script()[1:])
    [record] = artifact.provenance["reuses"]
    assert record["result"] == "succeeded" and record["outputs"] == []
    # the source declares four outputs; the one-action segment demanded none of them
    assert all(e["problems"] == {} for e in events(log, "outputs_validated"))


def test_a_reused_extraction_carries_only_its_own_output():
    approved(checkout_artifact())
    overview = checkout_script()[:11]
    artifact, planner, sessions, log = run(overview + [reuse_id("checkout_review.v1@s15"),
                                                       ScriptedStep("done", expect="Checkout: Overview")])
    [record] = artifact.provenance["reuses"]
    assert record["outputs"] == ["total"] and sorted(artifact.outputs) == ["total"]


def test_full_replay_of_the_source_still_requires_its_global_success_condition():
    source = checkout_artifact()
    broken = replace(source, success={"text_contains": "A screen this flow never reaches"})
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay_engine(broken, dict(PARAMS), FakeSurface(), Policy(allowed_hosts=HOSTS),
                           Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "failure" and result.outcome_code == "success_condition_not_met"


def test_segment_success_uses_the_actions_own_checkpoint_and_never_the_sources():
    from src.cua.library import build_segment, candidate_from, segment_success

    source = capability("shop_login_flow", checkout_nodes()[:4], {"text_contains": "Products"},
                        params={"username": "standard_user", "password": "secret_sauce"})
    path = approved(source)
    entry = CatalogEntry(path=str(path), digest=sha256_of(path), artifact=replace(source, status="approved"))
    nodes = {node.id: node for node in source.nodes if node.kind == "action"}
    checkpointed = candidate_from(entry, nodes["s4"], {"text_contains": "Swag Labs"})
    assert segment_success(checkpointed, [nodes["s4"]]) == {"text_contains": "Products"}      # its own checkpoint
    assert build_segment(entry, checkpointed).success == {"text_contains": "Products"}
    bare = candidate_from(entry, nodes["s2"], {"text_contains": "Swag Labs"})
    assert segment_success(bare, [nodes["s2"]]) == {"url_contains": ""}                        # asserts nothing
    assert build_segment(entry, bare).success == {"url_contains": ""}
    assert source.success == {"text_contains": "Products"}                                     # the source is untouched
