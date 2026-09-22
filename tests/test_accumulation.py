"""Output accumulation across pages and the back action: discovered once, replayed with no model."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import discover
from src.cua.artifact import ArtifactError, from_dict, linear_path, load_artifact, save_artifact, to_dict, validate
from src.cua.escalation import Escalator, NoOperator, SessionControl
from src.cua.evidence import RunLog
from src.cua.merge import merge_traces
from src.cua.models import Scenario, ScenarioTrace
from src.cua.policy import Policy, redact
from src.cua.replay import ReplayContext, replay, store_items
from tests.catalog_surface import CATALOG, CatalogSurface
from tests.context import Observation
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

HOSTS = ["catalog.test"]
SCORE = r"Score: (\d+)"


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


def detail_round(item: str, *reads: ScriptedStep) -> list[ScriptedStep]:
    """Open one result, read from it, come back through the browser history."""
    return [ScriptedStep("click", "link", item, expect="Details"), *reads, ScriptedStep("back", expect="Results")]


def score_read(mode="append", name="scores", optional=False) -> ScriptedStep:
    return ScriptedStep("extract", "text", text="Score:", output_name=name, pattern=SCORE, output_mode=mode,
                        optional=optional)


def run_discovery(script, surface=None, name="catalog_scores", output_contract=None):
    log = RunLog("discovery")
    escalator = Escalator(NoOperator(), SessionControl(), log)
    artifact = discover(goal="read the score of the first three results", name=name, params={},
                        surface=surface or CatalogSurface(), planner=ScriptedPlanner(script),
                        policy=Policy(allowed_hosts=HOSTS), escalator=escalator, log=log, entry_url=CATALOG,
                        max_steps=30, output_contract=output_contract)
    return artifact, log


def run_replay(artifact, surface=None):
    log = RunLog("replay")
    surface = surface or CatalogSurface()
    result = replay(artifact, {}, surface, Policy(allowed_hosts=HOSTS), Escalator(NoOperator(), SessionControl(), log),
                    log)
    return result, log, surface


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


THREE_SCORES = [*detail_round("Item A", score_read()), *detail_round("Item B", score_read()),
                *detail_round("Item C", score_read()), ScriptedStep("done", expect="Results")]


# ---------- discovery ----------

def test_three_detail_pages_build_one_three_item_output_in_order():
    artifact, log = run_discovery(THREE_SCORES)
    validate(artifact)
    path = linear_path(artifact)
    assert [n.action.action for n in path] == ["navigate"] + ["click", "extract", "back"] * 3
    assert [n.action.mode for n in path] == [None] + [None, "append", None] * 3
    assert artifact.outputs == {"scores": {"type": "list", "required": True,
                                           "items": {"type": "integer", "pattern": SCORE},
                                           "description": "Values read one at a time from text controls",
                                           "example": ["71", "58", "90"]}}
    assert [e["value"] for e in events(log, "recorded_output")] == ["71", "58", "90"]
    # back is a history move: effect none, never repeated blindly unless its checkpoint can be verified first
    backs = [n for n in path if n.action.action == "back"]
    assert all(n.effect == "none" and n.retry_safety == "verify_before_retry" for n in backs)
    assert all(n.action.target is None and n.action.value is None and n.action.checkpoint == {"text_contains": "Results"}
               for n in backs)


def test_the_append_artifact_survives_serialization_and_the_file_boundary():
    artifact, _ = run_discovery(THREE_SCORES)
    data = json.loads(json.dumps(to_dict(artifact)))
    assert [n["action"].get("mode") for n in data["nodes"] if n["action"] and n["action"]["action"] == "extract"] == [
        "append"] * 3
    assert from_dict(data) == artifact
    saved = save_artifact(artifact)
    assert load_artifact(saved) == artifact


def test_a_second_set_into_a_recorded_output_is_refused_and_explained():
    script = [*detail_round("Item A", score_read(mode="set")),
              *detail_round("Item B", score_read(mode="set"),                   # refused: would overwrite
                            score_read(mode="set", name="second_score")),      # a new name is fine
              *detail_round("Item C", score_read(mode="append"),               # refused: scores is a scalar
                            score_read(mode="append", name="more_scores")),    # a fresh list is fine
              ScriptedStep("done", expect="Results")]
    artifact, log = run_discovery(script)
    failures = events(log, "action_failed")
    assert [f["performed"] for f in failures] == ["no", "no"]
    assert "output 'scores' is already recorded and a set would overwrite it" in failures[0]["reason"]
    assert "use output_mode 'append'" in failures[0]["reason"]
    assert "already recorded as a single integer value" in failures[1]["reason"]
    assert artifact.outputs["scores"]["example"] == "71" and artifact.outputs["second_score"]["example"] == "58"
    assert artifact.outputs["more_scores"]["example"] == ["90"]
    assert [n.action.value for n in linear_path(artifact) if n.action.action == "extract"] == [
        "scores", "second_score", "more_scores"]


def test_extract_many_appends_every_item_in_order_across_pages():
    facts = ScriptedStep("extract_many", "text", texts=["Score:", "Tag:"], output_name="facts", pattern=r": (.+)$",
                         output_mode="append")
    artifact, _ = run_discovery([*detail_round("Item A", facts), *detail_round("Item B", facts),
                                 ScriptedStep("done", expect="Results")], name="catalog_facts")
    validate(artifact)
    assert artifact.outputs["facts"]["example"] == ["71", "alpha", "58", "beta"]
    assert artifact.outputs["facts"]["items"] == {"type": "string", "pattern": r": (.+)$"}
    result, _, _ = run_replay(artifact)
    assert result.status == "success" and result.outputs == {"facts": ["71", "alpha", "58", "beta"]}


def test_a_declared_list_contract_is_kept_while_appending():
    contract = {"scores": {"type": "list", "required": True, "items": {"type": "integer", "pattern": SCORE}}}
    artifact, _ = run_discovery(THREE_SCORES, output_contract=contract)
    assert artifact.outputs["scores"]["items"] == {"type": "integer", "pattern": SCORE}
    assert artifact.outputs["scores"]["example"] == ["71", "58", "90"]


# ---------- replay ----------

def test_replay_rebuilds_the_ordered_list_across_pages_without_a_model():
    artifact, _ = run_discovery(THREE_SCORES)
    result, log, surface = run_replay(artifact)
    assert result.status == "success" and result.outputs == {"scores": [71, 58, 90]}
    assert [e["id"] for e in result.executed_path] == [f"s{i}" for i in range(1, 11)]
    assert surface.actions.count(("back",)) == 3 and surface.history == []
    extracted = events(log, "output_extracted")
    assert [(e["mode"], e["total"]) for e in extracted] == [("append", 1), ("append", 2), ("append", 3)]
    # a different catalog gives different values: the artifact reads, it does not remember
    changed = CatalogSurface({"Item A": {"score": "5"}, "Item B": {"score": "6"}, "Item C": {"score": "7"}})
    assert run_replay(artifact, changed)[0].outputs == {"scores": [5, 6, 7]}


def test_an_append_node_adds_its_items_exactly_once_however_often_it_is_attempted():
    artifact, _ = run_discovery(THREE_SCORES)
    log = RunLog("replay")
    ctx = ReplayContext(artifact, {}, CatalogSurface(), Policy(allowed_hosts=HOSTS),
                        Escalator(NoOperator(), SessionControl(), log), log)
    node = next(n for n in artifact.nodes if n.action.action == "extract")
    store_items(ctx, node, [71])
    store_items(ctx, node, [71])           # a retried attempt of the same node
    assert ctx.outputs == {"scores": [71]} and len(events(log, "output_already_appended")) == 1
    ctx.appended.clear(); ctx.outputs.clear()   # what a flow restart does: the list is rebuilt from nothing
    store_items(ctx, node, [71])
    assert ctx.outputs == {"scores": [71]}


def test_an_optional_append_that_cannot_read_a_page_adds_nothing_and_keeps_what_it_has():
    facts = ScriptedStep("extract_many", "text", texts=["Score:", "Tag:"], output_name="facts", pattern=r": (.+)$",
                         output_mode="append", optional=True)
    artifact, _ = run_discovery([*detail_round("Item A", facts), *detail_round("Item B", facts),
                                 ScriptedStep("done", expect="Results")], name="catalog_facts")
    untagged = CatalogSurface({"Item A": {"score": "1", "tag": "x"}, "Item B": {"score": "2", "tag": ""},
                               "Item C": {"score": "3", "tag": "y"}})
    result, log, _ = run_replay(artifact, untagged)
    assert result.status == "success" and result.outputs == {"facts": ["1", "x"]}
    assert [e["target_index"] for e in events(log, "optional_output_absent")] == [1]


def test_back_with_no_history_is_a_not_performed_action_failure():
    artifact, _ = run_discovery(THREE_SCORES)
    surface = CatalogSurface()
    surface.navigate(CATALOG)
    surface.history.clear()
    with pytest.raises(Exception, match="no previous page"):
        surface.back()


# ---------- the contract in the artifact ----------

def catalog_artifact(**changes):
    artifact, _ = run_discovery(THREE_SCORES)
    return artifact


def rejects(artifact, message):
    with pytest.raises(ArtifactError, match=message):
        validate(artifact)


def test_validation_pins_the_append_contract():
    artifact = catalog_artifact()
    extract = next(n for n in artifact.nodes if n.action.action == "extract")
    extract.action.mode = "prepend"
    rejects(artifact, "unknown output mode 'prepend'")
    extract.action.mode = "append"
    artifact.outputs["scores"] = {"type": "integer", "required": True, "pattern": SCORE, "description": "", "example": "1"}
    rejects(artifact, "append adds to a list, but output 'scores' is 'integer'")
    artifact = catalog_artifact()
    back = next(n for n in artifact.nodes if n.action.action == "back")
    back.action.mode = "append"
    rejects(artifact, "only extract and extract_many carry an output mode")
    artifact = catalog_artifact()
    back = next(n for n in artifact.nodes if n.action.action == "back")
    back.action.value = CATALOG
    rejects(artifact, "back carries no target and no value")


def test_an_artifact_without_the_mode_key_keeps_set_semantics():
    artifact = catalog_artifact()
    data = to_dict(artifact)
    for node in data["nodes"]:
        if node["action"] and node["action"]["action"] == "extract":
            del node["action"]["mode"]
    data["outputs"]["scores"] = {"type": "integer", "required": True, "pattern": SCORE, "description": "", "example": "1"}
    loaded = from_dict(data)
    validate(loaded)
    assert all(n.action.mode is None for n in loaded.nodes if n.action)
    result, _, _ = run_replay(loaded)
    assert result.outputs == {"scores": 90}       # three sets of one scalar: the last read wins, as before


def test_merge_and_redaction_preserve_the_mode():
    artifact = catalog_artifact()
    other = catalog_artifact()
    other.provenance["run_id"] = "discovery-second"
    traces = [ScenarioTrace(scenario=Scenario(name=f"run{i}", params={}, sensitive=[]), artifact=a, run_id=f"d{i}",
                            planner="scripted") for i, a in enumerate((artifact, other))]
    merged = merge_traces(traces, selectors=[], campaign_id="campaign-1")
    validate(merged)
    assert [n.action.mode for n in linear_path(merged) if n.action.action == "extract"] == ["append"] * 3
    assert merged.outputs["scores"]["example"] == ["71", "58", "90"]
    assert redact(to_dict(artifact), ("secret",))["outputs"]["scores"]["example"] == ["71", "58", "90"]
    assert [n["action"]["mode"] for n in redact(to_dict(artifact), ())["nodes"]
            if n["action"] and n["action"]["action"] == "extract"] == ["append"] * 3
