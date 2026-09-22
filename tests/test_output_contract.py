"""A caller-declared output contract for direct discovery: authoritative for every declared output, enforced before
`done` is accepted, carried into the artifact, and re-checked when a replay claims success."""
import json
from types import SimpleNamespace

import pytest

from src.cua import agent as agent_module
from src.cua.__main__ import build_parser, cmd_discover
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import ArtifactError, linear_path, output_problems, validate
from src.cua.campaign import CampaignError, load_output_contract, outputs_from_dict
from src.cua.escalation import Escalator, NoOperator, SessionControl
from src.cua.evidence import RunLog
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.catalog_surface import CATALOG, CatalogSurface
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

HOSTS = ["catalog.test"]
SCORE = r"Score: (\d+)"
THREE_SCORES = {"scores": {"type": "list", "required": True, "min_items": 3, "max_items": 3,
                           "items": {"type": "integer", "pattern": SCORE}}}


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


def detail_round(item, *reads):
    return [ScriptedStep("click", "link", item, expect="Details"), *reads, ScriptedStep("back", expect="Results")]


def score_read(name="scores", mode="append", optional=False, pattern=SCORE):
    return ScriptedStep("extract", "text", text="Score:", output_name=name, pattern=pattern, output_mode=mode,
                        optional=optional)


DONE = ScriptedStep("done", expect="Results")


def run_discovery(script, contract, surface=None, name="catalog_scores"):
    log = RunLog("discovery")
    artifact = discover(goal="read the scores of the first three results", name=name, params={},
                        surface=surface or CatalogSurface(), planner=ScriptedPlanner(script),
                        policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(NoOperator(), SessionControl(), log),
                        log=log, entry_url=CATALOG, max_steps=40, output_contract=contract)
    return artifact, log


def run_replay(artifact, surface=None):
    log = RunLog("replay")
    result = replay(artifact, {}, surface or CatalogSurface(), Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    return result, log


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def results(action_failed_events):
    return [e["reason"] for e in action_failed_events]


# ---------- the contract file ----------

def test_the_contract_file_uses_the_artifact_output_shape_and_rejects_bad_cardinality(tmp_path):
    path = tmp_path / "outputs.json"
    path.write_text(json.dumps(THREE_SCORES))
    assert load_output_contract(path) == THREE_SCORES
    path.write_text(json.dumps({"outputs": THREE_SCORES}))          # a campaign spec's outputs section works too
    assert load_output_contract(path) == THREE_SCORES
    for bad, message in [({"min_items": 4, "max_items": 3}, "min_items 4 exceeds max_items 3"),
                         ({"max_items": 0}, "max_items must be at least 1"),
                         ({"min_items": -1}, "min_items must be a non-negative integer"),
                         ({"min_items": True}, "min_items must be a non-negative integer"),
                         ({"max_items": "3"}, "max_items must be a non-negative integer")]:
        spec = {"scores": {**THREE_SCORES["scores"], **bad}}
        spec["scores"] = {k: v for k, v in spec["scores"].items() if k not in ("min_items", "max_items")} | bad
        with pytest.raises(CampaignError, match=message):
            outputs_from_dict(spec)
    with pytest.raises(CampaignError, match="only a list output carries min_items or max_items"):
        outputs_from_dict({"score": {"type": "integer", "pattern": SCORE, "min_items": 1}})
    path.write_text("{}")
    with pytest.raises(CampaignError, match="declares no outputs"):
        load_output_contract(path)
    path.write_text("{not json")
    with pytest.raises(CampaignError, match="cannot read output contract"):
        load_output_contract(path)


def test_omitted_cardinality_keeps_the_existing_list_contract():
    plain = outputs_from_dict({"scores": {"type": "list", "items": {"type": "integer", "pattern": SCORE}}})
    assert plain == {"scores": {"type": "list", "required": True, "items": {"type": "integer", "pattern": SCORE}}}
    assert output_problems(plain, {"scores": [1]}) == {} and output_problems(plain, {"scores": [1, 2, 3, 4]}) == {}


def test_the_cli_loads_the_contract_and_hands_it_to_discovery(tmp_path, monkeypatch, capsys):
    from src.cua import __main__ as cli
    path = tmp_path / "outputs.json"
    path.write_text(json.dumps(THREE_SCORES))
    received = {}

    def fake_discovery(*args, **kwargs):
        received.update(kwargs)
        kwargs["log"].event("discovery_started")
        raise DiscoveryFailed("stop here")

    monkeypatch.setattr(cli, "discover_with_reuse", fake_discovery)
    monkeypatch.setattr(cli, "make_planner", lambda args: SimpleNamespace(name="fake"))
    monkeypatch.setattr(cli, "PlaywrightSurface", lambda **kwargs: CatalogSurface())
    args = build_parser().parse_args(["discover", "--goal", "g", "--url", CATALOG, "--name", "catalog_scores",
                                      "--output-contract", str(path), "--quiet"])
    assert cmd_discover(args) == 2 and received["output_contract"] == THREE_SCORES
    args = build_parser().parse_args(["discover", "--goal", "g", "--url", CATALOG, "--name", "catalog_scores",
                                      "--output-contract", str(tmp_path / "missing.json"), "--quiet"])
    assert cmd_discover(args) == 2 and "Cannot load output contract" in capsys.readouterr().out
    args = build_parser().parse_args(["discover", "--goal", "g", "--url", CATALOG, "--name", "catalog_scores", "--quiet"])
    received.clear()
    assert cmd_discover(args) == 2 and received["output_contract"] is None


# ---------- discovery under a contract ----------

def test_done_is_rejected_while_a_required_output_is_missing_and_accepted_once_it_is_complete():
    artifact, log = run_discovery([DONE, *detail_round("Item A", score_read()), DONE,
                                   *detail_round("Item B", score_read()), *detail_round("Item C", score_read()), DONE],
                                  THREE_SCORES)
    rejected = events(log, "done_rejected")
    assert [e["outputs"] for e in rejected] == [{"scores": "missing"}, {"scores": "1 item(s), at least 3 required"}]
    assert artifact.outputs["scores"]["example"] == ["71", "58", "90"]
    assert artifact.outputs["scores"] == {**THREE_SCORES["scores"], "example": ["71", "58", "90"],
                                          "description": artifact.outputs["scores"]["description"]}
    validate(artifact)


def test_the_history_names_only_the_missing_or_invalid_outputs():
    seen = []

    class Spy(ScriptedPlanner):
        def decide(self, goal, params, observation, history, candidates=()):
            seen.append([a.result for a in history if a.kind == "done"])
            return super().decide(goal, params, observation, history, candidates)

    contract = {**THREE_SCORES, "names": {"type": "list", "required": True, "items": {"type": "string", "pattern": None}}}
    contract["names"]["items"] = {"type": "string", "pattern": "(.+)"}
    log = RunLog("discovery")
    discover(goal="g", name="n", params={}, surface=CatalogSurface(),
             planner=Spy([ScriptedStep("extract_many", "link", texts=["Item A", "Item B", "Item C"], output_name="names",
                                       pattern="(.+)"), DONE, *detail_round("Item A", score_read()), DONE,
                          *detail_round("Item B", score_read()), *detail_round("Item C", score_read()), DONE]),
             policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(NoOperator(), SessionControl(), log), log=log,
             entry_url=CATALOG, max_steps=40, output_contract=contract)
    assert seen[2] == ["not done: the declared outputs are not complete yet: scores (missing); record the missing or "
                       "incomplete outputs before finishing"]
    assert "names" not in seen[2][0] and "71" not in json.dumps(seen)


def test_a_required_output_cannot_be_made_optional_or_retyped_by_the_planner():
    loose = score_read(optional=True, pattern=r"(\d+)")
    artifact, _ = run_discovery([*detail_round("Item A", loose), *detail_round("Item B", loose),
                                 *detail_round("Item C", loose), DONE], THREE_SCORES)
    spec = artifact.outputs["scores"]
    assert spec["required"] is True and spec["items"] == {"type": "integer", "pattern": SCORE}
    assert (spec["min_items"], spec["max_items"]) == (3, 3)
    assert all(n.action.mode == "append" for n in linear_path(artifact) if n.action.action == "extract")


def test_a_complete_list_refuses_another_append_and_says_it_is_complete():
    artifact, log = run_discovery([*detail_round("Item A", score_read()), *detail_round("Item B", score_read()),
                                   *detail_round("Item C", score_read()), *detail_round("Item A", score_read()), DONE],
                                  THREE_SCORES)
    [refused] = events(log, "action_failed")
    assert refused["performed"] == "no"
    assert "output 'scores' is already complete (3 of at most 3 items); do not append to it again" in refused["reason"]
    assert artifact.outputs["scores"]["example"] == ["71", "58", "90"]
    assert sum(1 for n in linear_path(artifact) if n.action.action == "extract") == 3


def test_a_multi_item_append_that_would_overflow_is_refused_whole():
    many = ScriptedStep("extract_many", "text", texts=["Score:", "Tag:"], output_name="facts", pattern=r": (.+)$",
                        output_mode="append")
    contract = {"facts": {"type": "list", "required": True, "max_items": 3, "items": {"type": "string", "pattern": r": (.+)$"}}}
    artifact, log = run_discovery([*detail_round("Item A", many), *detail_round("Item B", many),
                                   *detail_round("Item C", ScriptedStep("extract", "text", text="Score:", output_name="facts",
                                                                        pattern=r": (.+)$", output_mode="append")), DONE],
                                  contract, name="catalog_facts")
    [refused] = events(log, "action_failed")
    assert "holds 2 of at most 3 items; appending 2 more would exceed the contract, so nothing was appended" in refused["reason"]
    assert artifact.outputs["facts"]["example"] == ["71", "alpha", "90"]     # all or nothing, then one more fits


def test_incremental_appends_of_one_item_each_complete_the_list_and_replay():
    artifact, _ = run_discovery([*detail_round("Item A", score_read()), *detail_round("Item B", score_read()),
                                 *detail_round("Item C", score_read()), DONE], THREE_SCORES)
    result, log = run_replay(artifact)
    assert result.status == "success" and result.outputs == {"scores": [71, 58, 90]}
    assert events(log, "outputs_validated") == [{**events(log, "outputs_validated")[0], "problems": {}}]


def test_a_two_item_list_cannot_complete_when_exactly_three_are_required():
    with pytest.raises(DiscoveryFailed, match="human aborted discovery"):
        run_discovery([*detail_round("Item A", score_read()), *detail_round("Item B", score_read()), DONE], THREE_SCORES)


def test_the_shape_of_a_declared_output_is_enforced_before_acting():
    artifact, log = run_discovery([*detail_round("Item A", score_read(mode="set")),           # a list: set refused
                                   *detail_round("Item A", score_read()), *detail_round("Item B", score_read()),
                                   *detail_round("Item C", score_read()), DONE], THREE_SCORES)
    [refused] = events(log, "action_failed")
    assert "output 'scores' is declared as a list; read it with extract_many, or add one item at a time" in refused["reason"]
    scalar = {"first_score": {"type": "integer", "required": True, "pattern": SCORE}}
    artifact, log = run_discovery([*detail_round("Item A", score_read(name="first_score")),
                                   *detail_round("Item A", score_read(name="first_score", mode="set")), DONE], scalar)
    [refused] = events(log, "action_failed")
    assert "output 'first_score' is declared as a single integer value, so nothing can be appended" in refused["reason"]
    assert artifact.outputs["first_score"]["example"] == "71"


def test_four_aligned_required_lists_of_exactly_three_items_replay_successfully():
    three = lambda item_type, pattern: {"type": "list", "required": True, "min_items": 3, "max_items": 3,
                                        "items": {"type": item_type, "pattern": pattern}}
    contract = {"names": three("string", "(.+)"), "scores": three("integer", SCORE),
                "tags": three("string", r"Tag: (.+)"), "codes": three("string", r"Code: (.+)")}
    per_page = lambda: [score_read(), ScriptedStep("extract", "text", text="Tag:", output_name="tags", pattern=r"Tag: (.+)",
                                                   output_mode="append"),
                        ScriptedStep("extract", "text", text="Code:", output_name="codes", pattern=r"Code: (.+)",
                                     output_mode="append")]
    surface = CatalogSurface({"Item A": {"score": "71", "tag": "alpha", "code": "A-1"},
                              "Item B": {"score": "58", "tag": "beta", "code": "B-2"},
                              "Item C": {"score": "90", "tag": "gamma", "code": "C-3"}})
    script = [ScriptedStep("extract_many", "link", texts=["Item A", "Item B", "Item C"], output_name="names", pattern="(.+)"),
              *detail_round("Item A", *per_page()), *detail_round("Item B", *per_page()), *detail_round("Item C", *per_page()),
              DONE]
    artifact, _ = run_discovery(script, contract, surface=surface, name="catalog_table")
    validate(artifact)
    assert {name: spec["example"] for name, spec in artifact.outputs.items()} == {
        "names": ["Item A", "Item B", "Item C"], "scores": ["71", "58", "90"], "tags": ["alpha", "beta", "gamma"],
        "codes": ["A-1", "B-2", "C-3"]}
    fresh = CatalogSurface({"Item A": {"score": "1", "tag": "x", "code": "X-1"}, "Item B": {"score": "2", "tag": "y", "code": "Y-2"},
                            "Item C": {"score": "3", "tag": "z", "code": "Z-3"}})
    result, _ = run_replay(artifact, fresh)
    assert result.status == "success"
    assert result.outputs == {"names": ["Item A", "Item B", "Item C"], "scores": [1, 2, 3], "tags": ["x", "y", "z"],
                              "codes": ["X-1", "Y-2", "Z-3"]}


# ---------- replay re-checks the contract ----------

def test_replay_returns_a_structured_failure_for_an_incomplete_or_malformed_accumulated_output():
    artifact, _ = run_discovery([*detail_round("Item A", score_read()), *detail_round("Item B", score_read()),
                                 *detail_round("Item C", score_read()), DONE], THREE_SCORES)
    short = CatalogSurface({"Item A": {"score": "5"}, "Item B": {"score": "6"}, "Item C": {"score": "7"}})
    artifact.outputs["scores"]["min_items"] = artifact.outputs["scores"]["max_items"] = 2   # a stricter caller
    result, log = run_replay(artifact, short)
    assert result.status == "failure" and result.outcome_code == "outputs_incomplete" and result.step_id == "success"
    assert result.expected == "outputs ['scores'] satisfying the declared contract"
    assert result.observed == "scores: 3 item(s), at most 2 allowed" and "5" not in result.observed
    assert events(log, "outputs_validated")[0]["problems"] == {"scores": "3 item(s), at most 2 allowed"}
    del artifact.outputs["scores"]["min_items"], artifact.outputs["scores"]["max_items"]
    artifact.outputs["scores"]["items"] = {"type": "integer", "pattern": r"Score: (.+)"}   # reads a word as text
    words = CatalogSurface({"Item A": {"score": "five"}, "Item B": {"score": "6"}, "Item C": {"score": "7"}})
    result, _ = run_replay(artifact, words)
    assert (result.status, result.outcome_code, result.observed) == ("failure", "outputs_incomplete",
                                                                     "scores: item 0 is not a integer")


def test_output_problems_covers_scalars_lists_and_optional_outputs():
    outputs = {"total": {"type": "number", "required": True}, "note": {"type": "string", "required": False},
               "ids": {"type": "list", "required": False, "min_items": 2, "items": {"type": "string"}}}
    assert output_problems(outputs, {"total": 32.39}) == {}
    assert output_problems(outputs, {"total": "32.39", "note": None}) == {}
    assert output_problems(outputs, {}) == {"total": "missing"}
    assert output_problems(outputs, {"total": "thirty"}) == {"total": "not a number"}
    assert output_problems(outputs, {"total": 1, "ids": ["a"]}) == {"ids": "1 item(s), at least 2 required"}
    assert output_problems(outputs, {"total": 1, "ids": "a"}) == {"ids": "not a list"}
    assert output_problems(outputs, {"total": True}) == {"total": "not a number"}
    # an example recorded as an input placeholder is the input's value: typed by the input, never "not a number"
    assert output_problems(outputs, {"total": "{{deposit}}", "ids": ["{{first}}", "x"]}) == {}
    assert output_problems(outputs, {"total": "{{deposit}}.00"}) == {}     # "500" typed, "500.00" read back


# ---------- without a contract nothing changes ----------

def test_discovery_without_a_contract_still_infers_its_own_definitions():
    artifact, log = run_discovery([*detail_round("Item A", score_read(optional=True)), DONE], None)
    assert events(log, "done_rejected") == []
    assert artifact.outputs["scores"]["required"] is False and "min_items" not in artifact.outputs["scores"]
    assert run_replay(artifact)[0].status == "success"


def test_artifact_validation_rejects_bad_cardinality_and_cardinality_on_scalars():
    artifact, _ = run_discovery([*detail_round("Item A", score_read()), *detail_round("Item B", score_read()),
                                 *detail_round("Item C", score_read()), DONE], THREE_SCORES)
    artifact.outputs["scores"]["min_items"] = 4
    with pytest.raises(ArtifactError, match="min_items 4 exceeds max_items 3"):
        validate(artifact)
    artifact.outputs["scores"]["min_items"] = 1.5
    with pytest.raises(ArtifactError, match="min_items must be a non-negative integer"):
        validate(artifact)
    artifact.outputs["scores"] = {"type": "integer", "required": True, "pattern": SCORE, "max_items": 3}
    with pytest.raises(ArtifactError, match="only a list output carries min_items or max_items"):
        validate(artifact)


def test_undeclared_output_names_are_refused_before_extraction_and_the_valid_names_are_listed():
    contract = {**THREE_SCORES, "names": {"type": "list", "required": True, "items": {"type": "string", "pattern": "(.+)"}}}
    artifact, log = run_discovery([*detail_round("Item A", score_read(name="study_1_score"),       # refused
                                                 score_read()),
                                   *detail_round("Item B", score_read(name="score"),               # refused
                                                 score_read()),
                                   *detail_round("Item C", score_read()),
                                   ScriptedStep("extract_many", "link", texts=["Item A", "Item B", "Item C"],
                                                output_name="names", pattern="(.+)"), DONE], contract)
    refused = events(log, "action_failed")
    assert [f["performed"] for f in refused] == ["no", "no"]
    for failure, name in zip(refused, ("study_1_score", "score")):
        assert f"output '{name}' is not declared in the output contract" in failure["reason"]
        assert "the only valid output names are ['names', 'scores']" in failure["reason"]
    assert sorted(artifact.outputs) == ["names", "scores"] and artifact.outputs["scores"]["example"] == ["71", "58", "90"]


def test_the_clinical_trials_contract_file_declares_five_required_lists_of_exactly_three():
    contract = load_output_contract("scenarios/clinical_trials_outputs.json")
    assert sorted(contract) == ["enrollments", "locations", "nct_numbers", "sponsors", "titles"]
    for spec in contract.values():
        assert spec["type"] == "list" and spec["required"] is True and (spec["min_items"], spec["max_items"]) == (3, 3)
        assert spec["items"]["type"] in ("string", "integer") and spec["items"]["pattern"]
    assert contract["enrollments"]["items"]["type"] == "integer"


def test_the_usaspending_contract_declares_five_required_three_item_string_lists():
    contract = load_output_contract("scenarios/usaspending_outputs.json")
    assert sorted(contract) == ["award_amounts", "award_ids", "awarding_agencies", "recipients", "start_dates"]
    for name, spec in contract.items():
        assert spec["type"] == "list" and spec["required"] is True, name
        assert (spec["min_items"], spec["max_items"]) == (3, 3), name
        assert spec["items"]["type"] == "string" and spec["items"]["pattern"], name
    # exactly three items: fewer and more are both refused, and each pattern reads its displayed shape
    for values, problem in [(["a", "b"], "2 item(s), at least 3 required"), (["a"] * 4, "4 item(s), at most 3 allowed")]:
        assert output_problems(contract, {**{n: ["x", "y", "z"] for n in contract}, "award_ids": values}) == {
            "award_ids": problem}
    import re
    amount = re.compile(contract["award_amounts"]["items"]["pattern"])
    assert [amount.search(t).group(1) for t in ("$1,200,000", "Total: $99.95", "$7")] == ["$1,200,000", "$99.95", "$7"]
    date = re.compile(contract["start_dates"]["items"]["pattern"])
    for shown, expected in [("2026-07-31", "2026-07-31"), ("Start: 7/31/2026", "7/31/2026"),
                            ("31 July 2026", "31 July 2026"), ("July 31, 2026", "July 31, 2026")]:
        assert date.search(shown).group(1) == expected, shown
    assert amount.search("no amount here") is None and date.search("no date here") is None
