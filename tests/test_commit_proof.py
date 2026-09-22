"""A form-local commit is deterministic proof; arbitrary page change still is not.

When explicit expected text is contradicted, discovery may still record the action if the acted control's
own field group shows a commit: an editable field gave up its value and one new selected token carries it.
Both halves are required, the token must be new, carry its own removal or selection semantics, and sit in
the acted control's group. A toast, unrelated copy, a pre-existing chip, the value staying in the field and
two ambiguous new tokens are all refused. Offline, on the fake shop; no model and no live site.
"""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import ArtifactError, linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.replay import replay
from tests.context import ENTRY, HOSTS, PARAMS, Escalator, Policy, RecordingOperator, RunLog, SessionControl
from tests.fake_surface import FakeSurface
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

GROUP = "Filter by Keyword"
FIELD = "Search using keywords"
ADD = "Add"
KEYWORD = "renewable energy"


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


def commit_surface(**spec):
    """A field and a button that commits it, configured generically; no site logic anywhere."""
    surface = FakeSurface()
    surface.commits[ADD] = {"field": FIELD, "group": GROUP, **spec}
    return surface


def script(expect="Keyword:", value="{{keyword}}"):
    return [ScriptedStep("type", "textbox", FIELD, context=GROUP, value=value),
            ScriptedStep("click", "button", ADD, context=GROUP, expect=expect),
            ScriptedStep("done", expect="Swag Labs")]


def run(surface, steps=None, operator=None, params=None):
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover(goal="add a keyword filter", name="keyword_filter",
                        params={**PARAMS, "keyword": KEYWORD, **(params or {})}, surface=surface,
                        planner=ScriptedPlanner(steps or script()), policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(operator or RecordingOperator("resume"), SessionControl(), log), log=log,
                        entry_url=ENTRY, sensitive={"password"}, max_steps=20)
    return artifact, log


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def committed_nodes(artifact):
    return [n for n in linear_path(artifact)
            if n.action.action == "click" and n.action.target.strategies[0].get("name") == ADD]


# ---------- 1, 2, 9. the commit is proof, replays without a model, and stores a placeholder ----------

def test_a_cleared_field_and_a_new_removable_token_verify_and_record_the_click_once():
    surface = commit_surface(removable=True)
    artifact, log = run(surface)
    [added] = committed_nodes(artifact)
    assert added.action.checkpoint == {"selection_present": "{{keyword}}"}    # declared input stored as a placeholder
    assert events(log, "action_effect_proven")[0]["proof"] == "the field was committed into a new removable token"
    assert events(log, "action_effect_unverified") == []
    validate(artifact)


def test_replay_verifies_the_committed_state_without_a_model():
    artifact, _ = run(commit_surface(removable=True))
    fresh = commit_surface(removable=True)
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, {**PARAMS, "keyword": KEYWORD}, fresh, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success"
    assert {"selection_present": "{{keyword}}"} in [e["checkpoint"] for e in events(log, "checkpoint_passed")]
    assert fresh.tokens and fresh.tokens[-1]["text"] == KEYWORD          # replay really performed the commit


def test_a_committed_token_checkpoint_fails_replay_when_no_token_appears():
    artifact, _ = run(commit_surface(removable=True))
    inert = commit_surface(removable=True, token=False)                  # the button no longer commits anything
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, {**PARAMS, "keyword": KEYWORD}, inert, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "failure" and result.outcome_code == "checkpoint_not_met"


# ---------- 3, 4, 5, 6, 7. what is not proof ----------

def test_a_pre_existing_token_is_not_proof():
    surface = commit_surface(removable=True, token=False)
    surface.tokens.append({"text": KEYWORD, "group": GROUP, "removable": True, "selected": False})
    artifact, log = run(surface, operator=RecordingOperator("resume"))
    assert committed_nodes(artifact) == []
    assert events(log, "action_effect_unverified")[0]["cause"] == "action_effect_unverified"
    assert events(log, "commit_not_proven")[-1]["reason"].startswith("no new selected token")


def test_the_value_merely_remaining_in_the_textbox_is_not_proof():
    surface = commit_surface(removable=True, clears=False)
    artifact, log = run(surface, operator=RecordingOperator("resume"))
    assert committed_nodes(artifact) == []
    assert events(log, "commit_not_proven")[-1]["reason"] == "no editable field in the group gave up its value"


def test_unrelated_new_text_or_a_toast_is_not_proof():
    surface = commit_surface(token=False, toast="Filter applied")
    artifact, log = run(surface, operator=RecordingOperator("resume"))
    assert committed_nodes(artifact) == []
    assert events(log, "action_effect_unverified") != []


def test_a_token_outside_the_acted_controls_group_is_not_proof():
    surface = commit_surface(removable=True, outside="Some other panel")
    artifact, log = run(surface, operator=RecordingOperator("resume"))
    assert committed_nodes(artifact) == []
    assert events(log, "commit_not_proven")[-1]["reason"].startswith("no new selected token")


def test_two_ambiguous_new_tokens_are_refused():
    surface = commit_surface(removable=True, extra_tokens=1)
    artifact, log = run(surface, operator=RecordingOperator("resume"))
    assert committed_nodes(artifact) == []
    assert "2 new tokens appeared" in events(log, "commit_not_proven")[-1]["reason"]


def test_a_new_token_without_removal_or_selection_semantics_is_not_proof():
    surface = commit_surface(removable=False, selected=False)
    artifact, log = run(surface, operator=RecordingOperator("resume"))
    assert committed_nodes(artifact) == []
    assert events(log, "action_effect_unverified") != []


# ---------- 8. equivalent deterministic proof ----------

def test_a_new_selected_state_provides_equivalent_proof():
    surface = commit_surface(removable=False, selected=True)
    artifact, log = run(surface)
    [added] = committed_nodes(artifact)
    assert added.action.checkpoint == {"selection_present": "{{keyword}}"}
    assert events(log, "action_effect_proven")[0]["proof"] == "the field was committed into a new removable token"


def test_a_new_stored_form_value_provides_equivalent_proof():
    surface = commit_surface(token=False, stored=True)
    artifact, log = run(surface)
    [added] = committed_nodes(artifact)
    assert added.action.checkpoint is None          # a stored value is evidence, not a visible predicate
    assert events(log, "action_effect_proven")[0]["proof"] == "the field was committed into a new stored value"


# ---------- 10. sensitive values ----------

def test_a_sensitive_commit_is_not_written_down_and_keeps_the_unverified_handoff():
    surface = commit_surface(removable=True)
    steps = [ScriptedStep("type", "textbox", FIELD, context=GROUP, value="{{password}}"),
             ScriptedStep("click", "button", ADD, context=GROUP, expect="Keyword:"),
             ScriptedStep("done", expect="Swag Labs")]
    artifact, log = run(surface, steps=steps, operator=RecordingOperator("resume"))
    text = log.path.read_text()
    assert "secret_sauce" not in text and "secret_sauce" not in json.dumps(artifact.__dict__, default=str)
    assert committed_nodes(artifact) == []                      # cannot be represented safely: no node
    assert events(log, "commit_not_proven")[-1]["reason"] == "the committed token cannot be recorded safely"


# ---------- 11. no valid proof keeps the previous guarantees ----------

def test_without_commit_proof_the_action_stays_unrecorded_and_is_not_repeated():
    surface = commit_surface(token=False)
    blind = ScriptedStep("click", "button", ADD, context=GROUP, expect="Keyword:")
    steps = [ScriptedStep("type", "textbox", FIELD, context=GROUP, value="{{keyword}}"), blind, blind,
             ScriptedStep("done", expect="Swag Labs")]
    artifact, log = run(surface, steps=steps, operator=RecordingOperator("resume"))
    assert committed_nodes(artifact) == []
    assert surface.actions.count(("click", ADD, GROUP)) == 1     # the second attempt never reached the page
    assert events(log, "unverified_repeat_refused") != []


# ---------- 12, 13. the vocabulary stays closed and the rule stays site-agnostic ----------

def test_replay_refuses_a_checkpoint_condition_it_cannot_verify():
    artifact, _ = run(commit_surface(removable=True))
    node = committed_nodes(artifact)[0]
    node.action.checkpoint = {"vibes_match": "something"}
    with pytest.raises(ArtifactError, match="unknown condition"):
        validate(artifact)


def test_no_checkpoint_condition_name_is_erased_by_the_redaction_rules():
    # A condition whose own key reads as sensitive ("token", "secret", ...) would have its value replaced
    # wholesale in every log and artifact, leaving a checkpoint that looks recorded and asserts nothing.
    from src.cua.artifact import CHECKPOINT_KEYS
    from src.cua.policy import SENSITIVE_KEY_PATTERN, redact

    for key in CHECKPOINT_KEYS:
        assert not SENSITIVE_KEY_PATTERN.search(key), f"checkpoint condition {key!r} reads as a sensitive key"
        assert redact({key: "a value"}) == {key: "a value"}


def test_the_commit_rule_names_no_site_capability_or_selector():
    import ast
    from pathlib import Path
    banned = ["usaspending", "saucedemo", "keyword filter", "filter by keyword", "renewable energy",
              "search using keywords", "advanced search", "#filter", "prime awards"]
    for path in sorted(Path("src/cua").glob("*.py")):
        tree = ast.parse(path.read_text())
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                      and getattr(node, "body", None) and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                for word in banned:
                    assert word not in node.value.lower(), f"{path} carries {word!r} in a live string"
