"""What a person does during a discovery handoff becomes part of the artifact; what a person does during a replay
handoff stays evidence. Offline, on the fake shop; no model."""
import copy
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import DiscoveryFailed, RestartDiscovery, discover
from src.cua.artifact import linear_path, to_dict, validate
from src.cua.escalation import NoOperator
from src.cua.models import GraphEdge, Guard
from src.cua.policy import Policy
from src.cua.replay import replay
from src.cua.reuse import discover_with_reuse
from tests.context import ENTRY, HOSTS, PARAMS, Escalator, RecordingOperator, RunLog, SessionControl
from tests.fake_surface import FakeSurface, el
from tests.scripted_planner import MONEY, ScriptedPlanner, ScriptedStep, checkout_script

LOGIN = checkout_script()[:next(i for i, step in enumerate(checkout_script()) if step.name == "Login") + 1]
STUCK = ScriptedStep("stuck")
CART_URL = ENTRY + "cart.html"
PARAMS_CART = {**PARAMS, "cart_url": CART_URL}     # the cart's address is a declared parameter of this flow


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


def run(script, operator, surface=None, params=None):
    log = RunLog("discovery", secrets=("secret_sauce",))
    surface = surface or FakeSurface()
    artifact = discover(goal="reach the checkout overview and read the total", name="handoff_flow",
                        params=dict(params or PARAMS_CART), surface=surface, planner=ScriptedPlanner(script),
                        policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(operator, SessionControl(), log), log=log,
                        entry_url=ENTRY, sensitive={"password"}, max_steps=25)
    return artifact, surface, log


def replay_fresh(artifact, operator=None, irreversible="confirm", params=None):
    log = RunLog("replay", secrets=("secret_sauce",))
    surface = FakeSurface()
    result = replay(artifact, dict(params or PARAMS_CART), surface, Policy(allowed_hosts=HOSTS),
                    Escalator(operator or NoOperator(), SessionControl(), log), log, irreversible_policy=irreversible)
    return result, surface, log


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def actions_of(artifact):
    return [n.action.action for n in linear_path(artifact)]


# ---------- navigate -> resume -> automated extraction ----------

def test_a_human_navigation_is_recorded_in_position_and_the_artifact_replays_without_a_person():
    operator = RecordingOperator("resume", manual=[("navigate", CART_URL)])
    artifact, _, log = run(LOGIN + [ScriptedStep("click", "button", "Add to cart", context="{{product_name}}"), STUCK,
                                    ScriptedStep("click", "button", "Checkout", expect="Checkout: Your Information"),
                                    ScriptedStep("done", expect="Checkout: Your Information")], operator)
    validate(artifact)
    path = linear_path(artifact)
    assert actions_of(artifact) == ["navigate", "type", "type", "click", "click", "navigate", "click"]
    human = path[5]
    assert human.action.value == "{{cart_url}}" and human.action.checkpoint == {"url_contains": "/cart.html"}
    assert (human.effect, human.retry_safety) == ("none", "verify_before_retry")
    assert artifact.provenance["human_steps"] == [human.id] and artifact.provenance["interventions"] == 1
    [recorded] = events(log, "recorded_human_step")
    assert recorded["step_id"] == human.id and recorded["action"] == "navigate"
    result, surface, _ = replay_fresh(artifact)
    assert result.status == "success" and ("navigate", CART_URL) in surface.actions
    assert surface.screen == "info"


def test_human_click_and_type_record_usable_ladders_and_placeholders():
    operator = RecordingOperator("resume", manual=[("type", "Username", "standard_user"),
                                                   ("type", "Password", "secret_sauce"), ("click", "Login")])
    artifact, _, log = run([STUCK, ScriptedStep("done", expect="Products")], operator)
    validate(artifact)
    path = linear_path(artifact)
    assert actions_of(artifact) == ["navigate", "type", "type", "click"]
    assert [n.action.value for n in path[1:3]] == ["{{username}}", "{{password}}"]
    assert path[1].action.target.strategies[0] == {"kind": "role", "role": "textbox", "name": "Username"}
    assert path[3].action.target.strategies[0] == {"kind": "role", "role": "button", "name": "Login"}
    assert path[3].action.checkpoint == {"url_contains": "/inventory.html"}
    assert artifact.provenance["human_steps"] == ["s2", "s3", "s4"]
    dumped = json.dumps(to_dict(artifact))
    assert "secret_sauce" not in dumped and "standard_user" not in dumped
    assert "secret_sauce" not in log.path.read_text()
    assert replay_fresh(artifact, params={**PARAMS_CART, "username": "standard_user"})[0].status == "success"


def test_a_sensitive_value_that_is_not_a_declared_parameter_cannot_be_recorded():
    operator = RecordingOperator("resume", manual=[("type", "Password", "hunter2")])
    with pytest.raises(DiscoveryFailed, match="unrecordable_human_action: the value typed into textbox 'Password' is "
                                              "sensitive by its field name"):
        run([STUCK], operator)


# ---------- restart, abort and confirmations ----------

def test_restart_discards_the_abandoned_attempt_and_its_human_actions():
    surfaces = []

    def factory():
        surfaces.append(FakeSurface())
        return surfaces[-1]

    operator = RecordingOperator("restart", manual=[("type", "Username", "someone"), ("click", "Login")])
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover_with_reuse(factory, None, goal="g", name="handoff_flow", params=dict(PARAMS_CART),
                                   planner=ScriptedPlanner([STUCK] + LOGIN + [ScriptedStep("done", expect="Products")]),
                                   policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(operator, SessionControl(), log),
                                   log=log, entry_url=ENTRY, sensitive={"password"}, auto_reuse=False)
    assert len(surfaces) == 2 and events(log, "discovery_restarted")[0]["reason"].startswith("operator chose restart")
    assert actions_of(artifact) == ["navigate", "type", "type", "click"] and "human_steps" not in artifact.provenance
    assert events(log, "recorded_human_step") == [] and "someone" not in json.dumps(to_dict(artifact))


def test_restart_without_a_session_owner_surfaces_as_a_restart_request():
    with pytest.raises(RestartDiscovery, match="operator chose restart"):
        run([STUCK], RecordingOperator("restart", manual=[("click", "Login")]))


def test_abort_and_confirmation_answers_create_no_action_nodes():
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run([STUCK], RecordingOperator("abort", manual=[("click", "Login")]))
    approver = RecordingOperator("approve")
    artifact, _, log = run(checkout_script()[:-5] + [ScriptedStep("click", "button", "Finish", expect="Thank you"),
                                                     ScriptedStep("done", expect="Thank you")], approver)
    assert approver.requests[0].kind == "confirm" and "human_steps" not in artifact.provenance
    assert events(log, "recorded_human_step") == []
    denier = RecordingOperator("deny")
    artifact, _, log = run(checkout_script()[:-5] + [ScriptedStep("click", "button", "Finish"),
                                                     ScriptedStep("done", expect="Checkout: Overview")], denier)
    assert denier.requests[0].kind == "confirm" and actions_of(artifact).count("click") == 5 and events(log, "recorded_human_step") == []


# ---------- replay handoffs never touch the artifact ----------

def test_replay_interventions_do_not_modify_the_artifact():
    artifact, _, _ = run(LOGIN + [ScriptedStep("done", expect="Products")], NoOperator())
    login = linear_path(artifact)[3]
    login.action.target.strategies = [{"kind": "role", "role": "button", "name": "Log in (renamed)"}]
    before = copy.deepcopy(to_dict(artifact))
    operator = RecordingOperator("resume", manual=[("click", "Login")])
    result, _, log = replay_fresh(artifact, operator=operator)
    assert result.status == "success" and result.interventions[0]["human_actions"] == [{"action": "click", "target": "Login"}]
    assert to_dict(artifact) == before and events(log, "recorded_human_step") == []


# ---------- failed or uncertain human actions ----------

def test_a_human_action_the_surface_lost_track_of_fails_discovery_clearly():
    surface = FakeSurface(faults=["uncertain_click:Login"])
    operator = RecordingOperator("resume", manual=[("type", "Username", "standard_user"),
                                                   ("type", "Password", "secret_sauce"), ("click", "Login")])
    with pytest.raises(DiscoveryFailed, match="unrecordable_human_action: the operator's click on button 'Login' may not "
                                              "have completed"):
        run([STUCK], operator, surface=surface)


def test_a_human_action_that_never_happened_is_not_recorded_and_discovery_continues():
    surface = FakeSurface(faults=["blocked_click:Login"])
    operator = RecordingOperator("resume", manual=[("click", "Login"), ("type", "Username", "standard_user")])
    artifact, _, log = run([STUCK, ScriptedStep("type", "textbox", "Password", value="{{password}}"),
                            ScriptedStep("click", "button", "Login", expect="Products"),
                            ScriptedStep("done", expect="Products")], operator, surface=surface)
    assert actions_of(artifact) == ["navigate", "type", "type", "click"]
    assert artifact.provenance["human_steps"] == ["s2"]                    # only the typing the person really did
    [finished] = events(log, "intervention_finished")
    assert finished["human_actions"][0]["error"].startswith("click on button 'Login' was not performed")


# ---------- irreversible or unknown human actions stay gated ----------

def test_a_recorded_irreversible_human_click_keeps_its_effect_and_replay_gates_it():
    operator = RecordingOperator("resume", manual=[("click", "Finish")])
    artifact, _, _ = run(checkout_script()[:-5] + [STUCK, ScriptedStep("done", expect="Thank you")], operator)
    finish = linear_path(artifact)[-1]
    assert finish.action.target.strategies[0]["name"] == "Finish"
    assert (finish.effect, finish.retry_safety) == ("irreversible", "never_retry")
    result, surface, _ = replay_fresh(artifact, irreversible="deny")
    assert result.status == "failure" and result.outcome_code == "irreversible_denied"
    assert ("click", "Finish", "") not in surface.actions
    approved, surface, _ = replay_fresh(artifact, operator=RecordingOperator("approve"), irreversible="confirm")
    assert approved.status == "success" and surface.actions.count(("click", "Finish", "")) == 1


# ---------- the shape of the failed run: two automated sections joined by a manual navigation ----------

def test_manual_navigation_between_two_automated_sections_is_recorded_and_required():
    operator = RecordingOperator("resume", manual=[("navigate", CART_URL)])
    section_one = LOGIN + [ScriptedStep("click", "button", "Add to cart", context="{{product_name}}")]
    section_two = [ScriptedStep("click", "button", "Checkout", expect="Checkout: Your Information"),
                   ScriptedStep("type", "textbox", "First Name", value="{{first_name}}"),
                   ScriptedStep("type", "textbox", "Last Name", value="{{last_name}}"),
                   ScriptedStep("type", "textbox", "Zip/Postal Code", value="{{postal_code}}"),
                   ScriptedStep("click", "button", "Continue", expect="Checkout: Overview"),
                   ScriptedStep("extract", "text", text="Total:", output_name="total", pattern=MONEY),
                   ScriptedStep("done", expect="Checkout: Overview")]
    artifact, _, _ = run(section_one + [STUCK] + section_two, operator)
    validate(artifact)
    human = linear_path(artifact)[5]
    assert human.action.action == "navigate" and human.action.value == "{{cart_url}}"
    result, _, _ = replay_fresh(artifact)
    assert result.status == "success" and result.outputs == {"total": 32.39}

    # Without that node the second section cannot start: the very failure the fix removes.
    without = copy.deepcopy(artifact)
    without.nodes = [n for n in without.nodes if n.id != human.id]
    incoming = next(e for e in without.edges if e.target == human.id)
    outgoing = next(e for e in without.edges if e.source == human.id)
    without.edges = [e for e in without.edges if human.id not in (e.source, e.target)] + [
        GraphEdge(source=incoming.source, target=outgoing.target, guards=[Guard(kind="always")], priority=0)]
    without.provenance.pop("human_steps", None)
    validate(without)
    broken, _, _ = replay_fresh(without)
    assert broken.status == "failure" and broken.outcome_code == "target_not_found" and broken.step_id == "s7"


# ---------- abandoned exploratory actions before a human recovery ----------

DETAILS = ScriptedStep("click", "button", "Details", expect="Preview")
CLOSE = ScriptedStep("click", "button", "Close")


def test_popup_actions_that_led_nowhere_are_pruned_and_the_human_navigation_takes_their_place():
    surface = FakeSurface(extra_elements=[el("button", "Details")])
    operator = RecordingOperator("resume", manual=[("navigate", CART_URL)])
    artifact, _, log = run(LOGIN + [DETAILS, CLOSE, DETAILS, CLOSE, STUCK,
                                    ScriptedStep("click", "button", "Continue Shopping", expect="Products"),
                                    ScriptedStep("done", expect="Products")], operator, surface=surface)
    validate(artifact)
    assert actions_of(artifact) == ["navigate", "type", "type", "click", "navigate", "click"]
    human = linear_path(artifact)[4]
    assert human.action.value == "{{cart_url}}" and artifact.provenance["human_steps"] == [human.id]
    [pruned] = events(log, "speculative_suffix_pruned")
    assert pruned["node_ids"] == ["s5", "s6"] and pruned["superseded_by"] == "human navigate"
    assert "Preview" not in json.dumps(pruned) and "Details" not in json.dumps(pruned)
    assert len(events(log, "recorded_step")) == 3 + 2 + 1                     # every attempt stays in the log
    assert len(events(log, "recorded_human_step")) == 1
    assert len(events(log, "recorded_recoverable_outcome")) == 2
    result, replay_surface, _ = replay_fresh(artifact)
    assert result.status == "success" and result.interventions == []
    assert ("click", "Details", "") not in replay_surface.actions and replay_surface.screen == "inventory"


def test_state_the_human_continued_from_is_kept_and_a_recorded_output_is_a_boundary():
    surface = FakeSurface(extra_elements=[el("button", "Details")])
    operator = RecordingOperator("resume", manual=[("navigate", CART_URL)])
    section = checkout_script()[:-4]                                             # up to the overview screen
    artifact, _, log = run(section + [ScriptedStep("extract", "text", text="Total:", output_name="total", pattern=MONEY),
                                      DETAILS, CLOSE, STUCK, ScriptedStep("done", expect="Your Cart")], operator, surface=surface)
    details = [e for e in events(log, "recorded_step") if e["action"] == "click"][-1]["step_id"]
    assert [e["node_ids"] for e in events(log, "speculative_suffix_pruned")] == [[details]]
    kept = actions_of(artifact)
    assert kept[-2:] == ["extract", "navigate"] and "total" in artifact.outputs      # the output stays, the popup goes
    # a click that built state (the cart) is kept when the person navigates onward from it
    operator = RecordingOperator("resume", manual=[("navigate", CART_URL)])
    artifact, _, log = run(LOGIN + [ScriptedStep("click", "button", "Add to cart", context="{{product_name}}"), STUCK,
                                    ScriptedStep("click", "button", "Checkout", expect="Checkout: Your Information"),
                                    ScriptedStep("done", expect="Checkout: Your Information")], operator)
    assert actions_of(artifact) == ["navigate", "type", "type", "click", "click", "navigate", "click"]
    assert events(log, "speculative_suffix_pruned") == [] and events(log, "speculative_suffix_kept")[0]["node_ids"] == ["s5"]


def test_an_abandoned_suffix_that_cannot_be_proven_harmless_fails_discovery():
    # The planner's expectation was contradicted and the screen changed: neither provably harmless nor continued.
    # The control's own state flipped, so the step is recorded; the wrong guess still marks it as abandoned.
    surface = FakeSurface()
    surface.toggles["Compact view"] = ("pressed", False)
    operator = RecordingOperator("resume", manual=[("navigate", ENTRY + "cart.html")])
    script = [ScriptedStep("click", "button", "Compact view", expect="Products"), STUCK]
    with pytest.raises(DiscoveryFailed, match="abandoned_actions_unprovable: the operator's navigation supersedes"):
        run(script, operator, surface=surface)
