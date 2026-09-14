"""Discovery loop: observe -> decide -> policy -> act -> verify -> record."""
import pytest

from src.cua import agent as agent_module
from src.cua.agent import DiscoveryFailed, discover
from src.cua.escalation import NoOperator
from src.cua.planner import OpenAIPlanner, action_from_tool_input, render_observation
from tests.context import (HOSTS, PARAMS, Element, Escalator, Observation, Policy, RecordingOperator, RunLog,
                           SessionControl)
from tests.fake_surface import ENTRY, FakeSurface
from tests.scripted_planner import ScriptedPlanner, ScriptedStep, checkout_script


def run_discovery(script, surface=None, params=None, operator=None, max_steps=25):
    log = RunLog("discovery", secrets=("secret_sauce",))
    escalator = Escalator(operator or NoOperator(), SessionControl(), log)
    artifact = discover(goal="reach the checkout overview and read the totals", name="checkout_review",
                        params=params or dict(PARAMS), surface=surface or FakeSurface(),
                        planner=ScriptedPlanner(script), policy=Policy(allowed_hosts=HOSTS),
                        escalator=escalator, log=log, entry_url=ENTRY, sensitive={"password"}, max_steps=max_steps)
    return artifact, log


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


def test_discovery_records_a_fully_parameterized_flow():
    artifact, _ = run_discovery(checkout_script())
    actions = [s.action for s in artifact.steps]
    assert actions == ["navigate", "type", "type", "click", "click", "click", "click", "type", "type", "type",
                       "click", "extract", "extract", "extract", "extract"]
    assert [s.value for s in artifact.steps if s.action == "type"] == [
        "{{username}}", "{{password}}", "{{first_name}}", "{{last_name}}", "{{postal_code}}"]
    assert artifact.steps[4].target.strategies[0] == {"kind": "role", "role": "button", "name": "Add to cart",
                                                      "context": "{{product_name}}"}   # qualified and parameterized
    assert artifact.steps[11].target.strategies[0]["name"] == "View details for {{product_name}}"
    assert artifact.steps[12].target.strategies[0] == {"kind": "text", "text": "Item total:"}
    assert artifact.success == {"text_contains": "Checkout: Overview"}
    assert artifact.status == "draft" and artifact.version == 1


def test_discovery_records_checkpoints_for_the_important_states():
    artifact, _ = run_discovery(checkout_script())
    checkpoints = {s.id: s.checkpoint["text_contains"] for s in artifact.steps if s.checkpoint}
    assert checkpoints["s4"] == "Products"                       # login succeeded, product list visible
    assert checkpoints["s5"] == "Remove"                         # requested product added
    assert checkpoints["s6"] == "Your Cart"                      # cart page reached
    assert checkpoints["s7"] == "Checkout: Your Information"
    assert checkpoints["s11"] == "Checkout: Overview"            # information accepted, overview reached


def test_discovery_records_typed_outputs_and_declared_outcomes():
    artifact, _ = run_discovery(checkout_script())
    assert artifact.outputs["product_name"]["type"] == "string"
    assert artifact.outputs["product_name"]["example"] == "{{product_name}}"
    assert artifact.outputs["total"] == {"type": "number", "required": True, "pattern": r"\$\s*([\d.]+)",
                                         "description": "Text read from text ''", "example": "32.39"}
    outcomes = {o["code"]: o for o in artifact.outcomes}
    assert outcomes["user_locked_out"]["detect"] == {"text_contains": "locked out"}
    assert outcomes["product_not_found"]["detect"] == {"text_missing": "{{product_name}}"}
    assert all(o["source"] == "planner" for o in outcomes.values())


def test_password_never_reaches_the_planner_the_artifact_or_the_log():
    seen = {}

    class SpyPlanner(ScriptedPlanner):
        def decide(self, goal, params, observation, history):
            seen.update(params)
            return super().decide(goal, params, observation, history)

    log = RunLog("discovery", secrets=("secret_sauce",))
    surface = FakeSurface()
    artifact = discover(goal="g", name="checkout_review", params=dict(PARAMS), surface=surface,
                        planner=SpyPlanner(checkout_script()), policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(NoOperator(), SessionControl(), log), log=log, entry_url=ENTRY,
                        sensitive={"password"})
    assert seen["password"] == "{{password}}" and seen["username"] == "standard_user"
    assert surface.typed["Password"] == "secret_sauce"          # the real value was typed
    assert artifact.inputs["password"]["sensitive"] is True
    assert "secret_sauce" not in str(artifact) and "secret_sauce" not in log.path.read_text()


def test_dialog_dismissal_is_a_recoverable_outcome_not_a_step():
    artifact, _ = run_discovery(checkout_script(), surface=FakeSurface(show_notice=True))
    assert all("Accept" not in str(s.target) for s in artifact.steps)
    recoverable = next(o for o in artifact.outcomes if o["kind"] == "recoverable")
    assert recoverable["source"] == "observed"
    assert recoverable["recover"]["target"]["strategies"][0]["name"] == "Accept"


def test_planner_asking_for_finish_needs_human_approval():
    operator = RecordingOperator(disposition="deny")
    script = checkout_script()
    script.insert(-1, ScriptedStep("click", "button", "Finish", expect="Thank you"))
    surface = FakeSurface()
    artifact, log = run_discovery(script, surface=surface, operator=operator)
    assert operator.requests[0].kind == "confirm" and "Finish" in operator.requests[0].reason
    assert ("click", "Finish", "") not in surface.actions and surface.screen == "overview"
    assert all(s.risk == "safe" for s in artifact.steps)
    assert '"decision": "confirm"' in log.path.read_text()


def test_denied_action_is_fed_back_and_not_recorded():
    script = [ScriptedStep("navigate", value="https://evil.example.com/")] + checkout_script()
    surface = FakeSurface()
    artifact, log = run_discovery(script, surface=surface)
    assert ("navigate", "https://evil.example.com/") not in surface.actions
    assert all(s.value != "https://evil.example.com/" for s in artifact.steps)
    assert '"decision": "deny"' in log.path.read_text()


def test_repeated_denials_stop_discovery():
    script = [ScriptedStep("navigate", value="https://evil.example.com/")] * 3
    with pytest.raises(DiscoveryFailed, match="disallowed"):
        run_discovery(script)


def test_unmet_expectation_keeps_the_step_but_drops_its_checkpoint():
    script = checkout_script()
    script[3] = ScriptedStep("click", "button", "Login", expect="Welcome back")   # wrong guess, real navigation
    artifact, log = run_discovery(script)
    assert '"expectation_failed"' in log.path.read_text()
    assert artifact.steps[3].action == "click" and artifact.steps[3].checkpoint is None


def test_slow_entry_page_is_retried_not_fatal(monkeypatch):
    monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)
    artifact, log = run_discovery(checkout_script(), surface=FakeSurface(faults=["slow_entry"]))
    assert len(artifact.steps) == 15
    assert '"transient_error"' in log.path.read_text()


def test_stuck_planner_escalates_and_abort_ends_discovery():
    operator = RecordingOperator(disposition="abort")
    with pytest.raises(DiscoveryFailed, match="aborted"):
        run_discovery([ScriptedStep("stuck")], operator=operator)
    assert operator.requests[0].kind == "stuck"


def test_max_steps_is_a_stopping_condition():
    forever = [ScriptedStep("type", "textbox", "Username", value="x")] * 10
    with pytest.raises(DiscoveryFailed, match="steps"):
        run_discovery(forever, max_steps=3)


def test_model_tool_call_is_bound_to_an_observed_element():
    observation = Observation(url=ENTRY, elements=[Element("textbox", "Username"), Element("button", "Login")])
    action = action_from_tool_input({"kind": "click", "element_index": 1, "expect": "Products", "reason": "",
                                     "outcomes": []}, observation)
    assert action.kind == "click" and action.target.name == "Login" and action.expect == "Products"
    bad = action_from_tool_input({"kind": "click", "element_index": 7, "reason": ""}, observation)
    assert bad.kind == "stuck"


def test_prompt_shows_the_enclosing_item_only_for_ambiguous_controls():
    observation = FakeSurface()
    observation.logged_in, observation.screen = True, "inventory"
    rendered = render_observation(observation.observe())
    assert 'button "Add to cart" (in: Sauce Labs Backpack)' in rendered
    assert 'link "cart"' in rendered and "(in:" not in rendered.split('link "cart"')[1].split("\n")[0]


def test_openai_planner_requests_exactly_one_strict_action():
    class ToolCall:
        type = "function_call"
        name = "choose_action"
        arguments = '{"kind":"click","element_index":1,"value":null,"output_name":null,"pattern":null,"optional":false,"expect":"Products","reason":"Log in","outcomes":[]}'

    class Responses:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return type("Response", (), {"output": [ToolCall()], "output_text": ""})()

    client = type("Client", (), {"responses": Responses()})()
    planner = OpenAIPlanner("test-openai-model", client=client)
    observation = Observation(url=ENTRY, elements=[Element("textbox", "Username"), Element("button", "Login")])

    action = planner.decide("log in", {}, observation, [])

    assert action.kind == "click" and action.target.name == "Login"
    assert client.responses.kwargs["parallel_tool_calls"] is False
    assert client.responses.kwargs["tool_choice"] == {"type": "function", "name": "choose_action"}
    tool = client.responses.kwargs["tools"][0]
    assert tool["type"] == "function" and tool["strict"] is True
    assert tool["parameters"]["additionalProperties"] is False
