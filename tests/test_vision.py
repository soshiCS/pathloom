"""The bounded screenshot-vision fallback for discovery: trigger rules, budgets, validation, policy, provider
payloads, artifact recording, and the guarantee that replay never touches it. Offline, on the fake shop."""
import base64
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.cua import agent as agent_module
from src.cua.__main__ import build_parser
from src.cua.agent import DiscoveryFailed, discover
from src.cua.escalation import NoOperator
from src.cua.models import Observation, ScreenshotFrame
from src.cua.planner import (CHOOSE_VISUAL_ACTION_TOOL, MIN_VISION_CONFIDENCE, ClaudePlanner, OpenAIPlanner,
                             render_visual_prompt, visual_decision_from_tool_input)
from src.cua.replay import replay
from src.cua.surface import locator_for
from tests.context import (ENTRY, HOSTS, PARAMS, Element, Escalator, Policy, RecordingOperator, RunLog,
                           SessionControl)
from tests.fake_surface import TINY_PNG, VIEWPORT, FakeSurface
from tests.scripted_planner import NO_TARGET, ScriptedStep, ScriptedVisionPlanner, checkout_script, visual_click

# The fake shop draws every control at (10, 10, 50, 20); a proposal there is grounded to the real Login button.
LOGIN_VIA_VISION = visual_click("Login", expect="Products", box=(10, 10, 50, 20))
MISSING_CONTROL = ScriptedStep("stuck", stuck_cause="missing_control")


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


def stuck_then(script_after: list[ScriptedStep], *stucks: int) -> list[ScriptedStep]:
    """Type the credentials, then get stuck `stucks` times (the script cannot see Login), then continue."""
    return [ScriptedStep("type", "textbox", "Username", value="{{username}}"),
            ScriptedStep("type", "textbox", "Password", value="{{password}}"),
            *[MISSING_CONTROL] * (stucks[0] if stucks else 1)] + script_after


def after_login() -> list[ScriptedStep]:
    return checkout_script()[4:]                   # from Add to cart onward, ending with done


def run(script, decisions, surface=None, vision=True, max_attempts=2, operator=None, max_steps=25):
    planner = ScriptedVisionPlanner(script, decisions)
    log = RunLog("discovery", secrets=("secret_sauce",))
    surface = surface or FakeSurface()
    artifact = discover(goal="reach the checkout overview", name="checkout_review", params=dict(PARAMS),
                        surface=surface, planner=planner, policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(operator or NoOperator(), SessionControl(), log), log=log,
                        entry_url=ENTRY, sensitive={"password"}, max_steps=max_steps,
                        vision=planner if vision else None, max_vision_attempts=max_attempts)
    return artifact, planner, surface, log


def events(log, name):
    return [json.loads(l) for l in log.path.read_text().splitlines() if f'"event": "{name}"' in l]


# ---------- the happy path: a visual click becomes an ordinary, verified, replayable step ----------

def test_vision_fills_the_gap_and_the_step_replays_without_any_planner():
    artifact, planner, surface, log = run(stuck_then(after_login()), [LOGIN_VIA_VISION])
    assert len(planner.frames) == 1 and planner.calls[0]["remaining"] == 1
    assert planner.calls[0]["params"]["password"] == "{{password}}"           # placeholders only
    assert (planner.calls[0]["width"], planner.calls[0]["height"]) == VIEWPORT
    assert ("click", "Login", "") in surface.actions and surface.screen == "overview"

    step = artifact.steps[3]
    assert step.action == "click" and step.checkpoint == {"text_contains": "Products"} and step.risk == "safe"
    assert step.target.strategies == [{"kind": "role", "role": "button", "name": "Login"},   # grounded: same box
                                      {"kind": "coords", "x": 35, "y": 20, "exact": True,
                                       "viewport": {"width": VIEWPORT[0], "height": VIEWPORT[1]},
                                       "scroll": {"x": 0, "y": 0}}]
    assert artifact.provenance["vision_fallback"] == {"attempts": 1, "max_attempts": 2, "planner": "scripted+vision",
                                                      "steps": ["s4"]}
    text = log.path.read_text()
    for event in ("vision_fallback_requested", "vision_fallback_decided", "vision_action_verified", "recorded_step"):
        assert f'"{event}"' in text
    assert events(log, "vision_action_verified")[0] == {**events(log, "vision_action_verified")[0], "verified": True,
                                                         "kind": "click", "expect": "Products"}
    assert events(log, "recorded_step")[2]["targeting"] == "vision"
    assert base64.b64encode(TINY_PNG).decode() not in text and "secret_sauce" not in text   # no image, no secret

    replay_surface = FakeSurface()                          # replay: no planner object exists at all
    replay_log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(PARAMS), replay_surface, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success" and result.outputs["total"] == 32.39
    assert '"rung": 0' in replay_log.path.read_text()       # the best-effort role rung resolved on the fake


# ---------- trigger rules ----------

def test_vision_is_never_called_when_structured_discovery_can_proceed():
    artifact, planner, _, log = run(checkout_script(), [LOGIN_VIA_VISION])
    assert planner.frames == [] and len(artifact.steps) == 15
    assert "vision_fallback" not in artifact.provenance
    assert not any(json.loads(l)["event"].startswith("vision_") for l in log.path.read_text().splitlines())


def test_vision_can_be_disabled():
    with pytest.raises(DiscoveryFailed, match="aborted"):
        run(stuck_then(after_login()), [LOGIN_VIA_VISION], vision=False)


def test_vision_is_not_attempted_for_denials_confirmations_or_completed_goals():
    # A denied navigation, a refused risky click, and a rejected "done" all feed back to the planner; none is stuck.
    script = [ScriptedStep("navigate", value="https://evil.example.com/")] + checkout_script()
    script.insert(-1, ScriptedStep("click", "button", "Finish", expect="Thank you"))
    script.insert(-1, ScriptedStep("done", expect="Not on screen"))
    _, planner, surface, log = run(script, [LOGIN_VIA_VISION], operator=RecordingOperator(disposition="deny"))
    assert planner.frames == [] and ("click", "Finish", "") not in surface.actions
    assert '"decision": "deny"' in log.path.read_text() and '"done_rejected"' in log.path.read_text()


# ---------- budgets and loop prevention ----------

def test_one_attempt_per_unchanged_screen_then_escalation():
    operator = RecordingOperator(disposition="abort")
    with pytest.raises(DiscoveryFailed, match="aborted"):
        run(stuck_then([], 3), [NO_TARGET, LOGIN_VIA_VISION], operator=operator, max_attempts=5)
    # The second stuck on the same screen never reached the planner; the human was asked instead.
    assert operator.requests[0].kind == "stuck"


def test_no_target_escalates_instead_of_looping():
    operator = RecordingOperator(disposition="abort")
    with pytest.raises(DiscoveryFailed, match="aborted"):
        _, planner, _, _ = run(stuck_then([], 2), [NO_TARGET, NO_TARGET], operator=operator, max_attempts=5)
    assert len(operator.requests) == 1


def test_total_budget_is_enforced_across_changing_screens():
    # Login by vision (screen changes), then stuck again on the inventory screen: budget of 1 is spent.
    operator = RecordingOperator(disposition="abort")
    script = stuck_then([MISSING_CONTROL] + after_login())
    with pytest.raises(DiscoveryFailed, match="aborted"):
        run(script, [LOGIN_VIA_VISION, visual_click("Add to cart", expect="Remove")], operator=operator, max_attempts=1)
    log_text = operator.requests[0].reason
    assert "stuck" in log_text


def test_budget_exhaustion_is_logged_once():
    operator = RecordingOperator(disposition="abort")
    script = stuck_then([MISSING_CONTROL] + after_login())
    try:
        run(script, [LOGIN_VIA_VISION], operator=operator, max_attempts=1)
    except DiscoveryFailed:
        pass
    # captured through the operator's log: re-run to inspect the log directly
    planner = ScriptedVisionPlanner(stuck_then([MISSING_CONTROL] + after_login()), [LOGIN_VIA_VISION])
    log = RunLog("discovery", secrets=("secret_sauce",))
    with pytest.raises(DiscoveryFailed):
        discover(goal="g", name="checkout_review", params=dict(PARAMS), surface=FakeSurface(), planner=planner,
                 policy=Policy(allowed_hosts=HOSTS),
                 escalator=Escalator(RecordingOperator("abort"), SessionControl(), log),
                 log=log, entry_url=ENTRY, sensitive={"password"}, vision=planner, max_vision_attempts=1)
    assert len(events(log, "vision_budget_exhausted")) == 1 and len(planner.frames) == 1


# ---------- validation of proposals ----------

def test_out_of_bounds_low_confidence_and_unverifiable_proposals_are_rejected_without_clicking():
    width, height = VIEWPORT
    good = {"kind": "visual_click", "role": "button", "name": "Login", "x": 300, "y": 200, "width": 120,
            "height": 40, "value": None, "expect": "Products", "reason": "r", "confidence": 0.9}
    cases = [
        ({**good, "x": width - 10}, "not inside"), ({**good, "y": -1}, "not inside"),
        ({**good, "width": 0}, "not inside"), ({**good, "x": float("nan")}, "not a finite number"),
        ({**good, "x": "300"}, "not a finite number"), ({**good, "x": True}, "not a finite number"),
        ({**good, "confidence": MIN_VISION_CONFIDENCE - 0.1}, "below"), ({**good, "expect": None}, "no expected"),
        ({**good, "kind": "visual_type", "value": None}, "needs a value"),
        ({**good, "kind": "extract"}, "unknown kind"),
    ]
    for data, message in cases:
        decision = visual_decision_from_tool_input(data, width, height)
        assert decision.kind == "no_target" and message in decision.rejected, (data, decision)
    accepted = visual_decision_from_tool_input(good, width, height)
    assert accepted.kind == "visual_click" and accepted.target.expect == "Products"
    assert visual_decision_from_tool_input({**good, "kind": "no_target"}, width, height).rejected is None

    rejected = visual_click("Login", expect="Products", box=(790, 590, 50, 50))    # outside the fake viewport
    operator = RecordingOperator(disposition="abort")
    with pytest.raises(DiscoveryFailed, match="aborted"):
        run(stuck_then([]), [rejected], operator=operator)
    # The scripted decision bypasses tool-input validation; the agent's own guard is the run log below.
    assert operator.requests[0].kind == "stuck"


def test_a_visual_action_that_cannot_be_verified_escalates_and_is_not_recorded():
    operator = RecordingOperator(disposition="abort")
    unverifiable = visual_click("Login", expect="Welcome back, friend")
    with pytest.raises(DiscoveryFailed, match="aborted"):
        run(stuck_then([]), [unverifiable], operator=operator)
    assert "could not be verified" in operator.requests[0].reason
    planner = ScriptedVisionPlanner(stuck_then([]), [visual_click("Login", expect="Welcome back, friend")])
    surface = FakeSurface()
    log = RunLog("discovery", secrets=("secret_sauce",))
    with pytest.raises(DiscoveryFailed):
        discover(goal="g", name="checkout_review", params=dict(PARAMS), surface=surface, planner=planner,
                 policy=Policy(allowed_hosts=HOSTS),
                 escalator=Escalator(RecordingOperator("abort"), SessionControl(), log),
                 log=log, entry_url=ENTRY, sensitive={"password"}, vision=planner)
    assert surface.actions.count(("click", "Login", "")) == 1                    # clicked once, never repeated
    assert events(log, "vision_action_verified")[0]["verified"] is False
    assert not any(e.get("targeting") == "vision" for e in events(log, "recorded_step"))


# ---------- policy ----------

def test_a_visual_action_still_passes_through_policy():
    # A blocked control proposed by vision is denied like any other; the screen is unchanged afterwards, so
    # no second visual attempt is made and the run goes to a person.
    blocked = visual_click("Sign up", expect="Products")
    planner = ScriptedVisionPlanner(stuck_then(after_login()), [blocked, LOGIN_VIA_VISION])
    surface, log = FakeSurface(), RunLog("discovery", secrets=("secret_sauce",))
    with pytest.raises(DiscoveryFailed, match="aborted"):
        discover(goal="g", name="checkout_review", params=dict(PARAMS), surface=surface, planner=planner,
                 policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(NoOperator(), SessionControl(), log),
                 log=log, entry_url=ENTRY, sensitive={"password"}, vision=planner, max_vision_attempts=3)
    assert surface.actions.count(("click", "Sign up", "")) == 0 and len(planner.frames) == 1
    assert any(e["decision"] == "deny" for e in events(log, "policy_checked"))
    # After the denial the script's next miss is an ordinary not-found, not a perception report: no second look.
    assert events(log, "vision_fallback_skipped")[0]["reason"].startswith("stuck cause is 'planner'")


def test_risky_visual_actions_cannot_bypass_confirmation():
    finish = visual_click("Finish", expect="Thank you")
    operator = RecordingOperator(disposition="deny")
    with pytest.raises(DiscoveryFailed):
        run(stuck_then([MISSING_CONTROL] * 2), [finish, finish, finish], operator=operator, max_attempts=3,
            max_steps=6)
    assert operator.requests[0].kind == "confirm" and "Finish" in operator.requests[0].reason
    # (the deny disposition is not "abort", so discovery keeps asking until its step budget ends)


# ---------- provider support ----------

def test_planner_without_vision_support_falls_back_to_the_human_cleanly():
    from tests.scripted_planner import ScriptedPlanner
    planner = ScriptedPlanner(stuck_then([]))                                  # no decide_visually
    log = RunLog("discovery")
    with pytest.raises(DiscoveryFailed, match="aborted"):
        discover(goal="g", name="checkout_review", params=dict(PARAMS), surface=FakeSurface(), planner=planner,
                 policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(NoOperator(), SessionControl(), log),
                 log=log, entry_url=ENTRY, vision=planner)
    assert events(log, "vision_fallback_unavailable")[0]["reason"].startswith("surface or planner")


def frame() -> ScreenshotFrame:
    return ScreenshotFrame(png=TINY_PNG, width=800, height=600, path="masked.png")


def test_openai_image_request_carries_the_masked_image_and_no_secret():
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            call = SimpleNamespace(type="function_call", name="choose_visual_action", arguments=json.dumps(
                {"kind": "visual_click", "role": "button", "name": "Login", "x": 300, "y": 200, "width": 120,
                 "height": 40, "value": None, "expect": "Products", "reason": "r", "confidence": 0.8}))
            return SimpleNamespace(output=[call])

    planner = OpenAIPlanner(model="gpt-6-astra", client=SimpleNamespace(responses=Responses()))
    observation = Observation(url=ENTRY, elements=[Element("textbox", "Username")])
    decision = planner.decide_visually("goal", {"username": "standard_user", "password": "{{password}}"},
                                       observation, [], frame(), remaining_attempts=1)
    assert decision.kind == "visual_click" and decision.target.name == "Login"
    assert captured["model"] == "gpt-6-astra" and captured["tools"][0]["name"] == "choose_visual_action"
    content = captured["input"][0]["content"]
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"] == "data:image/png;base64," + base64.b64encode(TINY_PNG).decode()
    assert "800 x 600" in content[0]["text"] and "{{password}}" in content[0]["text"]
    assert "secret_sauce" not in json.dumps(captured, default=str)


def test_anthropic_image_request_carries_the_masked_image_and_no_secret():
    captured = {}

    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            block = SimpleNamespace(type="tool_use", input={
                "kind": "no_target", "role": "", "name": "", "x": 0, "y": 0, "width": 0, "height": 0, "value": None,
                "expect": None, "reason": "nothing", "confidence": 0.0})
            return SimpleNamespace(content=[block], stop_reason="tool_use")

    planner = ClaudePlanner.__new__(ClaudePlanner)
    planner.client, planner.model = SimpleNamespace(messages=Messages()), "m"
    planner.vision_model, planner.name = "m", "claude:m"
    decision = planner.decide_visually("goal", {"password": "{{password}}"}, Observation(url=ENTRY, elements=[]),
                                       [], frame(), remaining_attempts=0)
    assert decision.kind == "no_target" and decision.reason == "nothing"
    assert captured["tools"] == [CHOOSE_VISUAL_ACTION_TOOL]
    image, text = captured["messages"][0]["content"]
    assert image["source"] == {"type": "base64", "media_type": "image/png", "data": base64.b64encode(TINY_PNG).decode()}
    assert "Visual attempts remaining after this one: 0" in text["text"]
    assert "secret_sauce" not in json.dumps(captured, default=str)


def test_provider_without_image_support_returns_unavailable():
    class Responses:
        def create(self, **kwargs):
            raise RuntimeError("model does not accept images")

    planner = OpenAIPlanner(model="text-only", client=SimpleNamespace(responses=Responses()))
    decision = planner.decide_visually("g", {}, Observation(url=ENTRY, elements=[]), [], frame(), 1)
    assert decision.kind == "unavailable" and "does not accept images" in decision.reason


# ---------- replay stays model-free; artifacts stay compatible ----------

def test_replay_engines_never_import_or_reference_vision():
    probe = ("import sys, src.cua.replay, src.cua.graph_replay, src.cua.stability, src.cua.approval; "
             "print(sorted(m for m in sys.modules if m in ('src.cua.planner', 'anthropic', 'openai')))")
    loaded = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert loaded.strip() == "[]"
    for module in ("src/cua/replay.py", "src/cua/graph_replay.py"):
        source = open(module).read()
        assert "decide_visually" not in source and "viewport_screenshot" not in source


def test_coordinate_rungs_without_viewport_metadata_are_unchanged():
    old = locator_for(Element("button", "Login", box=(10, 20, 30, 40)))
    assert old.strategies[-1] == {"kind": "coords", "x": 25, "y": 40}
    new = locator_for(Element("button", "Login", box=(10, 20, 30, 40)), viewport=(800, 600))
    assert new.strategies[-1] == {"kind": "coords", "x": 25, "y": 40, "viewport": {"width": 800, "height": 600}}


def test_cli_exposes_the_vision_flags_as_opt_in():
    parser = build_parser()
    args = parser.parse_args(["discover", "--goal", "g", "--url", "u", "--name", "n"])
    assert args.vision_fallback is False and args.max_vision_attempts == 2
    args = parser.parse_args(["discover-campaign", "--spec", "s.json", "--vision-fallback",
                              "--max-vision-attempts", "3"])
    assert args.vision_fallback is True and args.max_vision_attempts == 3


# ---------- trigger classification: only a perception gap may reach vision ----------

def provider_planner(make_response):
    class Responses:
        def create(self, **kwargs):
            if not isinstance(kwargs.get("input"), str):                # only the visual call sends content blocks
                raise AssertionError("an image request was sent although vision must not have been triggered")
            return make_response(kwargs)
    return OpenAIPlanner(model="m", client=SimpleNamespace(responses=Responses()))


def tool_reply(arguments: str):
    return SimpleNamespace(output=[SimpleNamespace(type="function_call", name="choose_action", arguments=arguments)])


def discovery_with(planner, surface, operator=None):
    log = RunLog("discovery", secrets=("secret_sauce",))
    with pytest.raises(DiscoveryFailed):
        discover(goal="g", name="checkout_review", params=dict(PARAMS), surface=surface, planner=planner,
                 policy=Policy(allowed_hosts=HOSTS),
                 escalator=Escalator(operator or NoOperator(), SessionControl(), log),
                 log=log, entry_url=ENTRY, sensitive={"password"}, vision=planner, max_steps=3)
    return log


@pytest.mark.parametrize("label, make_response", [
    ("no tool call", lambda kw: SimpleNamespace(output=[], output_text="I would rather not")),
    ("invalid JSON", lambda kw: tool_reply("{not json")),
    ("invalid index", lambda kw: tool_reply(json.dumps({"kind": "click", "element_index": 99, "reason": ""}))),
    ("ordinary stuck", lambda kw: tool_reply(json.dumps({"kind": "stuck", "stuck_cause": "other", "reason": "?"}))),
])
def test_provider_failures_and_ordinary_stuck_states_never_capture_a_screenshot(label, make_response):
    surface = FakeSurface()
    log = discovery_with(provider_planner(make_response), surface)
    assert surface.viewport_shots == [] and events(log, "vision_fallback_requested") == [], label
    assert events(log, "vision_fallback_skipped")[0]["reason"].startswith("stuck cause is")
    assert not any(e["decision"] for e in events(log, "policy_checked"))


def test_anthropic_refusal_is_a_provider_stuck_state_not_a_perception_gap():
    class Messages:
        def create(self, **kwargs):
            assert all(isinstance(m["content"], str) for m in kwargs["messages"]), "no image may be sent"
            return SimpleNamespace(content=[], stop_reason="refusal")

    planner = ClaudePlanner.__new__(ClaudePlanner)
    planner.client, planner.model = SimpleNamespace(messages=Messages()), "m"
    planner.vision_model, planner.name = "m", "c"
    action = planner.decide("g", {}, Observation(url=ENTRY, elements=[]), [])
    assert action.kind == "stuck" and action.stuck_cause == "provider"
    surface = FakeSurface()
    log = discovery_with(planner, surface)
    assert surface.viewport_shots == [] and events(log, "vision_fallback_requested") == []


def test_only_an_explicit_missing_control_report_is_eligible():
    from src.cua.planner import action_from_tool_input
    observation = Observation(url=ENTRY, elements=[])
    cause = lambda data: action_from_tool_input({"reason": "", **data}, observation).stuck_cause
    assert cause({"kind": "stuck", "stuck_cause": "missing_control"}) == "perception"
    assert cause({"kind": "stuck", "stuck_cause": "other"}) == "planner"
    assert cause({"kind": "stuck", "reason": "the control is drawn on a canvas"}) == "planner"   # text is not a signal
    assert cause({"kind": "click", "element_index": 0}) == "provider"


# ---------- false verification and placeholders ----------

def test_an_expected_text_already_on_screen_rejects_the_proposal_without_clicking():
    already = visual_click("Login", expect="Swag Labs")                 # visible on the login page before any click
    planner = ScriptedVisionPlanner(stuck_then([]), [already])
    surface, log = FakeSurface(), RunLog("discovery", secrets=("secret_sauce",))
    with pytest.raises(DiscoveryFailed, match="aborted"):
        discover(goal="g", name="checkout_review", params=dict(PARAMS), surface=surface, planner=planner,
                 policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(NoOperator(), SessionControl(), log),
                 log=log, entry_url=ENTRY, sensitive={"password"}, vision=planner)
    assert ("click", "Login", "") not in surface.actions and len(planner.frames) == 1
    assert "already on screen" in events(log, "vision_target_rejected")[0]["reason"]
    assert events(log, "vision_action_verified") == [] and not any(e.get("targeting") == "vision"
                                                                   for e in events(log, "recorded_step"))


@pytest.mark.parametrize("decision", [
    visual_click("Login", expect="Products", value="{{missing}}"),              # a visual_type with an unknown value
    visual_click("Login", expect="Welcome {{missing}}"),                        # an unknown placeholder in expect
])
def test_unknown_placeholders_are_rejected_before_policy_and_before_acting(decision):
    planner = ScriptedVisionPlanner(stuck_then([]), [decision])
    surface, log = FakeSurface(), RunLog("discovery", secrets=("secret_sauce",))
    with pytest.raises(DiscoveryFailed, match="aborted"):                     # no KeyError escapes
        discover(goal="g", name="checkout_review", params=dict(PARAMS), surface=surface, planner=planner,
                 policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(NoOperator(), SessionControl(), log),
                 log=log, entry_url=ENTRY, sensitive={"password"}, vision=planner)
    assert not any(a[0] in ("click", "type") and a[1] == "Login" for a in surface.actions)
    assert "undeclared placeholders ['missing']" in events(log, "vision_target_rejected")[0]["reason"]
    assert not any(e["decision"] for e in events(log, "policy_checked")[2:])   # nothing after the two typed fields


def test_declared_placeholders_in_a_visual_type_are_substituted_when_acting():
    typed = visual_click("Username", expect="standard_user", role="textbox", value="{{username}}")
    login = visual_click("Login", expect="Products")
    script = [MISSING_CONTROL, ScriptedStep("type", "textbox", "Password", value="{{password}}"), MISSING_CONTROL]
    artifact, planner, surface, _ = run(script + after_login(), [typed, login], max_attempts=2)
    assert ("type", "Username", "standard_user") in surface.actions                # the real value was typed
    assert planner.calls[0]["params"]["password"] == "{{password}}"
    assert artifact.steps[1].value == "{{username}}"                              # recorded as the placeholder
    assert artifact.steps[1].checkpoint == {"text_contains": "{{username}}"}


# ---------- the visual locator ----------

def test_visual_locator_is_exact_coordinates_unless_grounded():
    from src.cua.agent import vision_locator
    frame = ScreenshotFrame(png=TINY_PNG, width=800, height=600, scroll_x=0, scroll_y=120)
    seen = Observation(url=ENTRY, elements=[Element("button", "Start", box=(10, 10, 50, 20))])
    ungrounded = Element("button", "Start", box=(300, 200, 120, 40), source="vision")     # same name, elsewhere
    assert vision_locator(ungrounded, seen, frame).strategies == [
        {"kind": "coords", "x": 360, "y": 220, "exact": True, "viewport": {"width": 800, "height": 600},
         "scroll": {"x": 0, "y": 120}}]
    grounded = Element("button", "Start", box=(20, 15, 30, 10), source="vision")          # overlaps the real one
    assert vision_locator(grounded, seen, frame).strategies[0] == {"kind": "role", "role": "button", "name": "Start"}
    assert vision_locator(grounded, seen, frame).strategies[1]["exact"] is True
