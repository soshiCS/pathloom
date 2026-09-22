"""A control that was on screen when the action was chosen and has gone before acting: a fast, structured
failure with nothing dispatched, and a typed missing-control stuck cause that still reaches the vision fallback."""
import json
import time

import pytest

from src.cua import agent as agent_module
from src.cua.agent import VISION_CAUSES, DiscoveryFailed, discover
from src.cua.artifact import build_linear, linear_path
from src.cua.escalation import NoOperator
from src.cua.models import ACTION_FAILURE_CAUSES, ActionError, Element, GraphAction, Locator
from src.cua.planner import action_from_tool_input
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.context import ENTRY, HOSTS, PARAMS, Escalator, Observation, RecordingOperator, RunLog, SessionControl, linear_node
from tests.fake_surface import FakeSurface, el
from tests.scripted_planner import ScriptedPlanner, ScriptedStep, checkout_script

LOGIN = checkout_script()[:next(i for i, s in enumerate(checkout_script()) if s.name == "Login") + 1]


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


class VanishingSurface(FakeSurface):
    """A notice whose Close button is listed once and is gone by the time anything is clicked."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.notice = el("button", "Dismiss")
        self.closed = self.notice
        self.gone_ref = None
        self.observed = 0

    def elements(self):
        rows = super().elements()
        # the control stays listed (the screen was read before it closed); only the page no longer holds it
        return rows + ([self.notice] if self.notice is not None else [self.closed])

    def observe(self):
        self.observed += 1
        return super().observe()

    def still_present(self, target):
        """The notice's button is in the page only while the notice is: everything else always is."""
        if self.gone_ref is not None and target.ref == self.gone_ref:
            return False
        return True

    def click(self, target):
        if not self.still_present(target):
            raise ActionError(f"click on {target.role} '{target.name}' was not performed: the control was on screen "
                              f"when it was chosen and is no longer in the page", performed="no", cause="stale_target")
        return super().click(target)

    def vanish(self):
        self.gone_ref, self.notice = self.notice.ref, None


def test_the_cause_is_in_the_closed_vocabulary():
    assert "stale_target" in ACTION_FAILURE_CAUSES
    assert ActionError("x", performed="no", cause="stale_target").cause == "stale_target"
    with pytest.raises(ValueError, match="cause must be one of"):
        ActionError("x", performed="no", cause="vanished")


def test_a_vanished_control_fails_fast_with_no_click_and_discovery_continues():
    surface = VanishingSurface()

    class Vanishing(ScriptedPlanner):
        """The planner picks the notice from a screen that still shows it; the notice then closes itself."""

        def decide(self, goal, params, observation, history, candidates=()):
            action = super().decide(goal, params, observation, history, candidates)
            if action.target is not None and action.target.name == "Dismiss":
                surface.vanish()          # between this decision and the click, the notice is gone
            return action

    log = RunLog("discovery", secrets=("secret_sauce",))
    started = time.monotonic()
    artifact = discover(goal="log in", name="stale_flow", params=dict(PARAMS), surface=surface,
                        planner=Vanishing([ScriptedStep("click", "button", "Dismiss")] + LOGIN +
                                          [ScriptedStep("done", expect="Products")]),
                        policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(RecordingOperator("resume"), SessionControl(), log),
                        log=log, entry_url=ENTRY, sensitive={"password"}, max_steps=20)
    assert time.monotonic() - started < 10                      # no long wait for a control that had gone
    events = [json.loads(line) for line in log.path.read_text().splitlines() if '"action_failed"' in line]
    assert [e["cause"] for e in events] == ["stale_target"] and [e["performed"] for e in events] == ["no"]
    assert not any(a[0] == "click" and a[1] == "Dismiss" for a in surface.actions)     # zero clicks
    assert [n.action.action for n in linear_path(artifact)] == ["navigate", "type", "type", "click"]
    assert all("Dismiss" not in str(n.action.target.strategies) for n in linear_path(artifact) if n.action.target)


def test_replay_performs_no_action_and_returns_the_structured_failure():
    surface = VanishingSurface()
    ladder = Locator(strategies=[{"kind": "role", "role": "button", "name": "Dismiss"},
                                 {"kind": "css", "selector": surface.notice.ref}])
    nodes = [linear_node("s1", GraphAction(action="navigate", target=None, value=ENTRY,
                                           checkpoint={"text_contains": "Swag Labs"})),
             linear_node("s2", GraphAction(action="click", target=ladder, value=None))]
    artifact = build_linear(name="stale_replay", goal="dismiss the notice", params={},
                            surface_meta={"kind": "web", "app": "shop", "entry_url": ENTRY, "allowed_hosts": HOSTS},
                            nodes=nodes, outputs={}, outcomes=[], success={"text_contains": "Swag Labs"},
                            run_id="test-run", sensitive=set(), planner_name="scripted")
    surface.vanish()
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, {}, surface, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "failure" and result.outcome_code in ("stale_target", "target_not_found")
    assert not any(a[0] == "click" for a in surface.actions)
    assert result.performed_attempts == 1                        # the entry navigation alone


# ---------- the typed stuck cause reaches vision; other causes do not ----------

def test_a_missing_control_cause_is_typed_and_vision_eligible():
    screen = Observation(url=ENTRY, elements=[])
    missing = action_from_tool_input({"kind": "stuck", "stuck_cause": "missing_control", "reason": "free text"}, screen)
    assert missing.stuck_cause == "perception" and missing.stuck_cause in VISION_CAUSES
    for cause in ("other", None):
        ordinary = action_from_tool_input({"kind": "stuck", "stuck_cause": cause, "reason": "a control is missing"}, screen)
        assert ordinary.stuck_cause == "planner" and ordinary.stuck_cause not in VISION_CAUSES
    from src.cua.planner import provider_stuck
    assert provider_stuck("the provider failed").stuck_cause == "provider"
    assert "provider" not in VISION_CAUSES and "planner" not in VISION_CAUSES


def test_the_fallback_acts_on_the_typed_cause_and_never_on_reason_text():
    from src.cua.models import Action
    from tests.scripted_planner import NO_TARGET, ScriptedVisionPlanner

    surface = FakeSurface()
    planner = ScriptedVisionPlanner([], [NO_TARGET])
    log = RunLog("discovery", secrets=("secret_sauce",))
    fallback = agent_module.Vision(planner, 2, "goal", {}, {}, surface, log)
    # the wording says a control is missing, but the typed cause is the planner's own confusion
    assert fallback.propose(Action(kind="stuck", stuck_cause="planner",
                                   reason="a required visible control is missing from the observation"),
                            surface.observe(), []) is None
    assert planner.frames == []                                  # no screenshot was ever taken
    skipped = [json.loads(l) for l in log.path.read_text().splitlines() if '"vision_fallback_skipped"' in l]
    assert skipped and "not a missing or unactionable control" in skipped[0]["reason"]
    # the typed perception cause does reach the model, whatever its wording says
    fallback.propose(Action(kind="stuck", stuck_cause="perception", reason="anything at all"), surface.observe(), [])
    assert len(planner.frames) == 1
