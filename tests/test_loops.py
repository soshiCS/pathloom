"""Loop detection: a planner repeating itself without progress gets one warning, then the structured handoff."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua import artifact as artifact_module
from src.cua.agent import LOOP_STOP_REPEATS, DiscoveryFailed, ProgressTracker, cycles_at_tail, discover
from src.cua.artifact import linear_path
from src.cua.escalation import NoOperator
from src.cua.models import InterventionResult
from src.cua.policy import Policy
from src.cua.models import Action
from tests.context import ENTRY, HOSTS, PARAMS, Escalator, RecordingOperator, RunLog, SessionControl
from tests.fake_surface import FakeSurface, el
from tests.scripted_planner import MONEY, ScriptedPlanner, ScriptedStep, checkout_script

LOGIN = checkout_script()[:next(i for i, step in enumerate(checkout_script()) if step.name == "Login") + 1]   # -> inventory
TYPE_USER = ScriptedStep("type", "textbox", "Username", value="{{username}}")
ADD = ScriptedStep("click", "button", "Add to cart", context="{{product_name}}")
REMOVE = ScriptedStep("click", "button", "Remove", context="{{product_name}}")


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


class Spy(ScriptedPlanner):
    def __init__(self, script):
        super().__init__(script)
        self.seen: list[list[str]] = []

    def decide(self, goal, params, observation, history, candidates=()):
        self.seen.append([a.result for a in history])
        return super().decide(goal, params, observation, history, candidates)


def run(script, surface=None, operator=None, max_steps=20):
    planner = Spy(script)
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover(goal="reach the cart", name="looping", params=dict(PARAMS), surface=surface or FakeSurface(),
                        planner=planner, policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(operator or NoOperator(), SessionControl(), log), log=log, entry_url=ENTRY,
                        sensitive={"password"}, max_steps=max_steps)
    return artifact, planner, log


def decisions(log):
    return [(json.loads(l)["decision"], json.loads(l)["turn"]) for l in log.path.read_text().splitlines()
            if '"event": "planner_loop_detected"' in l]


def test_typing_the_same_value_into_the_same_field_is_warned_then_stopped_without_saving():
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run([TYPE_USER] * 6, operator=operator)
    request = operator.requests[0]
    assert request.reason.startswith("planner is stuck: no progress: the same action repeated 3 times")
    assert not artifact_module.ARTIFACTS_DIR.exists()


def test_the_warning_reaches_the_planner_once_and_the_stop_comes_at_the_third_repeat():
    planner = Spy([TYPE_USER] * 6)
    log = RunLog("discovery", secrets=("secret_sauce",))
    with pytest.raises(DiscoveryFailed):
        discover(goal="g", name="looping", params=dict(PARAMS), surface=FakeSurface(), planner=planner,
                 policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(RecordingOperator("abort"), SessionControl(), log),
                 log=log, entry_url=ENTRY, sensitive={"password"}, max_steps=20)
    # the first typing changes the field (progress of a kind); the second and third repeat an unchanged screen
    assert decisions(log) == [("warn", 3), ("stop", 4)]
    warned = [r for history in planner.seen for r in history if "WARNING" in r]
    assert len({*warned}) == 1 and "repeats the last 1 action(s) for the 2nd time" in warned[0]
    assert len(planner.seen) == 4                                # four decisions, far below max_steps


def test_two_states_alternating_are_a_cycle():
    surface = FakeSurface(extra_elements=[el("button", "Hide"), el("button", "Show")])   # toggles that change nothing lasting
    planner = Spy([ScriptedStep("click", "button", "Hide"), ScriptedStep("click", "button", "Show")] * 4)
    log = RunLog("discovery", secrets=("secret_sauce",))
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        discover(goal="g", name="looping", params=dict(PARAMS), surface=surface, planner=planner,
                 policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(RecordingOperator("abort"), SessionControl(), log),
                 log=log, entry_url=ENTRY, sensitive={"password"}, max_steps=25)
    assert decisions(log) == [("warn", 4), ("stop", 6)]
    assert json.loads([l for l in log.path.read_text().splitlines() if '"planner_loop_detected"' in l][-1])["period"] == 2
    assert len(planner.seen) == 6 and sum(1 for a in surface.actions if a[0] == "click") == 5
    warned = {r for history in planner.seen for r in history if "WARNING" in r}
    assert len(warned) == 1 and "repeats the last 2 action(s)" in next(iter(warned))


def test_different_targets_values_and_outputs_are_not_a_loop():
    script = LOGIN + [ScriptedStep("click", "button", "Add to cart", context="Sauce Labs Backpack"),
                      ScriptedStep("click", "button", "Add to cart", context="Sauce Labs Bike Light"),
                      ScriptedStep("type", "textbox", "First Name", value="a", skip_if_absent=True),
                      ScriptedStep("click", "link", "cart", expect="Your Cart"),
                      ScriptedStep("click", "button", "Checkout", expect="Checkout: Your Information"),
                      ScriptedStep("type", "textbox", "First Name", value="{{first_name}}"),
                      ScriptedStep("type", "textbox", "Last Name", value="{{last_name}}"),
                      ScriptedStep("type", "textbox", "Zip/Postal Code", value="{{postal_code}}"),
                      ScriptedStep("click", "button", "Continue", expect="Checkout: Overview"),
                      ScriptedStep("extract", "text", text="Item total:", output_name="subtotal", pattern=MONEY),
                      ScriptedStep("extract", "text", text="Tax:", output_name="tax", pattern=MONEY),
                      ScriptedStep("extract", "text", text="Total:", output_name="total", pattern=MONEY),
                      ScriptedStep("done", expect="Checkout: Overview")]
    artifact, planner, log = run(script)
    assert decisions(log) == [] and not any("WARNING" in r for h in planner.seen for r in h)
    assert sorted(artifact.outputs) == ["subtotal", "tax", "total"]


def test_recording_an_output_resets_the_window():
    """Only a new output clears the loop history: a checkpoint or a URL change is what a cycle produces."""
    reach_overview = checkout_script()[:11]                     # login .. Continue: the overview screen
    total = ScriptedStep("extract", "text", text="Total:", output_name="total", pattern=MONEY)
    # a real two-screen cycle: the overview's Cancel goes back to the products list, the cart link returns
    cancel = ScriptedStep("click", "button", "Cancel", expect="Products")
    cart = ScriptedStep("click", "link", "cart", expect="Your Cart")
    checkout = ScriptedStep("click", "button", "Checkout", expect="Checkout: Your Information")

    # without an output between them, three alternations of the same two clicks stop the run
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run(reach_overview + [cancel, cart, checkout] * 4, operator=operator, max_steps=40)
    assert operator.requests[0].reason.startswith("planner is stuck: no progress: the same 3 actions repeated 3 times")

    # with an output recorded first, the window is cleared and the same pair is not yet a stop
    artifact, planner, log = run(reach_overview + [total, cancel, cart, checkout, cancel, cart, checkout,
                                                   ScriptedStep("done", expect="Checkout: Your Information")],
                                 max_steps=30)
    assert [json.loads(l)["why"] for l in log.path.read_text().splitlines() if '"progress_made"' in l] == [
        "output recorded"]
    assert sorted(artifact.outputs) == ["total"] and decisions(log) == []      # the window started over


def test_an_action_whose_effect_may_not_be_reversible_is_not_repeated_even_once():
    class Approver:
        def __init__(self):
            self.requests = []

        def handle(self, request, surface):
            self.requests.append(request)
            if request.kind == "confirm":
                return InterventionResult(resolved=True, human_actions=[], note="ok", disposition="approve")
            return InterventionResult(resolved=False, human_actions=[], note="stop", disposition="abort")

    surface = FakeSurface(extra_elements=[el("button", "Close")])           # ambiguous: no container evidence
    operator = Approver()
    planner = Spy([ScriptedStep("click", "button", "Close")] * 3)
    log = RunLog("discovery", secrets=("secret_sauce",))
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        discover(goal="g", name="looping", params=dict(PARAMS), surface=surface, planner=planner,
                 policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(operator, SessionControl(), log), log=log,
                 entry_url=ENTRY, sensitive={"password"}, max_steps=10)
    assert sum(1 for a in surface.actions if a[0] == "click") == 1        # confirmed once, never repeated blindly
    assert [r.kind for r in operator.requests] == ["confirm", "stuck"]
    assert decisions(log) == [("stop", 2)]


def test_cycle_counting_is_exact():
    assert cycles_at_tail([1, 1, 1], 1) == 3 and cycles_at_tail([2, 1, 1], 1) == 2 and cycles_at_tail([1, 2, 1, 2], 2) == 2
    assert cycles_at_tail([1, 2, 1, 2, 1], 2) == 2 and cycles_at_tail([1], 2) == 0 and cycles_at_tail([], 1) == 0
    tracker = ProgressTracker({"username": "standard_user"})
    assert LOOP_STOP_REPEATS == 3


# ---------- a two-screen cycle whose steps each pass a checkpoint and change the page ----------

SEARCH = ScriptedStep("click", "link", "cart", expect="Your Cart")            # products -> cart
REFINE = ScriptedStep("click", "button", "Continue Shopping", expect="Products")   # cart -> products


def alternating(times, search=SEARCH, refine=REFINE):
    return LOGIN + [step for _ in range(times) for step in (search, refine)]


def performed(surface, name):
    return sum(1 for a in surface.actions if a[0] == "click" and a[1] == name)


def test_a_two_screen_cycle_is_stopped_early_even_though_every_step_verifies():
    surface = FakeSurface()
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run(alternating(6), surface=surface, operator=operator, max_steps=40)
    reason = operator.requests[0].reason
    assert reason.startswith("planner is stuck: no progress: the same 2 actions repeated 3 times")
    # every step of the cycle passed its own checkpoint and changed the page, and it was still caught
    assert performed(surface, "cart") == 3 and performed(surface, "Continue Shopping") == 2


def test_the_cycle_is_caught_well_before_the_step_budget():
    surface = FakeSurface()
    planner = Spy(alternating(8))
    log = RunLog("discovery", secrets=("secret_sauce",))
    with pytest.raises(DiscoveryFailed):
        discover(goal="reach the cart", name="looping", params=dict(PARAMS), surface=surface, planner=planner,
                 policy=Policy(allowed_hosts=HOSTS),
                 escalator=Escalator(RecordingOperator("abort"), SessionControl(), log), log=log, entry_url=ENTRY,
                 sensitive={"password"}, max_steps=40)
    assert len(planner.seen) == len(LOGIN) + 5          # login, two full cycles, then the fifth click stops
    assert len(planner.seen) < 40                       # far short of the step budget
    assert [d for d, _ in decisions(log)] == ["warn", "stop"]


def test_rewording_the_reason_or_the_expectation_does_not_evade_detection():
    varied = []
    for index in range(6):
        varied.append(ScriptedStep("click", "link", "cart", expect="Your Cart" if index % 2 else "Cart"))
        varied.append(ScriptedStep("click", "button", "Continue Shopping",
                                   expect="Products" if index % 2 else "Sauce Labs Backpack"))
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run(LOGIN + varied, operator=operator, max_steps=40)
    assert operator.requests[0].reason.startswith("planner is stuck: no progress: the same 2 actions repeated 3 times")


def test_the_repeated_action_is_not_performed_again_once_the_cycle_is_proven():
    surface = FakeSurface()
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run(alternating(6), surface=surface, operator=operator, max_steps=40)
    clicks = [a for a in surface.actions if a[0] == "click" and a[1] in ("cart", "Continue Shopping")]
    assert len(clicks) == 5                       # the sixth, which proved the cycle, never reached the surface


def test_a_result_detail_result_workflow_with_back_and_outputs_is_not_a_loop():
    """Opening a page, recording something new there and coming back is the shape of real work, not a cycle."""
    rounds = []
    for index, name in enumerate(("Sauce Labs Backpack", "Sauce Labs Bike Light")):
        rounds += [ScriptedStep("click", "button", "Add to cart", context=name),
                   ScriptedStep("click", "link", "cart", expect="Your Cart"),
                   ScriptedStep("extract", "link", name=f"View details for {name}", output_name=f"item_{index}",
                                pattern="(.+)"),
                   ScriptedStep("back", expect="Products")]
    artifact, planner, log = run(LOGIN + rounds + [ScriptedStep("done", expect="Products")], max_steps=40)
    assert decisions(log) == [] and sorted(artifact.outputs) == ["item_0", "item_1"]
    assert [json.loads(l)["why"] for l in log.path.read_text().splitlines() if '"progress_made"' in l] == [
        "output recorded"] * 2


def test_the_same_action_on_materially_different_screens_is_not_a_loop():
    # "Add to cart" clicked on two different products: same wording, different enclosing item each time
    script = LOGIN + [ScriptedStep("click", "button", "Add to cart", context="Sauce Labs Backpack"),
                      ScriptedStep("click", "button", "Add to cart", context="Sauce Labs Bike Light"),
                      ScriptedStep("click", "link", "cart", expect="Your Cart"),
                      ScriptedStep("done", expect="Your Cart")]
    artifact, planner, log = run(script)
    assert decisions(log) == [] and artifact.name == "looping"


def test_a_restart_clears_the_attempt_local_loop_state():
    """A restart abandons the attempt; the fresh one starts with an empty window and is not stopped by it."""
    log = RunLog("discovery", secrets=("secret_sauce",))
    tracker = ProgressTracker(dict(PARAMS))
    surface = FakeSurface()
    surface.navigate(ENTRY)
    observation = surface.observe()
    action = Action(kind="click", target=next(e for e in observation.elements if e.name == "Login"))
    for _ in range(2):
        tracker.record(action, observation)
    assert tracker.check(action, observation)[0] == "stop"
    tracker.reset("human intervened", log)              # what a restart or a handoff does to the window
    assert tracker.entries == [] and tracker.warned is False
    assert tracker.check(action, observation)[0] is None
    assert [json.loads(l)["why"] for l in log.path.read_text().splitlines() if '"progress_made"' in l] == [
        "human intervened"]


def test_a_looping_discovery_saves_no_artifact_and_fails_structurally():
    import src.cua.artifact as artifact_module

    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted discovery") as failed:
        run(alternating(6), operator=operator, max_steps=40)
    assert not artifact_module.ARTIFACTS_DIR.exists()
    assert operator.requests[0].kind == "stuck"
    assert "no progress" in operator.requests[0].reason
