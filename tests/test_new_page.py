"""A click that opens one new page makes it the active page; several or a disallowed destination fail
structurally. Offline, through the fake surface's popup support."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.context import ENTRY, HOSTS, PARAMS, Escalator, RecordingOperator, RunLog, SessionControl
from tests.fake_surface import FakeSurface, el
from tests.scripted_planner import MONEY, ScriptedPlanner, ScriptedStep, checkout_script

LOGIN = checkout_script()[:next(i for i, s in enumerate(checkout_script()) if s.name == "Login") + 1]
DETAILS = [el("link", f"View details for {name}") for name in ("Sauce Labs Backpack", "Sauce Labs Bike Light")]


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


def shop(opens=None) -> FakeSurface:
    """The shop where each product's details link opens its own new page (the cart page stands in for it)."""
    surface = FakeSurface()
    surface.opens_page = dict(opens or {})
    return surface


def run(script, surface, operator=None):
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover(goal="read the cart from each detail page", name="detail_pages", params=dict(PARAMS),
                        surface=surface, planner=ScriptedPlanner(script), policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(operator or NoOperator(), SessionControl(), log), log=log,
                        entry_url=ENTRY, sensitive={"password"}, max_steps=20)
    return artifact, log


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


OPEN_ONE = {"View details for Sauce Labs Backpack": ("cart", ENTRY + "cart.html")}


def test_a_click_that_opens_one_new_page_makes_it_active_and_its_fields_readable():
    surface = shop(OPEN_ONE)
    artifact, log = run(LOGIN + [ScriptedStep("click", "link", "View details for Sauce Labs Backpack",
                                              expect="Your Cart"),
                                 ScriptedStep("done", expect="Your Cart")], surface)
    validate(artifact)
    assert surface.screen == "cart" and surface.opener == ("inventory", ENTRY + "inventory.html")
    click = linear_path(artifact)[4]
    assert click.action.action == "click" and click.action.checkpoint == {"text_contains": "Your Cart"}
    assert sum(1 for a in surface.actions if a[0] == "click" and a[1].startswith("View details")) == 1


def test_back_closes_the_new_page_and_returns_to_the_results_page():
    surface = shop(OPEN_ONE)
    artifact, _ = run(LOGIN + [ScriptedStep("click", "link", "View details for Sauce Labs Backpack", expect="Your Cart"),
                               ScriptedStep("back", expect="Products"),
                               ScriptedStep("done", expect="Products")], surface)
    assert surface.screen == "inventory" and surface.opener is None
    assert [n.action.action for n in linear_path(artifact)] == ["navigate", "type", "type", "click", "click", "back"]


def test_three_detail_pages_are_visited_in_order_and_replay_clicks_each_once():
    opens = {e.name: ("cart", ENTRY + "cart.html") for e in DETAILS}
    script = list(LOGIN)
    for element in DETAILS:
        script += [ScriptedStep("click", "link", element.name, expect="Your Cart"),
                   ScriptedStep("extract", "text", text="Your Cart", output_name=f"seen_{len(script)}",
                                pattern=r"(Your Cart)"),
                   ScriptedStep("back", expect="Products")]
    script.append(ScriptedStep("done", expect="Products"))
    artifact, _ = run(script, shop(opens))
    validate(artifact)
    fresh = shop(opens)
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(PARAMS), fresh, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success" and len(result.outputs) == 2
    for element in DETAILS:
        assert sum(1 for a in fresh.actions if a[0] == "click" and a[1] == element.name) == 1
    assert fresh.screen == "inventory" and fresh.opener is None


def test_several_new_pages_fail_structurally_without_choosing_one():
    surface = shop({"View details for Sauce Labs Backpack": 2})
    operator = RecordingOperator("abort")
    step = ScriptedStep("click", "link", "View details for Sauce Labs Backpack", expect="Your Cart")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run(LOGIN + [step, step], surface, operator=operator)
    assert surface.screen == "inventory"
    assert operator.requests[0].reason.startswith("planner is stuck: the same click on link 'View details")
    assert "opened 2 new pages" in operator.requests[0].reason


def test_a_destination_outside_the_allowlist_fails_closed():
    surface = shop({"View details for Sauce Labs Backpack": "elsewhere.example"})
    operator = RecordingOperator("abort")
    step = ScriptedStep("click", "link", "View details for Sauce Labs Backpack", expect="Your Cart")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run(LOGIN + [step, step], surface, operator=operator)
    assert surface.screen == "inventory" and surface.opener is None
    assert "outside the allowlist" in operator.requests[0].reason


def test_an_ordinary_same_page_click_is_unchanged():
    surface = shop()
    artifact, log = run(LOGIN + [ScriptedStep("click", "button", "Add to cart", context="{{product_name}}"),
                                 ScriptedStep("click", "link", "cart", expect="Your Cart"),
                                 ScriptedStep("done", expect="Your Cart")], surface)
    assert surface.opener is None and surface.screen == "cart"
    assert [n.action.action for n in linear_path(artifact)] == ["navigate", "type", "type", "click", "click", "click"]
