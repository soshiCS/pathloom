"""Checkpoints are evidence: text the action made appear, a typed value shown back, or a proven page change.
Text that was already on screen before the action proves nothing and is never recorded as its checkpoint."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import discover
from src.cua.artifact import linear_path, validate
from src.cua.escalation import Escalator, NoOperator, SessionControl
from src.cua.evidence import RunLog
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.context import HOSTS, PARAMS
from tests.fake_surface import ENTRY, FakeSurface
from tests.scripted_planner import ScriptedPlanner, ScriptedStep


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


LOGIN = [ScriptedStep("type", "textbox", "Username", value="{{username}}"),
         ScriptedStep("type", "textbox", "Password", value="{{password}}")]


def run_discovery(script, surface=None):
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover(goal="reach the cart", name="cart_visit", params=dict(PARAMS), surface=surface or FakeSurface(),
                        planner=ScriptedPlanner(script), policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(NoOperator(), SessionControl(), log), log=log, entry_url=ENTRY,
                        sensitive={"password"}, max_steps=20)
    return artifact, log


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def checkpoints(artifact):
    return [(n.action.action, n.action.checkpoint) for n in linear_path(artifact)]


def test_text_already_on_screen_is_not_recorded_as_the_checkpoint_of_a_typing_action():
    # "Swag Labs" is the site's banner: it is there before and after typing, so it proves nothing.
    artifact, log = run_discovery([ScriptedStep("type", "textbox", "Username", value="{{username}}", expect="Swag Labs"),
                                   ScriptedStep("done", expect="Swag Labs")])
    assert checkpoints(artifact)[1] == ("type", None)
    assert linear_path(artifact)[1].retry_safety == "never_retry"
    assert events(log, "expectation_failed") == []                     # the text was seen; it just is not evidence
    assert events(log, "checkpoint_not_evidence")[0]["expected"] == "Swag Labs"
    assert events(log, "recorded_step")[0]["checkpoint"] is None


def test_text_that_newly_appears_is_the_checkpoint():
    artifact, _ = run_discovery([*LOGIN, ScriptedStep("click", "button", "Login", expect="Products"),
                                 ScriptedStep("done", expect="Products")])
    login = linear_path(artifact)[3]
    assert login.action.checkpoint == {"text_contains": "Products"} and login.retry_safety == "verify_before_retry"


def test_a_typed_value_shown_back_by_the_control_is_the_proof_of_typing():
    artifact, _ = run_discovery([ScriptedStep("type", "textbox", "Username", value="{{username}}", expect="{{username}}"),
                                 ScriptedStep("done", expect="Swag Labs")])
    typed = linear_path(artifact)[1]
    assert typed.action.checkpoint == {"text_contains": "{{username}}"}     # parameterized, never the literal
    assert typed.retry_safety == "verify_before_retry"


def test_a_click_that_changes_the_page_is_proven_by_its_url_when_its_expected_text_was_already_there():
    # The product's name is on the inventory page and on the cart page: it cannot prove the cart opened. The
    # page change can, so the checkpoint becomes the new path, which replay verifies the same way.
    artifact, log = run_discovery([*LOGIN, ScriptedStep("click", "button", "Login", expect="Products"),
                                   ScriptedStep("click", "button", "Add to cart", context="{{product_name}}",
                                                expect="Products"),
                                   ScriptedStep("click", "link", "cart", expect="{{product_name}}"),
                                   ScriptedStep("done", expect="Your Cart")])
    validate(artifact)
    add, cart = linear_path(artifact)[4], linear_path(artifact)[5]
    assert add.action.checkpoint is None and add.retry_safety == "never_retry"   # same page, same banner: no proof
    assert cart.action.checkpoint == {"url_contains": "/cart.html"} and cart.retry_safety == "verify_before_retry"
    assert events(log, "checkpoint_from_url") == [{**events(log, "checkpoint_from_url")[0], "kind": "click",
                                                    "url_contains": "/cart.html"}]
    replay_log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(PARAMS), FakeSurface(), Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success"
    # a node with no checkpoint is logged as such, never as a verified one
    assert "s5" in [e["step_id"] for e in events(replay_log, "no_checkpoint_recorded")]
    assert [e["checkpoint"] for e in events(replay_log, "checkpoint_passed")][-1] == {"url_contains": "/cart.html"}


def test_a_real_navigation_is_proven_by_its_url_when_the_expected_text_is_generic():
    artifact, _ = run_discovery([*LOGIN, ScriptedStep("click", "button", "Login", expect="Products"),
                                 ScriptedStep("click", "button", "Add to cart", context="{{product_name}}"),
                                 ScriptedStep("navigate", value=ENTRY + "cart.html", expect="{{product_name}}"),
                                 ScriptedStep("done", expect="Your Cart")])
    navigate = linear_path(artifact)[5]
    assert navigate.action.action == "navigate" and navigate.action.checkpoint == {"url_contains": "/cart.html"}
    assert navigate.retry_safety == "verify_before_retry"


def test_a_contradicted_expectation_is_replaced_by_independent_proof_not_papered_over():
    artifact, log = run_discovery([*LOGIN, ScriptedStep("click", "button", "Login", expect="Checkout: Overview"),
                                   ScriptedStep("done", expect="Products")])
    login = linear_path(artifact)[3]
    assert events(log, "expectation_failed")[0]["expected"] == "Checkout: Overview"
    # the wrong guess is never recorded; the URL change the click really made is
    assert login.action.checkpoint == {"url_contains": "/inventory.html"}
    assert events(log, "action_effect_proven")[0]["proof"] == "the page address changed"


def test_the_planner_is_told_when_its_expected_text_was_not_evidence():
    seen = []

    class Spy(ScriptedPlanner):
        def decide(self, goal, params, observation, history, candidates=()):
            seen.append([a.result for a in history])
            return super().decide(goal, params, observation, history, candidates)

    log = RunLog("discovery", secrets=("secret_sauce",))
    discover(goal="g", name="n", params=dict(PARAMS), surface=FakeSurface(),
             planner=Spy([ScriptedStep("type", "textbox", "Username", value="{{username}}", expect="Swag Labs"),
                          ScriptedStep("done", expect="Swag Labs")]),
             policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(NoOperator(), SessionControl(), log), log=log,
             entry_url=ENTRY, sensitive={"password"})
    assert seen[1] == ["done; 'Swag Labs' was already on screen before the action, so it is not proof the action "
                       "worked and no checkpoint was recorded; next time expect text that only appears afterwards"]


# ---------- a checkpoint must be false before the action and true after it ----------

from src.cua.agent import url_checkpoint                                    # noqa: E402
from src.cua.models import Action, Element, Observation                    # noqa: E402

BASE = "https://shop.test/earthquakes/map/"


def proof(before_url, after_url, params=None):
    found = url_checkpoint(before_url, after_url, params or {})
    if found is None:
        return None
    found.pop("_part", None)
    return found


def holds(checkpoint, url, params=None):
    """What replay would conclude: a url_contains checkpoint is a substring test on the whole URL."""
    from src.cua.artifact import substitute

    return substitute(checkpoint["url_contains"], params or {}) in url


def test_a_query_only_change_never_records_the_unchanged_path():
    found = proof(BASE + "?a=1", BASE + "?a=1&range=week")
    assert found is not None and found != {"url_contains": "/earthquakes/map/"}
    assert not holds(found, BASE + "?a=1") and holds(found, BASE + "?a=1&range=week")


def test_a_fragment_only_change_never_records_the_unchanged_path():
    found = proof(BASE, BASE + "#settings")
    assert found == {"url_contains": "#settings"}
    assert not holds(found, BASE) and holds(found, BASE + "#settings")


def test_a_genuine_path_change_is_false_before_and_true_after():
    found = proof(BASE, "https://shop.test/earthquakes/detail/")
    assert found == {"url_contains": "/earthquakes/detail/"}
    assert not holds(found, BASE) and holds(found, "https://shop.test/earthquakes/detail/")


def test_an_unchanged_url_and_an_unrepresentable_change_record_nothing():
    assert proof(BASE, BASE) is None                                   # nothing changed at all
    assert proof(BASE + "?token=abc", BASE + "?token=def") is None     # only an unstable, sensitive key
    assert proof(BASE + "?utm_source=a", BASE + "?utm_source=b") is None
    assert proof(BASE + "?sid=1", BASE + "?sid=2") is None


def test_secrets_and_parameters_never_leak_through_a_url_checkpoint():
    params = {"city": "Boston", "password": "secret_sauce"}
    assert proof(BASE, BASE + "?password=secret_sauce") is None        # a sensitive key is never asserted
    assert proof(BASE, BASE + "?api_key=abcd1234efgh") is None         # nor a credential-shaped value
    # a registered secret is masked by redaction wherever it appears, so it can never become a checkpoint
    from src.cua.policy import redact
    assert redact("q=secret_sauce", ("secret_sauce",)) != "q=secret_sauce"
    assert proof(BASE, BASE + "?q=hunter2%s" % "abcdefghij1234") is None
    # a declared input's value is always stored as its placeholder, sensitive or not: no literal survives
    declared = proof(BASE, BASE + "?q=secret_sauce", params)
    assert declared == {"url_contains": "q={{password}}"} and "secret_sauce" not in str(declared)
    found = proof(BASE, BASE + "?where=Boston", params)
    assert found == {"url_contains": "where={{city}}"}                 # the input becomes its placeholder
    assert not holds(found, BASE, params) and holds(found, BASE + "?where=Boston", params)


def test_a_performed_click_with_no_representable_proof_is_recorded_conservatively():
    # the product's name is on both screens and only an unstable query key changes: no checkpoint at all
    artifact, log = run_discovery([*LOGIN, ScriptedStep("click", "button", "Login", expect="Products"),
                                   ScriptedStep("click", "button", "Add to cart", context="{{product_name}}",
                                                expect="Products"),
                                   ScriptedStep("done", expect="Products")])
    add = linear_path(artifact)[4]
    assert add.action.checkpoint is None
    assert (add.effect, add.retry_safety) == ("reversible", "never_retry")     # never repeated blindly


def test_replay_cannot_accept_a_failed_action_with_an_already_true_checkpoint():
    from src.cua.replay import checkpoint_met

    before = Observation(url=BASE + "?a=1", elements=[])
    stale = {"url_contains": "/earthquakes/map/"}
    assert checkpoint_met(stale, before, {})                    # the old rule would have recorded this
    fresh = proof(BASE + "?a=1", BASE + "?a=1&range=week")
    assert not checkpoint_met(fresh, before, {})                # the new rule never records a predicate like that
    after = Observation(url=BASE + "?a=1&range=week", elements=[])
    assert checkpoint_met(fresh, after, {})


def test_proof_after_failure_follows_the_same_invariant():
    import json

    from src.cua.agent import proof_after_failure

    class Screen:
        def __init__(self, url):
            self.url = url

        def observe(self):
            return Observation(url=self.url, elements=[])

    action = Action(kind="click", target=Element(role="button", name="Settings"), expect="Earthquakes")
    log = RunLog("discovery")
    # only the query changed, and the unchanged path is not proof: the lost action stays unproven
    before = Observation(url=BASE, elements=[Element(role="text", name="", text="Earthquakes")])
    assert proof_after_failure(action, "Earthquakes", before, Screen(BASE + "?range=week"), log, {}) == "url"
    assert proof_after_failure(action, "Earthquakes", before, Screen(BASE + "?sid=9"), log, {}) is None
    assert proof_after_failure(action, "Earthquakes", before, Screen(BASE), log, {}) is None


def test_navigation_and_back_still_record_their_page_change():
    artifact, _ = run_discovery([*LOGIN, ScriptedStep("click", "button", "Login", expect="Products"),
                                 ScriptedStep("click", "button", "Add to cart", context="{{product_name}}"),
                                 ScriptedStep("navigate", value=ENTRY + "cart.html", expect="{{product_name}}"),
                                 ScriptedStep("back", expect="Products"),
                                 ScriptedStep("done", expect="Products")])
    path = linear_path(artifact)
    navigate = next(n for n in path if n.action.action == "navigate" and n.id != "s1")
    assert navigate.action.checkpoint == {"url_contains": "/cart.html"}
    back = next(n for n in path if n.action.action == "back")
    assert back.action.checkpoint == {"text_contains": "Products"}


def test_production_code_carries_no_audit_specific_names():
    from pathlib import Path

    source = Path("src/cua/agent.py").read_text()
    for banned in ("earthquake", "usgs", "shop.test", "openstreetmap", "clinicaltrials", "usaspending"):
        assert banned not in source.lower(), banned
