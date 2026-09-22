"""A control whose selectedness lives only in a class token, in real Chromium.

Many custom controls carry their state nowhere a browser exposes it: no native selectedness, no ARIA, no
data attribute, no associated control. The class list is then the only evidence there is. It is read as
whole tokens against a closed vocabulary of words that mean selectedness, never as a string and never as
styling, and the token name is never written into the artifact. No model; a local file, no network."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import discover
from src.cua.artifact import linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.replay import replay
from tests.context import Escalator, Policy, RecordingOperator, RunLog, SessionControl
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]

# Each button starts with ordinary styling classes and adds (or removes) one token on click. The markup
# shape mirrors a real component library's: several presentational classes, then the state word.
PAGE = """<!doctype html><html><head><title>Periods</title><style>.hot { color: #900; }</style></head><body>
<h1>Periods</h1>
<p>Award Type</p>
<form aria-label="Choices">
  <button id="plain" class="button__sm button-type__tertiary-light" type="button">Plain year</button>
  <button id="dashed" class="btn btn--sm" type="button">Dashed year</button>
  <button id="styling" class="btn btn--sm" type="button">Styling year</button>
  <button id="trap1" class="btn unselected" type="button">Trap one</button>
  <button id="trap2" class="btn" type="button">Trap two</button>
  <button id="trap3" class="btn" type="button">Trap three</button>
  <button id="already" class="btn selected" type="button">Already year</button>
  <button id="drops" class="btn selected" type="button">Dropping year</button>
  <button id="many" class="btn" type="button">Many year</button>
  <button id="active" class="btn" type="button">Active year</button>
</form>
<script>
  const add = (id, token) => document.getElementById(id).addEventListener('click',
      (e) => e.currentTarget.classList.add(token));
  add('plain', 'selected');
  add('dashed', 'is-selected');
  add('styling', 'hot');                       // a presentational class, nothing to do with selection
  add('trap1', 'unselected');                  // substring of "selected", a different word
  add('trap2', 'selected-item');               // a compound token, not the state word
  add('trap3', 'button-selected-style');       // a styling token that merely contains the word
  add('active', 'active');                     // ambiguous: focused? running? enabled? never proof
  document.getElementById('many').addEventListener('click', (e) => {
    e.currentTarget.classList.add('chosen');
    e.currentTarget.classList.add('is-current');
  });
  document.getElementById('drops').addEventListener('click',
      (e) => e.currentTarget.classList.remove('selected'));
</script></body></html>"""


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    folder = tmp_path_factory.mktemp("classes")
    (folder / "periods.html").write_text(PAGE)
    return (folder / "periods.html").as_uri()


@pytest.fixture(scope="module")
def surface():
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=2500)
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    yield live
    live.close()


@pytest.fixture
def fresh(surface, site):
    surface.navigate(site)
    return surface


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 1.0)


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def control(fresh, name):
    return next(e for e in fresh.observe().elements if e.role == "button" and name in (e.name or ""))


def run(fresh, site, name, expect="Never shown on this page", operator=None):
    log = RunLog("discovery")
    steps = [ScriptedStep("click", "button", name, expect=expect), ScriptedStep("done", expect="Periods")]
    artifact = discover(goal="choose a period", name="period_choice", params={}, surface=fresh,
                        planner=ScriptedPlanner(steps), policy=Policy(allowed_hosts=[""]),
                        escalator=Escalator(operator or RecordingOperator("resume"), SessionControl(), log), log=log,
                        entry_url=site, max_steps=12)
    return artifact, log


def clicked(artifact, name):
    return [n for n in linear_path(artifact)
            if n.action.action == "click" and n.action.target.strategies[0].get("name") == name]


# ---------- the token is read, exactly ----------

@pytest.mark.parametrize("name, selector", [("Plain year", "#plain"), ("Dashed year", "#dashed")])
def test_an_exact_state_token_appearing_after_a_click_is_read_as_selected(fresh, name, selector):
    element = control(fresh, name)
    before = fresh.selected_state(element)
    assert before["known"] is False and before["class_token"] is False
    fresh._page.click(selector)
    after = fresh.selected_state(element)
    assert after["known"] is True and after["selected"] is True and after["source"] == "class-token"


def test_several_state_tokens_at_once_are_still_one_selected_reading(fresh):
    element = control(fresh, "Many year")
    fresh._page.click("#many")
    state = fresh.selected_state(element)
    assert state["known"] is True and state["selected"] is True and state["source"] == "class-token"


@pytest.mark.parametrize("name, selector", [
    ("Styling year", "#styling"),          # a presentational class
    ("Trap one", "#trap1"),                # "unselected": a different word
    ("Trap two", "#trap2"),                # "selected-item": a compound token
    ("Trap three", "#trap3"),              # "button-selected-style": styling that contains the word
    ("Active year", "#active"),            # "active": ambiguous, never proof on its own
])
def test_an_unrelated_or_trap_class_is_never_read_as_selected(fresh, name, selector):
    element = control(fresh, name)
    assert fresh.selected_state(element)["known"] is False
    fresh._page.click(selector)
    state = fresh.selected_state(element)
    assert state["known"] is False and state["selected"] is False


def test_a_control_that_already_carries_the_token_reads_as_selected_from_the_start(fresh):
    state = fresh.selected_state(control(fresh, "Already year"))
    assert state["known"] is True and state["selected"] is True


def test_a_token_being_removed_reads_as_no_longer_selected(fresh):
    element = control(fresh, "Dropping year")
    assert fresh.selected_state(element)["selected"] is True
    fresh._page.click("#drops")
    after = fresh.selected_state(element)
    assert after["known"] is False and after["selected"] is False


# ---------- discovery and replay ----------

def test_the_live_shape_with_no_expectation_records_target_selected_never_null(fresh, site):
    artifact, log = run(fresh, site, "Plain year", expect=None)
    [node] = clicked(artifact, "Plain year")
    assert node.action.checkpoint == {"target_selected": "true"}
    assert node.retry_safety == "verify_before_retry"
    assert events(log, "action_effect_proven")[0]["proof"] == "the control is now selected (class-token)"
    validate(artifact)


def test_the_live_shape_with_pre_existing_expected_text_records_target_selected_never_null(fresh, site):
    artifact, log = run(fresh, site, "Plain year", expect="Award Type")     # on screen from the start
    [node] = clicked(artifact, "Plain year")
    assert node.action.checkpoint == {"target_selected": "true"}
    assert events(log, "checkpoint_not_evidence")[0]["reason"] == "the text was already on screen before acting"


def test_no_class_name_is_ever_written_into_the_artifact(fresh, site):
    artifact, _ = run(fresh, site, "Plain year", expect=None)
    [node] = clicked(artifact, "Plain year")
    # The checkpoint is the abstract condition; the ladder names the control the way a person would.
    assert node.action.checkpoint == {"target_selected": "true"}
    written = json.dumps([node.action.checkpoint, [dict(s) for s in node.action.target.strategies]]).lower()
    for token in ("button__sm", "tertiary", "class", "is-selected", "chosen"):
        assert token not in written


def test_the_artifact_replays_in_real_chromium_without_a_model(fresh, site):
    artifact, _ = run(fresh, site, "Plain year", expect=None)
    fresh.navigate(site)
    log = RunLog("replay")
    result = replay(artifact, {}, fresh, Policy(allowed_hosts=[""]),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success"
    assert "selected" in (fresh._page.get_attribute("#plain", "class") or "")
    assert {"target_selected": "true"} in [e["checkpoint"] for e in events(log, "checkpoint_passed")]


def test_replay_fails_safely_when_the_token_never_appears(fresh, tmp_path):
    # Discovered against a page whose button commits; replayed against one where it does nothing.
    working = tmp_path / "working.html"
    working.write_text(PAGE)
    inert = tmp_path / "inert.html"
    inert.write_text(PAGE.replace("add('plain', 'selected');", ""))
    fresh.navigate(working.as_uri())
    artifact, _ = run(fresh, working.as_uri(), "Plain year", expect=None)
    artifact.surface["entry_url"] = inert.as_uri()
    artifact.nodes[0].action.value = inert.as_uri()
    log = RunLog("replay")
    result = replay(artifact, {}, fresh, Policy(allowed_hosts=[""]),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "failure" and result.outcome_code == "checkpoint_not_met"


def test_an_unrelated_class_change_is_not_proven_and_records_no_checkpoint(fresh, site):
    artifact, log = run(fresh, site, "Styling year", expect="Award Type")
    [node] = clicked(artifact, "Styling year")
    assert node.action.checkpoint is None
    assert events(log, "action_effect_proven") == []
