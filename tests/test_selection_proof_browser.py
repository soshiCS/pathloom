"""Selectable controls in real Chromium: native, ARIA and custom, proven only by browser state.

The page offers one of each shape a real filter panel uses, plus a control whose click changes only its
colour. No model; a local file, no network."""
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

PAGE = """<!doctype html><html><head><title>Periods</title><style>
  .on { background: #123; color: #fff; }
</style></head><body>
<h1>Periods</h1>
<form aria-label="Choices">
  <p><label for="box">Include drafts</label> <input id="box" type="checkbox"></p>
  <p><label for="radio">Quarterly</label> <input id="radio" type="radio" name="span"></p>
  <p><label for="pick">Range</label>
     <select id="pick"><option value="">None</option><option value="y">Year</option></select></p>
  <button id="pressed" type="button" aria-pressed="false">Compact</button>
  <div id="opt" role="option" aria-selected="false" tabindex="0">Fiscal 2025</div>
  <button id="custom" type="button" data-selected="false">Custom chip</button>
  <button id="styled" type="button">Styling only</button>
  <button id="wrapper" type="button" aria-controls="backing">Backed</button>
  <input id="backing" type="hidden" value="">
  <button id="plain" type="button">Plain</button>
</form>
<script>
  const flip = (el, name) => el.setAttribute(name, el.getAttribute(name) === 'true' ? 'false' : 'true');
  document.getElementById('pressed').addEventListener('click', (e) => flip(e.currentTarget, 'aria-pressed'));
  document.getElementById('opt').addEventListener('click', (e) => flip(e.currentTarget, 'aria-selected'));
  document.getElementById('custom').addEventListener('click', (e) => flip(e.currentTarget, 'data-selected'));
  document.getElementById('styled').addEventListener('click', (e) => e.currentTarget.classList.toggle('on'));
  document.getElementById('wrapper').addEventListener('click', () => {
    const backing = document.getElementById('backing');
    backing.value = backing.value ? '' : 'chosen';
  });
</script></body></html>"""


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    folder = tmp_path_factory.mktemp("periods")
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


def control(surface, text, roles=("button", "checkbox", "radio", "combobox", "option", "link")):
    """The interactive control with this name, not the label text that happens to read the same."""
    elements = surface.observe().elements
    named = [e for e in elements if e.role in roles and (text in (e.name or "") or text in (e.text or ""))]
    if named:
        return named[0]
    return next(e for e in elements if text in (e.name or "") or text in (e.text or ""))


def run(fresh, site, name, role="button", operator=None, expect="Never shown on this page"):
    log = RunLog("discovery")
    steps = [ScriptedStep("click", role, name, expect=expect),
             ScriptedStep("done", expect="Periods")]
    artifact = discover(goal="choose a period", name="period_choice", params={}, surface=fresh,
                        planner=ScriptedPlanner(steps), policy=Policy(allowed_hosts=[""]),
                        escalator=Escalator(operator or RecordingOperator("resume"), SessionControl(), log), log=log,
                        entry_url=site, max_steps=12)
    return artifact, log


def clicked(artifact, name):
    return [n for n in linear_path(artifact)
            if n.action.action == "click" and n.action.target.strategies[0].get("name") == name]


# ---------- the state reader, control shape by control shape ----------

@pytest.mark.parametrize("text, selector, source", [
    ("Include drafts", "#box", "checked"),
    ("Quarterly", "#radio", "checked"),
    ("Compact", "#pressed", "aria-pressed"),
    ("Fiscal 2025", "#opt", "aria-selected"),
    ("Custom chip", "#custom", "data-selected"),
])
def test_the_selected_state_is_read_from_browser_state_for_each_control_shape(fresh, text, selector, source):
    element = control(fresh, text)
    before = fresh.selected_state(element)
    assert before["known"] is True and before["selected"] is False and before["source"] == source
    fresh._page.click(selector)
    after = fresh.selected_state(element)
    assert after["known"] is True and after["selected"] is True and after["source"] == source


def test_a_label_reports_the_state_of_the_input_it_drives(fresh):
    label = next(e for e in fresh.observe().elements if (e.text or "") == "Include drafts")
    fresh._page.check("#box")
    state = fresh.selected_state(label)
    assert state["known"] is True and state["selected"] is True and "checked" in state["source"]


def test_a_native_option_reports_its_own_selectedness(fresh):
    # A closed select does not list its options as controls, so the option is addressed structurally,
    # exactly as a locator's css rung would address it.
    from src.cua.models import Element

    fresh._page.select_option("#pick", "y")
    option = Element(role="option", name="Year", text="Year",
                     ref="body:nth-of-type(1) > form:nth-of-type(1) > p:nth-of-type(3) > "
                         "select:nth-of-type(1) > option:nth-of-type(2)")
    state = fresh.selected_state(option)
    assert state["known"] is True and state["selected"] is True and state["source"] == "selected"


def test_a_control_backed_by_a_hidden_input_reports_that_inputs_state(fresh):
    backed = control(fresh, "Backed")
    assert fresh.selected_state(backed)["selected"] is False
    fresh._page.click("#wrapper")
    state = fresh.selected_state(backed)
    assert state["known"] is True and state["selected"] is True and state["source"] == "partner-stored"


def test_a_control_with_no_state_at_all_reports_unknown(fresh):
    assert fresh.selected_state(control(fresh, "Plain"))["known"] is False


def test_a_control_that_only_changes_colour_never_reports_selected(fresh):
    styled = control(fresh, "Styling only")
    assert fresh.selected_state(styled)["known"] is False
    fresh._page.click("#styled")
    assert fresh.selected_state(styled)["known"] is False     # the class changed; no state did


# ---------- discovery and replay ----------

@pytest.mark.parametrize("name, role", [("Compact", "button"), ("Custom chip", "button"),
                                        ("Include drafts", "checkbox"), ("Quarterly", "radio")])
def test_a_real_selection_proves_the_click_and_records_one_node(fresh, site, name, role):
    artifact, log = run(fresh, site, name, role=role)
    [node] = clicked(artifact, name)
    assert node.action.checkpoint == {"target_selected": "true"}
    assert "the control is now selected" in events(log, "action_effect_proven")[0]["proof"]
    validate(artifact)


def test_the_artifact_replays_in_real_chromium_without_a_model(fresh, site):
    artifact, _ = run(fresh, site, "Compact")
    fresh.navigate(site)
    log = RunLog("replay")
    result = replay(artifact, {}, fresh, Policy(allowed_hosts=[""]),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success"
    assert fresh._page.get_attribute("#pressed", "aria-pressed") == "true"
    assert {"target_selected": "true"} in [e["checkpoint"] for e in events(log, "checkpoint_passed")]


def test_a_styling_only_click_is_not_proven_and_is_not_repeated(fresh, site):
    artifact, log = run(fresh, site, "Styling only", operator=RecordingOperator("resume"))
    assert clicked(artifact, "Styling only") == []
    assert events(log, "action_effect_unverified")[0]["cause"] == "action_effect_unverified"
    assert events(log, "unverified_repeat_refused") != []


def test_a_control_already_selected_before_the_click_is_not_proven(fresh):
    # Deselecting is a real transition too, but it is not "became selected", so it is not this proof.
    from src.cua.agent import selection_transition
    from src.cua.models import Action

    fresh._page.click("#pressed")                 # on before the action under test
    element = control(fresh, "Compact")
    action = Action(kind="click", target=element)
    action.selected_before = fresh.selected_state(element)
    assert action.selected_before["selected"] is True
    fresh._page.click("#pressed")                 # the click turns it off
    log = RunLog("discovery")
    assert selection_transition(action, fresh, log) is None
    assert events(log, "selection_not_proven")[-1]["reason"] == "the control was already selected"


# ---------- the planner's expectation never decides whether a selection is proof ----------

def test_a_pre_existing_expected_text_does_not_hide_a_real_selection(fresh, site):
    """The live shape: the expected text was already on screen, so it proves nothing and the selection does."""
    artifact, log = run(fresh, site, "Compact", expect="Periods")     # the page heading, present all along
    [node] = clicked(artifact, "Compact")
    assert node.action.checkpoint == {"target_selected": "true"}      # never a null checkpoint
    assert events(log, "checkpoint_not_evidence")[0]["reason"] == "the text was already on screen before acting"
    assert events(log, "action_effect_proven")[0]["proof"] == "the control is now selected (aria-pressed)"
    validate(artifact)


def test_a_real_selection_with_no_expectation_is_recorded_and_replays(fresh, site):
    artifact, log = run(fresh, site, "Custom chip", expect=None)
    [node] = clicked(artifact, "Custom chip")
    assert node.action.checkpoint == {"target_selected": "true"}
    assert events(log, "checkpoint_not_evidence") == []
    fresh.navigate(site)
    replay_log = RunLog("replay")
    result = replay(artifact, {}, fresh, Policy(allowed_hosts=[""]),
                    Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success"
    assert fresh._page.get_attribute("#custom", "data-selected") == "true"


def test_a_controls_own_label_is_never_proof_but_its_selection_is(fresh, site):
    # A control's own label is on screen before and after: expecting it proves nothing about the click.
    artifact, log = run(fresh, site, "Compact", expect="Compact")
    [node] = clicked(artifact, "Compact")
    assert node.action.checkpoint == {"target_selected": "true"}
    assert events(log, "checkpoint_not_evidence") != []


def test_a_styling_only_click_with_pre_existing_text_records_no_checkpoint(fresh, site):
    artifact, log = run(fresh, site, "Styling only", expect="Periods")
    [node] = clicked(artifact, "Styling only")
    assert node.action.checkpoint is None            # nothing about it changed, so there is nothing to verify
    assert node.retry_safety == "never_retry"
    assert events(log, "action_effect_proven") == []
