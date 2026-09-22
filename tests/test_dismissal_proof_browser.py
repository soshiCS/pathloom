"""Dismissals in real Chromium: a modal, a panel and a sidebar, each proven by its own control vanishing.

The page mirrors the audited shape. Closing the sidebar removes it and its close button, and the results
area then enters a loading state, so no text the planner could predict has appeared yet. No model; a local
file, no network."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import discover
from src.cua.artifact import linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.context import Escalator, RecordingOperator, RunLog, SessionControl
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]

PAGE = """<!doctype html><html><head><title>Results</title></head><body>
<h1>Results</h1>

<aside id="sidebar" role="region" aria-label="Filters">
  <h2>Filters</h2>
  <button id="closeSidebar" type="button" aria-label="Close filters">x</button>
</aside>

<button id="openModal" type="button">Open details</button>
<div id="modal" role="dialog" aria-label="Details" hidden>
  <h2>Details</h2>
  <button id="closeModal" type="button" aria-label="Close">x</button>
</div>

<div id="menu" role="menu" aria-label="Options">
  <button id="closeMenu" type="button" aria-label="Close">x</button>
</div>

<!-- a close button in a region that names a destructive operation: never a dismissal -->
<section role="region" aria-label="Delete workspace">
  <button id="closeDanger" type="button" aria-label="Close">x</button>
</section>

<!-- a plain region with a bare close and no dismissal metadata of any kind -->
<div id="bare">
  <button id="closeBare" type="button">Close</button>
</div>

<!-- an ordinary control that vanishes because the page re-renders, dismissing nothing -->
<button id="apply" type="button">Apply</button>

<!-- two identically named controls: one is dismissed, the other stays -->
<div id="twinpanel" role="region" aria-label="Sorting">
  <button id="closeTwin" type="button" aria-label="Close sorting">x</button>
</div>
<button id="twinElsewhere" type="button" aria-label="Close sorting">x</button>

<div id="results"></div>
<script>
  const drop = (id) => document.getElementById(id).remove();
  document.getElementById('closeSidebar').addEventListener('click', () => {
    drop('sidebar');
    // the results area starts loading: nothing a planner could have predicted is on screen yet
    document.getElementById('results').textContent = 'Loading';
    setTimeout(() => { document.getElementById('results').textContent = 'Award Amount 1200000'; }, 4000);
  });
  document.getElementById('openModal').addEventListener('click',
      () => document.getElementById('modal').removeAttribute('hidden'));
  document.getElementById('closeModal').addEventListener('click', () => drop('modal'));
  document.getElementById('closeMenu').addEventListener('click', () => drop('menu'));
  document.getElementById('closeBare').addEventListener('click', () => drop('bare'));
  document.getElementById('apply').addEventListener('click', () => drop('apply'));
  document.getElementById('closeTwin').addEventListener('click', () => drop('twinpanel'));
</script></body></html>"""


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dismissals")
    (folder / "results.html").write_text(PAGE)
    return (folder / "results.html").as_uri()


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


def within(fresh, container_label, name="Close"):
    """The control with this name inside the container with that accessible label."""
    return next(e for e in fresh.observe().elements
                if (e.name or "") == name and container_label in (e.landmark_name or ""))


def run(fresh, site, steps, operator=None):
    log = RunLog("discovery")
    artifact = discover(goal="close the panel and read the results", name="panel_close", params={}, surface=fresh,
                        planner=ScriptedPlanner(steps), policy=Policy(allowed_hosts=[""]),
                        escalator=Escalator(operator or RecordingOperator("resume"), SessionControl(), log), log=log,
                        entry_url=site, max_steps=12)
    return artifact, log


def clicks_of(artifact, name):
    return [n for n in linear_path(artifact)
            if n.action.action == "click" and n.action.target.strategies[0].get("name") == name]


# ---------- the three container kinds ----------

@pytest.mark.parametrize("label", ["Filters", "Details", "Options"])
def test_closing_a_sidebar_modal_or_menu_is_a_proven_dismissal(fresh, label):
    from src.cua.models import Action

    if label == "Details":
        fresh._page.click("#openModal")
    element = within(fresh, label, "Close filters" if label == "Filters" else "Close")
    assert Policy(allowed_hosts=[""]).dismisses_interface(Action(kind="click", target=element))


def test_the_audited_shape_is_proven_although_the_results_are_still_loading(fresh, site):
    """The regression: the sidebar and its close button vanish, the results area shows only a loader.

    The results text arrives long after the expectation window closes, exactly as in the live run, so the
    planner's guess is contradicted and the disappearance is the only proof available.
    """
    steps = [ScriptedStep("click", "button", "Close filters", expect="Award Amount"),
             ScriptedStep("done", expect="Results")]
    artifact, log = run(fresh, site, steps)
    [node] = clicks_of(artifact, "Close filters")
    assert node.action.checkpoint == {"target_absent": "true"}
    assert events(log, "expectation_failed")[0]["expected"] == "Award Amount"
    assert events(log, "action_effect_proven")[0]["proof"] == "the dismissal's own control is gone"
    assert events(log, "action_effect_unverified") == []
    validate(artifact)


def test_a_panel_hiding_while_a_different_same_named_control_remains_is_proven(fresh):
    """Two controls share a name; only the clicked one goes. Identity is structural, never the label."""
    from src.cua.agent import dismissal_transition
    from src.cua.models import Action

    inside = next(e for e in fresh.observe().elements
                  if (e.name or "") == "Close sorting" and (e.landmark_name or "") == "Sorting")
    before = fresh.observe()
    action = Action(kind="click", target=inside)
    action.selected_before = None
    fresh.click(inside)
    after = fresh.observe()
    log = RunLog("discovery")
    proof = dismissal_transition(action, before, after, Policy(allowed_hosts=[""]), log)
    assert proof == {"checkpoint": {"target_absent": "true"}, "why": "the dismissal's own control is gone"}
    # the identically named control outside the panel is still on screen
    assert any((e.name or "") == "Close sorting" for e in after.elements)


# ---------- what disappearance never proves ----------

def test_an_ordinary_control_vanishing_on_rerender_is_not_proven(fresh, site):
    steps = [ScriptedStep("click", "button", "Apply", expect="Never shown"),
             ScriptedStep("done", expect="Results")]
    artifact, log = run(fresh, site, steps, operator=RecordingOperator("resume"))
    assert clicks_of(artifact, "Apply") == []
    assert events(log, "action_effect_unverified")[0]["cause"] == "action_effect_unverified"
    assert events(log, "action_effect_proven") == []


def test_a_close_in_a_destructive_container_is_not_a_dismissal(fresh):
    from src.cua.models import Action

    element = within(fresh, "Delete workspace")
    assert not Policy(allowed_hosts=[""]).dismisses_interface(Action(kind="click", target=element))


def test_a_bare_close_with_no_dismissal_metadata_is_not_a_dismissal(fresh):
    from src.cua.models import Action

    element = next(e for e in fresh.observe().elements
                   if (e.name or "") == "Close" and not (e.landmark_name or ""))
    assert not Policy(allowed_hosts=[""]).dismisses_interface(Action(kind="click", target=element))


# ---------- replay ----------

def test_the_artifact_replays_in_real_chromium_without_a_model(fresh, site):
    steps = [ScriptedStep("click", "button", "Close filters", expect="Award Amount"),
             ScriptedStep("done", expect="Results")]
    artifact, _ = run(fresh, site, steps)
    fresh.navigate(site)
    log = RunLog("replay")
    result = replay(artifact, {}, fresh, Policy(allowed_hosts=[""]),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success"
    assert fresh._page.query_selector("#sidebar") is None       # the sidebar really was closed
    assert {"target_absent": "true"} in [e["checkpoint"] for e in events(log, "checkpoint_passed")]


def test_replay_clicks_the_dismissal_exactly_once(fresh, site):
    steps = [ScriptedStep("click", "button", "Close filters", expect="Award Amount"),
             ScriptedStep("done", expect="Results")]
    artifact, _ = run(fresh, site, steps)
    log = RunLog("replay")
    result = replay(artifact, {}, fresh, Policy(allowed_hosts=[""]),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success"
    # The action is logged once and never repeated: the checkpoint it carries is satisfied first time.
    assert [e["action"] for e in events(log, "acted")].count("click") == 1
    assert events(log, "action_repeated") == []


def test_replay_fails_safely_when_the_control_survives(fresh, site, tmp_path):
    steps = [ScriptedStep("click", "button", "Close filters", expect="Award Amount"),
             ScriptedStep("done", expect="Results")]
    artifact, _ = run(fresh, site, steps)
    stuck = tmp_path / "stuck.html"
    stuck.write_text(PAGE.replace("drop('sidebar');", ""))
    artifact.surface["entry_url"] = stuck.as_uri()
    artifact.nodes[0].action.value = stuck.as_uri()
    log = RunLog("replay")
    result = replay(artifact, {}, fresh, Policy(allowed_hosts=[""]),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "failure" and result.outcome_code == "checkpoint_not_met"
    assert events(log, "target_absent_unmet")[-1]["reason"] == "the control is still on screen"
