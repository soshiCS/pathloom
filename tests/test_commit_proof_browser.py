"""A form-local commit in real Chromium: a keyword field that empties into a removable chip.

The page mirrors the shape that exposed the bug: a labelled filter group holding a text field, an add
button, and the chips it commits into, plus a decoy chip in a different group and a toast. No model; a
local file, no network."""
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

PAGE = """<!doctype html><html><head><title>Filters</title></head><body>
<h1>Filters</h1>
<form id="keywords" aria-label="Filter by keyword">
  <label for="kw">Search using keywords</label>
  <input id="kw" type="text">
  <button id="add" type="button">Add</button>
  <ul id="chips"></ul>
</form>
<form id="other" aria-label="Other filters">
  <ul id="elsewhere"><li>Pre-existing <button type="button" aria-label="Remove">x</button></li></ul>
</form>
<p id="toast"></p>
<script>
  document.getElementById('add').addEventListener('click', () => {
    const field = document.getElementById('kw');
    const value = field.value.trim();
    if (!value) return;
    const li = document.createElement('li');
    li.textContent = value;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.setAttribute('aria-label', 'Remove ' + value);
    remove.textContent = 'x';
    remove.addEventListener('click', () => li.remove());
    li.appendChild(remove);
    document.getElementById('chips').appendChild(li);
    field.value = '';                                  // the field gives up its value: the commit happened
    document.getElementById('toast').textContent = 'Filter applied';
  });
</script></body></html>"""

INERT = PAGE.replace("document.getElementById('chips').appendChild(li);", "")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    folder = tmp_path_factory.mktemp("filters")
    (folder / "filters.html").write_text(PAGE)
    (folder / "inert.html").write_text(INERT)
    return folder


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
    surface.navigate((site / "filters.html").as_uri())
    return surface


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 1.0)


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def script(expect):
    return [ScriptedStep("type", "textbox", "Search using keywords", value="{{keyword}}"),
            ScriptedStep("click", "button", "Add", expect=expect),
            ScriptedStep("done", expect="Filters")]


def run(fresh, entry, steps, operator=None):
    log = RunLog("discovery")
    artifact = discover(goal="add a keyword filter", name="keyword_filter", params={"keyword": "renewable energy"},
                        surface=fresh, planner=ScriptedPlanner(steps), policy=Policy(allowed_hosts=[""]),
                        escalator=Escalator(operator or RecordingOperator("resume"), SessionControl(), log), log=log,
                        entry_url=entry, max_steps=12)
    return artifact, log


def added_nodes(artifact):
    return [n for n in linear_path(artifact)
            if n.action.action == "click" and n.action.target.strategies[0].get("name") == "Add"]


def test_a_real_commit_into_a_removable_chip_is_proof_and_records_one_node(fresh, site):
    artifact, log = run(fresh, (site / "filters.html").as_uri(), script("Keyword:"))
    [added] = added_nodes(artifact)
    assert added.action.checkpoint == {"selection_present": "{{keyword}}"}
    assert events(log, "action_effect_proven")[0]["proof"] == "the field was committed into a new removable token"
    assert events(log, "action_effect_unverified") == []
    validate(artifact)


def test_the_committed_artifact_replays_in_real_chromium_without_a_model(fresh, site):
    artifact, _ = run(fresh, (site / "filters.html").as_uri(), script("Keyword:"))
    fresh.navigate((site / "filters.html").as_uri())
    log = RunLog("replay")
    result = replay(artifact, {"keyword": "renewable energy"}, fresh, Policy(allowed_hosts=[""]),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success"
    chips = fresh._page.evaluate("() => Array.from(document.querySelectorAll('#chips li')).map(li => li.textContent)")
    assert any("renewable energy" in chip for chip in chips)
    assert {"selection_present": "{{keyword}}"} in [e["checkpoint"] for e in events(log, "checkpoint_passed")]


def test_a_toast_and_a_chip_in_another_group_are_not_proof(fresh, site):
    artifact, log = run(fresh, (site / "inert.html").as_uri(), script("Keyword:"),
                        operator=RecordingOperator("resume"))
    # the field still empties and a toast appears, but nothing is committed inside the acted group
    assert added_nodes(artifact) == []
    assert events(log, "action_effect_unverified")[0]["cause"] == "action_effect_unverified"


def test_the_expected_text_appearing_still_wins_without_any_commit_reasoning(fresh, site):
    artifact, log = run(fresh, (site / "filters.html").as_uri(), script("Filter applied"))
    [added] = added_nodes(artifact)
    assert added.action.checkpoint == {"text_contains": "Filter applied"}
    assert events(log, "expectation_failed") == [] and events(log, "action_effect_proven") == []
