"""Nested chip markup in real Chromium: one logical token per removal control.

The page builds each chip the way component frameworks do, as several nested containers around a label and
a close button. Every one of those containers satisfies the candidate test, so this is the shape that made
perception report two tokens for one visible chip. No model; a local file, no network."""
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

# Each chip is <li><div><span>label</span><button>x</button></div></li>: the li, the div and (for the
# selected variant) the span all look like candidates, and all find the same close button.
PAGE = """<!doctype html><html><head><title>Filters</title></head><body>
<h1>Filters</h1>
<form id="keywords" aria-label="Filter by keyword">
  <label for="kw">Search using keywords</label>
  <input id="kw" type="text">
  <button id="add" type="button">Add</button>
  <ul id="chips"></ul>
</form>
<form id="other" aria-label="Other filters">
  <ul><li><div><span>Elsewhere</span><button type="button" aria-label="Remove">x</button></div></li></ul>
</form>
<form id="years" aria-label="Fiscal years">
  <ul><li role="option" aria-selected="true">FY 2026</li><li role="option" aria-selected="false">FY 2025</li></ul>
</form>
<script>
  const chip = (value) => {
    const li = document.createElement('li');
    const box = document.createElement('div');
    const label = document.createElement('span');
    label.textContent = value;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.setAttribute('aria-label', 'Remove ' + value);
    remove.textContent = 'x';
    remove.addEventListener('click', () => li.remove());
    box.appendChild(label);
    box.appendChild(remove);
    li.appendChild(box);
    return li;
  };
  document.getElementById('add').addEventListener('click', () => {
    const field = document.getElementById('kw');
    const value = field.value.trim();
    if (!value) return;
    document.getElementById('chips').appendChild(chip(value));
    field.value = '';
  });
</script></body></html>"""

# The same page, but one click commits two separate chips: two removal controls, genuinely ambiguous.
TWO = PAGE.replace("document.getElementById('chips').appendChild(chip(value));",
                   "document.getElementById('chips').appendChild(chip(value));"
                   "document.getElementById('chips').appendChild(chip(value + ' (also)'));")

# And a variant where the two chips read exactly the same: still two tokens, never merged by text.
SAME = PAGE.replace("document.getElementById('chips').appendChild(chip(value));",
                    "document.getElementById('chips').appendChild(chip(value));"
                    "document.getElementById('chips').appendChild(chip(value));")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    folder = tmp_path_factory.mktemp("chips")
    (folder / "one.html").write_text(PAGE)
    (folder / "two.html").write_text(TWO)
    (folder / "same.html").write_text(SAME)
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


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 1.0)


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def script():
    return [ScriptedStep("type", "textbox", "Search using keywords", value="{{keyword}}"),
            ScriptedStep("click", "button", "Add", expect="Applied Filters"),
            ScriptedStep("done", expect="Filters")]


def run(surface, entry, operator=None):
    surface.navigate(entry)
    log = RunLog("discovery")
    artifact = discover(goal="add a keyword filter", name="keyword_filter", params={"keyword": "renewable energy"},
                        surface=surface, planner=ScriptedPlanner(script()), policy=Policy(allowed_hosts=[""]),
                        escalator=Escalator(operator or RecordingOperator("resume"), SessionControl(), log), log=log,
                        entry_url=entry, max_steps=12)
    return artifact, log


def added_nodes(artifact):
    return [n for n in linear_path(artifact)
            if n.action.action == "click" and n.action.target.strategies[0].get("name") == "Add"]


def field_control(surface, name="Search using keywords"):
    return next(e for e in surface.observe().elements if e.role == "textbox" and name in (e.name or ""))


# ---------- 1, 3. one visible chip is one token, and the click is recorded ----------

def test_a_nested_chip_reports_exactly_one_token_and_records_one_action(surface, site):
    artifact, log = run(surface, (site / "one.html").as_uri())
    [added] = added_nodes(artifact)
    assert added.action.checkpoint == {"selection_present": "{{keyword}}"}
    assert events(log, "commit_not_proven") == []            # no duplicate ever reached the rule
    assert events(log, "action_effect_proven")[0]["proof"] == "the field was committed into a new removable token"
    validate(artifact)


def test_perception_reports_one_token_for_one_nested_chip(surface, site):
    surface.navigate((site / "one.html").as_uri())
    surface._page.fill("#kw", "renewable energy")
    surface._page.click("#add")
    state = surface.committed_state(field_control(surface))
    assert [t["text"] for t in state["tokens"]] == ["renewable energy"]
    assert state["tokens"][0]["removable"] is True


# ---------- 9. the removal glyph is not part of the value ----------

def test_the_removal_glyph_is_not_included_in_the_token_value(surface, site):
    surface.navigate((site / "one.html").as_uri())
    surface._page.fill("#kw", "renewable energy")
    surface._page.click("#add")
    [token] = surface.committed_state(field_control(surface))["tokens"]
    assert token["text"] == "renewable energy" and "x" not in token["text"].split()[-1][-1:]


# ---------- 4. the artifact replays without a model ----------

def test_the_artifact_replays_in_real_chromium_without_a_model(surface, site):
    artifact, _ = run(surface, (site / "one.html").as_uri())
    surface.navigate((site / "one.html").as_uri())
    log = RunLog("replay")
    result = replay(artifact, {"keyword": "renewable energy"}, surface, Policy(allowed_hosts=[""]),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success"
    assert {"selection_present": "{{keyword}}"} in [e["checkpoint"] for e in events(log, "checkpoint_passed")]


# ---------- 5, 6. genuinely separate chips stay separate ----------

def test_two_chips_with_different_removal_controls_remain_ambiguous(surface, site):
    artifact, log = run(surface, (site / "two.html").as_uri(), operator=RecordingOperator("resume"))
    assert added_nodes(artifact) == []
    assert "2 new tokens appeared" in events(log, "commit_not_proven")[-1]["reason"]


def test_two_chips_with_identical_text_remain_two_tokens(surface, site):
    surface.navigate((site / "same.html").as_uri())
    surface._page.fill("#kw", "renewable energy")
    surface._page.click("#add")
    state = surface.committed_state(field_control(surface))
    assert [t["text"] for t in state["tokens"]] == ["renewable energy", "renewable energy"]
    assert len({t["remove_ref"] for t in state["tokens"]}) == 2      # never merged by their words


# ---------- 7, 8. selected options, and scoping to the acted group ----------

def test_a_native_selected_option_without_a_removal_control_is_detected(surface, site):
    # Asked from the sibling option, as a click on one option would: a control never proves its own commit,
    # so the acted element itself is excluded from the tokens by design.
    surface.navigate((site / "one.html").as_uri())
    sibling = next(e for e in surface.observe().elements if "FY 2025" in (e.text or e.name or ""))
    state = surface.committed_state(sibling)
    assert state is not None
    assert [t["text"] for t in state["tokens"]] == ["FY 2026"]       # the unselected sibling is not a token
    assert state["tokens"][0]["selected"] is True and not state["tokens"][0]["removable"]


def test_a_chip_in_another_group_is_excluded(surface, site):
    surface.navigate((site / "one.html").as_uri())
    state = surface.committed_state(field_control(surface))
    assert [t["text"] for t in state["tokens"]] == []                # "Elsewhere" lives in the other form
