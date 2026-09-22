"""Custom autocompletes with no ARIA semantics, in real Chromium: the suggestion container is found by its
association with the control, plain-text rows inside actionable parents are candidates, and only deterministic
widget state (value, active descendant, a stored value, a new chip) proves a click was accepted."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua import surface as surface_module
from src.cua.agent import discover
from src.cua.artifact import linear_path, to_dict, validate
from src.cua.escalation import NoOperator
from src.cua.models import ActionError
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.context import Escalator, RecordingOperator, RunLog, SessionControl
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]

AGENCIES = ["Environmental Protection Agency (EPA)", "Environmental Protection Board (EPB)",
            "Agriculture Department (USDA)", "Energy Department (DOE)", "Energy Department (2 offices)"]

PAGE = """<!doctype html><html><head><title>Filters</title><style>
  body { font-family: sans-serif; }
  .field { position: relative; width: 420px; margin-bottom: 180px; }
  .menu { position: absolute; left: 0; top: 100%; width: 100%; border: 1px solid #ccc; background: #fff; z-index: 5; }
  .menu > div { padding: 4px; cursor: pointer; }
  .hidden { display: none; }
  .chip { display: inline-block; border: 1px solid #888; padding: 2px 6px; }
  #far { position: absolute; left: 0; top: 1800px; width: 420px; border: 1px solid #ccc; }
</style></head><body>
<h1>Filters</h1>
<p>Reference: Environmental Protection Agency (EPA) appears in this paragraph too.</p>
<div class="field">
  <input id="plain" placeholder="Agency">
  <div id="plain-menu" class="menu hidden"></div>
</div>
<div class="field">
  <input id="chips" placeholder="Chip agency">
  <div id="chips-menu" class="menu hidden"></div>
  <div id="chip-box"></div><input type="hidden" id="chips-value">
</div>
<div class="field">
  <input id="nested" placeholder="Nested agency">
  <div id="nested-menu" class="menu hidden"></div>
</div>
<div class="field">
  <input id="twins" placeholder="Twin menus">
  <div id="twins-a" class="menu hidden"></div>
</div>
<div id="twins-b" class="menu hidden"></div>
<div class="field"><input id="silent" placeholder="Silent agency"><div id="silent-menu" class="menu hidden"></div></div>
<div id="far">Environmental Protection Agency (EPA)</div>
<script>
  const show = (menu, rows) => { menu.innerHTML = ''; for (const row of rows) menu.appendChild(row); menu.classList.remove('hidden'); };
  const plainRow = (text, onPick) => { const d = document.createElement('div'); d.textContent = text;
    d.addEventListener('click', () => onPick(text)); return d; };
  const nestedRow = (text, onPick) => { const d = document.createElement('div'); const inner = document.createElement('span');
    inner.textContent = text; d.appendChild(inner); d.addEventListener('click', () => onPick(text)); return d; };
  const AGENCIES = __AGENCIES__;
  const wire = (inputId, menuId, makeRow, onPick, twinId) => {
    const input = document.getElementById(inputId), menu = document.getElementById(menuId);
    input.addEventListener('input', () => {
      const hits = AGENCIES.filter((name) => name.toLowerCase().includes(input.value.toLowerCase()));
      show(menu, hits.map((name) => makeRow(name, onPick)));
      if (twinId) show(document.getElementById(twinId), hits.map((name) => makeRow(name, onPick)));
    });
  };
  wire('plain', 'plain-menu', plainRow, (text) => { document.getElementById('plain').value = text;
    document.getElementById('plain-menu').classList.add('hidden'); });
  wire('chips', 'chips-menu', plainRow, (text) => {
    const chip = document.createElement('span'); chip.className = 'chip'; chip.textContent = text;
    const remove = document.createElement('button'); remove.textContent = 'Remove'; chip.appendChild(remove);
    document.getElementById('chip-box').appendChild(chip);
    document.getElementById('chips-value').value = text;
    document.getElementById('chips').value = '';
    document.getElementById('chips-menu').classList.add('hidden'); });
  wire('nested', 'nested-menu', nestedRow, (text) => { document.getElementById('nested').value = text;
    document.getElementById('nested-menu').classList.add('hidden'); });
  wire('twins', 'twins-a', plainRow, (text) => { document.getElementById('twins').value = text; }, 'twins-b');
  wire('silent', 'silent-menu', plainRow, () => { document.getElementById('silent-menu').classList.add('hidden'); });
</script></body></html>""".replace("__AGENCIES__", json.dumps(AGENCIES))


@pytest.fixture(scope="module")
def page_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("custom") / "filters.html"
    path.write_text(PAGE)
    return path.as_uri()


@pytest.fixture
def surface(page_file):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=2500)
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    live.page_url = page_file
    live.navigate(page_file)
    yield live
    live.close()


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 2.0)
    monkeypatch.setattr(surface_module, "SUGGESTION_TIMEOUT_MS", 2000)


def control(surface, placeholder):
    return next(e for e in surface.observe().elements if e.role == "textbox" and e.name == placeholder)


def control_by_role(surface, role, name):
    return next(e for e in surface.observe().elements if e.role == role and e.name == name)


def value_of(surface, element_id):
    return surface._page.evaluate("(id) => document.getElementById(id).value", element_id)


def clicks_on(surface, menu_id):
    return surface._page.evaluate("(id) => document.getElementById(id).classList.contains('hidden')", menu_id)


def test_a_non_aria_autocomplete_with_plain_text_rows_is_selected_by_its_plain_name(surface):
    surface.select(control(surface, "Agency"), "Environmental Protection Agency")
    assert value_of(surface, "plain") == "Environmental Protection Agency (EPA)"


def test_a_plain_suggestion_inside_an_actionable_parent_is_a_candidate(surface):
    surface.select(control(surface, "Nested agency"), "Agriculture Department")
    assert value_of(surface, "nested") == "Agriculture Department (USDA)"


def test_a_chip_with_its_own_removal_control_proves_acceptance(surface):
    surface.select(control(surface, "Chip agency"), "Energy Department (DOE)")
    assert value_of(surface, "chips") == "" and value_of(surface, "chips-value") == "Energy Department (DOE)"
    chips = surface._page.evaluate("() => Array.from(document.querySelectorAll('#chip-box .chip')).map(c => c.textContent)")
    assert chips == ["Energy Department (DOE)Remove"]


def test_several_matching_plain_suggestions_are_ambiguous_with_no_click(surface):
    with pytest.raises(ActionError) as refused:
        surface.select(control(surface, "Agency"), "Energy Department")
    assert refused.value.performed == "yes" and refused.value.cause == "ambiguous_selection"
    assert value_of(surface, "plain") == "Energy Department"          # only the typed text; nothing was chosen


def test_unrelated_matching_page_text_is_never_a_candidate(surface):
    # the paragraph and the far-away box both hold the exact agency text; neither is associated with the control
    with pytest.raises(ActionError) as failed:
        surface.select(control(surface, "Silent agency"), "Environmental Protection Agency")
    assert failed.value.performed == "yes" and "no widget state shows it was accepted" in str(failed.value)
    assert value_of(surface, "silent") == "Environmental Protection Agency"


def test_several_unlinked_containers_are_refused(surface):
    with pytest.raises(ActionError) as refused:
        surface.select(control(surface, "Twin menus"), "Agriculture Department")
    assert refused.value.performed == "yes" and refused.value.cause == "ambiguous_selection"
    assert "2 open popups hold a suggestion matching it" in str(refused.value)
    assert value_of(surface, "twins") == "Agriculture Department"          # only the typed text; nothing was chosen


def test_a_clicked_but_unverified_suggestion_records_no_node(surface):
    policy = Policy(allowed_hosts=[""])
    log = RunLog("discovery")
    artifact = discover(goal="filter by agency", name="agency_filter", params={"agency": "Environmental Protection Agency"},
                        surface=surface, planner=ScriptedPlanner([
                            ScriptedStep("select", "textbox", "Silent agency", value="{{agency}}"),
                            ScriptedStep("type", "textbox", "Agency", value="{{agency}}"),
                            ScriptedStep("done", expect="Filters")]),
                        policy=policy, escalator=Escalator(RecordingOperator("resume"), SessionControl(), log),
                        log=log, entry_url=surface.page_url)
    assert [n.action.action for n in linear_path(artifact)] == ["navigate", "type"]
    text = log.path.read_text()
    assert '"event": "action_failed"' in text and '"action": "select"' not in text


def test_discovery_records_the_selection_once_and_replay_needs_no_model(surface):
    policy = Policy(allowed_hosts=[""])
    log = RunLog("discovery")
    artifact = discover(goal="filter by agency", name="agency_filter", params={"agency": "Environmental Protection Agency"},
                        surface=surface, planner=ScriptedPlanner([
                            ScriptedStep("select", "textbox", "Agency", value="{{agency}}"),
                            ScriptedStep("done", expect="Filters")]),
                        policy=policy, escalator=Escalator(NoOperator(), SessionControl(), log), log=log,
                        entry_url=surface.page_url)
    validate(artifact)
    node = linear_path(artifact)[1]
    assert node.action.action == "select" and node.action.value == "{{agency}}"
    assert "Environmental" not in json.dumps(to_dict(artifact))
    assert sum(1 for line in log.path.read_text().splitlines() if '"kind": "select"' in line and '"acted"' in line) == 1

    surface.navigate(surface.page_url)
    replay_log = RunLog("replay")
    result = replay(artifact, {"agency": "Agriculture Department"}, surface, policy,
                    Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success" and value_of(surface, "plain") == "Agriculture Department (USDA)"


# ---------- three same-named links in different rows stay three distinct targets ----------

ROWS_PAGE = """<!doctype html><html><head><title>Results</title></head><body><h1>Results</h1>
<ul>
  <li><span>First notice</span> <a href="#a">Environmental Protection Agency</a></li>
  <li><span>Second notice</span> <a href="#b">Environmental Protection Agency</a></li>
  <li><span>Third notice</span> <a href="#c">Environmental Protection Agency</a></li>
</ul></body></html>"""


@pytest.fixture(scope="module")
def rows_page(tmp_path_factory):
    path = tmp_path_factory.mktemp("rows") / "results.html"
    path.write_text(ROWS_PAGE)
    return path.as_uri()


def test_three_same_named_links_record_distinct_targets_and_replay_without_collapsing(surface, rows_page):
    from src.cua.models import Action

    policy = Policy(allowed_hosts=[""])
    contract = {"agencies": {"type": "list", "required": True, "min_items": 3, "max_items": 3,
                             "items": {"type": "string", "pattern": "(.+)"}}}

    class RowPlanner(ScriptedPlanner):
        def decide(self, goal, params, observation, history, candidates=()):
            if not history:
                links = [e for e in observation.elements if e.role == "link"]
                return Action(kind="extract_many", targets=links, output_name="agencies", pattern="(.+)",
                              reason="scripted")
            return super().decide(goal, params, observation, history, candidates)

    log = RunLog("discovery")
    artifact = discover(goal="read each result's agency", name="row_agencies",
                        params={"agency": "Environmental Protection Agency"}, surface=surface,
                        planner=RowPlanner([ScriptedStep("done", expect="Results")]), policy=policy,
                        escalator=Escalator(NoOperator(), SessionControl(), log), log=log, entry_url=rows_page,
                        output_contract=contract)
    validate(artifact)                                          # duplicate-target validation still applies
    node = linear_path(artifact)[-1]
    assert node.action.action == "extract_many" and len(node.action.targets) == 3
    assert len({t.strategies[-1]["selector"] for t in node.action.targets}) == 3      # distinct structural rungs
    assert {t.strategies[0]["name"] for t in node.action.targets} == {"{{agency}}"}   # identical semantic rungs
    assert artifact.outputs["agencies"]["example"] == ["{{agency}}"] * 3

    surface.navigate(rows_page)
    replay_log = RunLog("replay")
    result = replay(artifact, {"agency": "Environmental Protection Agency"}, surface, policy,
                    Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success"
    assert result.outputs == {"agencies": ["Environmental Protection Agency"] * 3}    # three values, not deduplicated
    resolved = [json.loads(l) for l in replay_log.path.read_text().splitlines() if '"target_resolved"' in l]
    assert [e["target_index"] for e in resolved if "target_index" in e] == [0, 1, 2]


# ---------- a widget that filters only on real key events ----------

KEYS_PAGE = """<!doctype html><html><head><title>Key filter</title><style>
  .field { position: relative; width: 420px; margin-bottom: 200px; }
  .menu { position: absolute; left: 0; top: 100%; width: 100%; border: 1px solid #ccc; background: #fff; z-index: 5; }
  .menu > div { padding: 4px; cursor: pointer; }
  .hidden { display: none; }
  .chip { display: inline-block; border: 1px solid #888; padding: 2px 6px; }
</style></head><body>
<h1>Key filter</h1>
<div class="field">
  <input id="keys" placeholder="Agency" autocomplete="off">
  <div id="keys-menu" class="menu hidden"></div>
  <div id="keys-chips"></div><input type="hidden" id="keys-value">
</div>
<div class="field">
  <input id="silentkeys" placeholder="Silent" autocomplete="off">
  <div id="silentkeys-menu" class="menu hidden"></div>
</div>
<select id="native" aria-label="Native"><option value="">Choose</option><option value="a">Environmental Protection Agency (EPA)</option>
  <option value="b">Energy Department (DOE)</option></select>
<script>
  const OPTIONS = __OPTIONS__;
  // filters ONLY from key events: a programmatic value change never opens the list
  const wireKeys = (inputId, onPick) => {
    const input = document.getElementById(inputId), menu = document.getElementById(inputId + '-menu');
    let typed = '';
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Backspace' || e.key === 'Delete') typed = '';
      else if (e.key.length === 1) typed += e.key;
      setTimeout(() => {
        const hits = typed ? OPTIONS.filter((name) => name.toLowerCase().includes(typed.toLowerCase())) : [];
        menu.innerHTML = '';
        for (const name of hits) {
          const row = document.createElement('div'); row.textContent = name;
          row.addEventListener('click', () => { menu.classList.add('hidden'); onPick(input, name); });
          menu.appendChild(row);
        }
        menu.classList.toggle('hidden', hits.length === 0);
      }, 0);
    });
  };
  wireKeys('keys', (input, name) => {
    const chip = document.createElement('span'); chip.className = 'chip'; chip.textContent = name;
    const remove = document.createElement('button'); remove.textContent = 'Remove'; chip.appendChild(remove);
    document.getElementById('keys-chips').appendChild(chip);
    document.getElementById('keys-value').value = name;
    input.value = '';
  });
  wireKeys('silentkeys', () => {});            // the menu closes and nothing is stored: never accepted
</script></body></html>""".replace("__OPTIONS__", json.dumps(
    ["Environmental Protection Agency (EPA)", "Environmental Protection Board (EPB)",
     "Agency for Environmental Protection Agency support", "Energy Department (DOE)", "Energy Department (2 offices)"]))


@pytest.fixture(scope="module")
def keys_page(tmp_path_factory):
    path = tmp_path_factory.mktemp("keys") / "keys.html"
    path.write_text(KEYS_PAGE)
    return path.as_uri()


@pytest.fixture
def keys_surface(keys_page):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=2500)
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    live.page_url = keys_page
    live.navigate(keys_page)
    yield live
    live.close()


def keys_value(surface, element_id):
    return surface._page.evaluate("(id) => document.getElementById(id).value", element_id)


def test_a_widget_that_filters_only_on_key_events_is_selected(keys_surface):
    keys_surface.select(control(keys_surface, "Agency"), "Energy Department (DOE)")
    assert keys_value(keys_surface, "keys-value") == "Energy Department (DOE)"
    chips = keys_surface._page.evaluate(
        "() => Array.from(document.querySelectorAll('#keys-chips .chip')).map(c => c.textContent)")
    assert chips == ["Energy Department (DOE)Remove"]


def test_an_exact_option_wins_over_one_that_merely_contains_the_text(keys_surface):
    keys_surface.select(control(keys_surface, "Agency"), "Environmental Protection Agency (EPA)")
    assert keys_value(keys_surface, "keys-value") == "Environmental Protection Agency (EPA)"


def test_a_plain_name_matching_several_glossed_options_is_ambiguous_and_clicks_nothing(keys_surface):
    with pytest.raises(ActionError) as refused:
        keys_surface.select(control(keys_surface, "Agency"), "Energy Department")
    assert refused.value.performed == "yes" and refused.value.cause == "ambiguous_selection"
    assert keys_value(keys_surface, "keys-value") == ""


def test_a_value_no_option_matches_clicks_nothing(keys_surface):
    # this widget shows no rows at all for an unmatched query, so the bounded keyboard commit is tried once;
    # its Enter resolves nothing, so the selection still fails and the widget stores nothing
    with pytest.raises(ActionError) as failed:
        keys_surface.select(control(keys_surface, "Agency"), "Department of Nothing")
    assert failed.value.performed == "yes" and "no widget state shows it was accepted" in str(failed.value)
    assert keys_value(keys_surface, "keys-value") == ""


def test_a_clicked_option_with_no_accepted_state_fails_safely(keys_surface):
    # the plain name is typed, the glossed option is clicked, and the widget stores nothing: the field still
    # holds only what was typed, which is never proof that the option was accepted
    with pytest.raises(ActionError) as failed:
        keys_surface.select(control(keys_surface, "Silent"), "Environmental Protection Board")
    assert failed.value.performed == "yes" and "no widget state shows it was accepted" in str(failed.value)


def test_a_native_select_on_the_same_page_is_unchanged(keys_surface):
    keys_surface.select(control_by_role(keys_surface, "combobox", "Native"), "Energy Department (DOE)")
    assert keys_value(keys_surface, "native") == "b"


def test_typing_is_cleared_between_selections(keys_surface):
    keys_surface.select(control(keys_surface, "Agency"), "Environmental Protection Board (EPB)")
    keys_surface.select(control(keys_surface, "Agency"), "Energy Department (DOE)")
    assert keys_value(keys_surface, "keys-value") == "Energy Department (DOE)"
    assert keys_value(keys_surface, "keys") == ""


# ---------- fields that resolve the typed value only when Enter is pressed ----------

COMMIT_PAGE = """<!doctype html><html><head><title>Commit</title><style>
  .field { position: relative; width: 420px; margin-bottom: 200px; }
  .menu { position: absolute; left: 0; top: 100%; width: 100%; border: 1px solid #ccc; background: #fff; z-index: 5; }
  .menu > div { padding: 4px; cursor: pointer; }
  .hidden { display: none; }
</style></head><body>
<h1>Commit</h1>
<div class="field"><input id="resolve" placeholder="Resolving" autocomplete="off"><input type="hidden" id="resolve-value"></div>
<div class="field"><input id="inert" placeholder="Inert" autocomplete="off"></div>
<div class="field"><input id="goes" placeholder="Navigating" autocomplete="off"></div>
<div class="field"><input id="echo" placeholder="Echo" autocomplete="off"></div>
<div class="field">
  <input id="listed" placeholder="Listed" autocomplete="off"><div id="listed-menu" class="menu hidden"></div>
</div>
<script>
  const OPTIONS = __OPTIONS__;
  // resolves the raw query into a fuller value, only on Enter, with no suggestion list at all
  document.getElementById('resolve').addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    const input = document.getElementById('resolve');
    input.value = input.value + ', resolved address';
    document.getElementById('resolve-value').value = input.value;
  });
  document.getElementById('inert').addEventListener('keydown', () => {});          // Enter does nothing at all
  document.getElementById('goes').addEventListener('keydown', (e) => {             // Enter navigates, proves nothing
    if (e.key === 'Enter') { e.preventDefault(); location.hash = '#submitted'; }
  });
  document.getElementById('echo').addEventListener('keydown', (e) => {             // Enter re-sets the same text
    if (e.key !== 'Enter') return;
    const input = document.getElementById('echo'); input.value = input.value;
  });
  const listed = document.getElementById('listed'), menu = document.getElementById('listed-menu');
  let typed = '';
  listed.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { listed.value = 'committed by enter'; return; }        // must never be reached
    if (e.key === 'Backspace' || e.key === 'Delete') typed = '';
    else if (e.key.length === 1) typed += e.key;
    setTimeout(() => {
      const hits = OPTIONS.filter((name) => name.toLowerCase().includes(typed.toLowerCase()));
      menu.innerHTML = '';
      for (const name of hits) {
        const row = document.createElement('div'); row.textContent = name;
        row.addEventListener('click', () => { listed.value = name; menu.classList.add('hidden'); });
        menu.appendChild(row);
      }
      menu.classList.toggle('hidden', hits.length === 0);
    }, 0);
  });
</script></body></html>""".replace("__OPTIONS__", json.dumps(
    ["Energy Department (DOE)", "Energy Department (2 offices)", "Harbour Authority"]))


@pytest.fixture(scope="module")
def commit_page(tmp_path_factory):
    path = tmp_path_factory.mktemp("commit") / "commit.html"
    path.write_text(COMMIT_PAGE)
    return path.as_uri()


@pytest.fixture
def commit_surface(commit_page):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=2500, secrets=("hunter2",))
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    live.page_url = commit_page
    live.navigate(commit_page)
    yield live
    live.close()


@pytest.fixture(autouse=True)
def brief_wait(monkeypatch):
    monkeypatch.setattr(surface_module, "SUGGESTION_TIMEOUT_MS", 800)


def field_value(surface, element_id):
    return surface._page.evaluate("(id) => document.getElementById(id).value", element_id)


def test_a_no_popup_field_resolves_after_exactly_one_enter(commit_surface):
    commit_surface.select(control(commit_surface, "Resolving"), "10 Example Street")
    assert field_value(commit_surface, "resolve") == "10 Example Street, resolved address"
    assert field_value(commit_surface, "resolve-value") == "10 Example Street, resolved address"
    assert "keyboard commit" in commit_surface.last_selection_proof


def test_enter_is_never_used_when_a_matching_option_is_offered(commit_surface):
    commit_surface.select(control(commit_surface, "Listed"), "Harbour Authority")
    assert field_value(commit_surface, "listed") == "Harbour Authority"      # clicked, not committed by Enter


def test_enter_is_never_used_for_ambiguous_suggestions(commit_surface):
    with pytest.raises(ActionError) as refused:
        commit_surface.select(control(commit_surface, "Listed"), "Energy Department")
    assert refused.value.cause == "ambiguous_selection"
    assert field_value(commit_surface, "listed") == "Energy Department"      # only the typed text


def test_enter_is_never_used_when_visible_suggestions_do_not_match(commit_surface):
    with pytest.raises(ActionError) as failed:
        commit_surface.select(control(commit_surface, "Listed"), "Energy")   # matches two, then nothing exact
    assert failed.value.performed == "yes" and "committed by enter" != field_value(commit_surface, "listed")


def test_an_enter_that_changes_nothing_fails_safely(commit_surface):
    with pytest.raises(ActionError) as failed:
        commit_surface.select(control(commit_surface, "Inert"), "Nothing happens here")
    assert failed.value.performed == "yes" and "no widget state shows it was accepted" in str(failed.value)


def test_an_enter_that_only_navigates_is_an_unknown_outcome(commit_surface):
    with pytest.raises(ActionError) as failed:
        commit_surface.select(control(commit_surface, "Navigating"), "Somewhere else")
    assert failed.value.performed in ("yes", "unknown")
    assert "accepted" in str(failed.value) or "unknown" in str(failed.value)


def test_a_field_that_echoes_the_typed_query_is_not_proof(commit_surface):
    with pytest.raises(ActionError) as failed:
        commit_surface.select(control(commit_surface, "Echo"), "Exactly what I typed")
    assert failed.value.performed == "yes" and "no widget state shows it was accepted" in str(failed.value)


def test_a_committed_selection_records_and_replays_without_a_model(commit_surface):
    policy = Policy(allowed_hosts=[""])
    log = RunLog("discovery", secrets=("hunter2",))
    artifact = discover(goal="enter the address", name="address_entry", params={"address": "10 Example Street"},
                        surface=commit_surface, planner=ScriptedPlanner([
                            ScriptedStep("select", "textbox", "Resolving", value="{{address}}"),
                            ScriptedStep("done", expect="Commit")]),
                        policy=policy, escalator=Escalator(NoOperator(), SessionControl(), log), log=log,
                        entry_url=commit_surface.page_url)
    validate(artifact)
    node = linear_path(artifact)[1]
    assert node.action.action == "select" and node.action.value == "{{address}}"
    text = log.path.read_text()
    assert "hunter2" not in text                                              # secrets never reach the log
    assert '"how": "keyboard commit' in text                                   # the mechanism is recorded
    assert "10 Example Street" not in json.dumps(to_dict(artifact))          # the artifact keeps the placeholder

    commit_surface.navigate(commit_surface.page_url)
    replay_log = RunLog("replay", secrets=("hunter2",))
    result = replay(artifact, {"address": "42 Other Road"}, commit_surface, policy,
                    Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success"
    assert field_value(commit_surface, "resolve") == "42 Other Road, resolved address"
