"""The select action in real Chromium: a native select, an ARIA combobox with delayed suggestions, a plain text
input with a suggestion list, no match, duplicate labels, an uncertain post-action failure, and a model-free replay."""
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
from tests.context import Escalator, RunLog, SessionControl
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

pytestmark = pytest.mark.browser

PAGE = """<!doctype html><html><head><title>Choosers</title><style>[hidden]{display:none} li{cursor:pointer}</style></head><body>
<h1>Choosers</h1>
<select id="country" aria-label="Country"><option value="">Choose</option><option value="us">United States</option>
  <option value="uk">United Kingdom</option><option value="uy" disabled>Uruguay</option></select>
<p><label for="city">City</label> <input id="city" role="combobox" aria-autocomplete="list" aria-controls="city-list" aria-expanded="false"></p>
<ul id="city-list" role="listbox" hidden></ul>
<p><label for="plain">Plain</label> <input id="plain"></p>
<ul id="plain-list" role="listbox" hidden></ul>
<p><label for="vanish">Vanishing</label> <input id="vanish" role="combobox" aria-controls="vanish-list"></p>
<ul id="vanish-list" role="listbox" hidden></ul>
<p><label for="owned">Owned</label> <input id="owned" role="combobox" aria-owns="owned-list"></p>
<ul id="owned-list" role="listbox" hidden></ul>
<p><label for="lonely">Lonely</label> <input id="lonely"></p>
<ul id="lonely-list" role="listbox" hidden></ul>
<ul id="other-menu" role="menu" hidden><li role="menuitem">Boston, Massachusetts</li></ul>
<select id="twins" aria-label="Twins"><option value="">Choose</option><option value="a">Same</option><option value="b">Same</option>
  <option value="c">Only one</option></select>
<div id="picked"></div>
<script>
  const CITIES = [["Boston, Massachusetts", 1], ["Boston, Lincolnshire", 2], ["Bostonia, California", 3],
                  ["Springfield, Illinois", 4], ["Springfield, Illinois", 5],
                  ["Environmental Protection Agency (EPA)", 6], ["Environmental Protection Board (EPB)", 7],
                  ["Energy Department (DOE)", 8], ["Energy Department (2 offices)", 9]];
  const wire = (inputId, listId, delay, onPick, onShow) => {
    const input = document.getElementById(inputId), list = document.getElementById(listId);
    let pending = null;
    input.addEventListener('input', () => {
      const value = input.value.toLowerCase();
      list.innerHTML = ''; list.hidden = true; input.setAttribute('aria-expanded', 'false');
      if (pending) clearTimeout(pending);        // debounce, as an autocomplete does between keystrokes
      pending = setTimeout(() => {
        const hits = CITIES.filter(([name]) => name.toLowerCase().includes(value));
        for (const [name, id] of hits) {
          const li = document.createElement('li'); li.setAttribute('role', 'option'); li.textContent = name;
          li.addEventListener('click', () => { input.value = name; list.hidden = true; input.setAttribute('aria-expanded', 'false');
            document.getElementById('picked').textContent = 'picked:' + id; onPick && onPick(input); });
          list.appendChild(li);
        }
        if (hits.length) { list.hidden = false; input.setAttribute('aria-expanded', 'true'); onShow && onShow(); }
      }, delay);
    });
  };
  wire('city', 'city-list', 800, null);
  wire('plain', 'plain-list', 0, null);
  wire('vanish', 'vanish-list', 0, (input) => input.remove());
  wire('owned', 'owned-list', 0, null);
  wire('lonely', 'lonely-list', 0, null, () => { const menu = document.getElementById('other-menu'); if (menu) menu.hidden = false; });
  document.getElementById('other-menu').firstElementChild.addEventListener('click', () => { document.getElementById('picked').textContent = 'picked:menu'; });
</script></body></html>"""


@pytest.fixture(scope="module")
def surface(tmp_path_factory):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=2500)
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    page = tmp_path_factory.mktemp("choosers") / "choosers.html"
    page.write_text(PAGE)
    live.page_url = page.as_uri()
    yield live
    live.close()


@pytest.fixture
def fresh(surface):
    surface.navigate(surface.page_url)
    return surface


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 2.0)


def control(surface, name):
    return next(e for e in surface.observe().elements if e.name == name and e.role in ("combobox", "textbox"))


def value_of(surface, element_id):
    return surface._page.evaluate("(id) => document.getElementById(id).value", element_id)


def picked(surface):
    return surface._page.evaluate("() => document.getElementById('picked').textContent")


def test_a_native_select_chooses_by_visible_option_text(fresh):
    fresh.select(control(fresh, "Country"), "united kingdom")
    assert value_of(fresh, "country") == "uk"
    with pytest.raises(ActionError) as refused:
        fresh.select(control(fresh, "Country"), "Zzz")
    assert refused.value.performed == "no" and "none of the 4 options matches" in str(refused.value)
    with pytest.raises(ActionError) as disabled:
        fresh.select(control(fresh, "Country"), "Uruguay")
    assert disabled.value.performed == "no" and value_of(fresh, "country") == "uk"


def test_an_aria_combobox_with_delayed_suggestions_is_selected_and_verified(fresh):
    fresh.select(control(fresh, "City"), "Boston, Massachusetts")
    assert value_of(fresh, "city") == "Boston, Massachusetts" and picked(fresh) == "picked:1"


def test_a_typed_value_no_suggestion_matches_is_not_a_selection(fresh, monkeypatch):
    monkeypatch.setattr(surface_module, "SUGGESTION_TIMEOUT_MS", 1500)
    with pytest.raises(ActionError) as failed:
        fresh.select(control(fresh, "City"), "Zzz")
    # no rows are offered for this query, so the bounded keyboard commit runs once and resolves nothing
    assert failed.value.performed == "yes" and "no widget state shows it was accepted" in str(failed.value)
    assert value_of(fresh, "city") == "Zzz" and picked(fresh) == ""


def test_duplicate_exact_labels_are_ambiguous_and_nothing_is_clicked(fresh):
    with pytest.raises(ActionError) as refused:
        fresh.select(control(fresh, "Plain"), "Springfield, Illinois")
    assert refused.value.performed == "yes" and refused.value.cause == "ambiguous_selection"
    assert "more than one suggestion is a exact match" in str(refused.value)
    assert value_of(fresh, "plain") == "Springfield, Illinois" and picked(fresh) == ""     # typed, nothing chosen


def test_multiple_prefix_matches_are_ambiguous_and_a_unique_exact_match_wins_among_similar_options(fresh):
    with pytest.raises(ActionError) as refused:
        fresh.select(control(fresh, "City"), "Boston")                                     # three prefix matches
    assert refused.value.cause == "ambiguous_selection" and "prefix match" in str(refused.value) and picked(fresh) == ""
    fresh.navigate(fresh.page_url)
    fresh.select(control(fresh, "City"), "Boston, Lincolnshire")                              # exact among similar
    assert picked(fresh) == "picked:2"


def test_aria_owns_links_the_popup_like_aria_controls(fresh):
    fresh.select(control(fresh, "Owned"), "Bostonia")
    assert value_of(fresh, "owned") == "Bostonia, California" and picked(fresh) == "picked:3"


def test_an_unlinked_control_uses_the_single_visible_popup_and_refuses_when_several_are_open(fresh, monkeypatch):
    monkeypatch.setattr(surface_module, "SUGGESTION_TIMEOUT_MS", 1500)
    # Lonely has no aria link: typing opens its own list and an unrelated menu that also shows the value
    with pytest.raises(ActionError) as refused:
        fresh.select(control(fresh, "Lonely"), "Boston, Massachusetts")
    assert refused.value.cause == "ambiguous_selection"
    assert "2 open popups hold a suggestion matching it" in str(refused.value)
    assert picked(fresh) == ""                                                             # the menu's option was never clicked
    fresh.navigate(fresh.page_url)
    fresh._page.evaluate("() => document.getElementById('other-menu').remove()")
    fresh.select(control(fresh, "Lonely"), "Bostonia")                                       # now exactly one popup
    assert value_of(fresh, "lonely") == "Bostonia, California" and picked(fresh) == "picked:3"


def test_duplicate_native_select_labels_are_ambiguous_and_leave_the_select_untouched(fresh):
    with pytest.raises(ActionError) as refused:
        fresh.select(control(fresh, "Twins"), "Same")
    assert refused.value.performed == "no" and refused.value.cause == "ambiguous_selection"
    assert value_of(fresh, "twins") == ""
    fresh.select(control(fresh, "Twins"), "Only one")
    assert value_of(fresh, "twins") == "c"


def test_a_plain_text_input_with_a_suggestion_list_works_like_a_combobox(fresh):
    fresh.select(control(fresh, "Plain"), "Bostonia")
    assert value_of(fresh, "plain") == "Bostonia, California" and picked(fresh) == "picked:3"


def test_a_control_that_vanishes_after_the_choice_is_an_unknown_state_not_a_success(fresh):
    with pytest.raises(ActionError) as unknown:
        fresh.select(control(fresh, "Vanishing"), "Boston, Lincolnshire")
    assert unknown.value.performed == "unknown" and picked(fresh) == "picked:2"


def test_discovery_records_the_placeholder_and_replay_selects_without_a_model(fresh):
    policy = Policy(allowed_hosts=[""])
    log = RunLog("discovery")
    artifact = discover(goal="choose the city", name="choose_city", params={"city": "Boston, Massachusetts"}, surface=fresh,
                        planner=ScriptedPlanner([ScriptedStep("select", "textbox", "City", value="{{city}}"),
                                                 ScriptedStep("done", expect="Choosers")]),
                        policy=policy, escalator=Escalator(NoOperator(), SessionControl(), log), log=log,
                        entry_url=fresh.page_url)
    validate(artifact)
    node = linear_path(artifact)[1]
    assert node.action.action == "select" and node.action.value == "{{city}}" and node.effect == "reversible"
    assert "Boston" not in json.dumps(to_dict(artifact)) and '"value": "[typed]"' in log.path.read_text()
    fresh.navigate(fresh.page_url)
    replay_log = RunLog("replay")
    result = replay(artifact, {"city": "Bostonia"}, fresh, policy, Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success" and value_of(fresh, "city") == "Bostonia, California" and picked(fresh) == "picked:3"


def test_a_suggestion_with_a_trailing_acronym_is_selected_by_its_plain_name(fresh):
    fresh.select(control(fresh, "City"), "Environmental Protection Agency")
    assert value_of(fresh, "city") == "Environmental Protection Agency (EPA)" and picked(fresh) == "picked:6"


def test_two_glossed_suggestions_with_the_same_plain_name_stay_ambiguous(fresh, monkeypatch):
    monkeypatch.setattr(surface_module, "SUGGESTION_TIMEOUT_MS", 1500)
    with pytest.raises(ActionError) as refused:
        fresh.select(control(fresh, "City"), "Energy Department")
    assert refused.value.performed == "yes" and refused.value.cause == "ambiguous_selection"
    assert "exact without a trailing bracket match" in str(refused.value) and picked(fresh) == ""


def test_a_typed_value_that_no_suggestion_accepts_is_still_not_a_selection(fresh, monkeypatch):
    monkeypatch.setattr(surface_module, "SUGGESTION_TIMEOUT_MS", 1500)
    with pytest.raises(ActionError) as failed:
        fresh.select(control(fresh, "City"), "Environmental Protection Service")
    # no rows are offered for this query, so the bounded keyboard commit runs once and resolves nothing
    assert failed.value.performed == "yes" and "no widget state shows it was accepted" in str(failed.value)
    assert value_of(fresh, "city") == "Environmental Protection Service" and picked(fresh) == ""


def test_discovery_records_a_glossed_selection_only_after_acceptance_and_replays(fresh):
    policy = Policy(allowed_hosts=[""])
    log = RunLog("discovery")
    artifact = discover(goal="choose the agency", name="choose_agency", params={"agency": "Environmental Protection Agency"},
                        surface=fresh, planner=ScriptedPlanner([ScriptedStep("select", "textbox", "City", value="{{agency}}"),
                                                                ScriptedStep("done", expect="Choosers")]),
                        policy=policy, escalator=Escalator(NoOperator(), SessionControl(), log), log=log,
                        entry_url=fresh.page_url)
    validate(artifact)
    node = linear_path(artifact)[1]
    assert node.action.action == "select" and node.action.value == "{{agency}}"
    assert "Environmental" not in json.dumps(to_dict(artifact))
    fresh.navigate(fresh.page_url)
    replay_log = RunLog("replay")
    result = replay(artifact, {"agency": "Bostonia"}, fresh, policy, Escalator(NoOperator(), SessionControl(), replay_log),
                    replay_log)
    assert result.status == "success" and value_of(fresh, "city") == "Bostonia, California"
