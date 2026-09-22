"""Browser perception merges the page projection with Chromium's computed accessibility tree.

Runs a local headless Chromium on an inline page (no network); skips when Chromium is absent.
"""
import pytest

from src.cua.models import Locator
from src.cua.planner import render_observation
from src.cua.surface import locator_for

pytestmark = pytest.mark.browser

SECRET = "hunter2"
PAGE = f"""
<html><body>
  <p>Training password: {SECRET}</p>
  <label>Amount <input type="text" aria-label="Amount" required></label>
  <input type="password" aria-label="Password" value="{SECRET}">
  <input type="checkbox" aria-label="Newsletter">
  <input type="radio" name="t" aria-label="Savings" checked> <input type="radio" name="t" aria-label="Checking">
  <div role="checkbox" aria-checked="true" tabindex="0" id="alerts"
       onclick="this.setAttribute('aria-checked', this.getAttribute('aria-checked') === 'true' ? 'false' : 'true')">
    Email alerts
  </div>
  <a href="/details" role="button">Details</a>
  <button aria-pressed="false" id="mute">Mute</button>
  <button aria-expanded="false" aria-controls="menu">More</button>
  <select aria-label="Region"><option>East</option><option>West</option></select>
  <div role="tablist"><div role="tab" aria-selected="true" tabindex="0">Overview</div>
                      <div role="tab" aria-selected="false" tabindex="-1">History</div></div>
  <button disabled>Archive</button>
  <span class="title">Products</span>
</body></html>
"""


@pytest.fixture(scope="module")
def surface(tmp_path_factory):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, secrets=(SECRET,))
    except Exception as error:      # no browser binary in this environment
        pytest.skip(f"Chromium is not available: {error}")
    page = tmp_path_factory.mktemp("page") / "controls.html"
    page.write_text(PAGE)
    live.navigate(page.as_uri())
    yield live
    live.close()


@pytest.fixture
def reloaded(surface):
    surface.navigate(surface._page.url)      # every test starts from the page's initial state
    return surface


def by_name(observation, role, name):
    matches = [e for e in observation.elements if (e.role, e.name) == (role, name)]
    assert len(matches) == 1, [(e.role, e.name) for e in observation.elements]
    return matches[0]


def test_accessibility_only_controls_appear_with_their_states(reloaded):
    observation = reloaded.observe()
    assert reloaded.last_accessibility_error is None
    newsletter = by_name(observation, "checkbox", "Newsletter")
    assert newsletter.source == "ax" and newsletter.states == {"checked": "false"} and newsletter.text == ""
    assert by_name(observation, "radio", "Savings").states == {"checked": "true"}
    assert by_name(observation, "radio", "Checking").states == {"checked": "false"}
    assert by_name(observation, "tab", "Overview").states == {"selected": "true"}
    assert by_name(observation, "button", "Archive").states == {"disabled": "true"}
    assert by_name(observation, "textbox", "Amount").states == {"required": "true"}     # no readonly=false noise
    assert by_name(observation, "button", "More").states == {"expanded": "false"}


def test_accessibility_only_control_is_resolved_clicked_and_reobserved(reloaded):
    newsletter = by_name(reloaded.observe(), "checkbox", "Newsletter")
    ladder = locator_for(newsletter)                                    # what discovery would record
    assert ladder.strategies[0] == {"kind": "role", "role": "checkbox", "name": "Newsletter"}
    assert ladder.strategies[1]["kind"] == "css" and ladder.strategies[2]["kind"] == "coords"
    resolved = reloaded.resolve(Locator(strategies=ladder.strategies[:1]))   # the role rung alone
    assert resolved is not None and resolved.ref == newsletter.ref
    reloaded.click(resolved)
    assert by_name(reloaded.observe(), "checkbox", "Newsletter").states == {"checked": "true"}
    reloaded.click(reloaded.resolve(Locator(strategies=ladder.strategies[1:2])))   # the structural rung too
    assert by_name(reloaded.observe(), "checkbox", "Newsletter").states == {"checked": "false"}


def test_generic_text_is_upgraded_to_a_real_control(reloaded):
    observation = reloaded.observe()
    alerts = by_name(observation, "checkbox", "Email alerts")
    assert alerts.source == "ax" and alerts.states == {"checked": "true"} and alerts.text == "Email alerts"
    assert not any(e.role == "text" and e.text == "Email alerts" for e in observation.elements)
    reloaded.click(alerts)
    assert by_name(reloaded.observe(), "checkbox", "Email alerts").states == {"checked": "false"}


def test_native_dom_roles_survive_conflicting_aria_roles(reloaded):
    observation = reloaded.observe()
    details = by_name(observation, "link", "Details")
    assert details.source == "dom+ax" and details.states == {}
    assert not any(e.role == "button" and e.name == "Details" for e in observation.elements)


def test_dom_and_accessibility_views_of_one_element_are_deduplicated_and_ordered(reloaded):
    observation = reloaded.observe()
    refs = [e.ref for e in observation.elements]
    assert len(refs) == len(set(refs))                                  # one element per backing node
    region = by_name(observation, "combobox", "Region")
    assert region.source == "dom+ax" and region.text == "East" and region.states == {"expanded": "false"}
    assert by_name(observation, "button", "Mute").states == {"pressed": "false"}
    names = [e.name or e.text for e in observation.elements]
    assert names.index("Newsletter") < names.index("Details") < names.index("Products")   # document order


def test_states_reach_the_planner_prompt(reloaded):
    rendered = render_observation(reloaded.observe())
    assert 'checkbox "Newsletter" (checked=false)' in rendered
    assert 'checkbox "Email alerts" (checked=true)' in rendered
    assert 'button "Mute" (pressed=false)' in rendered and 'tab "Overview" (selected=true)' in rendered
    assert "required=false" not in rendered and "readonly" not in rendered


def test_password_values_and_registered_secrets_are_not_exposed(reloaded):
    observation = reloaded.observe()
    password = by_name(observation, "textbox", "Password")
    assert password.text == "" and password.states == {}
    everything = " ".join(f"{e.name} {e.text} {e.context}" for e in observation.elements)
    assert SECRET not in everything and SECRET not in render_observation(observation)
    assert "Training password: ••••••" in everything


def test_accessibility_failure_falls_back_to_the_projection(reloaded, monkeypatch):
    def broken():
        raise RuntimeError("CDP session lost")

    monkeypatch.setattr(reloaded, "_accessibility_nodes", broken)
    observation = reloaded.observe()
    assert reloaded.last_accessibility_error == "RuntimeError: CDP session lost"
    assert by_name(observation, "link", "Details").source == "dom"
    assert any(e.role == "text" and e.text == "Email alerts" for e in observation.elements)   # no upgrade
    assert not any(e.name == "Newsletter" for e in observation.elements)                    # not projected
    assert all(set(e.states) <= {"disabled", "covered"} for e in observation.elements)   # the projection's own evidence only
    monkeypatch.undo()
    assert by_name(reloaded.observe(), "checkbox", "Newsletter").source == "ax"               # recovered
    assert reloaded.last_accessibility_error is None


def snapshot(observation):
    return [(e.role, e.name, e.text, e.box, e.context, e.ref, dict(e.states), e.source) for e in observation.elements]


def test_a_failure_after_the_tree_was_read_returns_the_untouched_projection(reloaded, monkeypatch):
    with pytest.MonkeyPatch.context() as no_tree:                     # the pure projection, for comparison
        no_tree.setattr(reloaded, "_accessibility_nodes", lambda: (_ for _ in ()).throw(RuntimeError("off")))
        projection = snapshot(reloaded.observe())
    assert any(role == "text" and text == "Email alerts" for role, _, text, *_ in projection)

    def enrich_fails(pending):                                        # the tree was read, upgrades were staged ...
        assert pending, "the accessibility nodes were read and an enrichment was pending"
        raise RuntimeError("enrichment lost the page")

    monkeypatch.setattr(reloaded, "_enrich", enrich_fails)
    fallback = reloaded.observe()
    assert reloaded.last_accessibility_error == "RuntimeError: enrichment lost the page"
    assert snapshot(fallback) == projection                           # ... and none of them leaked out
    assert all(set(e.states) <= {"disabled", "covered"} and e.source == "dom" for e in fallback.elements)
    assert not any(e.name == "Newsletter" for e in fallback.elements)
    assert by_name(fallback, "link", "Details").source == "dom"
    monkeypatch.undo()
    recovered = reloaded.observe()
    assert reloaded.last_accessibility_error is None
    assert by_name(recovered, "checkbox", "Newsletter").source == "ax"
    assert by_name(recovered, "checkbox", "Email alerts").states == {"checked": "true"}
