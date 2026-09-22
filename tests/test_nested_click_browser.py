"""Visible text nested inside a clickable container, on a local page in real Chromium, no model.

The text itself cannot take the click (it does not receive pointer events), so the adapter climbs to the nearest
enclosing control that is actionable by its own semantics, or reports `unactionable_target` without touching
anything; the runtime then offers that state to the bounded vision fallback or to a person."""
import json
import re
from pathlib import Path

import pytest

from src.cua import agent as agent_module
from src.cua import surface as surface_module
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import linear_path
from src.cua.escalation import NoOperator
from src.cua.models import ActionError, Element, Locator
from src.cua.replay import replay
from tests.context import Escalator, Policy, RecordingOperator, RunLog, SessionControl
from tests.scripted_planner import ScriptedStep, ScriptedVisionPlanner, visual_click

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]

PAGE = """<!doctype html><html><head><title>Nested text</title><style>
  .box { position: relative; display: inline-block; padding: 6px; border: 1px solid #ccc; min-width: 220px; }
  .cap { position: absolute; left: 0; top: 0; width: 100%; height: 100%; }   /* a decoration laid over the text */
  #shade { position: absolute; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.05); }
</style></head><body>
<h1>Nested text</h1>
<p><a href="#linked" id="linked" class="box"><span>Nested in link</span><i class="cap"></i></a></p>
<p><button id="native" class="box"><span>Nested in button</span><i class="cap"></i></button></p>
<div role="button" tabindex="0" id="aria" class="box"><span>Nested in aria button</span><i class="cap"></i></div>
<div role="menu"><div role="menuitem" tabindex="0" id="item" class="box"><span>Nested in menu item</span><i class="cap"></i></div></div>
<div class="box" id="row"><span>Row text</span><i class="cap"></i></div>
<section id="section"><div class="box plain"><span>Plain text</span><i class="cap"></i></div><a href="#elsewhere">Elsewhere</a></section>
<div style="position:relative; width:260px"><a href="#covered" id="covered" class="box"><span>Covered text</span><i class="cap"></i></a><div id="shade"></div></div>
<p><label for="agree" id="agree-label" class="box"><span>Agree to terms</span><i class="cap"></i></label> <input type="checkbox" id="agree"></p>
<div id="out"></div>
<ul id="events"></ul>
<script>
  const log = (what) => { const li = document.createElement('li'); li.textContent = what; document.getElementById('events').appendChild(li); };
  const show = (text) => { document.getElementById('out').innerHTML = '<h2>' + text + '</h2>'; };
  document.getElementById('linked').addEventListener('click', (e) => { e.preventDefault(); log('link'); show('Link detail'); });
  document.getElementById('native').addEventListener('click', () => { log('button'); show('Button detail'); });
  document.getElementById('aria').addEventListener('click', () => { log('aria'); show('Aria detail'); });
  document.getElementById('item').addEventListener('click', () => { log('menuitem'); show('Menu detail'); });
  document.getElementById('row').addEventListener('click', () => { log('row'); show('Depth: 10 km'); });
  document.getElementById('section').addEventListener('click', () => { log('section'); show('Section detail'); });
  document.getElementById('covered').addEventListener('click', (e) => { e.preventDefault(); log('covered'); show('Covered detail'); });
  document.getElementById('shade').addEventListener('click', () => { log('shade'); });
  document.getElementById('agree').addEventListener('change', () => { log('agree'); });
</script></body></html>"""


@pytest.fixture(scope="module")
def surface(tmp_path_factory):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=2500)
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    page = tmp_path_factory.mktemp("nested") / "nested.html"
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


def text_inside(surface, container_id: str) -> Element:
    """The nested text as an element of its own, by structural reference: what a planner would be choosing."""
    ref, text, box = surface._page.evaluate("(id) => {" + surface_module.HELPERS_JS + """
      const el = document.getElementById(id).querySelector('span');
      return [cssPath(el), clean(el.textContent), box(el)]; }""", container_id)
    return Element(role="text", name="", text=text, box=tuple(box), ref=ref)


def page_events(surface) -> list[str]:
    return surface._page.evaluate("() => Array.from(document.querySelectorAll('#events li')).map(li => li.textContent)")


def shown(surface) -> str:
    return surface._page.evaluate("() => document.getElementById('out').textContent")


def listed(surface, role: str, text: str) -> Element:
    return next(e for e in surface.observe().elements if e.role == role and text in (e.name or e.text))


# ---------- 1, 2: the exact element first; a semantic ancestor only when the element cannot take the click ----------

@pytest.mark.parametrize("container, event, detail", [
    ("linked", "link", "Link detail"), ("native", "button", "Button detail"), ("aria", "aria", "Aria detail")])
def test_text_nested_in_a_native_link_button_or_role_button_is_actionable_as_itself(fresh, container, event, detail):
    """Step 1 of the order: the browser's own actionability accepts text inside a button, a link or a role=button
    (it retargets the hit test to that control), so the exact element is clicked and nothing is climbed."""
    assert fresh.click(text_inside(fresh, container)) is None
    assert page_events(fresh) == [event] and shown(fresh) == detail


def test_text_nested_in_an_explicit_interactive_role_the_browser_does_not_retarget_uses_that_ancestor(fresh):
    activated = fresh.click(text_inside(fresh, "item"))
    assert activated is not None and (activated.role, activated.name) == ("menuitem", "Nested in menu item")
    assert page_events(fresh) == ["menuitem"] and shown(fresh) == "Menu detail"


def test_discovery_records_the_ancestor_ladder_and_replay_needs_no_model(fresh):
    policy = Policy(allowed_hosts=[""])
    span = text_inside(fresh, "item")

    class NestedPlanner(ScriptedVisionPlanner):
        """Chooses the nested text by reference, the way a model that saw only the text would."""
        def decide(self, goal, params, observation, history, candidates=()):
            if not history:
                return agent_module.Action(kind="click", target=span, expect="Menu detail", reason="scripted")
            return super().decide(goal, params, observation, history, candidates)

    log = RunLog("discovery")
    artifact = discover(goal="open the menu item", name="nested_item", params={}, surface=fresh,
                        planner=NestedPlanner([ScriptedStep("done", expect="Menu detail")], []), policy=policy,
                        escalator=Escalator(NoOperator(), SessionControl(), log), log=log, entry_url=fresh.page_url)
    click = linear_path(artifact)[1]
    assert click.action.target.strategies[0] == {"kind": "role", "role": "menuitem", "name": "Nested in menu item"}
    assert click.action.target.strategies[1]["kind"] == "css" and span.ref not in json.dumps(click.action.target.strategies)
    assert click.action.checkpoint == {"text_contains": "Menu detail"}
    acted = next(json.loads(l) for l in log.path.read_text().splitlines() if '"event": "acted"' in l)
    assert acted["performed_on"] == "menuitem 'Nested in menu item'"
    fresh.navigate(fresh.page_url)
    replay_log = RunLog("replay")
    result = replay(artifact, {}, fresh, policy, Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success" and page_events(fresh) == ["menuitem"]


# ---------- 3: no usable semantics -> unactionable_target -> scripted vision -> verified -> replayed ----------

def test_a_javascript_only_row_is_unactionable_then_clicked_by_vision_and_replayed(fresh):
    policy = Policy(allowed_hosts=[""])
    span = text_inside(fresh, "row")
    with pytest.raises(ActionError) as refused:
        fresh.click(span)
    assert refused.value.performed == "no" and refused.value.cause == "unactionable_target"
    assert "no enclosing control is provably actionable" in str(refused.value) and page_events(fresh) == []

    class RowPlanner(ScriptedVisionPlanner):
        def decide(self, goal, params, observation, history, candidates=()):
            if not history:
                return agent_module.Action(kind="click", target=span, expect="Depth", reason="scripted")
            return super().decide(goal, params, observation, history, candidates)

    x, y, w, h = span.box
    planner = RowPlanner([ScriptedStep("done", expect="Depth")], [visual_click("Row text", expect="Depth", box=(x, y, w, h))])
    log = RunLog("discovery")
    artifact = discover(goal="open the row", name="nested_row", params={}, surface=fresh, planner=planner, policy=policy,
                        escalator=Escalator(NoOperator(), SessionControl(), log), log=log, entry_url=fresh.page_url,
                        vision=planner, max_vision_attempts=2)
    text = log.path.read_text()
    assert '"cause": "unactionable_target"' in text and '"stuck_cause": "actionability"' in text
    assert '"event": "vision_action_verified", "verified": true' in text
    click = linear_path(artifact)[1]
    assert click.action.target.strategies[0]["kind"] == "coords" and click.action.target.strategies[0]["exact"] is True
    assert click.action.checkpoint == {"text_contains": "Depth"} and artifact.provenance["vision_fallback"]["attempts"] == 1
    assert page_events(fresh) == ["row"]                      # exactly one physical click, by vision, no forced click
    fresh.navigate(fresh.page_url)
    replay_log = RunLog("replay")
    result = replay(artifact, {}, fresh, policy, Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success" and shown(fresh) == "Depth: 10 km"


# ---------- 4, 5: unrelated ancestors and overlays are never used ----------

def test_an_unrelated_ancestor_with_a_script_handler_is_never_selected(fresh):
    plain = fresh._page.evaluate("() => {" + surface_module.HELPERS_JS + """
      const el = document.querySelector('#section .plain span'); return [cssPath(el), clean(el.textContent), box(el)]; }""")
    with pytest.raises(ActionError) as refused:
        fresh.click(Element(role="text", name="", text=plain[1], box=tuple(plain[2]), ref=plain[0]))
    assert refused.value.performed == "no" and refused.value.cause == "unactionable_target"
    assert page_events(fresh) == [] and shown(fresh) == ""      # neither the section nor the sibling link was clicked


def test_an_overlay_over_the_control_is_never_bypassed(fresh):
    with pytest.raises(ActionError) as refused:
        fresh.click(text_inside(fresh, "covered"))
    assert refused.value.performed == "no" and refused.value.cause == "unactionable_target"
    assert "is not actionable either" in str(refused.value)
    assert page_events(fresh) == [] and shown(fresh) == ""      # the shade was not clicked through, nothing was forced


# ---------- 6: a click that may have been dispatched is not sent to vision or repeated ----------

def test_a_possibly_dispatched_click_is_handed_to_a_person_not_to_vision(fresh, monkeypatch):
    policy = Policy(allowed_hosts=[""])
    real_click = type(fresh).click

    def click_then_lose_track(self, target):
        real_click(self, target)                                # the click really reaches the page
        raise ActionError("click on the row did not complete: the adapter lost track of the page", performed="unknown")

    monkeypatch.setattr(type(fresh), "click", click_then_lose_track)
    planner = ScriptedVisionPlanner([ScriptedStep("click", "button", "Nested in button", expect="Something else")],
                                    [visual_click("Nested in button", expect="Button detail")])
    log = RunLog("discovery")
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        discover(goal="press", name="nested_button", params={}, surface=fresh, planner=planner, policy=policy,
                 escalator=Escalator(operator, SessionControl(), log), log=log, entry_url=fresh.page_url,
                 vision=planner, max_vision_attempts=2)
    text = log.path.read_text()
    assert '"vision_fallback_requested"' not in text and planner.frames == []
    assert page_events(fresh) == ["button"]                     # dispatched once, never repeated
    assert operator.requests[0].reason.startswith("planner is stuck: click on button 'Nested in button' ended in an unknown state")


# ---------- 7: choice controls and their labels are untouched ----------

def test_text_nested_in_a_label_still_activates_its_control_through_the_label(fresh):
    activated = fresh.click(text_inside(fresh, "agree-label"))
    assert activated is not None and activated.role == "label"
    assert fresh._page.evaluate("() => document.getElementById('agree').checked") is True
    assert page_events(fresh) == ["agree"]
    fresh.click(listed(fresh, "checkbox", "Agree to terms"))     # the direct choice path is unchanged
    assert fresh._page.evaluate("() => document.getElementById('agree').checked") is False


# ---------- 8: nothing forced, nothing scripted ----------

def test_the_ancestor_path_contains_no_forced_click_or_direct_activation():
    source = Path("src/cua/surface.py").read_text()
    adapter = source[source.index("class PlaywrightSurface"):]            # past the Surface protocol's stubs
    path = adapter[adapter.index("def click(self, target: Element)"):adapter.index("def _activate_choice")]
    assert "def _click_actionable_ancestor" in path
    assert "force=" not in path and "force = " not in path
    referenced = path[path.index("locator = self._page.locator(target.ref).first"):]     # the by-reference path
    assert "mouse.click" not in referenced and "coords" not in referenced
    js = surface_module.ACTIONABLE_ANCESTOR_JS
    assert ".click(" not in js and "dispatchEvent" not in js and ".focus(" not in js
    assert re.search(r"\.(checked|value|open)\s*=[^=]", js) is None and "evaluate(\"(el) => el.click" not in path
    assert path.count("ancestor.click()") == 1 and path.count("locator.click()") == 1   # child or parent, never both
