"""Controls an application builds from nonsemantic elements: visible, named by browser-visible metadata and
styled as pointer targets. Perceived once, clicked through the ordinary trial, recorded and replayed."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import discover
from src.cua.artifact import linear_path, to_dict, validate
from src.cua.escalation import NoOperator
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.context import Escalator, RunLog, SessionControl
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]

SECRET = "hunter2"
PAGE = f"""<!doctype html><html><head><title>Icons</title><style>
  .pointer {{ cursor: pointer; display: inline-block; width: 24px; height: 24px; border: 1px solid #888; }}
  .plain {{ display: inline-block; width: 24px; height: 24px; border: 1px solid #ccc; }}
  .clipbox {{ width: 20px; height: 20px; overflow: hidden; }}
  .inert-zone {{ }}
  #out {{ margin-top: 12px; }}
</style></head><body>
<h1>Icons</h1>
<i id="settings" class="pointer" aria-label="Settings" title="Settings"></i>
<div id="filters" class="pointer" title="Filters"></div>
<i id="unnamed" class="pointer"></i>
<i id="named-only" class="plain" aria-label="Static badge" title="Static badge"></i>
<i id="hidden" class="pointer" aria-label="Hidden icon" style="display:none"></i>
<div aria-hidden="true"><i id="assistive" class="pointer" aria-label="Assistive icon"></i></div>
<div inert><i id="inert" class="pointer" aria-label="Inert icon"></i></div>
<i id="disabled" class="pointer" aria-label="Disabled icon" aria-disabled="true"></i>
<div class="clipbox"><i id="clipped" class="pointer" aria-label="Clipped icon" style="position:absolute; left:400px"></i></div>
<i id="no-events" class="pointer" aria-label="Untouchable icon" style="pointer-events:none"></i>
<div id="outer" class="pointer" aria-label="Outer wrapper"><i id="inner" class="pointer" aria-label="Inner action"></i></div>
<div id="holder" class="pointer" aria-label="Holder"><button id="real">Real button</button></div>
<button id="native">Native</button>
<div role="button" tabindex="0" id="aria-btn" class="plain" aria-label="Aria button"></div>
<i id="secretish" class="pointer" aria-label="Open {SECRET} panel" title="Open {SECRET} panel"></i>
<div id="out"></div>
<script>
  const say = (what) => {{ document.getElementById('out').textContent = what; }};
  document.getElementById('settings').addEventListener('click', () => say('Settings opened'));
  document.getElementById('filters').addEventListener('click', () => say('Filters opened'));
  document.getElementById('inner').addEventListener('click', () => say('Inner clicked'));
</script></body></html>"""


@pytest.fixture(scope="module")
def page_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("icons") / "icons.html"
    path.write_text(PAGE)
    return path.as_uri()


@pytest.fixture
def surface(page_file):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=2500, secrets=(SECRET,))
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    live.page_url = page_file
    live.navigate(page_file)
    yield live
    live.close()


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 2.0)


def named(surface, name):
    return [e for e in surface.observe().elements if e.name == name]


def shown(surface):
    return surface._page.evaluate("() => document.getElementById('out').textContent")


def test_a_named_pointer_element_appears_once_as_an_actionable_control(surface):
    [settings] = named(surface, "Settings")
    assert settings.role == "button" and settings.ref and settings.box[2] > 0
    [filters] = named(surface, "Filters")          # named by title alone
    assert filters.role == "button"


def test_it_can_be_clicked_recorded_and_replayed_without_a_planner(surface):
    policy = Policy(allowed_hosts=[""])
    log = RunLog("discovery", secrets=(SECRET,))
    artifact = discover(goal="open the settings panel", name="icon_click", params={}, surface=surface,
                        planner=ScriptedPlanner([ScriptedStep("click", "button", "Settings", expect="Settings opened"),
                                                 ScriptedStep("done", expect="Settings opened")]),
                        policy=policy, escalator=Escalator(NoOperator(), SessionControl(), log), log=log,
                        entry_url=surface.page_url)
    validate(artifact)
    click = linear_path(artifact)[1]
    assert click.action.action == "click"
    assert click.action.target.strategies[0] == {"kind": "role", "role": "button", "name": "Settings"}
    assert shown(surface) == "Settings opened"

    surface.navigate(surface.page_url)
    replay_log = RunLog("replay", secrets=(SECRET,))
    result = replay(artifact, {}, surface, policy, Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success" and shown(surface) == "Settings opened"


def test_a_pointer_element_without_a_name_is_excluded(surface):
    refs = {e.ref for e in surface.observe().elements}
    unnamed = surface._page.evaluate("() => {" + __import__("src.cua.surface", fromlist=["HELPERS_JS"]).HELPERS_JS
                                     + " return cssPath(document.getElementById('unnamed')); }")
    assert unnamed not in refs


def test_a_named_element_without_actionability_evidence_is_excluded(surface):
    assert [e for e in named(surface, "Static badge") if e.role == "button"] == []


@pytest.mark.parametrize("name", ["Hidden icon", "Assistive icon", "Inert icon", "Disabled icon", "Clipped icon",
                                  "Untouchable icon"])
def test_unavailable_candidates_are_excluded(surface, name):
    assert [e for e in named(surface, name) if e.role == "button"] == []


def test_a_named_pointer_child_and_its_ancestor_do_not_both_appear(surface):
    assert len(named(surface, "Inner action")) == 1
    assert [e for e in named(surface, "Outer wrapper") if e.role == "button"] == []
    assert [e for e in named(surface, "Holder") if e.role == "button"] == []     # it contains a real control


def test_native_and_aria_controls_keep_their_roles_and_appear_once(surface):
    [native] = [e for e in surface.observe().elements if e.name == "Native"]
    assert native.role == "button"
    aria_ref = surface._page.evaluate("() => {" + __import__("src.cua.surface", fromlist=["HELPERS_JS"]).HELPERS_JS
                                      + " return cssPath(document.getElementById('aria-btn')); }")
    [aria] = [e for e in surface.observe().elements if e.ref == aria_ref]
    assert aria.role == "button"                          # its explicit ARIA role still wins
    refs = [e.ref for e in surface.observe().elements]
    assert len(refs) == len(set(refs))                    # nothing is listed twice


def test_secrets_in_names_and_titles_stay_masked(surface):
    observation = surface.observe()
    assert SECRET not in json.dumps([e.__dict__ for e in observation.elements])
    assert any("Open" in e.name and "panel" in e.name for e in observation.elements if e.role == "button")


def test_production_code_carries_no_site_or_framework_specifics():
    from pathlib import Path

    source = Path("src/cua/surface.py").read_text()
    for banned in ("ng-", "data-reactid", "__vue", "_ngcontent", "usaspending", "openstreetmap", "usgs"):
        assert banned not in source.lower(), banned
    rule = source[source.index("const namedPointerControl"):source.index("// Elements whose emitted text")]
    assert "force" not in rule and ".click(" not in rule and "dispatchEvent" not in rule
