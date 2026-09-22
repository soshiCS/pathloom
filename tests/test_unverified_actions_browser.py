"""An unprovable action in real Chromium: not recorded, not repeated, and never mistaken for a success.

The page offers four controls that a planner could easily mis-expect: one that does nothing at all, one
whose only effect is its own ARIA state, one that navigates, and one that changes text. A wrong expectation
on the inert control leaves no node and no second click; a wrong expectation on the others is replaced by
the deterministic proof the page really provides. No model; a local file, no network."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.replay import replay
from tests.context import Escalator, Policy, RecordingOperator, RunLog, SessionControl
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]

PAGE = """<!doctype html><html><head><title>Console</title></head><body>
<h1>Console</h1>
<button id="inert">Refresh cache</button>
<button id="toggle" aria-pressed="false">Compact view</button>
<button id="reveal">Show summary</button>
<p><a id="go" href="detail.html">Open detail</a></p>
<div id="out"></div>
<ul id="clicks"></ul>
<script>
  const note = (what) => { const li = document.createElement('li'); li.textContent = what; document.getElementById('clicks').appendChild(li); };
  document.getElementById('inert').addEventListener('click', () => { note('inert'); });
  document.getElementById('toggle').addEventListener('click', (e) => {
    note('toggle');
    const on = e.currentTarget.getAttribute('aria-pressed') === 'true';
    e.currentTarget.setAttribute('aria-pressed', on ? 'false' : 'true');
  });
  document.getElementById('reveal').addEventListener('click', () => {
    note('reveal'); document.getElementById('out').innerHTML = '<h2>Summary ready</h2>';
  });
  document.getElementById('go').addEventListener('click', () => { note('go'); });
</script></body></html>"""

DETAIL = """<!doctype html><html><head><title>Detail</title></head><body><h1>Detail page</h1></body></html>"""


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    folder = tmp_path_factory.mktemp("console")
    (folder / "console.html").write_text(PAGE)
    (folder / "detail.html").write_text(DETAIL)
    return (folder / "console.html").as_uri()


@pytest.fixture(scope="module")
def surface(site):
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


def clicks(surface) -> list[str]:
    return surface._page.evaluate("() => Array.from(document.querySelectorAll('#clicks li')).map(li => li.textContent)")


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def run(fresh, site, script, operator=None, max_steps=12):
    log = RunLog("discovery")
    artifact = discover(goal="prepare the console", name="console_prepare", params={}, surface=fresh,
                        planner=ScriptedPlanner(script), policy=Policy(allowed_hosts=[""]),
                        escalator=Escalator(operator or RecordingOperator("resume"), SessionControl(), log), log=log,
                        entry_url=site, max_steps=max_steps)
    return artifact, log


def names_in(artifact) -> list[str]:
    return [n.action.target.strategies[0].get("name") for n in linear_path(artifact) if n.action.target]


def test_a_real_click_whose_expectation_never_appears_is_not_recorded_and_not_repeated(fresh, site):
    blind = ScriptedStep("click", "button", "Refresh cache", expect="Cache refreshed")
    artifact, log = run(fresh, site, [blind, blind, ScriptedStep("done", expect="Console")])
    assert clicks(fresh) == ["inert"]                      # dispatched once in a real browser, never twice
    assert "Refresh cache" not in names_in(artifact)
    [unverified] = events(log, "action_effect_unverified")
    assert unverified["cause"] == "action_effect_unverified"
    assert events(log, "unverified_repeat_refused")[0]["attempts"] == 1
    validate(artifact)


def test_a_real_aria_pressed_transition_is_deterministic_proof(fresh, site):
    artifact, log = run(fresh, site, [ScriptedStep("click", "button", "Compact view", expect="Compact mode on"),
                                      ScriptedStep("done", expect="Console")])
    assert clicks(fresh) == ["toggle"]
    assert names_in(artifact).count("Compact view") == 1
    # The selection rule proves this transition and records a checkpoint replay can re-verify, which
    # supersedes the older reading that recognised the flip but could write nothing down.
    assert events(log, "action_effect_proven")[0]["proof"] == "the control is now selected (aria-pressed)"
    assert events(log, "action_effect_unverified") == []
    [toggle] = [n for n in linear_path(artifact)
                if n.action.target and n.action.target.strategies[0].get("name") == "Compact view"]
    assert toggle.action.checkpoint == {"target_selected": "true"}


def test_a_real_navigation_replaces_a_wrong_expectation_with_its_url(fresh, site):
    artifact, log = run(fresh, site, [ScriptedStep("click", "link", "Open detail", expect="Summary ready"),
                                      ScriptedStep("done", expect="Detail page")])
    opened = [n for n in linear_path(artifact)
              if n.action.target and n.action.target.strategies[0].get("name") == "Open detail"]
    assert len(opened) == 1 and opened[0].action.checkpoint["url_contains"].endswith("/detail.html")
    assert events(log, "action_effect_proven")[0]["proof"] == "the page address changed"


def test_newly_appearing_text_is_the_checkpoint_and_the_artifact_replays_without_a_model(fresh, site):
    artifact, log = run(fresh, site, [ScriptedStep("click", "button", "Show summary", expect="Summary ready"),
                                      ScriptedStep("extract", "heading", "Summary ready", output_name="state",
                                                   pattern=r"(.+)"),
                                      ScriptedStep("done", expect="Summary ready")])
    shown = [n for n in linear_path(artifact)
             if n.action.target and n.action.target.strategies[0].get("name") == "Show summary"]
    assert len(shown) == 1 and shown[0].action.checkpoint == {"text_contains": "Summary ready"}
    validate(artifact)
    log2 = RunLog("replay")
    result = replay(artifact, {}, fresh, Policy(allowed_hosts=[""]),
                    Escalator(NoOperator(), SessionControl(), log2), log2)
    assert result.status == "success" and result.outputs["state"] == "Summary ready"
    assert None not in [e["checkpoint"] for e in events(log2, "checkpoint_passed")]


def test_an_unprovable_action_hands_off_instead_of_guessing(fresh, site):
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed):
        run(fresh, site, [ScriptedStep("click", "button", "Refresh cache", expect="Cache refreshed"),
                          ScriptedStep("done", expect="Console")], operator=operator)
    assert clicks(fresh) == ["inert"] and [r.kind for r in operator.requests] == ["stuck"]
