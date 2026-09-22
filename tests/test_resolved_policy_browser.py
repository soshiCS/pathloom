"""Replay judges the control it resolved on the live page, not the locator stored in the artifact.

An approved bare `Close` inside a settings panel carries no dismissal evidence in its serialized locator;
only the resolved element knows its enclosing landmark. Judging the locator alone denied such an action
during automatic reuse while ordinary discovery allowed the very same button seconds later.
"""
import json
from dataclasses import replace as replace_fields

import pytest

from src.cua.artifact import build_linear, save_artifact
from src.cua.escalation import NoOperator
from src.cua.library import build_library, execute_candidate, find_candidates
from src.cua.models import GraphAction, Locator
from src.cua.policy import Policy
from src.cua.replay import replay
from src.cua.surface import HELPERS_JS
from tests.context import Escalator, RunLog, SessionControl, linear_node

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]

PAGE = """<!doctype html><html><head><title>Panels</title></head><body>
<h1>Panels</h1>
<aside id="settings"><h2>Display Settings</h2><p>Options</p><button id="close-settings">Close</button></aside>
<div role="region" aria-labelledby="ca"><h3 id="ca">Close Account</h3><button id="close-account">Close</button></div>
<div><button id="bare">Close</button></div>
<a href="https://elsewhere.test/away">Away</a>
</body></html>"""


@pytest.fixture(scope="module")
def page_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("panels") / "panels.html"
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


def path_of(surface, element_id):
    return surface._page.evaluate("(id) => {" + HELPERS_JS + " return cssPath(document.getElementById(id)); }", element_id)


def close_artifact(surface, element_id, name="panel_dismiss", checkpoint=None):
    ladder = Locator(strategies=[{"kind": "css", "selector": path_of(surface, element_id)}])
    nodes = [linear_node("s1", GraphAction(action="navigate", target=None, value=surface.page_url,
                                           checkpoint={"text_contains": "Panels"})),
             linear_node("s2", GraphAction(action="click", target=ladder, value=None, checkpoint=checkpoint))]
    return build_linear(name=name, goal="dismiss a panel", params={},
                        surface_meta={"kind": "web", "app": "panels", "entry_url": surface.page_url,
                                      "allowed_hosts": [""]},
                        nodes=nodes, outputs={}, outcomes=[], success={"text_contains": "Panels"},
                        run_id="test-run", sensitive=set(), planner_name="scripted")


def run_replay(surface, artifact, policy=None, irreversible="deny"):
    log = RunLog("replay")
    result = replay(artifact, {}, surface, policy or Policy(allowed_hosts=[""]),
                    Escalator(NoOperator(), SessionControl(), log), log, irreversible_policy=irreversible)
    return result, log


def decisions(log, step_id="s2"):
    return [json.loads(l) for l in log.path.read_text().splitlines()
            if '"policy_checked"' in l and json.loads(l).get("step_id") == step_id]


def test_a_reversible_close_in_a_panel_replays_under_deny_because_the_live_control_proves_it(surface):
    artifact = close_artifact(surface, "close-settings")
    assert artifact.nodes[1].effect == "reversible"
    result, log = run_replay(surface, artifact)
    assert result.status == "success"
    [decision] = decisions(log)
    assert decision["decision"] == "allow" and decision["evidence"] == "resolved"


def test_the_same_stored_action_is_automatically_reusable_under_deny(surface):
    artifact = close_artifact(surface, "close-settings", checkpoint={"text_contains": "Panels"})
    saved = save_artifact(replace_fields(artifact, status="approved", version=1), secrets=())
    surface.navigate(surface.page_url)
    log = RunLog("discovery")
    library = build_library(str(saved.parent), surface.page_url, [""], {}, set(), log, max_reuses=5)
    candidates = [c for c in find_candidates(library, surface.observe(), {}, surface, log, turn=1) if c.start_node == "s2"]
    assert candidates, "the stored Close action should be offered as a candidate"
    execution = execute_candidate(library, candidates[0], {}, surface, Policy(allowed_hosts=[""]), log, turn=1)
    assert execution.result is not None and execution.result.status == "success"
    assert execution.result.executed_path and execution.result.executed_path[0]["id"] == "s2"


def test_a_bare_close_with_no_dismissal_evidence_is_still_refused(surface):
    result, log = run_replay(surface, close_artifact(surface, "bare", name="bare_dismiss"))
    assert result.status == "failure" and result.outcome_code == "irreversible_denied"
    assert decisions(log)[0]["decision"] == "confirm"


def test_a_close_inside_a_resource_closing_container_is_still_refused(surface):
    result, log = run_replay(surface, close_artifact(surface, "close-account", name="account_dismiss"))
    assert result.status == "failure" and result.outcome_code == "irreversible_denied"
    assert "looks irreversible" in decisions(log)[0]["reason"]


def test_live_evidence_can_override_a_stale_safer_stored_effect(surface):
    # the artifact calls it reversible; the live control resolves into a resource closure, and the stricter view wins
    artifact = close_artifact(surface, "close-account", name="stale_effect")
    assert artifact.nodes[1].effect == "reversible"
    result, log = run_replay(surface, artifact, irreversible="confirm")
    assert result.status == "failure" and result.outcome_code in ("irreversible_not_confirmed", "irreversible_denied")
    assert surface.observe().url == surface.page_url          # nothing was clicked


def test_a_missing_target_performs_nothing(surface):
    artifact = close_artifact(surface, "close-settings", name="gone")
    artifact.nodes[1].action.target = Locator(strategies=[{"kind": "css", "selector": "button#not-here"}])
    result, log = run_replay(surface, artifact)
    assert result.status == "failure" and result.outcome_code == "target_not_found"
    assert result.performed_attempts == 1                      # only the entry navigation
    assert decisions(log) == []                                # the control was never judged: nothing resolved


def test_an_ambiguous_target_performs_nothing(surface):
    artifact = close_artifact(surface, "close-settings", name="ambiguous")
    artifact.nodes[1].action.target = Locator(strategies=[{"kind": "role", "role": "button", "name": "Close"}])
    result, log = run_replay(surface, artifact)
    assert result.status == "failure" and result.outcome_code == "ambiguous_target"
    assert result.performed_attempts == 1 and decisions(log) == []


def test_the_navigation_allowlist_is_unchanged(surface):
    artifact = close_artifact(surface, "close-settings", name="off_host")
    artifact.nodes[1].action = GraphAction(action="navigate", target=None, value="https://elsewhere.test/away")
    result, log = run_replay(surface, artifact)
    assert result.status == "failure" and result.outcome_code == "policy_denied"
    assert "outside the allowlist" in result.observed and surface.observe().url == surface.page_url
