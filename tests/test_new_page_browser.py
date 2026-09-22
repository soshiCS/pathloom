"""Links that open a new tab in real Chromium: the child page becomes active, `back` closes it, three details are
read in order, several popups and a disallowed destination fail structurally, and replay needs no planner."""
import http.server
import json
import threading

import pytest

from src.cua import agent as agent_module
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.models import ActionError
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.context import Escalator, RecordingOperator, RunLog, SessionControl
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]

AWARDS = {1: ("Agency One", "1200000"), 2: ("Agency Two", "3400000"), 3: ("Agency Three", "5600000")}


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("awards")
    rows = "".join(f"<p><a href='award{n}.html' target='_blank'>Award {n}</a></p>" for n in AWARDS)
    (root / "index.html").write_text(
        f"<!doctype html><html><head><title>Awards</title></head><body><h1>Award Search</h1>"
        f"{rows}"
        f"<a href='#' id='twins' onclick=\"window.open('award1.html');window.open('award2.html');return false\">Twins</a>"
        # a real, resolvable address that is simply not in the allowlist: the refusal must not depend on DNS
        f"<a href='http://127.0.0.2:9/away.html' target='_blank' id='away'>Away</a>"
        f"<a href='plain.html' id='plain'>Plain</a></body></html>")
    for n, (agency, amount) in AWARDS.items():
        (root / f"award{n}.html").write_text(
            f"<!doctype html><html><head><title>Award {n}</title></head><body><h1>Award Overview</h1>"
            f"<p>Awarding Agency: {agency}</p><p>Award Amount: {amount}</p></body></html>")
    (root / "plain.html").write_text("<!doctype html><html><head><title>Plain</title></head><body><h1>Plain Page</h1></body></html>")
    handler = lambda *a, **k: Quiet(*a, directory=str(root), **k)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/index.html"
    server.shutdown()
    server.server_close()


@pytest.fixture
def surface(site):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=3000, allowed_hosts=["127.0.0.1"])
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    live.entry = site
    live.navigate(site)
    yield live
    live.close()


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 2.0)


def link(surface, name):
    return next(e for e in surface.observe().elements if e.role == "link" and e.name == name)


def texts(surface):
    return " ".join(e.text for e in surface.observe().elements)


def test_a_result_link_opens_a_new_tab_that_becomes_active_and_exposes_detail_fields(surface):
    assert surface.click(link(surface, "Award 1")) is None
    observation = surface.observe()
    assert observation.url.endswith("award1.html") and "Award Overview" in " ".join(e.text for e in observation.elements)
    assert "Awarding Agency: Agency One" in texts(surface)
    assert surface._opener is not None                      # the results page is kept for `back`


def test_back_closes_the_child_page_and_returns_to_the_results(surface):
    surface.click(link(surface, "Award 2"))
    assert surface.observe().url.endswith("award2.html")
    surface.back()
    assert surface.observe().url.endswith("index.html") and "Award Search" in texts(surface)
    surface.click(link(surface, "Award 3"))
    assert "Agency Three" in texts(surface)


def test_several_new_pages_fail_structurally_and_stay_on_the_results_page(surface):
    with pytest.raises(ActionError) as refused:
        surface.click(link(surface, "Twins"))
    assert refused.value.performed == "unknown" and refused.value.cause == "ambiguous_selection"
    assert "2 new pages" in str(refused.value) and surface.observe().url.endswith("index.html")


def test_a_destination_outside_the_allowlist_is_closed_and_automation_stays_put(surface):
    with pytest.raises(ActionError) as refused:
        surface.click(link(surface, "Away"))
    assert refused.value.performed == "unknown" and "outside the allowlist" in str(refused.value)
    assert surface.observe().url.endswith("index.html") and surface._opener is None


def test_an_ordinary_same_page_link_is_unchanged(surface):
    surface.click(link(surface, "Plain"))
    assert surface.observe().url.endswith("plain.html") and surface._opener is None
    surface.back()
    assert surface.observe().url.endswith("index.html")


def test_discovery_and_replay_visit_three_new_tabs_in_order_without_a_planner(surface):
    policy = Policy(allowed_hosts=["127.0.0.1"])
    script = []
    for n in AWARDS:
        script += [ScriptedStep("click", "link", f"Award {n}", expect="Award Overview"),
                   ScriptedStep("extract", "text", text="Awarding Agency:", output_name=f"agency_{n}",
                                pattern=r"Awarding Agency: (.+)"),
                   ScriptedStep("back", expect="Award Search")]
    script.append(ScriptedStep("done", expect="Award Search"))
    log = RunLog("discovery")
    artifact = discover(goal="read each award's agency", name="award_agencies", params={}, surface=surface,
                        planner=ScriptedPlanner(script), policy=policy,
                        escalator=Escalator(NoOperator(), SessionControl(), log), log=log, entry_url=surface.entry)
    validate(artifact)
    assert [n.action.action for n in linear_path(artifact)] == ["navigate"] + ["click", "extract", "back"] * 3
    assert {f"agency_{n}": AWARDS[n][0] for n in AWARDS} == {k: v["example"] for k, v in artifact.outputs.items()}
    acted = [json.loads(l) for l in log.path.read_text().splitlines() if '"event": "acted"' in l]
    assert sum(1 for e in acted if e["kind"] == "click") == 3                      # each link clicked once

    surface.navigate(surface.entry)
    replay_log = RunLog("replay")
    result = replay(artifact, {}, surface, policy, Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success"
    assert result.outputs == {f"agency_{n}": AWARDS[n][0] for n in AWARDS}
    replayed = [json.loads(l) for l in replay_log.path.read_text().splitlines() if '"event": "acted"' in l]
    assert sum(1 for e in replayed if e["action"] == "click") == 3
