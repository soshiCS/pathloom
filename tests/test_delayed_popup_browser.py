"""New pages that arrive late, in real Chromium: the wait is for the event, not for one poll.

A browser may emit its `page` event several ticks after the click call returns. An empty list is therefore
not evidence of a same-page click until a bounded first-popup window has fully elapsed; only once a page
has appeared does the second question arise, whether more are coming. These tests drive the real algorithm
with genuinely delayed events rather than an instantly-opening link, so the timing is what is under test.
No model; a local HTTP server on loopback, no outside network."""
import http.server
import json
import threading

import pytest

from src.cua import agent as agent_module
from src.cua import surface as surface_module
from src.cua.agent import discover
from src.cua.artifact import linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.models import ActionError
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.context import Escalator, RecordingOperator, RunLog, SessionControl
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


# Each opener waits a stated number of milliseconds before opening, so the page event genuinely arrives
# after one or more polling intervals rather than during the click call.
PAGE = """<!doctype html><html><head><title>Results</title></head><body>
<h1>Results</h1>
<a href="#" id="late" onclick="setTimeout(() => window.open('child1.html'), 350); return false">Late one</a>
<a href="#" id="verylate" onclick="setTimeout(() => window.open('child1.html'), 500); return false">Very late one</a>
<a href="#" id="twolate" onclick="setTimeout(() => window.open('child1.html'), 200);
   setTimeout(() => window.open('child2.html'), 450); return false">Two late</a>
<a href="#" id="away" onclick="setTimeout(() => window.open('http://127.0.0.2:9/away.html'), 300); return false">Away late</a>
<a href="#" id="none" onclick="return false">Opens nothing</a>
<a href="plain.html" id="plain">Plain link</a>
</body></html>"""


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("delayed")
    (root / "index.html").write_text(PAGE)
    for n in (1, 2):
        (root / f"child{n}.html").write_text(
            f"<!doctype html><html><head><title>Child {n}</title></head><body>"
            f"<h1>Child Page {n}</h1><p>Detail {n}</p></body></html>")
    (root / "plain.html").write_text(
        "<!doctype html><html><head><title>Plain</title></head><body><h1>Plain Page</h1></body></html>")
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
    live.navigate(site)
    yield live
    live.close()


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 2.0)


def link(surface, name):
    return next(e for e in surface.observe().elements if e.role == "link" and e.name == name)


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


# ---------- the race itself ----------

def test_a_page_arriving_after_several_polling_intervals_is_still_adopted(surface):
    """The regression: the opener emits nothing for 350ms, far longer than one polling interval."""
    assert surface_module.POPUP_FIRST_MS > 350 > surface_module.POPUP_POLL_MS
    assert surface.click(link(surface, "Late one")) is None
    observed = surface.observe()
    assert observed.url.endswith("child1.html")
    assert "Child Page 1" in " ".join(e.text for e in observed.elements)
    assert surface._opener is not None                    # the opener is kept, so `back` can close the child


def test_a_page_arriving_near_the_end_of_the_first_window_is_still_adopted(surface):
    assert surface_module.POPUP_FIRST_MS > 500
    assert surface.click(link(surface, "Very late one")) is None
    assert surface.observe().url.endswith("child1.html")


def test_two_pages_emitted_on_separated_ticks_are_both_seen_and_refused(surface):
    # 200ms then 450ms: the second arrives well after the first, so only continued collection catches it.
    with pytest.raises(ActionError) as refused:
        surface.click(link(surface, "Two late"))
    assert refused.value.cause == "ambiguous_selection" and refused.value.performed == "unknown"
    assert surface.observe().url.endswith("index.html")   # nothing was chosen; the run stays put
    assert len(surface._page.context.pages) == 1          # both children were closed


def test_a_click_that_opens_nothing_is_an_ordinary_same_page_click(surface):
    assert surface.click(link(surface, "Opens nothing")) is None
    assert surface.observe().url.endswith("index.html")
    assert surface._opener is None


def test_an_ordinary_navigation_in_the_same_tab_is_unchanged(surface):
    assert surface.click(link(surface, "Plain link")) is None
    observed = surface.observe()
    assert observed.url.endswith("plain.html") and "Plain Page" in " ".join(e.text for e in observed.elements)
    assert surface._opener is None                        # a same-tab navigation is not an adopted popup


def test_a_late_child_outside_the_allowlist_is_closed_and_the_run_stays_on_the_opener(surface):
    with pytest.raises(ActionError) as refused:
        surface.click(link(surface, "Away late"))
    assert refused.value.performed == "unknown"
    assert surface.observe().url.endswith("index.html")
    assert len(surface._page.context.pages) == 1


# ---------- the waiting is bounded, and paid once per click ----------

def test_the_wait_for_a_first_page_is_bounded_and_small(surface):
    import time

    started = time.monotonic()
    surface.click(link(surface, "Opens nothing"))
    elapsed_ms = (time.monotonic() - started) * 1000
    # An ordinary click pays the whole first window, and no more than it plus a little slack.
    assert elapsed_ms >= surface_module.POPUP_FIRST_MS - surface_module.POPUP_POLL_MS
    assert elapsed_ms < surface_module.POPUP_FIRST_MS + surface_module.POPUP_SETTLE_MS + 1500


def test_an_empty_list_is_never_called_stable_before_the_first_window_elapses(surface):
    """The exact shape of the old bug: the loop exited after a single poll while the list was empty."""
    polls = []
    original = surface._page.wait_for_timeout

    def counting(ms):
        polls.append(ms)
        return original(ms)

    surface._page.wait_for_timeout = counting
    try:
        surface.click(link(surface, "Opens nothing"))
    finally:
        surface._page.wait_for_timeout = original
    waited = sum(polls)
    assert waited >= surface_module.POPUP_FIRST_MS       # not one poll, the whole window
    assert len(polls) >= surface_module.POPUP_FIRST_MS // surface_module.POPUP_POLL_MS


# ---------- discovery and replay ----------

def test_discovery_records_the_late_click_once_and_replay_opens_the_child_once(surface, site):
    log = RunLog("discovery")
    steps = [ScriptedStep("click", "link", "Late one", expect="Child Page 1"),
             ScriptedStep("extract", "text", text="Detail 1", output_name="detail", pattern=r"(.+)"),
             ScriptedStep("done", expect="Child Page 1")]
    artifact = discover(goal="open the first result", name="result_detail", params={}, surface=surface,
                        planner=ScriptedPlanner(steps), policy=Policy(allowed_hosts=["127.0.0.1"]),
                        escalator=Escalator(RecordingOperator("resume"), SessionControl(), log), log=log,
                        entry_url=site, max_steps=12)
    validate(artifact)
    clicks = [n for n in linear_path(artifact)
              if n.action.action == "click" and n.action.target.strategies[0].get("name") == "Late one"]
    assert len(clicks) == 1 and clicks[0].action.checkpoint == {"text_contains": "Child Page 1"}
    assert events(log, "action_effect_unverified") == []

    # Replayed on the same live session, returned to the entry page: the click must open the child again,
    # exactly once, with no planner involved.
    surface.navigate(site)
    replay_log = RunLog("replay")
    before_pages = len(surface._page.context.pages)
    result = replay(artifact, {}, surface, Policy(allowed_hosts=["127.0.0.1"]),
                    Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success" and result.outputs["detail"] == "Detail 1"
    assert surface.observe().url.endswith("child1.html")
    assert len(surface._page.context.pages) == before_pages + 1    # exactly one child, not two
