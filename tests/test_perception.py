"""Perception rules that only a real page can prove; runs Chromium on an inline page and skips if it is absent."""
import pytest

from src.cua.models import Locator

pytestmark = pytest.mark.browser

PAGE = """
<html><body>
  <div id="header">
    <a class="shopping_cart_link" data-test="shopping-cart-link" href="#">
      <span class="shopping_cart_badge" data-test="shopping-cart-badge">1</span>
    </a>
    <a class="cart_button" role="button" data-test="cart-button" style="display:inline-block;width:40px;height:40px">
      <span class="cart_badge">2</span>
    </a>
    <a id="menu-open" href="#" style="display:inline-block;width:24px;height:24px"></a>
    <a href="/inventory-item.html?id=4" role="button"
       aria-label="View details for Sauce Labs Backpack">Sauce Labs Backpack</a>
    <button data-test="add-to-cart-sauce-labs-backpack">Add to cart</button>
    <span class="title">Products</span>
  </div>
</body></html>
"""


@pytest.fixture(scope="module")
def surface(tmp_path_factory):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True)
    except Exception as error:      # no browser binary in this environment
        pytest.skip(f"Chromium is not available: {error}")
    page = tmp_path_factory.mktemp("page") / "shop.html"
    page.write_text(PAGE)
    live.navigate(page.as_uri())
    yield live
    live.close()


def test_icon_only_controls_are_named_by_their_test_id_or_id(surface):
    elements = {(e.role, e.name): e for e in surface.observe().elements}
    assert ("link", "shopping cart link") in elements          # the badge count "1" is not its name
    assert ("link", "menu open") in elements                   # an icon-only link named by its id
    assert ("button", "cart button") in elements               # an <a role="button"> without href, by its test id
    assert elements[("button", "cart button")].text == "2"
    assert ("button", "Add to cart") in elements
    # Regression: a real anchor with an href stays a link even when it carries role="button". Artifacts
    # recorded it as a link, and perception must not change between discovery and replay.
    assert ("link", "View details for Sauce Labs Backpack") in elements
    assert ("button", "View details for Sauce Labs Backpack") not in elements


def test_text_inside_a_control_is_not_a_separate_element(surface):
    observation = surface.observe()
    texts = [e.text for e in observation.elements if e.role == "text"]
    assert "Products" in texts and "1" not in texts and "2" not in texts   # badges belong to their controls
    cart = surface.resolve(Locator(strategies=[{"kind": "role", "role": "link", "name": "shopping cart link"}]))
    assert cart is not None and cart.text == "1"


def test_dom_paths_match_the_page_scripts_structural_path():
    """Offline: the CDP tree is mapped to the same tag:nth-of-type paths the page script emits."""
    from src.cua.surface import dom_paths
    element = lambda tag, node_id, children=(): {"nodeType": 1, "localName": tag, "backendNodeId": node_id,
                                                 "children": list(children)}
    tree = {"nodeType": 9, "children": [
        {"nodeType": 10, "nodeName": "html"},                       # the doctype is not an element
        element("html", 1, [
            element("head", 2),
            element("body", 3, [
                element("div", 4, [element("a", 5), {"nodeType": 3, "nodeValue": "text"}, element("a", 6)]),
                element("div", 7, [element("input", 8)]),
            ]),
        ]),
    ]}
    paths, order = dom_paths(tree)
    assert paths == {2: "head:nth-of-type(1)", 3: "body:nth-of-type(1)",
                     4: "body:nth-of-type(1) > div:nth-of-type(1)",
                     5: "body:nth-of-type(1) > div:nth-of-type(1) > a:nth-of-type(1)",
                     6: "body:nth-of-type(1) > div:nth-of-type(1) > a:nth-of-type(2)",
                     7: "body:nth-of-type(1) > div:nth-of-type(2)",
                     8: "body:nth-of-type(1) > div:nth-of-type(2) > input:nth-of-type(1)"}
    assert [order[paths[i]] for i in (2, 3, 4, 5, 6, 7, 8)] == [0, 1, 2, 3, 4, 5, 6]   # document order


# ---------- a submission whose navigation lands while the screen is being observed ----------

import http.server
import threading
from urllib.parse import parse_qs, urlparse

SEARCH_PAGE = """<!doctype html><html><head><title>Weather</title></head><body>
<h1>Search</h1>
<!-- The submit fires after the click's own new-page window has elapsed, so the navigation really is in
     flight while the next observation runs: that is the race this test exists to exercise. -->
<form id="f" action="/results" method="get" onsubmit="setTimeout(() => this.submit(), 900); return false;">
  <label>Location <input type="text" name="loc" aria-label="Location"></label>
  <button type="submit">Search</button>
</form></body></html>"""


class WeatherHandler(http.server.BaseHTTPRequestHandler):
    hits: dict = {"/": 0, "/results": 0}

    def do_GET(self):
        parsed = urlparse(self.path)
        WeatherHandler.hits[parsed.path] = WeatherHandler.hits.get(parsed.path, 0) + 1
        if parsed.path == "/results":
            loc = parse_qs(parsed.query).get("loc", [""])[0]
            body = (f"<!doctype html><html><head><title>Results</title></head><body><h1>Forecast for {loc}</h1>"
                    f"<p>Sunny</p><input type='checkbox' aria-label='Hourly'><a href='/'>Back</a></body></html>")
        else:
            body = SEARCH_PAGE
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def weather():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), WeatherHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


def test_a_submission_that_navigates_during_observation_is_recorded_once_and_replays_once(weather, surface):
    from src.cua import agent as agent_module
    from src.cua.agent import discover
    from src.cua.artifact import linear_path
    from src.cua.escalation import Escalator, NoOperator, SessionControl
    from src.cua.evidence import RunLog
    from src.cua.lifecycle import run_replay
    from src.cua.policy import Policy
    from tests.scripted_planner import ScriptedPlanner, ScriptedStep

    WeatherHandler.hits = {"/": 0, "/results": 0}
    live = surface                                      # one live browser per module: a second sync instance is refused
    retries_before = live.observation_retries
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(agent_module, "EXPECT_TIMEOUT_S", 8.0)
        log = RunLog("discovery")
        artifact = discover(goal="look up the forecast", name="forecast_lookup", params={"location": "Denver"},
                            surface=live, planner=ScriptedPlanner([
                                ScriptedStep("type", "textbox", "Location", value="{{location}}"),
                                ScriptedStep("click", "button", "Search", expect="Forecast"),
                                ScriptedStep("done", expect="Forecast")]),
                            policy=Policy(allowed_hosts=["127.0.0.1"]),
                            escalator=Escalator(NoOperator(), SessionControl(), log), log=log, entry_url=weather)
        final = live.observe()
        retries = live.observation_retries - retries_before
    assert WeatherHandler.hits["/results"] == 1 and WeatherHandler.hits["/"] == 1      # submitted once, never reloaded
    assert final.url.endswith("/results?loc=Denver") and "Forecast for Denver" in " ".join(e.text for e in final.elements)
    assert any(e.role == "checkbox" and e.name == "Hourly" and e.source == "ax" for e in final.elements)   # merged after
    assert retries >= 1                                                                   # the race really happened
    actions = [(n.action.action, n.action.checkpoint) for n in linear_path(artifact)]
    assert actions == [("navigate", {"text_contains": "Search"}), ("type", None), ("click", {"text_contains": "Forecast"})]
    click = linear_path(artifact)[2]
    assert (click.effect, click.retry_safety) == ("reversible", "verify_before_retry")
    text = log.path.read_text()
    assert text.count('"event": "acted", "kind": "click"') == 1 and '"reload_failed"' not in text
    assert "Execution context" not in text and '"discovery_failed"' not in text

    replay_log = RunLog("replay")
    result = run_replay(artifact, {"location": "Boston"}, live, Policy(allowed_hosts=["127.0.0.1"]),
                        Escalator(NoOperator(), SessionControl(), replay_log), replay_log, purpose="supervised")
    assert result.status == "success" and result.performed_attempts == 3
    assert WeatherHandler.hits["/results"] == 2 and WeatherHandler.hits["/"] == 2
    assert '"action_repeated"' not in replay_log.path.read_text()
