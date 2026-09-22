"""Browser regression: a results page, three detail pages read one at a time into one ordered list output, and
`back` through the real browser history; discovered once with a scripted planner, replayed with no model."""
import http.server
import json
import threading
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.browser

ITEMS = {"a": ("Item A", 71), "b": ("Item B", 58), "c": ("Item C", 90)}


class CatalogHandler(http.server.BaseHTTPRequestHandler):
    hits: dict = {}

    def do_GET(self):
        path = urlparse(self.path).path
        CatalogHandler.hits[path] = CatalogHandler.hits.get(path, 0) + 1
        key = path.rsplit("/", 1)[-1]
        if path.startswith("/items/") and key in ITEMS:
            name, score = ITEMS[key]
            body = (f"<!doctype html><html><head><title>Catalog</title></head><body><h1>Details</h1>"
                    f"<p>{name}</p><p>Score: {score}</p></body></html>")
        else:
            links = "".join(f"<li><a href='/items/{key}'>{name}</a></li>" for key, (name, _) in ITEMS.items())
            body = (f"<!doctype html><html><head><title>Catalog</title></head><body><h1>Results</h1>"
                    f"<ul>{links}</ul></body></html>")
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def catalog():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), CatalogHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


@pytest.fixture(scope="module")
def surface():
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True)
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    yield live
    live.close()


def test_detail_pages_are_read_into_one_list_and_left_through_the_browser_history(catalog, surface):
    from src.cua import agent as agent_module
    from src.cua.agent import discover
    from src.cua.artifact import linear_path, validate
    from src.cua.escalation import Escalator, NoOperator, SessionControl
    from src.cua.evidence import RunLog
    from src.cua.lifecycle import run_replay
    from src.cua.policy import Policy
    from tests.scripted_planner import ScriptedPlanner, ScriptedStep

    CatalogHandler.hits = {}
    policy = Policy(allowed_hosts=["127.0.0.1"])
    script = []
    for name in ("Item A", "Item B", "Item C"):
        script += [ScriptedStep("click", "link", name, expect="Details"),
                   ScriptedStep("extract", "text", text="Score:", output_name="scores", pattern=r"Score: (\d+)",
                                output_mode="append"),
                   ScriptedStep("back", expect="Results")]
    script.append(ScriptedStep("done", expect="Results"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(agent_module, "EXPECT_TIMEOUT_S", 8.0)
        log = RunLog("discovery")
        artifact = discover(goal="read the score of every result", name="catalog_scores", params={}, surface=surface,
                            planner=ScriptedPlanner(script), policy=policy,
                            escalator=Escalator(NoOperator(), SessionControl(), log), log=log, entry_url=catalog)
    validate(artifact)
    path = linear_path(artifact)
    assert [n.action.action for n in path] == ["navigate"] + ["click", "extract", "back"] * 3
    assert artifact.outputs["scores"]["example"] == ["71", "58", "90"]
    assert artifact.outputs["scores"]["items"] == {"type": "integer", "pattern": r"Score: (\d+)"}
    assert [n.action.checkpoint for n in path if n.action.action == "back"] == [{"text_contains": "Results"}] * 3
    assert [n.action.checkpoint for n in path if n.action.action == "click"] == [{"text_contains": "Details"}] * 3
    assert CatalogHandler.hits["/"] == 1               # back came from the history cache, not a new request
    text = log.path.read_text()
    assert text.count('"event": "acted", "kind": "back"') == 3 and '"discovery_failed"' not in text

    replay_log = RunLog("replay")
    result = run_replay(artifact, {}, surface, policy, Escalator(NoOperator(), SessionControl(), replay_log),
                        replay_log, purpose="supervised")
    assert result.status == "success" and result.outputs == {"scores": [71, 58, 90]}
    assert [e["id"] for e in result.executed_path] == [f"s{i}" for i in range(1, 11)]
    assert result.performed_attempts == 7               # navigate, three clicks, three backs; extracts are reads
    events = [json.loads(line) for line in replay_log.path.read_text().splitlines()]
    assert [e["total"] for e in events if e["event"] == "output_extracted"] == [1, 2, 3]
    assert surface.observe().url.rstrip("/") == catalog.rstrip("/")
