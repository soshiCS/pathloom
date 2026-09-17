"""The key proof of the vision fallback: a control drawn on a <canvas> that neither the DOM projection nor the
accessibility tree exposes is found by a scripted vision planner, verified through structured perception,
recorded as a coordinate step, and replayed by the deterministic engine with no planner at all.
Local Chromium serving a local page over a local socket; no network, no model."""
import http.server
import json
import threading

import pytest

from src.cua import agent as agent_module
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import save_artifact
from src.cua.escalation import NoOperator
from src.cua.lifecycle import load_artifact, run_replay
from src.cua.policy import Policy
from tests.context import Escalator, RunLog, SessionControl
from tests.scripted_planner import ScriptedStep, ScriptedVisionPlanner, visual_click

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]

SECRET = "hunter2"
PAGE = f"""<!doctype html><html><head><title>Canvas kiosk</title></head><body>
<h1>Kiosk</h1>
<p>Operator code: {SECRET}</p>
<button id="decoy" onclick="document.getElementById('out').textContent='Wrong button'">Start session</button>
<canvas id="c" width="400" height="200" style="display:block;border:1px solid #888"></canvas>
<div id="out"></div>
<div style="height:1500px"></div>
<script>
  const c = document.getElementById('c'), ctx = c.getContext('2d');
  ctx.fillStyle = '#eee'; ctx.fillRect(0, 0, 400, 200);
  ctx.fillStyle = '#1f3a5f'; ctx.fillRect(120, 70, 160, 60);
  ctx.fillStyle = '#fff'; ctx.font = '20px sans-serif'; ctx.fillText('Start session', 135, 107);
  c.addEventListener('click', (e) => {{
    const r = c.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    if (x >= 120 && x <= 280 && y >= 70 && y <= 130) {{
      document.getElementById('out').innerHTML = '<h2>Session started</h2><a href="#" id="next">Continue</a>';
    }}
  }});
</script></body></html>"""


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("kiosk")
    (root / "index.html").write_text(PAGE)
    handler = lambda *a, **k: Quiet(*a, directory=str(root), **k)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/index.html"
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def chromium():
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        PlaywrightSurface(headless=True).close()
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    return PlaywrightSurface


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 2.0)


def canvas_box(surface):
    """Where the canvas button is drawn, in viewport pixels: the vision planner 'sees' it here."""
    r = surface._page.evaluate("() => { const r = document.getElementById('c').getBoundingClientRect();"
                               " return [r.left, r.top]; }")
    return (int(r[0]) + 120, int(r[1]) + 70, 160, 60)


def test_canvas_control_is_discovered_by_vision_and_replayed_without_a_planner(site, chromium):
    surface = chromium(headless=True, secrets=(SECRET,))
    seen_at_capture = {}
    original = surface._page.screenshot

    def capture(**kwargs):                            # what the page showed at the moment of capture
        seen_at_capture["text"] = surface._page.evaluate("() => document.body.innerText")
        return original(**kwargs)

    surface._page.screenshot = capture
    try:
        surface.navigate(site)
        box = canvas_box(surface)
        structured = surface.observe()
        # A DOM decoy shares the guessed name, but does not sit under the canvas box: the canvas control itself
        # is invisible to DOM and accessibility perception.
        assert [e.name for e in structured.elements if "Start" in e.name] == ["Start session"]
        planner = ScriptedVisionPlanner([ScriptedStep("stuck", stuck_cause="missing_control"),
                                         ScriptedStep("done", expect="Session started")],
                                        [visual_click("Start session", expect="Session started", box=box)])
        log = RunLog("discovery", secrets=(SECRET,))
        artifact = discover(goal="start a kiosk session", name="kiosk_start", params={}, surface=surface,
                            planner=planner, policy=Policy(allowed_hosts=["127.0.0.1"]),
                            escalator=Escalator(NoOperator(), SessionControl(), log), log=log, entry_url=site,
                            vision=planner)
    finally:
        surface._page.screenshot = original
        surface.close()

    assert seen_at_capture["text"].count("••••••") == 1 and SECRET not in seen_at_capture["text"]
    assert len(planner.frames) == 1 and planner.frames[0].png[:8] == b"\x89PNG\r\n\x1a\n"
    assert planner.frames[0].width > 0 and planner.frames[0].path.endswith("-vision-attempt-1.png")
    node = artifact.nodes[1]
    assert node.action.action == "click" and node.action.checkpoint == {"text_contains": "Session started"}
    [coords] = node.action.target.strategies                                                 # exact coordinates only
    assert coords == {"kind": "coords", "x": box[0] + box[2] // 2, "y": box[1] + box[3] // 2, "exact": True,
                      "viewport": {"width": planner.frames[0].width, "height": planner.frames[0].height},
                      "scroll": {"x": 0, "y": 0}}
    assert planner.frames[0].scroll_y == 0
    assert artifact.provenance["vision_fallback"]["steps"] == ["s2"]
    text = log.path.read_text()
    assert SECRET not in text and "base64" not in text
    assert '"vision_action_verified"' in text and '"verified": true' in text

    path = save_artifact(artifact, secrets=(SECRET,))
    loaded = load_artifact(path)                           # a plain linear capability graph
    replay_surface = chromium(headless=True, secrets=(SECRET,))
    replay_log = RunLog("replay", secrets=(SECRET,))
    try:
        result = run_replay(loaded, {}, replay_surface, Policy(allowed_hosts=["127.0.0.1"]),
                            Escalator(NoOperator(), SessionControl(), replay_log), replay_log, purpose="supervised")
    finally:
        replay_surface.close()
    assert result.status == "success", result                    # the canvas was clicked, not the decoy button
    replay_text = replay_log.path.read_text()
    assert '"rung": 0' in replay_text and '"strategy": "coords"' in replay_text
    assert "Wrong button" not in replay_text
    replay_events = [json.loads(line) for line in replay_text.splitlines()]
    assert not any(e["event"].startswith("vision_") for e in replay_events)          # no model, no fallback
    assert [e["label"] for e in replay_events if e["event"] == "screenshot"] == ["final-success"]


def test_replay_refuses_the_coordinates_in_a_different_viewport(site, chromium):
    surface = chromium(headless=True)
    try:
        surface.navigate(site)
        box = canvas_box(surface)
        planner = ScriptedVisionPlanner([ScriptedStep("stuck", stuck_cause="missing_control"),
                                         ScriptedStep("done", expect="Session started")],
                                        [visual_click("Start session", expect="Session started", box=box)])
        log = RunLog("discovery")
        artifact = discover(goal="start a kiosk session", name="kiosk_start", params={}, surface=surface,
                            planner=planner, policy=Policy(allowed_hosts=["127.0.0.1"]),
                            escalator=Escalator(NoOperator(), SessionControl(), log), log=log, entry_url=site,
                            vision=planner)
    finally:
        surface.close()

    other = chromium(headless=True)
    other._page.set_viewport_size({"width": 1000, "height": 500})
    replay_log = RunLog("replay")
    try:
        result = run_replay(artifact, {}, other, Policy(allowed_hosts=["127.0.0.1"]),
                            Escalator(NoOperator(), SessionControl(), replay_log), replay_log, purpose="supervised")
        clicked = other._page.evaluate("() => document.getElementById('out').textContent")
    finally:
        other.close()
    assert result.status == "failure" and result.outcome_code == "target_not_found" and result.step_id == "s2"
    assert clicked == ""                                                     # nothing was clicked blindly
    assert result.interventions[0]["disposition"] == "abort"


def test_captures_are_css_pixels_even_on_a_high_density_display(site, chromium, tmp_path):
    surface = chromium(headless=True, device_scale_factor=2)
    try:
        surface.navigate(site)
        frame = surface.viewport_screenshot(str(tmp_path / "hidpi.png"))
        from src.cua.surface import png_dimensions
        assert (frame.width, frame.height) == surface.viewport_size() == png_dimensions(frame.png)
        assert (frame.scroll_x, frame.scroll_y) == (0, 0)
    finally:
        surface.close()


def test_exact_coordinates_resolve_only_in_the_recorded_viewport_and_scroll_position(site, chromium):
    from src.cua.models import Locator
    surface = chromium(headless=True)
    try:
        surface.navigate(site)
        width, height = surface.viewport_size()
        rung = {"kind": "coords", "x": 200, "y": 100, "exact": True, "viewport": {"width": width, "height": height},
                "scroll": {"x": 0, "y": 0}}
        point = surface.resolve(Locator(strategies=[rung]))
        assert point is not None and point.box == (200, 100, 0, 0) and point.ref == ""     # the point itself
        surface._page.evaluate("() => window.scrollTo(0, 120)")
        assert surface.resolve(Locator(strategies=[rung])) is None                           # scrolled: refused
        assert surface.resolve(Locator(strategies=[{**rung, "scroll": {"x": 0, "y": 120}}])).box == (200, 100, 0, 0)
        surface._page.evaluate("() => window.scrollTo(0, 0)")
        surface._page.set_viewport_size({"width": width - 100, "height": height})
        assert surface.resolve(Locator(strategies=[rung])) is None                           # other viewport: refused
        plain = {"kind": "coords", "x": 200, "y": 100}                     # no viewport binding: resolves anywhere
        assert surface.resolve(Locator(strategies=[plain])) is not None
    finally:
        surface.close()
