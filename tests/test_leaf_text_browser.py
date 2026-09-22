"""Visible leaf text in real Chromium: what the projection keeps, what it excludes, what the accessibility tree adds,
and the grounded visual extraction of values structured perception cannot reach (a shadow tree, a canvas)."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua import surface as surface_module
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.replay import replay
from tests.context import Escalator, Policy, RecordingOperator, RunLog, SessionControl
from tests.scripted_planner import ScriptedStep, ScriptedVisionPlanner, visual_extract

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]

SECRET = "hunter2"
PAGE = f"""<!doctype html><html><head><title>Leaf text</title><style>
  .row {{ display: flex; gap: 8px; width: 320px; }}
  .clip {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 38px; display: inline-block; }}
  .price::after {{ content: "$5.00"; }}
  .bullet::before {{ content: "•"; }}
  .flat {{ display: contents; }}
  .zero {{ width: 0; height: 0; overflow: hidden; }}
</style></head><body>
<h1>Leaf text</h1>
<div id="rows">
  <div class="row"><span>4.5</span><span>Place A</span><time>10:00</time><aside class="clip">36.3 km</aside></div>
  <div class="row"><span>4.6</span><span>Place B</span><time>11:00</time><aside class="clip">53.9 km</aside></div>
  <div class="row"><span>4.7</span><span>Place C</span><time>12:00</time><aside class="clip">57.8 km</aside></div>
</div>
<small>Fine print</small> <x-note>Custom note</x-note>
<span class="price" id="price"></span> <span class="bullet">Bulleted item</span>
<div class="flat">Contents text</div>
<p style="display:none">Hidden text</p>
<p aria-hidden="true">Assistive hidden</p>
<p style="position:absolute; left:-9999px">Offscreen text</p>
<div class="zero"><span>Zero area</span></div>
<div inert><span>Inert text</span></div>
<a href="#x"><b>Link bold</b></a> <h2><em>Heading em</em></h2>
<table><tr><th>Name</th><th>Qty</th></tr><tr><td><b>Widget</b></td><td>3</td></tr></table>
<aside id="code">Code {SECRET}</aside>
<div id="many"></div>
<x-stats id="stats"></x-stats>
<canvas id="c" width="200" height="40"></canvas>
<script>
  const many = document.getElementById('many');
  for (let i = 1; i <= 60; i++) {{ const d = document.createElement('div'); d.className = 'flat'; d.textContent = 'AX ' + i; many.appendChild(d); }}
  const root = document.getElementById('stats').attachShadow({{mode: 'open'}});
  root.innerHTML = '<style>.v{{margin:0 12px}}</style><div><span class="v">12.5 km</span><span class="v">53.9 km</span><span class="v">57.8 km</span></div>';
  const ctx = document.getElementById('c').getContext('2d');
  ctx.font = '16px sans-serif'; ctx.fillText('99.9 km', 10, 25);
</script></body></html>"""


@pytest.fixture(scope="module")
def surface(tmp_path_factory):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=2500, secrets=(SECRET,))
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    page = tmp_path_factory.mktemp("leaf") / "leaf.html"
    page.write_text(PAGE)
    live.page_url = page.as_uri()
    yield live
    live.close()


@pytest.fixture
def fresh(surface):
    surface.navigate(surface.page_url)
    surface._page.evaluate("() => window.scrollTo(0, 0)")
    return surface


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 2.0)


def texts(observation):
    return [e.text for e in observation.elements if e.role == "text"]


# ---------- 1: clipped leaf text in any tag ----------

def test_clipped_leaf_text_in_non_span_tags_is_observed(fresh):
    seen = texts(fresh.observe())
    for value in ("36.3 km", "53.9 km", "57.8 km", "10:00", "Fine print", "Custom note"):
        assert value in seen, value


# ---------- 2: accessibility static text supplements the projection ----------

def test_static_text_under_a_zero_box_wrapper_comes_from_the_accessibility_tree(fresh):
    contents = [e for e in fresh.observe().elements if e.text == "Contents text"]
    assert len(contents) == 1 and contents[0].source == "ax" and contents[0].box[2] > 0 and contents[0].name == ""


# ---------- 3: exclusions and de-duplication ----------

def test_hidden_offscreen_zero_area_inert_duplicate_and_decorative_text_stay_out(fresh):
    observation = fresh.observe()
    seen = texts(observation)
    for excluded in ("Hidden text", "Assistive hidden", "Offscreen text", "Zero area", "Inert text", "Link bold",
                     "Heading em", "Widget", "•", "• Bulleted item"):
        assert excluded not in seen, excluded
    assert "Bulleted item" in seen and "$5.00" in seen                     # generated text with a digit counts
    links = [e for e in observation.elements if e.role == "link" and e.text == "Link bold"]
    headings = [e for e in observation.elements if e.role == "heading" and e.text == "Heading em"]
    cells = [e for e in observation.elements if e.role == "cell" and e.text == "Widget"]
    assert len(links) == 1 and len(headings) == 1 and len(cells) == 1
    assert seen.count("36.3 km") == 1


# ---------- 4: caps ----------

def test_static_text_contribution_is_bounded_and_deterministic(fresh):
    first, second = fresh.observe(), fresh.observe()
    added = [e.text for e in first.elements if e.source == "ax" and e.text.startswith("AX ")]
    assert len(added) == surface_module.MAX_AX_TEXT_NODES - 1               # "Contents text" took one slot
    assert added == [f"AX {i}" for i in range(1, len(added) + 1)]
    assert [(e.role, e.text, e.ref) for e in first.elements] == [(e.role, e.text, e.ref) for e in second.elements]


# ---------- 9: secrets ----------

def test_secrets_stay_masked_in_leaf_text_and_under_a_box(fresh):
    observation = fresh.observe()
    assert f"Code {surface_module.SECRET_MASK}" in texts(observation) and SECRET not in json.dumps(
        [e.__dict__ for e in observation.elements])
    box = fresh._page.evaluate("() => { const r = document.getElementById('code').getBoundingClientRect();"
                               " return [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]; }")
    grounded = fresh.text_under(tuple(box))
    assert grounded is not None and grounded.text == f"Code {surface_module.SECRET_MASK}"


# ---------- 5, 6, 7, 8: grounded visual extraction ----------

DEPTHS = {"depths": {"type": "list", "required": True, "min_items": 3, "max_items": 3,
                     "items": {"type": "number", "pattern": r"(\d+(?:\.\d+)?) km"}}}
MISSING = ScriptedStep("stuck", stuck_cause="missing_data", output_name="depths")
DONE = ScriptedStep("done", expect="Leaf text")


def shadow_boxes(surface):
    return [tuple(b) for b in surface._page.evaluate(
        "() => Array.from(document.getElementById('stats').shadowRoot.querySelectorAll('.v')).map(el => {"
        " const r = el.getBoundingClientRect(); return [Math.round(r.x), Math.round(r.y), Math.round(r.width),"
        " Math.round(r.height)]; })")]


def run(surface, script, decisions, operator=None):
    planner = ScriptedVisionPlanner(script, decisions)
    log = RunLog("discovery", secrets=(SECRET,))
    artifact = discover(goal="read the three depths", name="depth_list", params={}, surface=surface, planner=planner,
                        policy=Policy(allowed_hosts=[""]), escalator=Escalator(operator or NoOperator(), SessionControl(), log),
                        log=log, entry_url=surface.page_url, vision=planner, max_vision_attempts=2, output_contract=DEPTHS)
    return artifact, log


def replay_on(surface, artifact, operator=None):
    log = RunLog("replay", secrets=(SECRET,))
    return replay(artifact, {}, surface, Policy(allowed_hosts=[""]), Escalator(operator or NoOperator(), SessionControl(), log),
                  log), log


def test_shadow_values_are_grounded_by_the_page_not_the_model_and_replay_without_a_model(fresh):
    assert not any("12.5 km" in e.text for e in fresh.observe().elements)   # structured perception cannot see them
    boxes = shadow_boxes(fresh)
    artifact, log = run(fresh, [MISSING, DONE], [visual_extract("depths", boxes, readings=["1.0", "2.0", "3.0"])])
    validate(artifact)
    assert artifact.outputs["depths"]["example"] == ["12.5", "53.9", "57.8"]
    node = linear_path(artifact)[-1]
    assert node.action.action == "extract_many" and [len(t.strategies) for t in node.action.targets] == [1, 1, 1]
    assert all(t.strategies[0]["kind"] == "coords" and t.strategies[0]["read"] for t in node.action.targets)
    assert "1.0" not in log.path.read_text().split('"vision_fallback_decided"')[1][:400]
    assert artifact.provenance["vision_fallback"]["grounded_extractions"][0]["ladders"] == [["coords"]] * 3
    result, replay_log = replay_on(fresh, artifact)
    assert result.status == "success" and result.outputs == {"depths": [12.5, 53.9, 57.8]}
    assert '"vision' not in replay_log.path.read_text()


def test_a_box_over_canvas_pixels_is_rejected_and_handed_off(fresh):
    box = tuple(fresh._page.evaluate("() => { const r = document.getElementById('c').getBoundingClientRect();"
                                     " return [Math.round(r.x) + 5, Math.round(r.y) + 5, 80, 25]; }"))
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run(fresh, [MISSING], [visual_extract("depths", [box] * 3)], operator=operator)
    assert operator.requests[0].reason.startswith("planner is stuck")


def test_viewport_and_scroll_drift_refuse_the_coordinate_rung_at_replay(fresh):
    boxes = shadow_boxes(fresh)
    artifact, _ = run(fresh, [MISSING, DONE], [visual_extract("depths", boxes)])
    page = fresh._page
    original = page.viewport_size
    try:
        page.set_viewport_size({"width": original["width"] + 100, "height": original["height"]})
        result, _ = replay_on(fresh, artifact, RecordingOperator("abort"))
        assert result.status == "failure" and result.outcome_code == "target_not_found"
    finally:
        page.set_viewport_size(original)
    # A replay begins with the entry navigation, so scroll drift can only arise mid-flow: the rung itself refuses it.
    fresh.navigate(fresh.page_url)
    ladder = linear_path(artifact)[-1].action.targets[0]
    assert fresh.resolve(ladder) is not None and fresh.resolve(ladder).text == "12.5 km"
    page.evaluate("() => { document.body.style.paddingTop = '2000px'; window.scrollTo(0, 400); }")
    assert fresh.scroll_position() == (0, 400) and fresh.resolve(ladder) is None
