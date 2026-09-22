"""Which controls perception lists, in real Chromium: hidden, inert and clipped ones are not on the operator's screen;
disabled and covered ones are listed but marked; a label, an icon child or an ancestor over a control is not cover."""
import pytest

pytestmark = pytest.mark.browser

PAGE = """<!doctype html><html><head><title>Availability</title><style>
  .clipbox { width: 120px; height: 30px; overflow: hidden; position: relative; }
  .cover { position: absolute; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.05); }
</style></head><body>
<h1>Availability</h1>
<button id="ready">Ready</button>
<button id="hidden" style="display:none">Hidden</button>
<div aria-hidden="true"><button id="assistive">Assistive hidden</button></div>
<div inert><button id="inert">Inert</button></div>
<button id="disabled" disabled>Disabled</button>
<div class="clipbox"><button id="clipped" style="position:absolute; left: 400px">Clipped away</button></div>
<div style="position:relative; width:200px; height:40px"><button id="covered">Covered</button><div class="cover"></div></div>
<div role="checkbox" aria-checked="false" tabindex="0" id="axonly" style="width:20px;height:20px;border:1px solid #333"></div>
<div style="position:relative; width:40px; height:24px"><div role="checkbox" aria-checked="false" tabindex="0" id="axcovered"
  style="width:20px;height:20px;border:1px solid #333"></div><div class="cover"></div></div>
<label style="position:relative; display:inline-block"><input type="checkbox" id="labelled" style="position:absolute; left:0; top:0">
  <span style="position:relative; display:inline-block; background:#fff; padding:2px 4px">Agree to terms</span></label>
<button id="iconed"><span style="display:inline-block; width:100%; height:100%">Star</span></button>
</body></html>"""


@pytest.fixture(scope="module")
def surface(tmp_path_factory):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=2500)
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    page = tmp_path_factory.mktemp("avail") / "avail.html"
    page.write_text(PAGE)
    live.navigate(page.as_uri())
    yield live
    live.close()


def by_name(observation, name):
    return [e for e in observation.elements if e.name == name]


def test_hidden_inert_and_clipped_controls_are_not_listed(surface):
    observation = surface.observe()
    for name in ("Hidden", "Assistive hidden", "Inert", "Clipped away"):
        assert by_name(observation, name) == [], name


def test_disabled_and_covered_controls_are_listed_but_marked(surface):
    observation = surface.observe()
    [disabled] = by_name(observation, "Disabled")
    assert disabled.states.get("disabled") == "true"
    [covered] = by_name(observation, "Covered")
    assert covered.states.get("covered") == "true"


def test_accessibility_only_controls_get_the_same_treatment(surface):
    elements = surface.observe().elements
    boxes = [e for e in elements if e.role == "checkbox" and e.source == "ax"]
    assert len(boxes) >= 2
    covered = [e for e in boxes if e.states.get("covered") == "true"]
    uncovered = [e for e in boxes if e.states.get("covered") != "true"]
    assert len(covered) == 1 and len(uncovered) >= 1


def test_a_label_an_icon_child_or_the_control_itself_over_the_control_is_not_cover(surface):
    observation = surface.observe()
    [agree] = [e for e in observation.elements if e.role == "checkbox" and e.name == "Agree to terms"]
    assert "covered" not in agree.states
    [star] = by_name(observation, "Star")
    assert "covered" not in star.states and "disabled" not in star.states
    [ready] = by_name(observation, "Ready")
    assert ready.states == {}
    surface.click(ready)                                   # genuinely actionable: the ordinary click path, nothing forced
