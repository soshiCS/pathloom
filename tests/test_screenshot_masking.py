"""Screenshots never show a registered secret: text nodes and form-control values are masked for the
capture and restored exactly afterwards, without firing application events. Local Chromium, no network."""
import pytest

pytestmark = pytest.mark.browser

SECRET = "hunter2"
MASK = "••••••"
PAGE = f"""
<html><body>
  <p id="copy">Training password: {SECRET} (keep it private)</p>
  <input id="plain" type="text" value="prefix-{SECRET}-suffix">
  <textarea id="notes">line one
contains {SECRET} here
line three</textarea>
  <input id="pw" type="password" value="{SECRET}">
  <input id="other" type="text" value="nothing secret">
  <select id="pick"><option>{SECRET}</option><option>safe</option></select>
  <script>
    window.__events = [];
    for (const el of document.querySelectorAll('input, textarea, select')) {{
      for (const kind of ['input', 'change', 'keydown', 'keyup']) {{
        el.addEventListener(kind, () => window.__events.push(el.id + ':' + kind));
      }}
    }}
  </script>
</body></html>
"""
CURRENT_JS = """() => ({
  copy: document.getElementById('copy').textContent,
  plain: document.getElementById('plain').value,
  notes: document.getElementById('notes').value,
  pw: document.getElementById('pw').value,
  other: document.getElementById('other').value,
  pick: document.getElementById('pick').options[0].textContent,
  events: window.__events.slice(),
})"""


@pytest.fixture(scope="module")
def surface(tmp_path_factory):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, secrets=(SECRET,))
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    page = tmp_path_factory.mktemp("page") / "secrets.html"
    page.write_text(PAGE)
    live.navigate(page.as_uri())
    yield live
    live.close()


@pytest.fixture
def fresh(surface):
    surface.navigate(surface._page.url)
    return surface


def current(surface) -> dict:
    return surface._page.evaluate(CURRENT_JS)


def test_secrets_are_masked_during_capture_and_restored_exactly(fresh, tmp_path):
    before = current(fresh)
    seen = {}
    original = fresh._page.screenshot

    def capture(**kwargs):                     # what the renderer draws at capture time, deterministically
        seen.update(current(fresh))
        return original(**kwargs)

    fresh._page.screenshot = capture
    try:
        fresh.screenshot(str(tmp_path / "masked.png"))
    finally:
        fresh._page.screenshot = original
    assert seen["copy"] == f"Training password: {MASK} (keep it private)"       # text node
    assert seen["plain"] == f"prefix-{MASK}-suffix"                             # input, partial string
    assert seen["notes"] == f"line one\ncontains {MASK} here\nline three"         # textarea, mid-string
    assert seen["pick"] == MASK                                                 # option label is a text node
    assert seen["pw"] == SECRET and seen["other"] == "nothing secret"           # dots already; untouched
    assert SECRET not in " ".join(str(v) for k, v in seen.items() if k not in ("pw", "events"))
    assert current(fresh) == before                                            # restored to the byte
    assert before["events"] == [] and current(fresh)["events"] == []           # no input/change/key events


def test_restoration_happens_even_when_capture_fails(fresh, tmp_path):
    before = current(fresh)

    def boom(**kwargs):
        raise RuntimeError("capture failed")

    original = fresh._page.screenshot
    fresh._page.screenshot = boom
    try:
        with pytest.raises(RuntimeError, match="capture failed"):
            fresh.screenshot(str(tmp_path / "never.png"))
    finally:
        fresh._page.screenshot = original
    assert current(fresh) == before and current(fresh)["events"] == []


def test_the_saved_image_differs_from_an_unmasked_capture(fresh, tmp_path):
    """Deterministic pixel check to complement the capture-time DOM check above: the masked capture is not
    the unmasked page, and both are real PNGs of the same viewport."""
    unmasked_path = tmp_path / "unmasked.png"
    fresh._page.screenshot(path=str(unmasked_path), full_page=True)      # the raw page, secret visible
    masked_path = tmp_path / "masked.png"
    fresh.screenshot(str(masked_path))
    again_path = tmp_path / "masked_again.png"
    fresh.screenshot(str(again_path))
    unmasked, masked, again = unmasked_path.read_bytes(), masked_path.read_bytes(), again_path.read_bytes()
    assert masked[:8] == b"\x89PNG\r\n\x1a\n" and len(masked) > 1000
    assert masked != unmasked                                          # masking changed what was drawn
    assert masked == again                                             # and it is reproducible
    restored = current(fresh)                                          # everything back, nothing fired
    assert restored["plain"] == f"prefix-{SECRET}-suffix" and restored["pick"] == SECRET
    assert restored["events"] == []
