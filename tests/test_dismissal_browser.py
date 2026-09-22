"""The safety policy's container evidence, as Chromium perceives it: native kind, enclosing landmark and its bounded
name, and the platform's own dismissal metadata. Local page, no model."""
import pytest

from src.cua.policy import Policy
from tests.context import Action

pytestmark = pytest.mark.browser

SECRET = "hunter2"
PAGE = f"""<!doctype html><html><head><title>Dismissals</title></head><body>
<h1>Dismissals</h1>
<aside id="settings"><h2>Earthquake Settings</h2><label><input type="checkbox"> Auto update</label>
  <button id="close-settings">Close</button></aside>
<dialog open id="welcome" aria-label="Welcome"><p>Hello</p><form method="dialog"><button id="close-welcome">Close</button></form>
  <button id="x" aria-label="Close">×</button></dialog>
<div role="dialog" aria-labelledby="ca-title" id="closure"><h3 id="ca-title">Close Account</h3>
  <p>Code {SECRET}</p><button id="close-closure">Close</button></div>
<form action="#" id="order"><h2>Checkout</h2><button id="submit-close">Close</button></form>
<div id="plain"><button id="bare">Close</button></div>
<nav id="menu"><button id="dismiss">Dismiss</button></nav>
</body></html>"""


@pytest.fixture(scope="module")
def surface(tmp_path_factory):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=2500, secrets=(SECRET,))
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    page = tmp_path_factory.mktemp("dismiss") / "dismiss.html"
    page.write_text(PAGE)
    live.navigate(page.as_uri())
    yield live
    live.close()


def button(surface, ref_id):
    ref = surface._page.evaluate("(id) => {" + __import__("src.cua.surface", fromlist=["HELPERS_JS"]).HELPERS_JS
                                 + " return cssPath(document.getElementById(id)); }", ref_id)
    return next(e for e in surface.observe().elements if e.ref == ref)


def test_perception_reports_native_kind_landmark_name_and_dismissal_metadata(surface):
    settings = button(surface, "close-settings")
    assert (settings.native, settings.landmark, settings.landmark_name, settings.dismisses) == (
        "button:button", "complementary", "Earthquake Settings", False)
    welcome = button(surface, "close-welcome")
    assert (welcome.landmark, welcome.landmark_name, welcome.dismisses) == ("dialog", "Welcome", True)
    icon = button(surface, "x")
    assert (icon.name, icon.text, icon.landmark) == ("Close", "×", "dialog")
    closure = button(surface, "close-closure")
    assert (closure.landmark, closure.landmark_name) == ("dialog", "Close Account")
    assert button(surface, "submit-close").native == "button:submit" and button(surface, "bare").landmark == ""
    assert button(surface, "dismiss").landmark == "navigation"


def test_secrets_never_reach_the_container_evidence(surface):
    # a landmark named by a labelledby heading; the container's copy holds the secret, its name does not
    elements = surface.observe().elements
    assert all(SECRET not in (e.landmark_name + e.context + e.name + e.text) for e in elements)


def test_the_policy_reads_the_evidence_deterministically(surface):
    policy = Policy(allowed_hosts=[""])
    verdict = lambda ref_id: policy.check(Action(kind="click", target=button(surface, ref_id)), surface.observe().url)
    assert verdict("close-settings").decision == "allow"                    # panel heading: proven dismissal
    assert verdict("close-welcome").decision == "allow"                     # <form method=dialog>: native dismissal
    assert verdict("x").decision == "allow"                                 # icon with an accessible name, in a dialog
    assert verdict("dismiss").decision == "allow"
    closure = verdict("close-closure")
    assert closure.decision == "confirm" and "looks irreversible" in closure.reason   # the container says Close Account
    assert verdict("submit-close").decision == "confirm"                    # a form's submit control, no dialog
    bare = verdict("bare")
    assert bare.decision == "confirm" and "nothing on the screen proves which" in bare.reason
