"""The observation transaction survives a document being replaced by a navigation in flight, tells that apart
from a stable document whose supplemental accessibility layer failed, and bounds core failures: decided by
exception types and the main-frame navigation counter, never by error text. Offline, on a fake page."""
from pathlib import Path

import pytest

from src.cua.models import TransientError

playwright = pytest.importorskip("playwright.sync_api")
PlaywrightError, PlaywrightTimeout = playwright.Error, playwright.TimeoutError

FORM = {"url": "http://app.test/", "title": "Search", "dialog": None, "elements": [
    {"role": "textbox", "name": "Location", "text": "", "box": [0, 0, 10, 10], "context": "", "ref": "input:nth-of-type(1)"},
    {"role": "button", "name": "Search", "text": "Search", "box": [0, 0, 10, 10], "context": "", "ref": "button:nth-of-type(1)"}]}
RESULTS = {"url": "http://app.test/results?loc=x", "title": "Results", "dialog": None, "elements": [
    {"role": "heading", "name": "Forecast for x", "text": "Forecast for x", "box": [0, 0, 10, 10], "context": "",
     "ref": "h1:nth-of-type(1)"}]}
MORE = {**RESULTS, "url": "http://app.test/results?page=2"}


class FakePage:
    """The slice of Playwright's Page the adapter touches during an observation, scripted per document.

    `fail` are exceptions the next evaluate calls raise. With `commit_on_error` each such failure is a
    navigation that destroyed the context: the next document becomes current, but the frame event only
    arrives on the next wait (as in the browser, where the event may follow the exception).
    """

    def __init__(self, documents: list, fail: list | None = None, commit_on_error: bool = False):
        self.documents = list(documents)
        self.fail = list(fail or [])
        self.commit_on_error = commit_on_error
        self.pending_event = False
        self.url = self.documents[0]["url"]
        self.main_frame = object()
        self.handlers: dict = {}
        self.goto_calls = self.load_waits = self.evaluate_calls = 0

    def on(self, event, handler):
        self.handlers.setdefault(event, []).append(handler)

    def commit(self):
        """A navigation lands: the next document is current and the main frame fires at once."""
        self.documents.pop(0)
        self.url = self.documents[0]["url"]
        self.fire()

    def fire(self):
        for handler in self.handlers.get("framenavigated", []):
            handler(self.main_frame)

    def wait_for_timeout(self, ms):
        if self.pending_event:                          # the delayed frame event arrives while waiting
            self.pending_event = False
            self.fire()

    def wait_for_load_state(self, state="load", timeout=None):
        self.load_waits += 1

    def goto(self, *args, **kwargs):
        self.goto_calls += 1

    def evaluate(self, script, *args):
        self.evaluate_calls += 1
        if self.fail:
            error = self.fail.pop(0)
            if self.commit_on_error and len(self.documents) > 1:
                self.documents.pop(0)
                self.url = self.documents[0]["url"]
                self.pending_event = True
            raise error
        if "querySelectorAll('*').length" in script:
            return len(self.documents[0]["elements"])
        return self.documents[0]


AX_CHECKBOX = ({"role": {"value": "checkbox"}, "name": {"value": "Hourly"}, "backendDOMNodeId": 7, "properties": []},)


def make_surface(page, tree=None):
    from src.cua.surface import PlaywrightSurface
    surface = PlaywrightSurface.__new__(PlaywrightSurface)
    surface._page, surface.timeout_ms, surface.secrets, surface.title = page, 1000, (), ""
    surface._last_document_status, surface._cdp, surface.last_accessibility_error = None, None, None
    surface._navigations, surface.observation_retries, surface.observation_failures = 0, 0, 0
    surface._accessibility_browser_error = False
    page.on("response", surface._remember_document_status)
    page.on("framenavigated", surface._remember_navigation)
    surface._accessibility_nodes = tree or (lambda: ([], {}, {}))
    return surface


def with_checkbox(page):
    """An accessibility tree adding one checkbox the projection did not list, with the page call it needs."""
    def tree():
        return list(AX_CHECKBOX), {7: "input:nth-of-type(2)"}, {"input:nth-of-type(2)": 5}
    original = page.evaluate

    def evaluate(script, *args):
        if script.startswith("(refs) =>"):
            return [{"visible": True, "box": [0, 0, 8, 8], "context": "", "text": ""}]
        return original(script, *args)

    page.evaluate = evaluate
    return tree


# ---------- a navigation in flight ----------

def test_a_context_destroyed_during_settle_is_retried_on_the_new_document_without_a_reload():
    page = FakePage([FORM, RESULTS], fail=[PlaywrightError("boom")], commit_on_error=True)   # unrelated wording
    surface = make_surface(page)
    observation = surface.observe()
    assert observation.url == RESULTS["url"] and [e.name for e in observation.elements] == ["Forecast for x"]
    assert page.goto_calls == 0 and surface.observation_retries == 1 and surface.observation_failures == 0
    assert page.load_waits == 3                 # settle (failed look), the wait for the navigation, settle (new look)


def test_a_navigation_committing_between_projection_and_merge_restarts_the_look():
    page = FakePage([FORM, RESULTS])
    calls = []

    def tree():
        calls.append(page.url)
        if len(calls) == 1:
            page.commit()                                  # the new document lands after the projection was read
        return [], {}, {}

    surface = make_surface(page, tree)
    observation = surface.observe()
    assert observation.url == RESULTS["url"] and surface.observation_retries == 1
    assert calls == [FORM["url"], RESULTS["url"]] and surface.last_accessibility_error is None
    assert page.goto_calls == 0


def test_a_navigation_storm_exhausts_the_bounded_attempts_into_a_transient_error():
    page = FakePage([FORM, RESULTS, MORE, MORE, MORE], fail=[PlaywrightError(str(i)) for i in range(4)],
                    commit_on_error=True)
    surface = make_surface(page)
    with pytest.raises(TransientError, match="kept navigating while it was being observed") as failed:
        surface.observe()
    assert failed.value.url is None and isinstance(failed.value.__cause__, PlaywrightError)
    assert surface.observation_retries == 3 and surface.observation_failures == 0 and page.goto_calls == 0
    assert page.load_waits == 5                 # three settles and two waits for a navigation to finish


# ---------- a stable document ----------

def test_an_accessibility_browser_error_on_a_stable_document_is_a_fallback_not_a_retry():
    page = FakePage([FORM])
    surface = make_surface(page, lambda: (_ for _ in ()).throw(PlaywrightError("Protocol error: Accessibility.enable")))
    observation = surface.observe()
    assert observation.url == FORM["url"] and [e.name for e in observation.elements] == ["Location", "Search"]
    assert not any(e.role == "checkbox" for e in observation.elements)          # no accessibility-only controls
    assert surface.last_accessibility_error.startswith("Error:")
    assert surface.observation_retries == 0 and surface.observation_failures == 0 and page.load_waits == 1
    assert page.goto_calls == 0


def test_an_accessibility_error_while_a_navigation_commits_discards_the_stale_projection():
    page = FakePage([FORM, RESULTS])
    enriched = with_checkbox(page)
    calls = []

    def tree():
        calls.append(page.url)
        if len(calls) == 1:                                # the document goes away under the accessibility call ...
            page.documents.pop(0)
            page.url = page.documents[0]["url"]
            page.pending_event = True                      # ... and the frame event follows a moment later
            raise PlaywrightError("Target closed")
        return enriched()

    surface = make_surface(page, tree)
    observation = surface.observe()
    assert observation.url == RESULTS["url"] and calls == [FORM["url"], RESULTS["url"]]
    assert surface.observation_retries == 1 and surface.last_accessibility_error is None
    assert any(e.role == "checkbox" and e.name == "Hourly" and e.source == "ax" for e in observation.elements)
    assert page.goto_calls == 0


def test_a_core_projection_error_without_navigation_is_bounded_and_never_a_fallback():
    page = FakePage([FORM], fail=[PlaywrightError("no context")] * 3)
    surface = make_surface(page)
    with pytest.raises(TransientError, match="could not be observed after 3 attempts") as failed:
        surface.observe()
    assert failed.value.url is None and isinstance(failed.value.__cause__, PlaywrightError)
    assert surface.observation_failures == 3 and surface.observation_retries == 0
    assert surface.last_accessibility_error is None and page.goto_calls == 0
    page = FakePage([FORM], fail=[PlaywrightError("hiccup")])           # one failure, then the stable document
    surface = make_surface(page)
    assert surface.observe().url == FORM["url"] and surface.observation_failures == 1


def test_a_document_that_never_loads_and_a_5xx_are_still_transient_errors_with_a_url():
    page = FakePage([FORM, RESULTS], fail=[PlaywrightError("gone")], commit_on_error=True)
    waits = []

    def never_loads(state="load", timeout=None):               # the first settle passes; the new document never loads
        waits.append(state)
        if len(waits) > 1:
            raise PlaywrightTimeout("slow")

    page.wait_for_load_state = never_loads
    with pytest.raises(TransientError, match="did not finish loading") as slow:
        make_surface(page).observe()
    assert slow.value.url == RESULTS["url"] and len(waits) == 2
    page = FakePage([FORM])
    surface = make_surface(page)
    surface._last_document_status = 503
    with pytest.raises(TransientError, match="HTTP 503") as failed:
        surface.observe()
    assert failed.value.url == FORM["url"] and surface._last_document_status is None


def test_playwright_timeouts_and_ordinary_errors_are_told_apart_by_type_only():
    source = Path("src/cua/surface.py").read_text()
    observe = source[source.index("    def observe(self)"):source.index("    def _observe_once")]
    merge = source[source.index("    def _merge_accessibility"):source.index("    def _enrich")]
    assert "Execution context" not in source and "most likely" not in source and "Protocol error" not in source
    for chunk in (observe, merge):
        assert "str(error)" not in chunk.replace('f"{type(error).__name__}: {error}"', "") and ".message" not in chunk
    page = FakePage([FORM, RESULTS], fail=[PlaywrightTimeout("slow evaluate")])
    surface = make_surface(page)
    with pytest.raises(PlaywrightTimeout):                # a timeout is not a race: it propagates as before
        surface.observe()
    assert surface.observation_retries == 0 and surface.observation_failures == 0
