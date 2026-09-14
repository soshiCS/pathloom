"""Surface interfaces and the Playwright browser adapter.

A surface hides whether automation is controlling a browser, desktop app, or another
user interface. Discovery and replay depend only on the Surface protocol.

Perception is deliberately *not* the DOM. `observe()` returns a flat list of controls
the way an operator would describe them (role, accessible name, visible text, the item
they belong to, box on screen). That is the same shape an accessibility tree or a
screenshot-grounded model would give for a desktop app, which keeps the recorded flow
surface-agnostic.
"""
from __future__ import annotations

import re
import time
from typing import Protocol

from .models import Element, Locator, Observation, TransientError

RENDER_SETTLE_MS = 5000   # longest we wait for client-side rendering to stop changing the page
RENDER_POLL_MS = 400
LABEL_PREFIX = re.compile(r"^([A-Za-z][A-Za-z /]{0,40}):")   # "Item total: $ 29.99" -> "Item total:"


class Surface(Protocol):
    """Operations every concrete UI adapter must provide."""

    def observe(self) -> Observation: ...
    def resolve(self, locator: Locator) -> Element | None: ...
    def click(self, target: Element) -> None: ...
    def type(self, target: Element, value: str) -> None: ...
    def navigate(self, url: str) -> None: ...
    def screenshot(self, path: str) -> str: ...


def visible_text(observation: Observation) -> str:
    """Everything the operator can read on screen, as one string, for checkpoint matching."""
    parts = [observation.dialog or ""]
    parts += [element.text or element.name for element in observation.elements]
    return "\n".join(part for part in parts if part)


def is_ambiguous(element: Element, observation: Observation) -> bool:
    """True when several controls on screen share this role and name (six "Add to cart" buttons)."""
    same = [e for e in observation.elements if (e.role, e.name) == (element.role, element.name)]
    return len(same) > 1


def locator_for(element: Element, ambiguous: bool = False) -> Locator:
    """Build the locator ladder recorded for an element, most stable strategy first.

    1. role + accessible name (+ the item it belongs to, when the name alone is ambiguous):
       how a human refers to the control; survives restyling and reordering.
    2. visible text: for controls whose text is their identity, or a "Label:" prefix for
       label/value text such as "Total: $ 32.39".
    3. structural path: exact position in the layout; stable for slow-changing legacy apps.
    4. screen coordinates: last resort, works even without any structure (screenshot mode).
    """
    strategies: list[dict] = []
    if element.name:
        rung = {"kind": "role", "role": element.role, "name": element.name}
        if ambiguous and element.context:
            rung["context"] = element.context
        strategies.append(rung)
    elif element.text and element.role in ("link", "button", "heading"):
        strategies.append({"kind": "text", "text": element.text[:80]})
    elif element.text and element.role == "text" and LABEL_PREFIX.match(element.text):
        # A value-bearing text is located by its label, never by the value that changes per input.
        strategies.append({"kind": "text", "text": LABEL_PREFIX.match(element.text).group(0)})
    if element.ref:
        strategies.append({"kind": "css", "selector": element.ref})
    x, y, w, h = element.box
    if w and h:
        strategies.append({"kind": "coords", "x": x + w // 2, "y": y + h // 2})
    return Locator(strategies=strategies)


def role_matches(element: Element, strategy: dict) -> bool:
    """role + name, and the enclosing item when the strategy names one."""
    if (element.role, element.name) != (strategy["role"], strategy["name"]):
        return False
    return not strategy.get("context") or element.context == strategy["context"]


# JavaScript that walks the rendered page and reports controls as an operator sees them.
# It runs in the page, so it is the only part of the system that knows about HTML.
OBSERVE_JS = r"""
() => {
  const out = [];
  const isVisible = (el) => {
    const r = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    return r.width > 0 && r.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
  };
  const cssPath = (el) => {
    const parts = [];
    while (el && el.nodeType === 1 && el.tagName !== 'HTML') {
      let index = 1, sib = el;
      while ((sib = sib.previousElementSibling)) if (sib.tagName === el.tagName) index++;
      parts.unshift(el.tagName.toLowerCase() + ':nth-of-type(' + index + ')');
      el = el.parentElement;
    }
    return parts.join(' > ');
  };
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const ownText = (el) => clean(Array.from(el.childNodes)
      .filter(n => n.nodeType === 3).map(n => n.textContent).join(' '));
  const box = (el) => { const r = el.getBoundingClientRect();
      return [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]; };
  // A control's accessible name: explicit label, its text, an image's alt, else where it leads.
  const controlName = (el) => {
    const explicit = el.getAttribute('aria-label') || el.getAttribute('title') || el.value || el.textContent;
    if (clean(explicit)) return explicit;
    const img = el.querySelector('img[alt]');
    if (img && clean(img.alt)) return img.alt;
    const href = el.getAttribute('href') || '';
    return href.split('/').pop().replace(/[?#].*$/, '').replace(/\.[a-z]+$/, '');
  };
  // Accessible name for a form field: real label, aria/placeholder, else the cell to its left.
  const fieldName = (el) => {
    if (el.id) { const l = document.querySelector('label[for="' + el.id + '"]'); if (l) return clean(l.textContent); }
    if (el.getAttribute('aria-label')) return clean(el.getAttribute('aria-label'));
    if (el.placeholder) return clean(el.placeholder);
    const cell = el.closest('td'); const prev = cell && cell.previousElementSibling;
    if (prev) return clean(prev.textContent);
    return clean(el.name);
  };
  // Table cells are named by their row header + column header ("Backpack Price").
  const cellName = (td) => {
    const row = td.parentElement; const table = td.closest('table');
    const rowHeader = row.cells[0] !== td ? clean(row.cells[0].textContent) : '';
    let colHeader = '';
    const headerRow = table && table.rows[0];
    if (headerRow && headerRow !== row && headerRow.cells[td.cellIndex] && headerRow.cells[td.cellIndex].tagName === 'TH')
      colHeader = clean(headerRow.cells[td.cellIndex].textContent);
    return clean(rowHeader + ' ' + colHeader);
  };
  // The item a control belongs to: the first title-like text (heading, link, label) found in
  // the nearest enclosing block that has one. That is how an operator says "the Add to cart
  // button for the Backpack".
  const contextOf = (el) => {
    let node = el.parentElement;
    for (let depth = 0; node && node !== document.body && depth < 8; depth++, node = node.parentElement) {
      for (const cand of node.querySelectorAll('h1,h2,h3,h4,h5,h6,[role=heading],a,label,th,legend,summary,dt')) {
        if (cand === el || cand.contains(el) || el.contains(cand)) continue;
        const label = clean(cand.textContent) || clean(cand.getAttribute('aria-label'));
        if (label) return label;
      }
    }
    return '';
  };
  const dialogEl = document.querySelector('[role="dialog"], dialog[open]');
  const dialog = dialogEl && isVisible(dialogEl) ? clean(dialogEl.innerText) : null;

  const push = (el, role, name, text) => out.push({role, name: clean(name), text: clean(text), box: box(el),
                                                   context: contextOf(el), ref: cssPath(el)});
  for (const el of document.body.querySelectorAll('*')) {
    if (!isVisible(el)) continue;
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'a' && el.hasAttribute('href')) push(el, 'link', controlName(el), el.textContent);
    else if (tag === 'button' || (tag === 'input' && (type === 'submit' || type === 'button')))
      push(el, 'button', controlName(el), el.value || el.textContent);
    else if (tag === 'input' && ['text', 'password', 'search', 'number', 'email', 'tel', ''].includes(type))
      push(el, 'textbox', fieldName(el), type === 'password' ? '' : el.value);   // secrets are never perceived
    else if (tag === 'textarea') push(el, 'textbox', fieldName(el), el.value);
    else if (tag === 'select') push(el, 'combobox', fieldName(el), el.value);
    else if (/^h[1-6]$/.test(tag)) push(el, 'heading', el.textContent, el.textContent);
    else if (tag === 'td' && !el.querySelector('table, input, button, a')) push(el, 'cell', cellName(el), el.textContent);
    else if (['p', 'li', 'span', 'div', 'label', 'th'].includes(tag) && ownText(el)) push(el, 'text', '', ownText(el));
  }
  return {url: location.href, title: document.title, dialog, elements: out};
}
"""

# Temporarily blank out text nodes containing a secret before a screenshot, then restore them.
MASK_JS = r"""
(secrets) => {
  window.__cuaMasked = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    let text = node.textContent, changed = false;
    for (const secret of secrets) {
      if (secret && text.includes(secret)) { text = text.split(secret).join('••••••'); changed = true; }
    }
    if (changed) { window.__cuaMasked.push([node, node.textContent]); node.textContent = text; }
  }
}
"""
UNMASK_JS = "() => { for (const [node, text] of (window.__cuaMasked || [])) node.textContent = text; window.__cuaMasked = []; }"


class PlaywrightSurface:
    """Real browser adapter backed by Playwright (Chromium)."""

    def __init__(self, headless: bool = True, timeout_ms: int = 15000, secrets: tuple[str, ...] = ()):
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=headless)
        self._page = self._browser.new_context().new_page()
        self._page.set_default_timeout(timeout_ms)
        self.timeout_ms = timeout_ms
        self.secrets = tuple(s for s in secrets if s)   # values that must never appear in evidence
        self.title = ""
        self._last_document_status: int | None = None
        # Track the status of the last top-level document so a 5xx after a click is noticed.
        self._page.on("response", self._remember_document_status)

    def _remember_document_status(self, response) -> None:
        if response.request.is_navigation_request() and response.frame == self._page.main_frame:
            self._last_document_status = response.status

    # ---------- perception ----------

    def observe(self) -> Observation:
        self._settle()
        status = self._last_document_status
        if status is not None and status >= 500:
            self._last_document_status = None
            raise TransientError(f"server returned HTTP {status}", url=self._page.url)
        raw = self._page.evaluate(OBSERVE_JS)
        self.title = raw["title"]
        elements = [Element(role=e["role"], name=e["name"], text=e["text"], box=tuple(e["box"]),
                            context=e["context"], ref=e["ref"]) for e in raw["elements"]]
        return Observation(url=raw["url"], elements=elements, dialog=raw["dialog"])

    def _settle(self) -> None:
        """Wait for a just-triggered navigation to finish, then for client-side rendering to stop.

        Modern pages keep painting after the load event, so we poll the size of what is
        perceived until two consecutive samples agree (bounded by RENDER_SETTLE_MS).
        """
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        try:
            self._page.wait_for_timeout(150)
            self._page.wait_for_load_state("load", timeout=self.timeout_ms)
        except PlaywrightTimeout as error:
            raise TransientError("page did not finish loading", url=self._page.url) from error
        previous = -1
        deadline = time.time() + RENDER_SETTLE_MS / 1000
        while time.time() < deadline:
            current = self._page.evaluate("() => document.body ? document.body.querySelectorAll('*').length : 0")
            if current == previous:
                return
            previous = current
            self._page.wait_for_timeout(RENDER_POLL_MS)

    def resolve(self, locator: Locator) -> Element | None:
        """Try each strategy in the ladder; return the first element that resolves."""
        for strategy in locator.strategies:
            element = self._resolve_one(strategy)
            if element is not None:
                return element
        return None

    def _resolve_one(self, strategy: dict) -> Element | None:
        kind = strategy.get("kind")
        if kind in ("role", "text"):
            # Match against perception, not markup, so the same rule works on any surface.
            for element in self.observe().elements:
                if kind == "role" and role_matches(element, strategy):
                    return element
                if kind == "text" and strategy["text"] and strategy["text"] in element.text:
                    return element
            return None
        if kind == "css":
            handle = self._page.query_selector(strategy["selector"])
            if handle is None or not handle.is_visible():
                return None
            box = handle.bounding_box() or {"x": 0, "y": 0, "width": 0, "height": 0}
            return Element(role="unknown", name="", text=(handle.inner_text() or "")[:200],
                           box=(int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"])),
                           ref=strategy["selector"])
        if kind == "coords":
            # Screenshot-style fallback: whatever control is drawn at that point on screen.
            x, y = strategy["x"], strategy["y"]
            for element in self.observe().elements:
                ex, ey, ew, eh = element.box
                if ex <= x <= ex + ew and ey <= y <= ey + eh and element.role != "text":
                    return element
            return Element(role="unknown", name="", box=(x, y, 1, 1))
        return None

    # ---------- action ----------

    def click(self, target: Element) -> None:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        try:
            if target.ref:
                self._page.locator(target.ref).first.click()
            else:
                x, y, w, h = target.box
                self._page.mouse.click(x + w / 2, y + h / 2)
        except PlaywrightTimeout as error:
            raise TransientError(f"click on {target.role} '{target.name}' timed out", url=self._page.url) from error

    def type(self, target: Element, value: str) -> None:
        if target.ref:
            self._page.locator(target.ref).first.fill(value)
        else:
            self.click(target)
            self._page.keyboard.type(value)

    def navigate(self, url: str) -> None:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        try:
            response = self._page.goto(url, wait_until="load")
        except PlaywrightTimeout as error:
            raise TransientError(f"navigation to {url} timed out", url=url) from error
        if response is not None and response.status >= 500:
            raise TransientError(f"server returned HTTP {response.status}", url=url)

    def screenshot(self, path: str) -> str:
        """Capture the page with any registered secret masked out of the rendered text."""
        if self.secrets:
            self._page.evaluate(MASK_JS, list(self.secrets))
        try:
            self._page.screenshot(path=path, full_page=True)
        finally:
            if self.secrets:
                self._page.evaluate(UNMASK_JS)
        return path

    def close(self) -> None:
        self._browser.close()
        self._playwright.stop()
