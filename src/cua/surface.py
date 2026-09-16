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
from dataclasses import replace
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


def clean_text(value) -> str:
    return " ".join(str(value or "").split())


def role_matches(element: Element, strategy: dict) -> bool:
    """role + name, and the enclosing item when the strategy names one."""
    if (element.role, element.name) != (strategy["role"], strategy["name"]):
        return False
    return not strategy.get("context") or element.context == strategy["context"]


# JavaScript that walks the rendered page and reports controls as an operator sees them.
# It runs in the page, so it is the only part of the system that knows about HTML. The helpers are
# shared by the projection (OBSERVE_JS) and by the enrichment of accessibility-only nodes (ENRICH_JS).
HELPERS_JS = r"""
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
  // A control's accessible name: explicit label, its text, an image's alt, its test id or id
  // (an icon-only control such as a cart link), else where it leads. A bare number is a badge
  // (a cart count), not a name.
  const humanize = (s) => clean((s || '').replace(/[-_]+/g, ' '));
  const controlName = (el) => {
    const explicit = clean(el.getAttribute('aria-label') || el.getAttribute('title') || el.value || el.textContent);
    if (explicit && !/^\d+$/.test(explicit)) return explicit;
    const img = el.querySelector('img[alt]');
    if (img && clean(img.alt)) return img.alt;
    const testId = humanize(el.getAttribute('data-test') || el.getAttribute('data-testid') || el.id || el.getAttribute('name'));
    if (testId) return testId;
    if (explicit) return explicit;
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
"""

OBSERVE_JS = "() => {" + HELPERS_JS + r"""
  const out = [];
  const dialogEl = document.querySelector('[role="dialog"], dialog[open]');
  const dialog = dialogEl && isVisible(dialogEl) ? clean(dialogEl.innerText) : null;

  const push = (el, role, name, text) => out.push({role, name: clean(name), text: clean(text), box: box(el),
                                                   context: contextOf(el), ref: cssPath(el)});
  const CONTROL = 'a[href], button, [role=button], [role=link]';
  for (const el of document.body.querySelectorAll('*')) {
    if (!isVisible(el)) continue;
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    const aria = (el.getAttribute('role') || '').toLowerCase();
    // The tag's native role wins (an <a href> is a link whatever ARIA role it also carries, so
    // perception stays the same between discovery and replay); an ARIA role rescues elements with
    // no native control role, such as an <a role="button"> without href.
    if (tag === 'a' && el.hasAttribute('href')) push(el, 'link', controlName(el), el.textContent);
    else if (tag === 'button' || (tag === 'input' && (type === 'submit' || type === 'button')))
      push(el, 'button', controlName(el), el.value || el.textContent);
    else if (aria === 'button' || aria === 'link') push(el, aria, controlName(el), el.textContent);
    else if (tag === 'input' && ['text', 'password', 'search', 'number', 'email', 'tel', ''].includes(type))
      push(el, 'textbox', fieldName(el), type === 'password' ? '' : el.value);   // secrets are never perceived
    else if (tag === 'textarea') push(el, 'textbox', fieldName(el), el.value);
    else if (tag === 'select') push(el, 'combobox', fieldName(el), el.value);
    else if (/^h[1-6]$/.test(tag)) push(el, 'heading', el.textContent, el.textContent);
    else if (tag === 'td' && !el.querySelector('table, input, button, a')) push(el, 'cell', cellName(el), el.textContent);
    else if (['p', 'li', 'span', 'div', 'label', 'th'].includes(tag) && ownText(el) && !el.closest(CONTROL))
      push(el, 'text', '', ownText(el));   // text inside a control is the control's, not a separate thing to click
  }
  return {url: location.href, title: document.title, dialog, elements: out};
}
"""

# Geometry, visibility, text and enclosing item for elements the accessibility tree found and the
# projection did not list. One call for all of them. A password field's value is never read.
ENRICH_JS = "(refs) => {" + HELPERS_JS + r"""
  return refs.map((ref) => {
    let el = null;
    try { el = document.querySelector(ref); } catch (e) { el = null; }
    if (!el) return null;
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    let text = '';
    if (tag === 'input' || tag === 'select' || tag === 'textarea') {
      // a password is never read; a checkbox's or radio's value ("on") says nothing to an operator
      text = (type === 'password' || type === 'checkbox' || type === 'radio') ? '' : (el.value || '');
    } else text = clean(el.innerText || el.textContent);
    return {visible: isVisible(el), box: box(el), context: contextOf(el), text: clean(text)};
  });
}
"""

# Temporarily mask registered secrets before a screenshot, then restore them exactly. Two kinds of
# rendered text carry a value: text nodes (page copy, option labels, so a <select> shows the masked
# label too) and the current value of text-like form controls (<input>, <textarea>), which is not a
# text node. Values are assigned directly, which fires no input/change event and touches no app
# state; a password field renders as dots already and is left alone. The originals live only in
# page memory between the two calls.
MASK_JS = r"""
(secrets) => {
  const mask = (text) => {
    let changed = false;
    for (const secret of secrets) {
      if (secret && text.includes(secret)) { text = text.split(secret).join('••••••'); changed = true; }
    }
    return changed ? text : null;
  };
  window.__cuaMasked = [];
  window.__cuaMaskedValues = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const masked = mask(node.textContent);
    if (masked !== null) { window.__cuaMasked.push([node, node.textContent]); node.textContent = masked; }
  }
  const TEXT_TYPES = ['text', 'search', 'email', 'tel', 'url', 'number', ''];
  for (const el of document.querySelectorAll('input, textarea')) {
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (el.tagName.toLowerCase() === 'input' && !TEXT_TYPES.includes(type)) continue;   // passwords stay dots
    const masked = mask(el.value || '');
    if (masked !== null) { window.__cuaMaskedValues.push([el, el.value]); el.value = masked; }
  }
}
"""
UNMASK_JS = r"""
() => {
  for (const [node, text] of (window.__cuaMasked || [])) node.textContent = text;
  for (const [el, value] of (window.__cuaMaskedValues || [])) el.value = value;
  window.__cuaMasked = [];
  window.__cuaMaskedValues = [];
}
"""


# Accessibility roles worth surfacing as controls (Chromium's computed role -> the role we perceive).
AX_ROLES = {
    "checkbox": "checkbox", "radio": "radio", "switch": "switch", "combobox": "combobox", "listbox": "listbox",
    "option": "option", "menuitem": "menuitem", "menuitemcheckbox": "menuitem", "menuitemradio": "menuitem",
    "tab": "tab", "slider": "slider", "spinbutton": "spinbutton", "button": "button", "togglebutton": "button",
    "link": "link", "textbox": "textbox", "searchbox": "textbox", "treeitem": "treeitem",
}
# Computed states worth showing; anything else Chromium reports (focusable, invalid=false, ...) is noise.
# For the first four a false value is information (an unticked box, a collapsed menu); for the rest only
# true says anything, so required=false or readonly=false is dropped.
AX_STATES = ("checked", "expanded", "selected", "pressed", "disabled", "required", "readonly")
AX_STATES_ONLY_WHEN_TRUE = ("disabled", "required", "readonly")
MAX_AX_NODES = 400          # bound on accessibility candidates merged per observation (large pages)
SECRET_MASK = "••••••"


def dom_paths(root: dict) -> tuple[dict[int, str], dict[str, int]]:
    """From a CDP DOM.getDocument tree: backend node id -> structural path, and path -> document order.

    The path is built exactly like the page script's `cssPath` (tag:nth-of-type(n) from the body
    down, the <html> element excluded), so an accessibility node and a projected element that share
    a backing element share a `ref` and can be merged without a per-node protocol call.
    """
    paths: dict[int, str] = {}
    order: dict[str, int] = {}

    def walk(node: dict, prefix: str | None) -> None:
        counts: dict[str, int] = {}
        for child in node.get("children") or []:
            if child.get("nodeType") != 1:
                continue
            tag = (child.get("localName") or child.get("nodeName") or "").lower()
            counts[tag] = counts.get(tag, 0) + 1
            if prefix is None:                      # the <html> element: not part of any path
                walk(child, "" if tag == "html" else None)
                continue
            part = f"{tag}:nth-of-type({counts[tag]})"
            path = part if prefix == "" else f"{prefix} > {part}"
            if "backendNodeId" in child:
                paths[child["backendNodeId"]] = path
            order[path] = len(order)
            walk(child, path)

    walk(root, None)
    return paths, order


def ax_states(node: dict) -> dict:
    states = {}
    for prop in node.get("properties") or []:
        name = prop.get("name")
        if name not in AX_STATES:
            continue
        value = (prop.get("value") or {}).get("value")
        value = str(value).lower() if isinstance(value, bool) else str(value)
        if name in AX_STATES_ONLY_WHEN_TRUE and value != "true":
            continue
        states[name] = value
    return states


class PlaywrightSurface:
    """Real browser adapter backed by Playwright (Chromium).

    Perception merges two structured sources: the page projection (OBSERVE_JS: visible text,
    geometry, structural refs, action handles) and Chromium's computed accessibility tree read
    over CDP (computed roles, accessible names, states such as checked or expanded). The merge is
    deterministic, keyed by the backing element's structural path; see `_merge_accessibility`.
    """

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
        self._cdp = None                          # CDP session for the accessibility tree, opened lazily
        self.last_accessibility_error: str | None = None   # why the last observation was DOM-only, if it was
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
        elements = self._merge_accessibility(elements)
        return Observation(url=raw["url"], elements=[self._masked(e) for e in elements],
                           dialog=self._mask(raw["dialog"]) if raw["dialog"] else raw["dialog"])

    # ---------- accessibility tree ----------

    def _accessibility_nodes(self) -> tuple[list[dict], dict[int, str], dict[str, int]]:
        """Chromium's computed accessibility tree and the DOM paths to map it back: two protocol calls."""
        if self._cdp is None:
            self._cdp = self._page.context.new_cdp_session(self._page)
            self._cdp.send("Accessibility.enable")
        document = self._cdp.send("DOM.getDocument", {"depth": -1})
        paths, order = dom_paths(document["root"])
        tree = self._cdp.send("Accessibility.getFullAXTree")
        return tree.get("nodes") or [], paths, order

    def _merge_accessibility(self, elements: list[Element]) -> list[Element]:
        """Fold accessibility nodes into the projection; on any failure return the projection unchanged.

        Rules, applied per accessibility node whose computed role is in AX_ROLES and whose backing
        element is known: a node backing an element the projection already lists keeps that
        element's native role, name, text and geometry and only gains states (a generic `text`
        element, however, is upgraded to the accessibility role and name); a node backing an
        element the projection skipped becomes a new element, enriched with geometry, visible text
        and enclosing item by one page call. Everything is then ordered by document position.

        The merge is transactional: it works on copies, and the caller's elements are never
        touched, so a failure at any stage (tree, mapping, enrichment) hands back exactly the
        projection that came in.
        """
        try:
            nodes, paths, order = self._accessibility_nodes()
            merged = [replace(element, states=dict(element.states)) for element in elements]   # copies only
            by_ref = {element.ref: element for element in merged}
            pending: list[tuple[str, str, str, dict]] = []
            seen: set[str] = set()
            for node in nodes:
                role = AX_ROLES.get(str((node.get("role") or {}).get("value", "")).lower())
                ref = paths.get(node.get("backendDOMNodeId"))
                if node.get("ignored") or role is None or ref is None or ref in seen:
                    continue
                seen.add(ref)
                if len(seen) > MAX_AX_NODES:
                    break
                name = clean_text((node.get("name") or {}).get("value", ""))
                states = ax_states(node)
                existing = by_ref.get(ref)
                if existing is None:
                    pending.append((ref, role, name, states))
                elif existing.role == "text":
                    existing.role, existing.name = role, name or existing.text
                    existing.states, existing.source = states, "ax"
                else:
                    existing.states, existing.source = states, "dom+ax"
            merged = merged + self._enrich(pending)
            big = len(order) + 1
            result = sorted(merged, key=lambda element: order.get(element.ref, big))
        except Exception as error:   # a supplemental source must never cost the observation
            self.last_accessibility_error = f"{type(error).__name__}: {error}"
            return elements
        self.last_accessibility_error = None
        return result

    def _enrich(self, pending: list[tuple[str, str, str, dict]]) -> list[Element]:
        if not pending:
            return []
        details = self._page.evaluate(ENRICH_JS, [ref for ref, _, _, _ in pending])
        added = []
        for (ref, role, name, states), detail in zip(pending, details):
            if not detail or not detail["visible"]:
                continue
            added.append(Element(role=role, name=name or detail["text"], text=detail["text"], box=tuple(detail["box"]),
                                 context=detail["context"], ref=ref, states=states, source="ax"))
        return added

    # ---------- secrets ----------

    def _mask(self, text: str) -> str:
        for secret in self.secrets:
            if secret in text:
                text = text.replace(secret, SECRET_MASK)
        return text

    def _masked(self, element: Element) -> Element:
        if not self.secrets:
            return element
        element.name, element.text, element.context = (self._mask(element.name), self._mask(element.text),
                                                       self._mask(element.context))
        return element

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
        """Capture the page with any registered secret masked out of the rendered text and form values.

        Masking and restoration are the two page calls above; restoration runs in `finally`, so a
        failed capture never leaves a masked value behind.
        """
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
