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

import json
import re
import struct
import time
from contextlib import contextmanager
from urllib.parse import urlparse
from dataclasses import replace
from typing import Protocol

from .models import ActionError, Element, Locator, Observation, ScreenshotFrame, TransientError

RENDER_SETTLE_MS = 5000   # longest we wait for client-side rendering to stop changing the page
RENDER_POLL_MS = 400
MAX_OBSERVE_ATTEMPTS = 3  # looks at the screen: while a navigation keeps replacing the document, or after a core failure
NAVIGATION_CONFIRM_MS = 500   # how long a browser error may wait for the navigation event that explains it
NAVIGATION_POLL_MS = 50
LABEL_PREFIX = re.compile(r"^([A-Za-z][A-Za-z /]{0,40}):")   # "Item total: $ 29.99" -> "Item total:"


class Surface(Protocol):
    """Operations every concrete UI adapter must provide."""

    def observe(self) -> Observation: ...
    def resolve(self, locator: Locator, exclude: frozenset[str] = frozenset()) -> Element | None: ...
    def matches(self, strategy: dict) -> list[Element]: ...
    def click(self, target: Element) -> Element | None: ...
    def type(self, target: Element, value: str) -> None: ...
    def select(self, target: Element, value: str) -> None: ...
    def navigate(self, url: str) -> None: ...
    def back(self) -> None: ...
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


def locator_for(element: Element, ambiguous: bool = False, viewport: tuple[int, int] | None = None) -> Locator:
    """Build the locator ladder recorded for an element, most stable strategy first.

    1. role + accessible name (+ the item it belongs to, when the name alone is ambiguous):
       how a human refers to the control; survives restyling and reordering.
    2. visible text: for controls whose text is their identity, or a "Label:" prefix for
       label/value text such as "Total: $ 32.39".
    3. structural path: exact position in the layout; stable for slow-changing legacy apps.
    4. screen coordinates: last resort, works even without any structure (screenshot mode).
       With `viewport`, the rung records the viewport the coordinates belong to, so replay can
       refuse to use them in a different one.
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
        rung = {"kind": "coords", "x": x + w // 2, "y": y + h // 2}
        if viewport:
            rung["viewport"] = {"width": viewport[0], "height": viewport[1]}
        strategies.append(rung)
    return Locator(strategies=strategies)


def png_dimensions(png: bytes) -> tuple[int, int]:
    """Width and height from a PNG's IHDR chunk (the first chunk, right after the signature)."""
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("not a PNG image")
    width, height = struct.unpack(">II", png[16:24])
    return width, height


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
  // Content that is on the page but not for the operator: hidden or inert subtrees, boxes fully
  // clipped by an overflow-hiding ancestor, and boxes laid out entirely outside the document.
  const hiddenAway = (el) => !!el.closest('[aria-hidden="true"], [inert]');
  const overlaps = (a, b) => Math.min(a.right, b.right) > Math.max(a.left, b.left)
                            && Math.min(a.bottom, b.bottom) > Math.max(a.top, b.top);
  const clippedAway = (el, rect) => {
    const r = rect || el.getBoundingClientRect();
    const doc = document.documentElement;
    const page = {left: -window.scrollX, top: -window.scrollY,
                  right: Math.max(doc.scrollWidth, doc.clientWidth) - window.scrollX,
                  bottom: Math.max(doc.scrollHeight, doc.clientHeight) - window.scrollY};
    if (!overlaps(r, page)) return true;
    for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
      const cs = window.getComputedStyle(node);
      if ((cs.overflow !== 'visible' || cs.overflowX !== 'visible' || cs.overflowY !== 'visible')
          && !overlaps(r, node.getBoundingClientRect())) return true;
    }
    return false;
  };
  // Text a stylesheet draws into an element (::before/::after with a string content). Only
  // rendered strings that carry a letter or digit count: bullets, arrows and icons are decoration.
  const generatedText = (el) => {
    const strings = [];
    for (const side of ['::before', '::after']) {
      const content = window.getComputedStyle(el, side).content || '';
      const m = /^"(.*)"$|^'(.*)'$/.exec(content);
      if (!m) continue;
      const s = clean(m[1] !== undefined ? m[1] : m[2]);
      if (s && /[\p{L}\p{N}]/u.test(s)) strings.push(s.slice(0, 80));
    }
    return strings.join(' ');
  };
  // Evidence for the safety policy about where a control sits: its native kind, the nearest enclosing
  // dialog, menu, panel or region with a bounded name, and whether the platform itself says the control
  // dismisses that container (a dialog form, a popover hide target, a framework dismiss attribute).
  const LANDMARKS = 'dialog, [role=dialog], [role=alertdialog], [aria-modal="true"], [role=menu], [role=listbox], '
      + '[role=tooltip], aside, [role=complementary], nav, [role=navigation], [role=region], details, [popover]';
  const LANDMARK_ROLES = {dialog: 'dialog', aside: 'complementary', nav: 'navigation', details: 'details'};
  const nativeOf = (el) => {
    const tag = el.tagName.toLowerCase();
    if (tag === 'button') {
      // a button submits only inside a form; elsewhere its default "submit" type means nothing
      const type = (el.type || 'submit').toLowerCase();
      return 'button:' + (type === 'submit' && !el.form ? 'button' : type);
    }
    if (tag === 'input') return tag + ':' + ((el.type || '').toLowerCase() || 'text');
    return tag;
  };
  const landmarkOf = (el) => {
    const host = el.parentElement && el.parentElement.closest(LANDMARKS);
    if (!host) return ['', ''];
    const explicit = (host.getAttribute('role') || '').toLowerCase();
    const role = explicit || (host.hasAttribute('popover') ? 'popover' : (LANDMARK_ROLES[host.tagName.toLowerCase()] || 'region'));
    let name = clean(host.getAttribute('aria-label'));
    if (!name && host.getAttribute('aria-labelledby')) {
      const by = document.getElementById(host.getAttribute('aria-labelledby'));
      name = by ? clean(by.textContent) : '';
    }
    if (!name) {
      const heading = host.querySelector('h1, h2, h3, h4, h5, h6, [role=heading], legend, summary');
      name = heading ? clean(heading.textContent) : '';
    }
    return [role, name.slice(0, 80)];
  };
  const dismissesOf = (el) => {
    if (el.matches('[data-dismiss], [data-bs-dismiss], [popovertargetaction="hide"]')) return true;
    const method = ((el.getAttribute('formmethod') || (el.form && el.form.getAttribute('method')) || '')).toLowerCase();
    return method === 'dialog' && !!el.closest('dialog');
  };
  // Whether a control can take interaction right now: '' when it can, else why not. Hidden, inert and
  // clipped controls are not on the operator's screen at all; a disabled or covered one is on screen but
  // cannot be operated. The hit test at the control's centre tolerates its own descendants, its ancestors
  // (a transparent child), its labels and anything inside the same control, so a label or an icon over
  // the control never counts as cover; a control outside the viewport cannot be hit-tested and is not
  // called covered.
  const CONTROL_SELECTOR = 'a[href], button, [role=button], [role=link], input, select, textarea, [role=checkbox], '
      + '[role=radio], [role=combobox], [role=textbox], [role=option], [role=menuitem], [role=tab], [role=switch]';
  const availability = (el) => {
    if (el.disabled || el.getAttribute('aria-disabled') === 'true' || el.closest('fieldset:disabled')) return 'disabled';
    if (hiddenAway(el)) return 'hidden';
    if (clippedAway(el)) return 'clipped';
    const r = el.getBoundingClientRect();
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    if (cx < 0 || cy < 0 || cx > window.innerWidth || cy > window.innerHeight) return '';
    const hit = document.elementFromPoint(cx, cy);
    if (!hit || hit === el || el.contains(hit) || hit.contains(el)) return '';
    const labels = Array.from(el.labels || []);
    if (labels.some((label) => label === hit || label.contains(hit))) return '';
    const around = hit.closest('label');
    if (around && (around.control === el || around.contains(el))) return '';
    const own = el.closest(CONTROL_SELECTOR), theirs = hit.closest(CONTROL_SELECTOR);
    if (own && own === theirs) return '';
    return 'covered';
  };
  // A control an application built from a nonsemantic element: no interactive tag, no ARIA control role,
  // no tabindex, yet visible, named by browser-visible metadata and styled as a pointer target. Everything
  // here comes from the platform's own view of the element; no framework internals, listener registries or
  // markup conventions are consulted. Bounded by POINTER_CONTROL_BUDGET per observation.
  const pointerName = (el) => clean(el.getAttribute('aria-label') || el.getAttribute('title'));
  const namedPointerControl = (el) => {
    if (el.matches(CONTROL_SELECTOR) || el.hasAttribute('tabindex')) return false;   // already a control
    const name = pointerName(el);
    if (!name) return false;                                  // an unnamed pointer target is not offered
    const style = window.getComputedStyle(el);
    if (style.cursor !== 'pointer' || style.pointerEvents === 'none') return false;
    if (el.getAttribute('aria-disabled') === 'true' || el.closest('[aria-disabled="true"]')) return false;
    if (availability(el)) return false;                       // hidden, clipped, disabled or covered
    // the nearest such element wins: never an ancestor that merely contains a control or another candidate
    if (el.querySelector(CONTROL_SELECTOR)) return false;
    for (const inner of el.querySelectorAll('*')) {
      if (inner === el) continue;
      const innerStyle = window.getComputedStyle(inner);
      if (pointerName(inner) && innerStyle.cursor === 'pointer' && innerStyle.pointerEvents !== 'none') return false;
    }
    return true;
  };
  // Elements whose emitted text already covers everything inside them.
  const REPRESENTED = 'a[href], button, [role=button], [role=link], h1, h2, h3, h4, h5, h6, select, textarea, option';
  const SKIP_TAGS = ['script', 'style', 'noscript', 'template', 'title', 'option', 'optgroup', 'iframe', 'object',
                     'input', 'select', 'textarea', 'head', 'meta', 'link', 'br', 'hr', 'img', 'video', 'audio',
                     'canvas', 'svg', 'path'];
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

  const emitted = new Set();
  const push = (el, role, name, text) => {
    const states = {};
    if (role !== 'text') {
      const why = availability(el);
      if (why === 'hidden' || why === 'clipped') return;      // not on the operator's screen
      if (why) states[why] = 'true';                             // disabled or covered: shown, not operable now
    }
    emitted.add(el); const [landmark, landmarkName] = landmarkOf(el);
    out.push({role, name: clean(name), text: clean(text), box: box(el), context: contextOf(el), ref: cssPath(el),
              native: nativeOf(el), landmark, landmark_name: landmarkName, dismisses: dismissesOf(el), states}); };
  const CONTROL = 'a[href], button, [role=button], [role=link]';
  let generatedBudget = 40;
  let pointerBudget = 40;        // named pointer controls offered per observation
  // Visible leaf text: an element's own text (plus rendered generated content), whatever its tag,
  // unless something already emitted covers it (a control, heading or cell around it), or it is
  // hidden away, clipped or offscreen. A tag decides nothing: a value in an <aside>, <small>,
  // <time> or custom element is as real to the operator as one in a <span>.
  const leafText = (el, tag) => {
    if (SKIP_TAGS.includes(tag) || el.closest(REPRESENTED)) return '';
    const cell = el.closest('td');
    if (cell && emitted.has(cell)) return '';
    let text = ownText(el);
    if (generatedBudget > 0) {
      const generated = generatedText(el);
      if (generated) { generatedBudget--; text = clean(text + ' ' + generated); }
    }
    if (!text || hiddenAway(el) || clippedAway(el)) return '';
    return text;
  };
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
    else if (pointerBudget > 0 && namedPointerControl(el)) { pointerBudget--; push(el, 'button', pointerName(el), el.textContent); }
    else { const text = leafText(el, tag); if (text) push(el, 'text', '', text); }
  }
  return {url: location.href, title: document.title, dialog, elements: out};
}
"""

# Geometry, visibility, text and enclosing item for elements the accessibility tree found and the
# projection did not list. One call for all of them. A password field's value is never read.
ENRICH_JS = "(refs) => {" + HELPERS_JS + r"""
  return refs.map(([ref, hint]) => {
    let el = null;
    try { el = document.querySelector(ref); } catch (e) { el = null; }
    if (!el) return null;
    const represented = !!el.closest(REPRESENTED) || hiddenAway(el);
    if (hint) {
      // A static text node the accessibility tree reported: its own rendered box, not its parent's.
      for (const node of el.childNodes) {
        if (node.nodeType !== 3 || !clean(node.textContent).includes(hint)) continue;
        const range = document.createRange(); range.selectNodeContents(node);
        const r = range.getBoundingClientRect();
        const visible = r.width > 0 && r.height > 0 && !represented && !clippedAway(el, r);
        return {visible, box: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)],
                context: contextOf(el), text: clean(node.textContent), represented};
      }
      return null;
    }
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    let text = '';
    if (tag === 'input' || tag === 'select' || tag === 'textarea') {
      // a password is never read; a checkbox's or radio's value ("on") says nothing to an operator
      text = (type === 'password' || type === 'checkbox' || type === 'radio') ? '' : (el.value || '');
    } else text = clean(el.innerText || el.textContent);
    const why = isVisible(el) ? availability(el) : 'hidden';
    return {visible: isVisible(el) && why !== 'hidden' && why !== 'clipped', box: box(el), context: contextOf(el),
            text: clean(text), represented, availability: why};
  });
}
"""

# Selecting a value: what kind of chooser a control is, its options, the suggestions it shows and what it
# holds afterwards. Options are matched by their visible text (exact first, then prefix, then containing),
# the first match winning deterministically; nothing here clicks, selects or types.
SELECT_KIND_JS = "(el) => {" + HELPERS_JS + r"""
  const tag = el.tagName.toLowerCase();
  const role = (el.getAttribute('role') || '').toLowerCase();
  if (tag === 'select') return {kind: 'select', editable: !el.disabled};
  const editable = (tag === 'input' || tag === 'textarea' || el.isContentEditable) && !el.disabled && !el.readOnly;
  if (role === 'combobox' || el.getAttribute('aria-autocomplete') || el.getAttribute('list')) return {kind: 'combobox', editable};
  if (editable) return {kind: 'textbox', editable};
  return {kind: '', editable: false};
}
"""
SELECT_OPTIONS_JS = r"(el) => Array.from(el.options).map((o) => ({label: (o.label || o.textContent || '').replace(/\s+/g, ' ').trim(), value: o.value, disabled: o.disabled}))"
# The suggestions a chooser shows right now. A popup the control references (aria-controls / aria-owns) is
# used when it exists and is visible, and only such popups are searched. An unlinked control falls back to the
# visible listbox, menu or datalist popups on the page only when there is exactly one; several is ambiguous and
# none is no popup. Options are reported with their text and structural reference; nothing is clicked.
SUGGESTIONS_JS = "(el) => {" + HELPERS_JS + r"""
  // The container a chooser's suggestions live in. Preferred: the popup the control names through
  // aria-controls / aria-owns, used only when it exists and is visible. Otherwise a container is
  // considered only when it is a plausible suggestion popup for THIS control: it holds candidate rows,
  // it is visible, it is not an ancestor of the control (never the page or a whole form), and it sits
  // directly under or over the control's own horizontal span. Exactly one such container may be used;
  // several are ambiguous. Arbitrary page text is never searched.
  const NEAR_PX = 24;
  const rect = (node) => node.getBoundingClientRect();
  const rowsIn = (scope) => {
    const semantic = Array.from(scope.querySelectorAll('[role=option], [role=menuitem], option'));
    if (semantic.length) return semantic;
    // A custom widget: rows are the actionable descendants, or the leaf elements whose own text is the
    // suggestion and whose nearest ancestor inside the container is actionable (a click target).
    const actionable = Array.from(scope.querySelectorAll('li, a[href], button, [role=button], [role=link], [tabindex], [onclick], [data-value]'));
    const rows = actionable.filter((node) => clean(node.textContent) && !actionable.some((other) => other !== node && node.contains(other)));
    if (rows.length) return rows;
    return Array.from(scope.children).filter((node) => clean(node.textContent));
  };
  const usable = (scope) => isVisible(scope) && !scope.contains(el) && rowsIn(scope).length > 0;
  const associated = (scope) => {
    const a = rect(el), b = rect(scope);
    const overlaps = Math.min(a.right, b.right) > Math.max(a.left, b.left);
    const below = b.top >= a.bottom - NEAR_PX && b.top <= a.bottom + NEAR_PX * 4;
    const above = b.bottom <= a.top + NEAR_PX && b.bottom >= a.top - NEAR_PX * 4;
    return overlaps && (below || above || (b.top >= a.top && b.bottom <= a.bottom + NEAR_PX * 8));
  };
  const ids = ((el.getAttribute('aria-controls') || '') + ' ' + (el.getAttribute('aria-owns') || '')).split(/\s+/).filter(Boolean);
  const named = ids.map((id) => document.getElementById(id)).filter((node) => node && isVisible(node));
  const linked = named.length > 0;
  let scopes = named;
  if (!linked) {
    const seenScope = new Set();
    const candidates = [];
    for (const scope of document.querySelectorAll('[role=listbox], [role=menu], [role=presentation], datalist, ul, ol, div, table'))
      if (usable(scope) && associated(scope)) candidates.push(scope);
    // Keep only the outermost container of each nested group, so one popup is not counted many times.
    for (const scope of candidates)
      if (!candidates.some((other) => other !== scope && other.contains(scope))) { if (!seenScope.has(scope)) { seenScope.add(scope); scopes.push(scope); } }
  }
  const popups = scopes.length;
  const seen = new Set();
  const groups = [];
  for (const scope of scopes) {
    const rows = scope.matches('[role=option], [role=menuitem]') ? [scope] : rowsIn(scope);
    const options = [];
    for (const row of rows) {
      if (seen.has(row) || row.getAttribute('aria-disabled') === 'true' || row.disabled) continue;
      if (row.tagName.toLowerCase() !== 'option' && !isVisible(row)) continue;
      seen.add(row);
      options.push({text: clean(row.label || row.textContent), ref: cssPath(row),
                    selected: row.getAttribute('aria-selected') === 'true'});
    }
    if (options.length) groups.push(options);
  }
  // A linked control trusts the popup it names; an unlinked one hands back every plausible container's
  // candidates separately, and the caller keeps the container that holds exactly one match.
  const options = linked ? groups.flat() : [];
  return {linked, popups, options, groups};
}
"""
# What the widget holds after a suggestion was clicked. Deterministic browser state only: the control's own
# value, the active descendant, the values of any hidden inputs the widget stores its choice in, and the text
# of chips or tags that carry their own removal control (a selected token, not page copy).
ACCEPTED_JS = "(el) => {" + HELPERS_JS + r"""
  const value = clean(el.value !== undefined ? el.value : el.textContent);
  const active = el.getAttribute('aria-activedescendant');
  const activeText = active && document.getElementById(active) ? clean(document.getElementById(active).textContent) : '';
  const widget = el.closest('form, [role=search], [role=combobox], fieldset, section, div') || document.body;
  const stored = Array.from(widget.querySelectorAll('input[type=hidden], select, input'))
      .filter((node) => node !== el && node.value).map((node) => clean(node.value));
  const chips = [];
  for (const chip of widget.querySelectorAll('[role=listitem], li, span, div')) {
    if (!isVisible(chip) || chip.contains(el)) continue;
    const remove = chip.querySelector('button, [role=button], a[href], [aria-label*="emove" i], [title*="emove" i]');
    if (!remove || remove.contains(chip)) continue;
    const own = clean(chip.textContent);
    if (own && own.length < 200) chips.push(own);
  }
  return {value, active_text: activeText, stored, chips, expanded: el.getAttribute('aria-expanded'),
          connected: el.isConnected};
}
"""


# The committed state of the field group an action ran in: what a form-local commit changes, and nothing else.
# Deterministic browser state only. The container is the acted control's nearest *meaningful* group (a form,
# search landmark, fieldset, group or labelled region) - never a bare div and never the document - so a token
# that appears elsewhere on the page can never be mistaken for this action's effect. Editable fields report
# their own values; tokens are elements that carry their own removal or selection semantics, each with the
# structural reference and accessible name replay needs to find it again. Nothing is clicked or changed.
COMMITTED_JS = "(el) => {" + HELPERS_JS + r"""
  const GROUPS = 'form, [role=search], [role=group], fieldset, [role=region], section[aria-label], section[aria-labelledby]';
  const group = el.closest(GROUPS);
  if (!group) return {scoped: false, fields: [], tokens: [], stored: []};
  const fields = [];
  for (const node of group.querySelectorAll('input, textarea, select, [contenteditable=true]')) {
    if (!isVisible(node)) continue;
    if (node.type === 'hidden' || node.type === 'password') continue;   // never read a secret's contents
    fields.push({ref: cssPath(node), value: clean(node.value !== undefined ? node.value : node.textContent)});
  }
  const stored = [];
  for (const node of group.querySelectorAll('input[type=hidden], select')) {
    if (node.value) stored.push({ref: cssPath(node), value: clean(node.value)});
  }
  // ---- tokens: collect every candidate first, then canonicalize ----
  // Broad candidate tags are deliberate: a chip is rarely marked up semantically. That means several
  // nested ancestors around one chip all find the same removal control, so the candidates cannot be
  // deduplicated while walking (an outer wrapper is reached before the inner element it wraps). They are
  // collected first and collapsed afterwards, one logical token per removal control or selected element.
  const REMOVERS = 'button, [role=button], a[href], [aria-label*="emove" i], [title*="emove" i], [aria-label*="elete" i]';
  const candidates = [];
  for (const node of group.querySelectorAll('[role=listitem], [role=option], li, span, div, output')) {
    if (!isVisible(node) || node.contains(el) || node === el) continue;
    // A token is a selected thing, not page copy: it either carries its own removal control, or it
    // reports a native selected state of its own.
    const remove = node.querySelector(REMOVERS);
    const selected = node.getAttribute('aria-selected') === 'true' || node.getAttribute('aria-checked') === 'true';
    if ((!remove || remove.contains(node)) && !selected) continue;
    // The token's own value is its text without the controls it contains: a chip whose label is followed
    // by a close button's own glyph is that label plus a control, not a different value.
    let own = '';
    for (const child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) { own += child.textContent; continue; }
      if (child.nodeType !== Node.ELEMENT_NODE) continue;
      if (child === remove || child.contains(remove) || child.matches('button, [role=button], a[href]')) continue;
      own += child.textContent;
    }
    own = clean(own) || clean(node.textContent);
    if (!own || own.length > 200) continue;
    candidates.push({node, remove, selected, text: own});
  }
  // Every candidate is reported, in document order, each carrying the structural references the caller
  // needs to collapse them: `canonical_tokens` does that in one place, shared with replay.
  const tokens = candidates.map((candidate) => {
    const node = candidate.node, remove = candidate.remove;
    return {ref: cssPath(node), text: candidate.text, name: clean(node.getAttribute('aria-label') || ''),
            role: node.getAttribute('role') || node.tagName.toLowerCase(),
            remove_ref: remove ? cssPath(remove) : '',
            removable: !!(remove && !remove.contains(node)), selected: candidate.selected,
            remove_name: remove ? clean(remove.getAttribute('aria-label') || remove.getAttribute('title') || remove.textContent) : ''};
  });
  return {scoped: true, fields, tokens, stored};
}
"""


# A displayed option often adds a parenthesised gloss to the name a caller knows ("... (EPA)", "... (2 open)").
# Such a trailing bracket is stripped before the exact level runs, so the caller's plain name still matches
# exactly one option; the text inside the brackets is never matched against and nothing else is removed.
TRAILING_BRACKET = re.compile(r"\s*[\(\[\{][^\(\)\[\]\{\}]*[\)\]\}]\s*$")
MATCH_LEVELS = (("exact", lambda text, wanted: text == wanted),
                ("exact without a trailing bracket", lambda text, wanted: TRAILING_BRACKET.sub("", text) == wanted),
                ("prefix", lambda text, wanted: text.startswith(wanted)),
                ("containing", lambda text, wanted: wanted in text))


def match_option(options: list[dict], value: str, key: str = "label") -> tuple[str, dict | None, str]:
    """The option a person would choose for `value`, or why none can be chosen.

    Levels in order: exact match; exact once a trailing parenthesised gloss is stripped from the displayed
    text; prefix match; containing match. Text is compared with case and spacing normalized, and disabled
    options never count. At each level exactly one match is chosen; more than one is `ambiguous` (DOM order
    decides nothing) and stops the search; none moves to the next level.
    Returns (status, option, level): status is "match", "ambiguous" or "none".
    """
    wanted = " ".join(value.split()).lower()
    usable = [o for o in options if not o.get("disabled") and " ".join(str(o.get(key, "")).split())]
    for level, test in MATCH_LEVELS:
        found = [o for o in usable if test(" ".join(str(o[key]).split()).lower(), wanted)]
        if len(found) == 1:
            return "match", found[0], level
        if len(found) > 1:
            return "ambiguous", None, level
    return "none", None, ""


def matching_suggestion(shown: dict, value: str) -> tuple[str, dict | None, str]:
    """The one suggestion to click, across the containers a chooser offered.

    A control that names its popup (`aria-controls` / `aria-owns`) is matched over that popup's options as
    before. An unlinked control is matched inside each plausible container separately: containers with no
    match are ignored, exactly one container with exactly one match proceeds, a container whose own match is
    ambiguous is `ambiguous`, and matches in several containers are `ambiguous_popups` (the count is
    returned as the level). Document order never decides.
    """
    if shown.get("linked"):
        return match_option(shown["options"], value, key="text")
    hits = []
    for options in shown.get("groups") or []:
        status, option, level = match_option(options, value, key="text")
        if status == "ambiguous":
            return status, None, level
        if status == "match":
            hits.append((option, level))
    if len(hits) > 1:
        return "ambiguous_popups", None, str(len(hits))
    if not hits:
        return "none", None, ""
    return "match", hits[0][0], hits[0][1]


# Class tokens that mean "this one is selected", as whole tokens and nothing else. Deliberately small and
# unambiguous: every entry means selectedness in ordinary usage, in both the plain and the `is-` spelling
# that component conventions use. "active" is absent on purpose, because it equally means focused, running,
# enabled or hovered, and would turn an unrelated style change into false proof.
STATE_TOKENS = ("selected", "is-selected", "checked", "is-checked", "chosen", "is-chosen",
                "current", "is-current")

# Whether one control reads as selected, from browser state only. Deterministic sources, in the order a
# browser itself would answer the question: the element's own native selectedness (a checkbox or radio's
# `checked`, an option's `selected`), the ARIA state a control publishes about itself, a boolean state
# attribute a custom control sets alongside it, a control structurally associated with it, and finally an
# exact state word in its own class list. Computed style is deliberately absent: colour and weight are how
# a selection is shown, never what makes it true. A control that publishes no such state at all reports
# `known: false`, and the caller then has no proof rather than a false negative.
SELECTED_JS = "(el) => {" + HELPERS_JS + "const STATE_TOKENS = " + json.dumps(list(STATE_TOKENS)) + ";" + r"""
  // Boolean state attributes a custom control sets on itself. Standard ARIA first; `data-selected` and
  // `data-checked` are the conventional spellings of the same idea for a control that has no ARIA role.
  // They are attribute NAMES, not values, so no site's class names or framework ids are involved.
  const FLAGS = ['aria-pressed', 'aria-checked', 'aria-selected', 'data-selected', 'data-checked'];
  const readFlag = (node) => {
    for (const name of FLAGS) {
      const raw = node.getAttribute(name);
      if (raw === null) continue;
      const value = raw.trim().toLowerCase();
      if (value === 'true' || value === 'false') return {source: name, selected: value === 'true'};
    }
    // aria-current names which one of a set is current; "false" and absence both mean it is not.
    const current = node.getAttribute('aria-current');
    if (current !== null) {
      const value = current.trim().toLowerCase();
      return {source: 'aria-current', selected: value !== '' && value !== 'false'};
    }
    return null;
  };

  // 1. The element's own native selectedness.
  if (el.tagName === 'INPUT' && (el.type === 'checkbox' || el.type === 'radio')) {
    return {known: true, selected: !!el.checked, source: 'checked', ref: cssPath(el)};
  }
  if (el.tagName === 'OPTION') return {known: true, selected: !!el.selected, source: 'selected', ref: cssPath(el)};

  // 2. A state the control publishes about itself.
  const own = readFlag(el);
  if (own) return {known: true, selected: own.selected, source: own.source, ref: cssPath(el)};

  // 3. A control deterministically associated with it: the input a label drives, or the single native
  // control this element wraps. Association is structural (`for`, or containment of exactly one control),
  // never a guess from position or wording.
  let partner = null;
  if (el.tagName === 'LABEL') {
    partner = el.htmlFor ? document.getElementById(el.htmlFor) : el.querySelector('input, select');
  }
  if (!partner) {
    // Exactly one native control inside: the element is that control's visible wrapper. More than one
    // is ambiguous and proves nothing, so nothing is chosen.
    const inside = el.querySelectorAll('input[type=checkbox], input[type=radio], option');
    if (inside.length === 1) partner = inside[0];
  }
  if (!partner) {
    // A hidden form control the element deterministically drives: same `id` target, or exactly one
    // hidden input inside it. Its value is never read, only whether it is checked.
    const controls = el.getAttribute('aria-controls');
    const named = controls ? document.getElementById(controls.trim().split(/\s+/)[0]) : null;
    if (named && named.tagName === 'INPUT') partner = named;
    if (!partner) {
      const hidden = el.querySelectorAll('input[type=hidden]');
      if (hidden.length === 1) partner = hidden[0];
    }
  }
  if (partner) {
    if (partner.tagName === 'OPTION') {
      return {known: true, selected: !!partner.selected, source: 'partner-selected', ref: cssPath(partner)};
    }
    if (partner.tagName === 'INPUT' && (partner.type === 'checkbox' || partner.type === 'radio')) {
      return {known: true, selected: !!partner.checked, source: 'partner-checked', ref: cssPath(partner)};
    }
    if (partner.tagName === 'INPUT' && partner.type === 'hidden') {
      // A hidden control stores the selection as its value. Whether it holds one is deterministic; what
      // it holds is never read here, so a stored secret is never touched.
      const held = (partner.value || '').trim().toLowerCase();
      const off = held === '' || held === 'false' || held === '0' || held === 'off';
      return {known: true, selected: !off, source: 'partner-stored', ref: cssPath(partner)};
    }
    const flag = readFlag(partner);
    if (flag) return {known: true, selected: flag.selected, source: 'partner-' + flag.source, ref: cssPath(partner)};
  }

  // 4. Last resort: a semantic state token in the element's own class list. Many custom controls carry
  // their selectedness nowhere else. This is the weakest evidence here and is treated accordingly:
  //  - whole tokens only, compared against a closed vocabulary, so "unselected", "selected-item" and
  //    "button-selected-style" are different tokens and never match;
  //  - the vocabulary holds only words that mean selectedness. "active" is excluded: it equally means
  //    focused, running, enabled or merely hovered, so it is not evidence of a selection;
  //  - the token NAME is what is read, never the class string, and nothing about it is written down.
  // The presence of such a token means selected; its absence on a control that carried one before means
  // deselected, which the caller compares as a transition.
  const tokens = Array.from(el.classList || []).map((token) => token.toLowerCase());
  const hit = tokens.find((token) => STATE_TOKENS.includes(token));
  if (hit) return {known: true, selected: true, source: 'class-token', ref: cssPath(el), class_token: true};
  // No state of any kind. `known` stays false: this control publishes nothing, and the caller must not
  // treat it as "known to be unselected" by any other measure. `class_token` records that the class list
  // was read and held no state word, which is what lets a token appearing after the click read as a
  // transition without claiming the control had any state before it.
  return {known: false, selected: false, source: '', ref: cssPath(el), class_token: false};
}
"""


PATH_SEPARATOR = " > "      # how cssPath joins its segments: an ancestor path ends at one of these


def canonical_tokens(candidates: list[dict]) -> list[dict]:
    """One record per logical selected token, from the raw candidates perception collected.

    A chip is rarely marked up semantically, so the candidate walk deliberately accepts broad container
    tags. Several nested ancestors around one chip therefore all find the same removal control and all
    look like tokens. They cannot be collapsed during the walk: an outer wrapper is reached before the
    inner element it wraps, so the canonical one is not yet known when the wrapper is seen. They are
    collapsed here instead.

    The identity of a token is what makes it removable or selected, never its text: two chips may
    legitimately display the same words, and merging those would hide a real second selection. A candidate
    that offers a removal control is keyed by that control; one that is only natively selected is keyed by
    its own backing element. Within a key the canonical element is the innermost candidate that still
    carries the label, the nearest meaningful container around the removal control. Finally a candidate
    that merely contains another logical token is a wrapper, not a token of its own. Document order is
    preserved, because the candidates arrive in it.
    """
    chosen: dict[str, dict] = {}
    for candidate in candidates:
        key = token_key(candidate)
        held = chosen.get(key)
        if held is None or encloses(held, candidate):
            chosen[key] = candidate
    canonical = [candidate for candidate in candidates if chosen.get(token_key(candidate)) is candidate]
    return [candidate for candidate in canonical
            if not any(other is not candidate and encloses(candidate, other) for other in canonical)]


def token_key(token: dict) -> str:
    """What makes two observations the same logical token: its removal control, else its own element."""
    remove = str(token.get("remove_ref") or "")
    return f"remove:{remove}" if remove else f"self:{token.get('ref') or ''}"


def encloses(outer: dict, inner: dict) -> bool:
    """Whether one candidate's structural reference contains another's, by the path perception records."""
    return ref_encloses(str(outer.get("ref") or ""), str(inner.get("ref") or ""))


def ref_encloses(outer_ref: str, inner_ref: str) -> bool:
    """Whether one structural reference is an ancestor path of another.

    The separator is required, so a path is never read as the ancestor of a sibling whose own path merely
    starts with the same characters.
    """
    if not outer_ref or not inner_ref or outer_ref == inner_ref:
        return False
    return inner_ref.startswith(outer_ref + PATH_SEPARATOR)


def commit_proof(after: dict, before: dict, typed: str) -> str | None:
    """Which deterministic state shows a keyboard-committed field resolved the query, or None.

    The field's own value counts only when it is no longer the raw text that was typed: a field that
    still holds exactly the query has resolved nothing. A stored value, an active descendant or a chip
    that newly appeared counts as it does for a clicked suggestion. Nothing visible elsewhere on the
    page, and no navigation by itself, is proof.
    """
    query = " ".join(str(typed).split()).lower()
    value = " ".join(str(after.get("value", "")).split())
    if value and value.lower() != query:
        return "the field resolved into a different value"
    fresh_stored = [v for v in after.get("stored", []) if v not in before.get("stored", [])]
    if any(" ".join(str(v).split()).lower() != query for v in fresh_stored if str(v).strip()):
        return "a stored value"
    fresh_chips = [c for c in after.get("chips", []) if c not in before.get("chips", [])]
    if fresh_chips:
        return "a selected chip"
    if after.get("active_text", "").strip():
        return "the active descendant"
    return None


def acceptance_proof(after: dict, chosen: str, before: dict, typed: str = "") -> str | None:
    """Which deterministic widget state shows the clicked suggestion was accepted, or None.

    The control's own value now holding the option and differing from what was typed, the active descendant
    naming it, a stored value (a hidden input or select the widget writes its choice into) that now holds
    it, or a newly present chip or tag carrying its own removal control. Visible page copy, the typed text
    and the suggestion row itself are never proof: only state the widget itself changed counts, and a value
    that is still exactly what was typed proves nothing at all.
    """
    wanted = " ".join(chosen.split()).lower()

    def holds(text: str) -> bool:
        text = " ".join(str(text).split()).lower()
        return bool(text) and (text == wanted or text in wanted or wanted in text)

    value = " ".join(str(after.get("value", "")).split())
    typed_text = " ".join(str(typed).split()).lower()
    # The control's value proves acceptance when it now reads as the chosen option. A value that is still
    # only what was typed, and shorter than the option it is supposed to have accepted, proves nothing.
    if value and holds(value) and not (value.lower() == typed_text and value.lower() != wanted):
        return "value"
    if holds(after.get("active_text", "")):
        return "active descendant"
    fresh_stored = [v for v in after.get("stored", []) if v not in before.get("stored", [])]
    if any(holds(v) for v in fresh_stored):
        return "stored value"
    fresh_chips = [c for c in after.get("chips", []) if c not in before.get("chips", [])]
    if any(holds(c) for c in fresh_chips):
        return "chip"
    return None


def pick_option(options: list[dict], value: str, key: str = "label") -> dict | None:
    """The single option `match_option` chooses, or None when there is none or the choice is ambiguous."""
    status, option, _ = match_option(options, value, key)
    return option if status == "match" else None


# The page element with visible text of its own under a screenshot box: what a vision proposal is grounded on.
# Hit-testing pierces shadow roots; the element must own text whose rendered box overlaps the requested one, so
# a container whose text sits elsewhere, a canvas, or an image never qualifies. Nothing is clicked or changed.
TEXT_UNDER_JS = "(area) => {" + HELPERS_JS + r"""
  const [bx, by, bw, bh] = area;
  const wanted = {left: bx, top: by, right: bx + bw, bottom: by + bh};
  const cx = bx + bw / 2, cy = by + bh / 2;
  let hit = document.elementFromPoint(cx, cy);
  while (hit && hit.shadowRoot) {
    const inner = hit.shadowRoot.elementFromPoint(cx, cy);
    if (!inner || inner === hit) break;
    hit = inner;
  }
  if (!hit) return null;
  const ownRects = (el) => Array.from(el.childNodes).filter(n => n.nodeType === 3 && clean(n.textContent))
      .map(n => { const range = document.createRange(); range.selectNodeContents(n); return range.getBoundingClientRect(); });
  const textHere = (el) => ownText(el) && ownRects(el).some(r => r.width > 0 && r.height > 0 && overlaps(r, wanted));
  const search = (root) => {
    if (textHere(root)) return root;
    for (const child of root.querySelectorAll('*')) if (isVisible(child) && textHere(child)) return child;
    return null;
  };
  let found = null;
  for (let node = hit, depth = 0; node && node !== document.documentElement && depth < 4; depth++) {
    found = search(node);
    if (found) break;
    node = node.parentElement || (node.getRootNode() instanceof ShadowRoot ? node.getRootNode().host : null);
  }
  if (!found || !isVisible(found) || hiddenAway(found) || found.closest('input, textarea, select')) return null;
  const inShadow = found.getRootNode() instanceof ShadowRoot;
  return {text: ownText(found), box: box(found), context: inShadow ? '' : contextOf(found),
          ref: inShadow ? '' : cssPath(found)};
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
COMMIT_SETTLE_MS = 400         # how long a keyboard-committed field is given to resolve its value
KEYSTROKE_DELAY_MS = 20        # between keystrokes typed into a chooser, so key-driven filters keep up
# Two bounded windows, because "no page yet" and "no more pages" are different questions.
# POPUP_FIRST_MS is how long a click is given to produce its FIRST page before it counts as same-page: an
# absent page event is only meaningful once this has elapsed, since the browser may emit it several ticks
# after the click returns. POPUP_SETTLE_MS is how much longer siblings are collected once one has arrived,
# so a script opening two windows on separate ticks is seen as two. The cost is paid per click: every
# ordinary same-page click waits out POPUP_FIRST_MS before it is known to be same-page, so this bounds the
# floor on click latency and is deliberately kept small.
POPUP_FIRST_MS = 600           # bounded wait for a first new page; the latency every click pays
POPUP_SETTLE_MS = 500          # further collection once a first page has appeared
POPUP_POLL_MS = 100
SUGGESTION_TIMEOUT_MS = 4000   # how long a combobox or autocomplete may take to show its suggestions
SUGGESTION_POLL_MS = 150
MAX_AX_NODES = 400          # bound on accessibility candidates merged per observation (large pages)
MAX_AX_TEXT_NODES = 40      # bound on accessibility static-text nodes added per observation
AX_STATIC_TEXT = "statictext"
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
            if child.get("nodeType") == 3 and prefix and "backendNodeId" in child:
                paths[child["backendNodeId"]] = prefix       # a text node: its parent element's path
                continue
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


# What a native choice control is right now; null for anything that is not <input type=radio|checkbox>.
CHOICE_STATE_JS = r"""
(el) => {
  if (!(el instanceof HTMLInputElement) || !['radio', 'checkbox'].includes(el.type)) return null;
  return {type: el.type, checked: el.checked, disabled: el.disabled, connected: el.isConnected};
}
"""
# The labels the browser itself associates with the control: <label for=id> and a wrapping <label>.
CHOICE_LABELS_JS = "(el) => Array.from(el.labels || [])"
# Whether the keyboard may activate a choice control whose pointer paths are obstructed. Decided from
# DOM semantics only: the element's own kind and state, inert/hidden/modal context, and whether what the
# browser hit-tests at the control's centre belongs to the same choice structure. For a radio that
# structure is its native group (same name, same form owner, same type) and the group's associated labels;
# for a checkbox it is the checkbox itself and its own associated labels only, because independent
# checkboxes routinely share a name.
KEYBOARD_SAFETY_JS = r"""
(el) => {
  const no = (reason) => ({safe: false, reason});
  if (!(el instanceof HTMLInputElement) || !['radio', 'checkbox'].includes(el.type)) return no('not a native choice control');
  if (el.disabled) return no('disabled');
  if (!el.isConnected) return no('detached');
  if (el.closest('[inert]')) return no('inside an inert subtree');
  const r = el.getBoundingClientRect(); const cs = window.getComputedStyle(el);
  if (r.width === 0 || r.height === 0 || cs.visibility === 'hidden' || cs.display === 'none') return no('hidden');
  if (el.closest('[aria-hidden="true"]')) return no('aria-hidden');
  for (const modal of document.querySelectorAll('dialog[open], [role="dialog"][aria-modal="true"], [role="alertdialog"][aria-modal="true"]'))
    if (!modal.contains(el)) return no('a modal dialog is open elsewhere');
  const related = new Set([el, ...Array.from(el.labels || [])]);
  if (el.type === 'radio' && el.name) {
    const pool = el.form ? Array.from(el.form.elements) : Array.from(document.querySelectorAll('input'));
    for (const other of pool)
      if (other !== el && other instanceof HTMLInputElement && other.type === 'radio' && other.name === el.name
          && other.form === el.form) { related.add(other); for (const l of other.labels || []) related.add(l); }
  }
  const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  if (!hit) return no('the control is outside the viewport');
  const inStructure = Array.from(related).some((node) => node === hit || node.contains(hit));
  if (!inStructure) return no('an unrelated element covers the control');
  return {safe: true, reason: ''};
}
"""


# The nearest enclosing control that is actionable by its own semantics, for text that sits inside a control
# without being one. Only what the platform treats as a control counts: a native interactive element, an anchor
# with an href, a label bound to a control, or an explicit interactive ARIA role. A container that merely
# contains the text, or that reacts to clicks only through a script handler, is never chosen. Nothing here
# clicks, focuses or mutates anything.
INTERACTIVE_ROLES = ("button", "link", "menuitem", "menuitemcheckbox", "menuitemradio", "tab", "option", "checkbox",
                     "radio", "switch", "treeitem")
ACTIONABLE_ANCESTOR_JS = "(el) => {" + HELPERS_JS + r"""
  const roles = %s;
  const semantics = (node) => {
    const tag = node.tagName.toLowerCase();
    if (tag === 'a' && node.hasAttribute('href')) return ['link', 'anchor with href'];
    if (tag === 'button') return ['button', 'native button'];
    if (tag === 'input' && ['button', 'submit', 'reset', 'image'].includes(node.type)) return ['button', 'native button'];
    if (tag === 'input' && ['checkbox', 'radio'].includes(node.type)) return [node.type, 'native choice control'];
    if (tag === 'summary' && node.parentElement && node.parentElement.tagName === 'DETAILS') return ['button', 'native disclosure'];
    if (tag === 'label' && node.control) return ['label', 'label bound to a control'];
    if (tag === 'select') return ['combobox', 'native select'];
    const role = (node.getAttribute('role') || '').trim().toLowerCase();
    if (roles.includes(role)) return [role, 'explicit interactive role'];
    return null;
  };
  for (let node = el.parentElement; node && node !== document.body && node !== document.documentElement; node = node.parentElement) {
    const found = semantics(node);
    if (!found) continue;
    if (node.disabled || node.getAttribute('aria-disabled') === 'true') return null;
    return {role: found[0], why: found[1], name: controlName(node), text: clean(node.textContent).slice(0, 80),
            ref: cssPath(node), box: box(node)};
  }
  return null;
}
""" % list(INTERACTIVE_ROLES)


def diagnostic(error: Exception) -> str:
    """The first line of a browser error, for logs and messages. Never used to decide anything."""
    text = str(getattr(error, "message", "") or error).strip()
    return (text.splitlines() or [""])[0][:160]


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

    def __init__(self, headless: bool = True, timeout_ms: int = 15000, secrets: tuple[str, ...] = (),
                 device_scale_factor: float | None = None, allowed_hosts: list[str] | None = None):
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=headless)
        context_options = {"device_scale_factor": device_scale_factor} if device_scale_factor else {}
        self._page = self._browser.new_context(**context_options).new_page()
        self._page.set_default_timeout(timeout_ms)
        # Hosts a page this adapter opens by itself may live on. None means "the caller checks elsewhere"
        # (the policy always checks every action); a list closes the door on an unexpected popup target.
        self.allowed_hosts = list(allowed_hosts) if allowed_hosts is not None else None
        self._opener = None                       # the page a popup was opened from, for `back`
        self.last_selection_proof = ""            # how the last selection was accepted, for the log (never a value)
        self.timeout_ms = timeout_ms
        self.secrets = tuple(s for s in secrets if s)   # values that must never appear in evidence
        self.title = ""
        self._last_document_status: int | None = None
        self._cdp = None                          # CDP session for the accessibility tree, opened lazily
        self.last_accessibility_error: str | None = None   # why the last observation was DOM-only, if it was
        self._navigations = 0                     # top-level documents committed so far (see observe)
        self.observation_retries = 0              # observations restarted because the document was replaced
        self.observation_failures = 0             # core observation failures on a stable document
        self._accessibility_browser_error = False # the last merge fell back because the browser raised
        # Track the status of the last top-level document so a 5xx after a click is noticed.
        self._watch(self._page)

    def _watch(self, page) -> None:
        """Listen on a page this adapter drives: document status and top-level navigations."""
        page.on("response", self._remember_document_status)
        page.on("framenavigated", self._remember_navigation)

    def _remember_document_status(self, response) -> None:
        if response.request.is_navigation_request() and response.frame == self._page.main_frame:
            self._last_document_status = response.status

    def _remember_navigation(self, frame) -> None:
        if frame == self._page.main_frame:
            self._navigations += 1

    # ---------- perception ----------

    def observe(self) -> Observation:
        """One consistent look at the current document, tolerant of a navigation the last action started.

        The observation is a transaction (settle, status check, projection, accessibility merge). Whether
        a browser error means the document was replaced is decided by evidence, never by the error's
        text: the main frame's navigation counter must advance (it may do so shortly after the error, so
        a bounded wait is allowed). A confirmed navigation discards the whole look, including a stale
        projection the accessibility layer fell back to, and takes it again after the browser's own
        navigation reached the load state; the page is never reloaded or navigated here. A browser error
        with no navigation is a core failure of the look, retried within the same bounded budget. The
        accessibility layer stays supplemental: its browser error on a stable document leaves the DOM
        projection untouched and counts as nothing. Exhausting the budget is a TransientError without a
        url, so a replay does not reload either; a raw browser error never escapes.
        """
        from playwright.sync_api import Error as PlaywrightError, TimeoutError as PlaywrightTimeout

        for attempt in range(1, MAX_OBSERVE_ATTEMPTS + 1):
            started = self._navigations
            self._accessibility_browser_error = False
            try:
                observation = self._observe_once()
            except PlaywrightTimeout:
                raise                                       # a load that never finished: not a race
            except PlaywrightError as error:               # the page call failed: a navigation, or a core failure
                if self._navigation_confirmed(started):
                    self._retry_after_navigation(attempt, error)
                    continue
                self.observation_failures += 1
                if attempt == MAX_OBSERVE_ATTEMPTS:
                    raise TransientError(f"the page could not be observed after {attempt} attempts", url=None) from error
                continue
            if self._navigations != started or (self._accessibility_browser_error and self._navigation_confirmed(started)):
                self._retry_after_navigation(attempt, None)   # a new document committed during the look
                continue
            return observation
        raise AssertionError("unreachable")

    def _navigation_confirmed(self, started: int) -> bool:
        """Whether a new top-level document committed since the look began, waiting briefly for the event."""
        deadline = time.time() + NAVIGATION_CONFIRM_MS / 1000
        while self._navigations == started and time.time() < deadline:
            self._page.wait_for_timeout(NAVIGATION_POLL_MS)
        return self._navigations != started

    def _retry_after_navigation(self, attempt: int, cause: Exception | None) -> None:
        """Discard the look; let the navigation in flight finish, or give up after the last attempt."""
        self.observation_retries += 1
        if attempt == MAX_OBSERVE_ATTEMPTS:
            raise TransientError("the page kept navigating while it was being observed", url=None) from cause
        self._await_document()

    def _await_document(self) -> None:
        """Let the navigation in flight finish on its own; a document that never loads is a load failure."""
        from playwright.sync_api import Error as PlaywrightError, TimeoutError as PlaywrightTimeout

        try:
            self._page.wait_for_load_state("load", timeout=self.timeout_ms)
        except PlaywrightTimeout as error:
            raise TransientError("page did not finish loading", url=self._page.url) from error
        except PlaywrightError:
            return                                          # replaced again already: the next look will see

    def _observe_once(self) -> Observation:
        self._settle()
        status = self._last_document_status
        if status is not None and status >= 500:
            self._last_document_status = None
            raise TransientError(f"server returned HTTP {status}", url=self._page.url)
        raw = self._page.evaluate(OBSERVE_JS)
        self.title = raw["title"]
        elements = [Element(role=e["role"], name=e["name"], text=e["text"], box=tuple(e["box"]),
                            context=e["context"], ref=e["ref"], native=e.get("native", ""),
                            landmark=e.get("landmark", ""), landmark_name=e.get("landmark_name", ""),
                            dismisses=bool(e.get("dismisses")), states=dict(e.get("states") or {}))
                    for e in raw["elements"]]
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
        projection that came in. A browser error is noted for `observe`, which restarts the whole
        look only if a navigation is confirmed to have replaced the document meanwhile.
        """
        from playwright.sync_api import Error as PlaywrightError

        try:
            nodes, paths, order = self._accessibility_nodes()
            merged = [replace(element, states=dict(element.states)) for element in elements]   # copies only
            by_ref = {element.ref: element for element in merged}
            pending: list[tuple[str, str, str, dict]] = []
            seen: set[str] = set()
            texts = 0
            for node in nodes:
                ax_role = str((node.get("role") or {}).get("value", "")).lower()
                role = AX_ROLES.get(ax_role)
                ref = paths.get(node.get("backendDOMNodeId"))
                name = clean_text((node.get("name") or {}).get("value", ""))
                if ax_role == AX_STATIC_TEXT and not node.get("ignored") and ref is not None and ref not in by_ref \
                        and name and texts < MAX_AX_NODES:
                    # Rendered text the projection did not list (a text node under a zero-box wrapper, or
                    # directly under the body): added only through its backing element, and at most
                    # MAX_AX_TEXT_NODES of them per screen (counted in `_enrich`, after the visibility check).
                    texts += 1
                    pending.append((ref, "text", name, {}))
                    continue
                if node.get("ignored") or role is None or ref is None or ref in seen:
                    continue
                seen.add(ref)
                if len(seen) > MAX_AX_NODES:
                    break
                states = ax_states(node)
                existing = by_ref.get(ref)
                if existing is None:
                    pending.append((ref, role, name, states))
                elif existing.role == "text":
                    existing.role, existing.name = role, name or existing.text
                    existing.states, existing.source = {**existing.states, **states}, "ax"
                else:
                    existing.states, existing.source = {**existing.states, **states}, "dom+ax"
            merged = merged + self._enrich(pending)
            big = len(order) + 1
            result = sorted(merged, key=lambda element: order.get(element.ref, big))
        except Exception as error:   # a supplemental source must never cost the observation
            self._accessibility_browser_error = isinstance(error, PlaywrightError)
            self.last_accessibility_error = f"{type(error).__name__}: {error}"
            return elements
        self.last_accessibility_error = None
        return result

    def _enrich(self, pending: list[tuple[str, str, str, dict]]) -> list[Element]:
        if not pending:
            return []
        details = self._page.evaluate(ENRICH_JS, [[ref, name if role == "text" else ""] for ref, role, name, _ in pending])
        added = []
        seen_text: set[tuple[str, str]] = set()
        for (ref, role, name, states), detail in zip(pending, details):
            if not detail or not detail["visible"]:
                continue
            if role == "text":
                key = (ref, detail["text"])
                if detail["represented"] or key in seen_text or len(seen_text) >= MAX_AX_TEXT_NODES:
                    continue
                seen_text.add(key)
                added.append(Element(role="text", name="", text=detail["text"], box=tuple(detail["box"]),
                                     context=detail["context"], ref=ref, source="ax"))
                continue
            if detail.get("availability"):
                states = {**states, detail["availability"]: "true"}
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
        element.landmark_name = self._mask(element.landmark_name)
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

    def resolve(self, locator: Locator, exclude: frozenset[str] = frozenset()) -> Element | None:
        """Try each strategy in the ladder; return the element of the first rung that resolves uniquely.

        A role or text rung succeeds only when exactly one perceived element matches all its fields; several
        matches make the rung ambiguous and the next, more specific rung decides, so document order never
        picks an element. `exclude` names structural references already taken by earlier targets of the same
        list: a rung whose own result is one of them is passed over the same way.
        """
        for strategy in locator.strategies:
            element = self._resolve_one(strategy)
            if element is None or (element.ref and element.ref in exclude):
                continue
            return element
        return None

    def matches(self, strategy: dict) -> list[Element]:
        """Every perceived element a role or text rung matches (all its fields applied); [] for other rungs."""
        kind = strategy.get("kind")
        if kind not in ("role", "text"):
            return []
        found = []
        for element in self.observe().elements:
            if kind == "role" and role_matches(element, strategy):
                found.append(element)
            elif kind == "text" and strategy["text"] and strategy["text"] in element.text:
                found.append(element)
        return found

    def _resolve_one(self, strategy: dict) -> Element | None:
        kind = strategy.get("kind")
        if kind in ("role", "text"):
            # Match against perception, not markup, so the same rule works on any surface; one match or none.
            found = self.matches(strategy)
            return found[0] if len(found) == 1 else None
        if kind == "css":
            # Prefer the element as perception sees it: the projection carries the control's role, name and
            # the runtime-only evidence (native kind, enclosing landmark, dismissal metadata) that a bare
            # handle cannot, and the safety policy judges a control by exactly that.
            perceived = next((e for e in self.observe().elements if e.ref == strategy["selector"]), None)
            if perceived is not None:
                return perceived
            handle = self._page.query_selector(strategy["selector"])
            if handle is None or not handle.is_visible():
                return None
            box = handle.bounding_box() or {"x": 0, "y": 0, "width": 0, "height": 0}
            return Element(role="unknown", name="", text=(handle.inner_text() or "")[:200],
                           box=(int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"])),
                           ref=strategy["selector"])
        if kind == "coords":
            # Screenshot-style fallback: whatever control is drawn at that point on screen. A rung that
            # names the viewport it was recorded in is only honoured in that viewport: page layouts do
            # not scale linearly, so guessing would click the wrong thing.
            recorded = strategy.get("viewport")
            if recorded and self.viewport_size() != (recorded.get("width"), recorded.get("height")):
                return None
            x, y = strategy["x"], strategy["y"]
            if strategy.get("exact"):
                # A vision coordinate: the recorded mouse point itself, valid only at the recorded scroll
                # position. It is never widened to whatever element happens to contain the point.
                scroll = strategy.get("scroll") or {"x": 0, "y": 0}
                if self.scroll_position() != (scroll.get("x", 0), scroll.get("y", 0)):
                    return None
                if strategy.get("read"):
                    # A grounded extraction point: the page text rendered there now, or nothing.
                    return self.text_under((x, y, 1, 1))
                return Element(role="unknown", name="", box=(x, y, 0, 0), source="coords")
            for element in self.observe().elements:
                ex, ey, ew, eh = element.box
                if ex <= x <= ex + ew and ey <= y <= ey + eh and element.role != "text":
                    return element
            return Element(role="unknown", name="", box=(x, y, 1, 1))
        return None

    # ---------- action ----------
    #
    # Every physical action has two phases. Phase one is inspection and actionability (an explicit
    # `trial=True` check, a state read, a label lookup, a preflight): a failure there means nothing was
    # dispatched, `performed="no"`. Phase two begins the moment the real click, fill or key press is
    # issued: any browser error from then on is `performed="unknown"`, and a readable post-state that
    # did not change is `performed="yes"`. The classification comes from which phase failed, never
    # from the wording of the error; error text is kept only as a sanitized diagnostic.

    def still_present(self, target: Element) -> bool:
        """Whether the control observed a moment ago is still in the page, asked once and without waiting.

        A transient notice can close itself between the observation an action was chosen from and the action
        itself; waiting the full action timeout for a control that is provably gone helps nobody. Read-only,
        and never used to shorten the wait for a control that was not already observed: the check is a
        single immediate query for the structural reference perception recorded.
        """
        from playwright.sync_api import Error as PlaywrightError

        if not target.ref:
            return True                    # nothing structural to check: the ordinary path decides
        try:
            return self._page.query_selector(target.ref) is not None
        except PlaywrightError:
            return True                    # cannot tell: let the ordinary path run

    def selected_state(self, target: Element) -> dict | None:
        """Whether this control reads as selected, from browser state alone, or None when it cannot be read.

        Deterministic sources only: native selectedness, the ARIA state the control publishes about
        itself, a boolean state attribute a custom control sets, or a structurally associated control (a
        label's input, a single wrapped native control, a hidden control it drives). Computed style is
        never consulted: colour is how a selection is shown, not what makes it true.
        """
        from playwright.sync_api import Error as PlaywrightError

        if not target or not target.ref:
            return None
        try:
            handle = self._page.locator(target.ref).first.element_handle(timeout=self.timeout_ms)
            if handle is None:
                return None
            return handle.evaluate(SELECTED_JS)
        except PlaywrightError:
            return None                    # cannot tell: the caller falls back to no proof

    def committed_state(self, target: Element) -> dict | None:
        """The committed state of the field group `target` sits in, or None when it has no such group.

        Read-only and deterministic: the editable fields' own values, the values stored in hidden inputs
        and selects, and the selected tokens (each carrying its own removal control or a native selected
        state). Scoped to the nearest meaningful container so nothing elsewhere on the page can be read as
        this control's doing. A password field's contents are never read.
        """
        from playwright.sync_api import Error as PlaywrightError

        if not target or not target.ref:
            return None
        try:
            handle = self._page.locator(target.ref).first.element_handle(timeout=self.timeout_ms)
            if handle is None:
                return None
            state = handle.evaluate(COMMITTED_JS)
        except PlaywrightError:
            return None                    # cannot tell: the caller falls back to no proof
        if not state.get("scoped"):
            return None
        # Perception reports every candidate; one logical token per removal control is what a caller sees.
        state["tokens"] = canonical_tokens(state.get("tokens") or [])
        return state

    def click(self, target: Element) -> Element | None:
        """Click through the element's own structural reference (or its recorded point).

        A native radio or checkbox takes the label-aware path (`_activate_choice`); everything else
        is Playwright's ordinary click after an explicit actionability trial, never forced. When the
        trial proves the element itself cannot take the click, the nearest enclosing control that is
        provably actionable (`_actionable_ancestor`) is clicked instead and returned, so the caller
        records that control; otherwise the failure is `performed="no"`, cause `unactionable_target`.
        A click that opens exactly one new page makes that page active (`_adopt_popup`); the click
        itself is never repeated because the original page looked unchanged.
        """
        from playwright.sync_api import Error as PlaywrightError

        what = f"click on {target.role} '{target.name}'"
        if not self.still_present(target):
            raise ActionError(f"{what} was not performed: the control was on screen when it was chosen and is no "
                              f"longer in the page", performed="no", cause="stale_target")
        if not target.ref:
            try:
                x, y, w, h = target.box
                with self._popups() as opened:
                    self._page.mouse.click(x + w / 2, y + h / 2)
            except PlaywrightError as error:
                raise ActionError(f"{what} did not complete: {diagnostic(error)}", performed="unknown",
                                  url=self._page.url) from error
            self._adopt_popup(opened, what)
            return None
        locator = self._page.locator(target.ref).first
        if self._activate_choice(locator, what):
            return None
        try:
            locator.click(trial=True)
        except PlaywrightError as error:
            return self._click_actionable_ancestor(locator, what, diagnostic(error))
        try:
            with self._popups() as opened:
                locator.click()
        except PlaywrightError as error:
            raise ActionError(f"{what} did not complete: {diagnostic(error)}", performed="unknown",
                              url=self._page.url) from error
        self._adopt_popup(opened, what)
        return None

    @contextmanager
    def _popups(self):
        """Collect the pages the browser context opens while the block runs, in the order they appeared."""
        opened: list = []

        def remember(page) -> None:
            opened.append(page)

        context = self._page.context
        context.on("page", remember)
        try:
            yield opened
            # A page event can arrive several ticks after the click returns, so an empty list is not
            # evidence of a same-page click until the first window has fully elapsed. Only once a page has
            # appeared does the second question arise: are there more? A script opening two windows emits
            # them separately, so collection continues until the count stops growing.
            waited = 0
            while not opened and waited < POPUP_FIRST_MS:
                self._page.wait_for_timeout(POPUP_POLL_MS)
                waited += POPUP_POLL_MS
            if opened:
                # Siblings are collected for the whole settle window, not until growth pauses: a script
                # that opens two windows may leave a gap of several ticks between them, and stopping at
                # the first quiet poll would see one page where there are two. The window is short and is
                # only ever paid by a click that already produced a page.
                waited = 0
                while waited < POPUP_SETTLE_MS:
                    self._page.wait_for_timeout(POPUP_POLL_MS)
                    waited += POPUP_POLL_MS
        finally:
            context.remove_listener("page", remember)

    def _adopt_popup(self, opened: list, what: str) -> None:
        """Make a single new page the active one, or fail structurally.

        Exactly one new page: it is waited for with the ordinary bounded load handling, its address is
        checked against the allowlist, and it becomes the page every later observation, screenshot,
        accessibility read, extraction and action uses; the opener is kept so `back` can close it.
        Several new pages are an `ambiguous_selection` failure and none is chosen. A destination outside
        the allowlist is closed and the run stays on the original page (`performed="unknown"`: the click
        did reach the page).
        """
        from playwright.sync_api import Error as PlaywrightError, TimeoutError as PlaywrightTimeout

        if not opened:
            return
        if len(opened) > 1:
            for page in opened:
                self._close_quietly(page)
            raise ActionError(f"{what} opened {len(opened)} new pages and none of them can be chosen safely",
                              performed="unknown", cause="ambiguous_selection")
        popup = opened[0]
        try:
            popup.wait_for_load_state("load", timeout=self.timeout_ms)
        except PlaywrightTimeout:
            pass                      # a slow child page: `observe` waits again on the active page
        except PlaywrightError as error:
            self._close_quietly(popup)
            raise ActionError(f"{what} opened a page that could not be read: {diagnostic(error)}",
                              performed="unknown", url=self._page.url) from error
        host = urlparse(popup.url).hostname or ""
        if self.allowed_hosts is not None and host not in self.allowed_hosts:
            self._close_quietly(popup)
            raise ActionError(f"{what} opened a page on host '{host}', which is outside the allowlist; it was "
                              f"closed and nothing was done there", performed="unknown", url=self._page.url)
        popup.set_default_timeout(self.timeout_ms)
        self._watch(popup)
        self._opener, self._page = self._page, popup
        self._cdp = None                          # the accessibility session belongs to the old page
        self._navigations += 1

    @staticmethod
    def _close_quietly(page) -> None:
        from playwright.sync_api import Error as PlaywrightError

        try:
            page.close()
        except PlaywrightError:
            pass

    def _click_actionable_ancestor(self, locator, what: str, why_not: str) -> Element:
        """The element's own trial failed, so nothing was dispatched. Find the nearest enclosing control that is
        provably actionable by its semantics alone (`ACTIONABLE_ANCESTOR_JS`: a native control, an anchor with
        href, a label bound to a control, or an explicit interactive ARIA role; never a container that merely
        holds the text or carries a script handler), give it the same trial, and only then click it once.
        Any failure before that click is `performed="no"` with cause `unactionable_target`.
        """
        from playwright.sync_api import Error as PlaywrightError

        try:
            found = locator.evaluate(ACTIONABLE_ANCESTOR_JS)
        except PlaywrightError as error:
            raise ActionError(f"{what} was not performed: {why_not}; its enclosing controls could not be inspected "
                              f"({diagnostic(error)})", performed="no", cause="unactionable_target") from error
        if not found:
            raise ActionError(f"{what} was not performed: {why_not}; no enclosing control is provably actionable",
                              performed="no", cause="unactionable_target")
        ancestor = self._page.locator(found["ref"]).first
        via = f"{found['role']} '{found['name']}' ({found['why']})"
        try:
            ancestor.click(trial=True)
        except PlaywrightError as error:
            raise ActionError(f"{what} was not performed: {why_not}; its enclosing {via} is not actionable either "
                              f"({diagnostic(error)})", performed="no", cause="unactionable_target") from error
        try:
            ancestor.click()
        except PlaywrightError as error:
            raise ActionError(f"{what} through its enclosing {via} did not complete: {diagnostic(error)}",
                              performed="unknown", url=self._page.url) from error
        return Element(role=found["role"], name=found["name"], text=found["text"], box=tuple(found["box"]),
                       ref=found["ref"])

    def _activate_choice(self, locator, what: str) -> bool:
        """Activate a native radio or checkbox the way a person does; False when `locator` is not one.

        Order: the browser-associated visible label that passes the actionability trial, else the
        input itself after the same trial, else the guarded keyboard path (`_keyboard_activate`).
        The element handle taken at the start is used throughout, so a control the page replaces
        is noticed rather than re-resolved to a look-alike. Inspection failures are `performed="no"`;
        after a click or key press begins, failures are `"unknown"`; a readable unchanged post-state
        is `"yes"`. Nothing is ever forced or clicked through script.
        """
        from playwright.sync_api import Error as PlaywrightError

        try:
            handle = locator.element_handle()
            state = handle.evaluate(CHOICE_STATE_JS)
        except PlaywrightError as error:
            raise ActionError(f"{what} was not performed: {diagnostic(error)}", performed="no") from error
        if state is None:
            return False
        if state["disabled"]:
            raise ActionError(f"{what} was not performed: the {state['type']} is disabled", performed="no")
        if state["type"] == "radio" and state["checked"]:
            return True                                   # already selected: satisfied, not toggled
        how = self._activate_by_pointer(handle, what)
        if how is None:
            how = self._keyboard_activate(handle, what)
        try:
            after = handle.evaluate(CHOICE_STATE_JS)
        except PlaywrightError as error:
            raise ActionError(f"{what} ({how}) was dispatched but the control could not be re-read: "
                              f"{diagnostic(error)}", performed="unknown") from error
        if after is None or not after["connected"]:
            raise ActionError(f"{what} ({how}) was dispatched but the control left the document before its state "
                              f"could be verified", performed="unknown")
        satisfied = after["checked"] if state["type"] == "radio" else after["checked"] != state["checked"]
        if not satisfied:
            raise ActionError(f"{what} ({how}) was performed but the {state['type']} did not change state",
                              performed="yes")
        return True

    def _activate_by_pointer(self, handle, what: str) -> str | None:
        """Click the first visible associated label that passes the trial, else the input after its own trial.

        Returns how it was done, or None when neither pointer path is actionable (nothing dispatched).
        """
        from playwright.sync_api import Error as PlaywrightError

        try:
            labels = [h.as_element() for h in handle.evaluate_handle(CHOICE_LABELS_JS).get_properties().values()]
            candidates = [("label", label) for label in labels if label is not None and label.is_visible()]
        except PlaywrightError as error:
            raise ActionError(f"{what} was not performed: {diagnostic(error)}", performed="no") from error
        candidates.append(("input", handle))
        for how, element in candidates:
            try:
                element.click(trial=True)
            except PlaywrightError:
                continue                                  # covered, detached or otherwise not actionable
            try:
                element.click()
            except PlaywrightError as error:
                raise ActionError(f"{what} ({how} click) did not complete: {diagnostic(error)}",
                                  performed="unknown") from error
            return f"{how} click"
        return None

    def _keyboard_activate(self, handle, what: str) -> str:
        """The guarded keyboard path for a native choice control whose pointer paths are obstructed.

        Safety, decided from DOM semantics before any key is sent (`KEYBOARD_SAFETY_JS`): the target is
        a native enabled radio or checkbox, visible, not inert and not behind an open modal elsewhere;
        and whatever the browser hit-tests at the control's centre is part of the same choice structure
        (the control and its own labels; for a radio also its native group of same name, form and type
        and that group's labels), never an unrelated overlay or an independent same-named checkbox. The
        control is then focused normally and must be `document.activeElement`.
        Only then is the native activation key (Space) pressed; from that moment on failures are unknown.
        """
        from playwright.sync_api import Error as PlaywrightError

        try:
            safety = handle.evaluate(KEYBOARD_SAFETY_JS)
        except PlaywrightError as error:
            raise ActionError(f"{what} was not performed: {diagnostic(error)}", performed="no") from error
        if not safety["safe"]:
            raise ActionError(f"{what} was not performed: pointer paths are obstructed and the keyboard path is "
                              f"not safe ({safety['reason']})", performed="no")
        try:
            handle.focus()
            focused = handle.evaluate("(el) => document.activeElement === el")
        except PlaywrightError as error:
            raise ActionError(f"{what} was not performed: the control could not be focused: {diagnostic(error)}",
                              performed="no") from error
        if not focused:
            raise ActionError(f"{what} was not performed: the control did not take focus", performed="no")
        try:
            self._page.keyboard.press("Space")
        except PlaywrightError as error:
            raise ActionError(f"{what} (keyboard) did not complete: {diagnostic(error)}", performed="unknown") from error
        return "keyboard"

    def type(self, target: Element, value: str) -> None:
        """Fill through the element's reference: an explicit preflight (attached, visible, enabled, editable)
        decides `performed="no"`; a failure of the real fill is `"unknown"`."""
        from playwright.sync_api import Error as PlaywrightError

        what = f"typing into {target.role} '{target.name}'"
        if not self.still_present(target):
            raise ActionError(f"{what} was not performed: the control was on screen when it was chosen and is no "
                              f"longer in the page", performed="no", cause="stale_target")
        if not target.ref:
            self.click(target)
            self._page.keyboard.type(value)
            return
        locator = self._page.locator(target.ref).first
        try:
            locator.wait_for(state="visible")
            ready = locator.is_enabled() and locator.is_editable()
        except PlaywrightError as error:
            raise ActionError(f"{what} was not performed: {diagnostic(error)}", performed="no") from error
        if not ready:
            raise ActionError(f"{what} was not performed: the control is not enabled and editable", performed="no")
        try:
            locator.fill(value)
        except PlaywrightError as error:
            raise ActionError(f"{what} did not complete: {diagnostic(error)}", performed="unknown",
                              url=self._page.url) from error

    def select(self, target: Element, value: str) -> None:
        """Choose `value` in a chooser: a native <select> by option text, or a combobox / autocomplete text
        input by typing the value, waiting a bounded time for visible suggestions, clicking the matching one
        through its own semantics and verifying the control accepted it.

        `performed="no"` until something is typed or chosen; a typed value with no accepted suggestion is
        `performed="yes"` (the field holds text, nothing was selected); a failure once an option click or a
        native selection has been issued is `"unknown"`.
        """
        from playwright.sync_api import Error as PlaywrightError

        what = f"selecting a value in {target.role} '{target.name}'"
        if not self.still_present(target):
            raise ActionError(f"{what} was not performed: the control was on screen when it was chosen and is no "
                              f"longer in the page", performed="no", cause="stale_target")
        if not target.ref:
            raise ActionError(f"{what} was not performed: the control has no structural reference", performed="no")
        locator = self._page.locator(target.ref).first
        try:
            handle = locator.element_handle()
            kind = handle.evaluate(SELECT_KIND_JS)
        except PlaywrightError as error:
            raise ActionError(f"{what} was not performed: {diagnostic(error)}", performed="no") from error
        if not kind["kind"] or not kind["editable"]:
            raise ActionError(f"{what} was not performed: the control is not an enabled chooser or text field",
                              performed="no")
        if kind["kind"] == "select":
            self._select_native(handle, value, what)
            return
        self._select_suggestion(handle, value, what)

    def _select_native(self, handle, value: str, what: str) -> None:
        from playwright.sync_api import Error as PlaywrightError

        try:
            options = handle.evaluate(SELECT_OPTIONS_JS)
        except PlaywrightError as error:
            raise ActionError(f"{what} was not performed: {diagnostic(error)}", performed="no") from error
        status, match, level = match_option(options, value)
        if status == "ambiguous":
            raise ActionError(f"{what} was not performed: more than one of the {len(options)} options is a {level} "
                              f"match for the value; nothing was chosen", performed="no", cause="ambiguous_selection")
        if match is None:
            raise ActionError(f"{what} was not performed: none of the {len(options)} options matches the value",
                              performed="no")
        try:
            handle.select_option(value=match["value"])
            chosen = handle.evaluate("(el) => el.value")
        except PlaywrightError as error:
            raise ActionError(f"{what} did not complete: {diagnostic(error)}", performed="unknown",
                              url=self._page.url) from error
        if chosen != match["value"]:
            raise ActionError(f"{what} was performed but the control did not keep the option", performed="yes")

    def _select_suggestion(self, handle, value: str, what: str) -> None:
        from playwright.sync_api import Error as PlaywrightError

        try:
            handle.click(trial=True)
            before = handle.evaluate(ACCEPTED_JS)
        except PlaywrightError as error:
            raise ActionError(f"{what} was not performed: {diagnostic(error)}", performed="no") from error
        # Type the value the way a person does. Many autocompletes filter their list from key events alone
        # and never react to a value set programmatically, so the suggestions would never appear.
        try:
            handle.click()
            handle.press("ControlOrMeta+a")
            handle.press("Delete")                  # clear whatever the field held, without touching the DOM
        except PlaywrightError as error:
            raise ActionError(f"{what} was not performed: {diagnostic(error)}", performed="no") from error
        try:
            handle.type(value, delay=KEYSTROKE_DELAY_MS)
        except PlaywrightError as error:
            raise ActionError(f"{what} did not complete: {diagnostic(error)}", performed="unknown",
                              url=self._page.url) from error
        deadline = time.time() + SUGGESTION_TIMEOUT_MS / 1000
        match = None
        while match is None:
            try:
                shown = handle.evaluate(SUGGESTIONS_JS)
            except PlaywrightError as error:
                raise ActionError(f"{what}: the value was typed but the suggestions could not be read: "
                                  f"{diagnostic(error)}", performed="yes") from error
            status, match, level = matching_suggestion(shown, value)
            if status == "ambiguous_popups":
                raise ActionError(f"{what}: the value was typed but {level} open popups hold a suggestion matching "
                                  f"it and the control names none of them; nothing was chosen", performed="yes",
                                  cause="ambiguous_selection")
            if status == "ambiguous":
                raise ActionError(f"{what}: the value was typed but more than one suggestion is a {level} match for "
                                  f"it; nothing was chosen", performed="yes", cause="ambiguous_selection")
            if match is None and time.time() >= deadline:
                if status == "none" and not (shown.get("options") or shown.get("groups")):
                    # Nothing was ever offered: some fields resolve the typed value only when Enter is pressed.
                    # One attempt, accepted only on deterministic new state (`_commit_by_keyboard`).
                    return self._commit_by_keyboard(handle, value, what, before)
                raise ActionError(f"{what}: the value was typed but visible suggestions did not match it within "
                                  f"{SUGGESTION_TIMEOUT_MS // 1000}s; typed text is not a selection", performed="yes")
            if match is None:
                self._page.wait_for_timeout(SUGGESTION_POLL_MS)
        option = self._page.locator(match["ref"]).first
        try:
            option.click(trial=True)
        except PlaywrightError as error:
            raise ActionError(f"{what}: the matching suggestion is not actionable ({diagnostic(error)}); typed text "
                              f"is not a selection", performed="yes") from error
        try:
            option.click()
            accepted = handle.evaluate(ACCEPTED_JS)
        except PlaywrightError as error:
            raise ActionError(f"{what} did not complete after choosing the suggestion: {diagnostic(error)}",
                              performed="unknown", url=self._page.url) from error
        if not accepted["connected"]:
            raise ActionError(f"{what}: the suggestion was chosen but the control left the document before its state "
                              f"could be verified", performed="unknown", url=self._page.url)
        proof = acceptance_proof(accepted, match["text"], before, typed=value)
        if not proof:
            raise ActionError(f"{what}: the suggestion was chosen but no widget state shows it was accepted",
                              performed="yes")
        self.last_selection_proof = f"suggestion clicked, proven by {proof}"

    def _commit_by_keyboard(self, handle, value: str, what: str, before: dict) -> None:
        """Last resort for a field that offers no suggestions at all: press Enter exactly once.

        Only reached when the bounded wait saw no suggestion rows whatsoever; a visible list that is
        ambiguous or simply does not match keeps its existing refusal, and Enter is never tried there.
        Acceptance still needs deterministic new state (`acceptance_proof` against the state captured
        before typing): the field resolving into something other than the raw query, an active
        descendant, a stored value or a new chip. The typed text itself, unrelated page changes and a
        navigation alone prove nothing; an effect without proof is an unknown performed action.
        """
        from playwright.sync_api import Error as PlaywrightError

        try:
            handle.press("Enter")
        except PlaywrightError as error:
            raise ActionError(f"{what} did not complete when the value was submitted: {diagnostic(error)}",
                              performed="unknown", url=self._page.url) from error
        self._page.wait_for_timeout(COMMIT_SETTLE_MS)
        try:
            accepted = handle.evaluate(ACCEPTED_JS)
        except PlaywrightError as error:
            # The control is gone (a navigation, a re-render): something happened and nothing proves what.
            raise ActionError(f"{what}: the value was submitted but the control could not be re-read "
                              f"({diagnostic(error)}); the outcome is unknown", performed="unknown",
                              url=self._page.url) from error
        if not accepted["connected"]:
            raise ActionError(f"{what}: the value was submitted but the control left the document before its state "
                              f"could be verified", performed="unknown", url=self._page.url)
        proof = commit_proof(accepted, before, typed=value)
        if proof is None:
            raise ActionError(f"{what}: the value was submitted but no widget state shows it was accepted",
                              performed="yes")
        self.last_selection_proof = f"keyboard commit, proven by {proof}"

    def navigate(self, url: str) -> None:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        try:
            response = self._page.goto(url, wait_until="load")
        except PlaywrightTimeout as error:
            raise TransientError(f"navigation to {url} timed out", url=url) from error
        if response is not None and response.status >= 500:
            raise TransientError(f"server returned HTTP {response.status}", url=url)

    def back(self) -> None:
        """Return to the previous page through the browser's own history, never by re-requesting a URL.

        On a page this adapter adopted from a click, going back means closing that child page and making
        its opener active again: its own history holds nothing earlier. Otherwise the browser's history
        is used: with no previous entry the browser does nothing, which the adapter reports as a
        not-performed action failure; a timeout or a browser error after the request was made is an
        unknown state, because the history move may already have happened. A 5xx is a load failure.
        """
        from playwright.sync_api import Error as PlaywrightError, TimeoutError as PlaywrightTimeout

        if self._opener is not None:
            child, self._page, self._opener = self._page, self._opener, None
            self._close_quietly(child)
            self._cdp = None
            self._navigations += 1
            return
        url = self._page.url
        try:
            response = self._page.go_back(wait_until="load")
        except PlaywrightTimeout as error:
            raise ActionError("back timed out before the previous page loaded", performed="unknown", url=url) from error
        except PlaywrightError as error:
            raise ActionError(f"back failed: {diagnostic(error)}", performed="unknown", url=url) from error
        if response is None and self._page.url == url:
            raise ActionError("back was not performed: there is no previous page in the browser history",
                              performed="no", url=url)
        if response is not None and response.status >= 500:
            raise TransientError(f"server returned HTTP {response.status}", url=self._page.url)

    def text_under(self, box: tuple[int, int, int, int]) -> Element | None:
        """The page element with visible text of its own rendered under a viewport box (see TEXT_UNDER_JS),
        masked like any perceived text; None when nothing text-bearing is drawn there (a canvas, an image)."""
        from playwright.sync_api import Error as PlaywrightError

        try:
            found = self._page.evaluate(TEXT_UNDER_JS, [int(v) for v in box])
        except PlaywrightError:
            return None
        if not found or not found["text"]:
            return None
        return self._masked(Element(role="text", name="", text=found["text"], box=tuple(found["box"]),
                                    context=found["context"], ref=found["ref"]))

    def viewport_size(self) -> tuple[int, int]:
        size = self._page.viewport_size or {"width": 0, "height": 0}
        return int(size["width"]), int(size["height"])

    def scroll_position(self) -> tuple[int, int]:
        x, y = self._page.evaluate("() => [Math.round(window.scrollX), Math.round(window.scrollY)]")
        return int(x), int(y)

    def viewport_screenshot(self, path: str) -> ScreenshotFrame:
        """A masked capture of the viewport only, in CSS pixels, so image coordinates are mouse coordinates.

        `scale="css"` keeps the image at the CSS size even on a high-density display; the PNG header
        is checked against the viewport so a mismatch is an error rather than a silent offset. Same
        masking and restoration as `screenshot`; the bytes are returned for the vision planner and
        the masked image is the only copy written (as evidence at `path`). The scroll position is
        recorded because the coordinates are only meaningful at that position.
        """
        if self.secrets:
            self._page.evaluate(MASK_JS, list(self.secrets))
        try:
            png = self._page.screenshot(path=path, full_page=False, scale="css")
        finally:
            if self.secrets:
                self._page.evaluate(UNMASK_JS)
        width, height = self.viewport_size()
        png_width, png_height = png_dimensions(png)
        if (png_width, png_height) != (width, height):
            raise ValueError(f"screenshot is {png_width}x{png_height} pixels but the viewport is {width}x{height}: "
                             f"coordinates would not match the mouse")
        scroll_x, scroll_y = self.scroll_position()
        return ScreenshotFrame(png=png, width=width, height=height, path=path, scroll_x=scroll_x, scroll_y=scroll_y)

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
