"""An in-memory Surface for tests: a tiny state machine shaped like a demo web shop.

It implements the same Surface protocol as the Playwright adapter, so discovery, replay,
policy, and escalation can be tested without a browser, an LLM, or the network. Screens
are plain lists of Elements; actions move between screens following the shop's rules
(login, product list, cart, checkout information, checkout overview, order complete).
"""
from __future__ import annotations

from dataclasses import replace

from src.cua.models import ActionError, ScreenshotFrame

from .context import Element, Locator, Observation, TransientError

# A valid 1x1 PNG; the fake has no pixels worth looking at, only bytes to hash and hand over.
TINY_PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478"
                         "9c63f8ffff3f0005fe02fe0d8b0a9c0000000049454e44ae426082")
VIEWPORT = (800, 600)

ENTRY = "https://www.saucedemo.com/"
USERS = {"standard_user": "secret_sauce", "locked_out_user": "secret_sauce"}
PRODUCTS = {"Sauce Labs Backpack": 29.99, "Sauce Labs Bike Light": 9.99}
TAX_RATE = 0.08
LOCKED_OUT = "Epic sadface: Sorry, this user has been locked out."
BAD_LOGIN = "Epic sadface: Username and password do not match any user in this service"


def el(role: str, name: str, text: str = "", context: str = "", states: dict | None = None) -> Element:
    """A fake control; its structural reference is distinct per control, as a real page path would be."""
    identity = name or text[:40]
    return Element(role=role, name=name, text=text or name, box=(10, 10, 50, 20), context=context,
                   ref=f"{role}:{identity}:{context}", states=dict(states or {}))


class FakeSurface:
    """Deterministic stand-in for the shop."""

    def __init__(self, show_notice: bool = False, faults: list[str] | None = None,
                 extra_elements: list[Element] | None = None):
        self.url = "about:blank"
        self.screen = "blank"
        self.dialog: str | None = None
        self.notice_pending = show_notice        # a cookie-style notice on the first page view
        # "slow_entry" (next navigate); "transient" | "verification" (on login); action faults, consumed in order:
        # "blocked_click:<name>" (never dispatched), "uncertain_click:<name>" (dispatched, then the surface loses
        # track), "lost_click:<name>" (not dispatched, but the surface cannot tell)
        self.faults = list(faults or [])
        self.extra_elements = list(extra_elements or [])
        self.typed: dict[str, str] = {}
        self.options: dict[str, list[str]] = {}   # choosers: control name -> the suggestion texts it offers
        self.popups: dict[str, int] = {}          # choosers: control name -> unlinked visible popups (default one)
        self.opens_page: dict[str, tuple] = {}    # control name -> (screen, url) a click opens in a new page
        # control name -> native state key ("checked", "expanded", ...) a click flips, and its current value
        self.toggles: dict[str, tuple[str, bool]] = {}
        self.inert: set[str] = set()             # controls that are real and clickable but change nothing at all
        # A generic "commit" widget: clicking `button` moves what was typed in `field` into a token in `group`.
        # commits: button name -> {"field", "group", "removable", "selected", "clears", "stored", "extra_tokens",
        #                          "toast", "outside"}
        self.commits: dict[str, dict] = {}
        # Selectable controls: name -> {"source", "selected", "toggles", "partner", "styling_only", "unknown"}.
        # `source` is the evidence the surface reports it from, exactly as a browser would name it.
        self.selectable: dict[str, dict] = {}
        # Dismissible interface chrome: panel name -> {"control", "landmark", "dismisses", "body",
        # "reveals", "twin"}. Clicking its control removes the panel and its control from the screen,
        # which is what a real sidebar or modal close does.
        self.panels: dict[str, dict] = {}
        self.dismissed: set[str] = set()
        self.tokens: list[dict] = []             # committed tokens now on screen: {"text", "group", "removable", "selected"}
        self.stored: dict[str, str] = {}         # hidden/stored values, keyed by a stable reference
        self.toast = ""                          # transient page copy a commit may also show
        self.opener: tuple | None = None          # the page a popup was opened from, for `back`
        self.selected: dict[str, str] = {}
        self.cart: list[str] = []
        self.logged_in = False
        self.error = ""
        self.actions: list[tuple] = []           # every call, for assertions
        self.history: list[tuple[str, str]] = []  # (screen, url) pages left behind, oldest first, like a browser
        self.screenshots = 0
        self.viewport_shots: list[str] = []      # every viewport capture handed to a vision planner
        self.grounding: dict[tuple, str | None] = {}   # box -> text prefix of the element drawn there (None: nothing)

    # ---------- screens ----------

    def elements(self) -> list[Element]:
        builders = {"blank": lambda: [], "login": self.login_screen, "inventory": self.inventory_screen,
                    "cart": self.cart_screen, "info": self.info_screen, "overview": self.overview_screen,
                    "complete": lambda: [el("text", "", "Thank you for your order!")]}
        tokens = []
        for index, token in enumerate(self.tokens):
            states = {"selected": "true"} if token.get("selected") else {}
            item = el("listitem", "", token["text"], context=token["group"], states=states)
            item.ref = f"token:{index}:{token['group']}"
            tokens.append(item)
            if token.get("removable"):
                remove = el("button", f"Remove {token['text']}", context=token["group"])
                # The same path shape a real surface records, so containment and sibling tests behave
                # here exactly as they do in a browser.
                remove.ref = f"token:{index}:{token['group']} > button:nth-of-type(1)"
                tokens.append(remove)
        commit_controls = []
        for name, spec in self.commits.items():
            field = el("textbox", spec["field"], self.typed.get(spec["field"], ""), context=spec["group"])
            commit_controls += [field, el("button", name, context=spec["group"])]
        toasts = [el("text", "", self.toast)] if self.toast else []
        selectable = [el(spec.get("role", "button"), name, context=spec.get("group", ""))
                      for name, spec in self.selectable.items()]
        panels = []
        for name, spec in self.panels.items():
            if name in self.dismissed:
                if spec.get("reveals"):
                    panels.append(el("text", "", spec["reveals"]))
                # A control elsewhere that merely shares the dismissal's name: a different control.
                if spec.get("twin"):
                    twin = el(spec.get("role", "button"), spec["control"], context="Another region")
                    twin.ref = f"div:nth-of-type({9 + list(self.panels).index(name)}) > button:nth-of-type(1)"
                    panels.append(twin)
                continue
            if spec.get("body"):
                panels.append(el("text", "", spec["body"], context=name))
            control = el(spec.get("role", "button"), spec["control"], context=name)
            control.ref = f"div:nth-of-type({1 + list(self.panels).index(name)}) > button:nth-of-type(1)"
            control.landmark = spec.get("landmark", "region")
            control.landmark_name = name
            control.dismisses = bool(spec.get("dismisses", True))
            control.native = spec.get("native", "button:button")
            panels.append(control)
            if spec.get("twin"):
                twin = el(spec.get("role", "button"), spec["control"], context="Another region")
                twin.ref = f"div:nth-of-type({9 + list(self.panels).index(name)}) > button:nth-of-type(1)"
                panels.append(twin)
        toggles = [el("button", name, states={key: value}) for name, (key, value) in self.toggles.items()]
        inert = [el("button", name) for name in sorted(self.inert)]
        return (builders[self.screen]() + commit_controls + tokens + toasts + selectable + panels + toggles
                + inert + self.extra_elements)

    def login_screen(self) -> list[Element]:
        screen = [el("text", "", "Swag Labs"), el("textbox", "Username", self.typed.get("Username", " ")),
                  el("textbox", "Password", " "),            # a password's value is never perceived
                  el("button", "Login")]
        return screen + ([el("text", "", self.error)] if self.error else [])

    def inventory_screen(self) -> list[Element]:
        screen = [el("text", "", "Products"), el("link", "cart", str(len(self.cart)) if self.cart else " ")]
        for name, price in PRODUCTS.items():
            button = "Remove" if name in self.cart else "Add to cart"
            screen += [el("link", f"View details for {name}", name), el("text", "", f"$ {price}"),
                       el("button", button, context=name)]
        return screen

    def cart_screen(self) -> list[Element]:
        screen = [el("text", "", "Your Cart")]
        for name in self.cart:
            screen += [el("link", f"View details for {name}", name), el("button", "Remove", context=name)]
        return screen + [el("button", "Continue Shopping"), el("button", "Checkout")]

    def info_screen(self) -> list[Element]:
        screen = [el("text", "", "Checkout: Your Information")]
        screen += [el("textbox", field, self.typed.get(field, " ")) for field in ("First Name", "Last Name", "Zip/Postal Code")]
        screen += [el("button", "Cancel"), el("button", "Continue")]
        return screen + ([el("text", "", self.error)] if self.error else [])

    def overview_screen(self) -> list[Element]:
        subtotal = sum(PRODUCTS[name] for name in self.cart)
        tax = round(subtotal * TAX_RATE, 2)
        screen = [el("text", "", "Checkout: Overview")]
        screen += [el("link", f"View details for {name}", name) for name in self.cart]
        screen += [el("text", "", f"Item total: $ {subtotal:.2f}"), el("text", "", f"Tax: $ {tax:.2f}"),
                   el("text", "", f"Total: $ {subtotal + tax:.2f}"), el("button", "Cancel"), el("button", "Finish")]
        return screen

    def dialog_elements(self) -> list[Element]:
        """Controls that belong to the open dialog, listed like any other control, inside a dialog landmark."""
        if not self.dialog:
            return []
        name = "Accept" if "cookies" in self.dialog else "Close" if "Preview" in self.dialog else "Verify"
        return [replace(el("button", name), landmark="dialog", landmark_name=self.dialog[:40])]

    def observe(self) -> Observation:
        if self.screen == "error":
            raise TransientError("server returned HTTP 503", url=self.url)
        return Observation(url=self.url, elements=self.dialog_elements() + self.elements(), dialog=self.dialog)

    # ---------- Surface protocol ----------

    def resolve(self, locator: Locator, exclude: frozenset = frozenset()) -> Element | None:
        for strategy in locator.strategies:
            found = self.resolve_one(strategy)
            if found is None or found.ref in exclude:
                continue                     # ambiguous, missing or taken: the next, more specific rung decides
            return found
        return None

    def matches(self, strategy: dict) -> list[Element]:
        """Every element a role or text rung matches; a css rung is answered by `resolve_one`."""
        found = []
        for element in self.dialog_elements() + self.elements():
            if strategy["kind"] == "role" and (element.role, element.name) == (strategy["role"], strategy["name"]) \
                    and (not strategy.get("context") or element.context == strategy["context"]):
                found.append(element)
            elif strategy["kind"] == "text" and strategy["text"] in element.text:
                found.append(element)
        return found

    def resolve_one(self, strategy: dict) -> Element | None:
        if strategy["kind"] in ("role", "text"):
            found = self.matches(strategy)
            return found[0] if len(found) == 1 else None
        for element in self.dialog_elements() + self.elements():
            if strategy["kind"] == "css" and element.ref == strategy["selector"]:
                return element
        return None

    def navigate(self, url: str) -> None:
        if self.faults and self.faults[0] == "slow_entry":
            self.faults.pop(0)
            raise TransientError("navigation timed out", url=url)
        self.actions.append(("navigate", url))
        self.remember_page()
        self.url = url
        self.error = ""
        if url.endswith("inventory.html") and self.logged_in:
            self.screen = "inventory"
        elif url.endswith("cart.html") and self.logged_in:
            self.screen = "cart"
        else:
            self.screen = "login"
            if self.notice_pending:
                self.dialog = "We use cookies to improve your experience."

    def remember_page(self) -> None:
        """Push the page about to be left onto the history, as a browser does on every navigation."""
        if self.screen != "blank":
            self.history.append((self.screen, self.url))

    def back(self) -> None:
        self.actions.append(("back",))
        if self.opener is not None:               # close the adopted page and return to its opener
            (self.screen, self.url), self.opener = self.opener, None
            self.dialog, self.error = None, ""
            return
        if not self.history:
            raise ActionError("back was not performed: there is no previous page in the browser history",
                              performed="no", url=self.url)
        self.screen, self.url = self.history.pop()
        self.dialog, self.error = None, ""

    def type(self, target: Element, value: str) -> None:
        self.actions.append(("type", target.name, value))
        self.typed[target.name] = value

    def select(self, target: Element, value: str) -> None:
        """A chooser configured in `options` (name -> suggestion texts): "select_unknown:<name>" makes the choice
        end in an unknown state; a value no suggestion matches leaves the typed text and is not a selection."""
        from src.cua.surface import match_option
        options = self.options.get(target.name)
        if options is None:
            raise ActionError(f"selecting a value in {target.role} '{target.name}' was not performed: the control is "
                              f"not an enabled chooser or text field", performed="no")
        self.actions.append(("select", target.name, value))
        self.typed[target.name] = value
        if self.faults and self.faults[0] == f"select_unknown:{target.name}":
            self.faults.pop(0)
            raise ActionError(f"selecting a value in {target.role} '{target.name}' did not complete after choosing the "
                              f"suggestion", performed="unknown", url=self.url)
        popups = self.popups.get(target.name, 1)
        if popups > 1:
            raise ActionError(f"selecting a value in {target.role} '{target.name}': the value was typed but {popups} "
                              f"unrelated popups are open and the control names none of them; nothing was chosen",
                              performed="yes", cause="ambiguous_selection")
        status, match, level = match_option([{"label": text} for text in options], value)
        if status == "ambiguous":
            raise ActionError(f"selecting a value in {target.role} '{target.name}': the value was typed but more than one "
                              f"suggestion is a {level} match for it; nothing was chosen", performed="yes",
                              cause="ambiguous_selection")
        if match is None:
            raise ActionError(f"selecting a value in {target.role} '{target.name}': the value was typed but no suggestion "
                              f"matching it appeared within 4s; typed text is not a selection", performed="yes")
        self.typed[target.name] = match["label"]
        self.selected[target.name] = match["label"]

    def click(self, target: Element) -> None:
        """Returns None: the fake never activates an enclosing control in place of the target."""
        fault = self.faults[0] if self.faults and ":" in self.faults[0] else ""
        kind, _, name = fault.partition(":")
        if name == target.name:
            self.faults.pop(0)
            if kind == "blocked_click":
                raise ActionError(f"click on {target.role} '{target.name}' was not performed: a label intercepts "
                                  f"pointer events", performed="no")
            if kind == "unactionable_click":
                raise ActionError(f"click on {target.role} '{target.name}' was not performed: the element cannot take "
                                  f"a pointer action; no enclosing control is provably actionable", performed="no",
                                  cause="unactionable_target")
            if kind == "lost_click":
                # The wording deliberately quotes actionability phrases: the state, not the text, must decide.
                raise ActionError(f"click on {target.role} '{target.name}' timed out (log: waiting for element to be "
                                  f"visible; <div> intercepts pointer events; retrying click action)",
                                  performed="unknown", url=self.url)
            if kind == "uncertain_click":
                self.perform_click(target)
                raise ActionError(f"click on {target.role} '{target.name}' timed out", performed="unknown", url=self.url)
        self.perform_click(target)

    def perform_click(self, target: Element) -> None:
        self.actions.append(("click", target.name, target.context))
        self.error = ""
        page = (self.screen, self.url)
        opened = self.opens_page.get(target.name)
        if opened is not None:
            if isinstance(opened, int):            # several new pages: nothing can be chosen
                raise ActionError(f"click on {target.role} '{target.name}' opened {opened} new pages and none of "
                                  f"them can be chosen safely", performed="unknown", cause="ambiguous_selection")
            if isinstance(opened, str):            # a destination outside the allowlist: closed, nothing done there
                raise ActionError(f"click on {target.role} '{target.name}' opened a page on host '{opened}', which is "
                                  f"outside the allowlist; it was closed and nothing was done there", performed="unknown")
            self.opener = page
            self.screen, self.url = opened
            return
        self.dispatch_click(target)
        if (self.screen, self.url) != page:
            self.history.append(page)

    def dispatch_click(self, target: Element) -> None:
        if target.name in self.commits:
            self.commit(target.name)
            return
        panel = next((name for name, spec in self.panels.items()
                      if spec["control"] == target.name
                      and target.ref == f"div:nth-of-type({1 + list(self.panels).index(name)}) > button:nth-of-type(1)"),
                     None)
        if panel is not None:
            self.dismiss(panel)
            return
        if target.name in self.selectable:
            spec = self.selectable[target.name]
            if spec.get("styling_only"):
                return                       # it looks different; nothing about its state changed
            if spec.get("toggles", True):
                spec["selected"] = not spec.get("selected")
            if spec.get("reveals"):
                self.toast = spec["reveals"]     # text the click brings about, absent before it
            return
        if target.name in self.toggles:
            key, value = self.toggles[target.name]
            self.toggles[target.name] = (key, not value)
            return
        if target.name in self.inert:
            return                              # dispatched, and nothing on the page moves
        if target.name == "Accept":
            self.dialog, self.notice_pending = None, False
        elif target.name in ("Verify", "Close"):
            self.dialog = None
        elif target.name == "Details":
            self.dialog = "Preview of the item"          # a popup that leads nowhere; Close dismisses it
        elif target.name == "Login":
            self.login()
        elif target.name == "Add to cart":
            self.cart.append(target.context)
        elif target.name == "Remove":
            self.cart.remove(target.context)
        elif target.name == "cart":
            self.screen, self.url = "cart", ENTRY + "cart.html"
        elif target.name == "Checkout":
            self.screen, self.url = "info", ENTRY + "checkout-step-one.html"
        elif target.name == "Continue":
            self.continue_checkout()
        elif target.name == "Finish":
            self.screen, self.url = "complete", ENTRY + "checkout-complete.html"
        elif target.name in ("Cancel", "Continue Shopping"):
            self.screen, self.url = "inventory", ENTRY + "inventory.html"

    def selected_state(self, target: Element) -> dict | None:
        """Whether this control reads as selected, the way a real surface reports it."""
        name = target.name if target is not None else ""
        if name in self.toggles:
            # A configured toggle publishes its own state, exactly as an ARIA control does in a browser.
            key, value = self.toggles[name]
            source = {"checked": "aria-checked", "selected": "aria-selected", "pressed": "aria-pressed"}.get(key)
            if source is None:
                return {"known": False, "selected": False, "source": "", "ref": target.ref}
            return {"known": True, "selected": bool(value), "source": source, "ref": target.ref}
        spec = self.selectable.get(name)
        if spec is None:
            return None
        if spec.get("unknown"):
            return {"known": False, "selected": False, "source": "", "ref": target.ref}
        return {"known": True, "selected": bool(spec.get("selected")),
                "source": spec.get("source", "aria-pressed"), "ref": spec.get("partner") or target.ref}

    def dismiss(self, panel: str) -> None:
        """Close a panel: it and its own control leave the screen, as a real dismissal does."""
        self.dismissed.add(panel)

    def commit(self, name: str) -> None:
        """Move what was typed in the widget's field into the group as a selected token."""
        spec = self.commits[name]
        value = self.typed.get(spec["field"], "")
        self.toast = spec.get("toast", "")
        if spec.get("clears", True):
            self.typed[spec["field"]] = ""
        if spec.get("stored"):
            self.stored[f"{spec['group']}/stored"] = value
        group = spec.get("outside") or spec["group"]
        for _ in range(spec.get("extra_tokens", 0)):
            self.tokens.append({"text": f"{value} (also)", "group": group,
                                "removable": spec.get("removable", True), "selected": spec.get("selected", False)})
        if spec.get("token", True):
            self.tokens.append({"text": value, "group": group, "removable": spec.get("removable", True),
                                "selected": spec.get("selected", False)})

    def committed_state(self, target: Element) -> dict | None:
        """The field-group state around `target`: what the real surface reads from the page, offline.

        Scoped by the control's own group, so a token in another group is invisible here, exactly as the
        real container scoping makes it invisible there.
        """
        group = target.context if target is not None else ""
        if not group:
            return None
        fields = [{"ref": f"field:{name}:{group}", "value": value}
                  for name, value in sorted(self.typed.items())
                  if any(spec["group"] == group and spec["field"] == name for spec in self.commits.values())]
        tokens = [{"ref": f"token:{index}:{token['group']}", "text": token["text"], "name": "",
                   "role": "listitem", "removable": bool(token.get("removable")),
                   "remove_ref": f"token:{index}:{token['group']} > button:nth-of-type(1)"
                                 if token.get("removable") else "",
                   "selected": bool(token.get("selected")), "remove_name": ""}
                  for index, token in enumerate(self.tokens) if token["group"] == group]
        stored = [{"ref": ref, "value": value} for ref, value in sorted(self.stored.items())
                  if ref.startswith(f"{group}/")]
        return {"scoped": True, "fields": fields, "tokens": tokens, "stored": stored}

    def login(self) -> None:
        username, password = self.typed.get("Username", ""), self.typed.get("Password", "")
        if username == "locked_out_user":
            self.error = LOCKED_OUT
        elif USERS.get(username) != password:
            self.error = BAD_LOGIN
        else:
            self.logged_in = True
            fault = self.faults.pop(0) if self.faults and ":" not in self.faults[0] else None   # action faults stay queued
            if fault == "transient":
                self.screen = "error"
            elif fault == "verification":
                self.dialog = "Please verify you are human to continue."
            else:
                self.screen, self.url = "inventory", ENTRY + "inventory.html"

    def continue_checkout(self) -> None:
        for field, label in (("First Name", "First Name"), ("Last Name", "Last Name"), ("Zip/Postal Code", "Postal Code")):
            if not self.typed.get(field, "").strip():
                self.error = f"Error: {label} is required"
                return
        self.screen, self.url = "overview", ENTRY + "checkout-step-two.html"

    def screenshot(self, path: str) -> str:
        self.screenshots += 1
        return path

    def viewport_size(self) -> tuple[int, int]:
        return VIEWPORT

    def scroll_position(self) -> tuple[int, int]:
        return (0, 0)

    def text_under(self, box: tuple) -> Element | None:
        """What the fake 'draws' under a box: the element whose text starts with the configured prefix."""
        prefix = self.grounding.get(tuple(box))
        if not prefix:
            return None
        return next((e for e in self.elements() if e.text.startswith(prefix)), None)

    def viewport_screenshot(self, path: str) -> ScreenshotFrame:
        """A frame whose bytes depend on the screen, so a fingerprint changes when the screen does."""
        self.screenshots += 1
        self.viewport_shots.append(path)
        stamp = f"{self.screen}|{self.dialog}|{self.error}|{sorted(self.typed.items())}".encode()
        return ScreenshotFrame(png=TINY_PNG + stamp, width=VIEWPORT[0], height=VIEWPORT[1], path=path)
