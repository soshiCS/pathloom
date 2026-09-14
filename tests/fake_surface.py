"""An in-memory Surface for tests: a tiny state machine shaped like a demo web shop.

It implements the same Surface protocol as the Playwright adapter, so discovery, replay,
policy, and escalation can be tested without a browser, an LLM, or the network. Screens
are plain lists of Elements; actions move between screens following the shop's rules
(login, product list, cart, checkout information, checkout overview, order complete).
"""
from __future__ import annotations

from .context import Element, Locator, Observation, TransientError

ENTRY = "https://www.saucedemo.com/"
USERS = {"standard_user": "secret_sauce", "locked_out_user": "secret_sauce"}
PRODUCTS = {"Sauce Labs Backpack": 29.99, "Sauce Labs Bike Light": 9.99}
TAX_RATE = 0.08
LOCKED_OUT = "Epic sadface: Sorry, this user has been locked out."
BAD_LOGIN = "Epic sadface: Username and password do not match any user in this service"


def el(role: str, name: str, text: str = "", context: str = "") -> Element:
    return Element(role=role, name=name, text=text or name, box=(10, 10, 50, 20), context=context,
                   ref=f"{role}:{name}:{context}")


class FakeSurface:
    """Deterministic stand-in for the shop."""

    def __init__(self, show_notice: bool = False, faults: list[str] | None = None,
                 extra_elements: list[Element] | None = None):
        self.url = "about:blank"
        self.screen = "blank"
        self.dialog: str | None = None
        self.notice_pending = show_notice        # a cookie-style notice on the first page view
        self.faults = list(faults or [])         # "slow_entry" (next navigate), "transient" | "verification" (on login)
        self.extra_elements = list(extra_elements or [])
        self.typed: dict[str, str] = {}
        self.cart: list[str] = []
        self.logged_in = False
        self.error = ""
        self.actions: list[tuple] = []           # every call, for assertions
        self.screenshots = 0

    # ---------- screens ----------

    def elements(self) -> list[Element]:
        builders = {"blank": lambda: [], "login": self.login_screen, "inventory": self.inventory_screen,
                    "cart": self.cart_screen, "info": self.info_screen, "overview": self.overview_screen,
                    "complete": lambda: [el("text", "", "Thank you for your order!")]}
        return builders[self.screen]() + self.extra_elements

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
        """Controls that belong to the open dialog, listed like any other control."""
        if not self.dialog:
            return []
        return [el("button", "Accept")] if "cookies" in self.dialog else [el("button", "Verify")]

    def observe(self) -> Observation:
        if self.screen == "error":
            raise TransientError("server returned HTTP 503", url=self.url)
        return Observation(url=self.url, elements=self.dialog_elements() + self.elements(), dialog=self.dialog)

    # ---------- Surface protocol ----------

    def resolve(self, locator: Locator) -> Element | None:
        for strategy in locator.strategies:
            for element in self.dialog_elements() + self.elements():
                if strategy["kind"] == "role" and (element.role, element.name) == (strategy["role"], strategy["name"]) \
                        and (not strategy.get("context") or element.context == strategy["context"]):
                    return element
                if strategy["kind"] == "text" and strategy["text"] in element.text:
                    return element
                if strategy["kind"] == "css" and element.ref == strategy["selector"]:
                    return element
        return None

    def navigate(self, url: str) -> None:
        if self.faults and self.faults[0] == "slow_entry":
            self.faults.pop(0)
            raise TransientError("navigation timed out", url=url)
        self.actions.append(("navigate", url))
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

    def type(self, target: Element, value: str) -> None:
        self.actions.append(("type", target.name, value))
        self.typed[target.name] = value

    def click(self, target: Element) -> None:
        self.actions.append(("click", target.name, target.context))
        self.error = ""
        if target.name == "Accept":
            self.dialog, self.notice_pending = None, False
        elif target.name == "Verify":
            self.dialog = None
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

    def login(self) -> None:
        username, password = self.typed.get("Username", ""), self.typed.get("Password", "")
        if username == "locked_out_user":
            self.error = LOCKED_OUT
        elif USERS.get(username) != password:
            self.error = BAD_LOGIN
        else:
            self.logged_in = True
            fault = self.faults.pop(0) if self.faults else None
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
