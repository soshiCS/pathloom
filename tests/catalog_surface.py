"""A deterministic multi-page catalog: a results page whose links open detail pages, and browser history.

It exists to exercise output accumulation across pages (`append`) and the `back` action; nothing in
it resembles any real site.
"""
from __future__ import annotations

from src.cua.models import ActionError, Element, Locator, Observation

CATALOG = "https://catalog.test/"
ITEMS = {"Item A": {"score": "71", "tag": "alpha"}, "Item B": {"score": "58", "tag": "beta"},
         "Item C": {"score": "90", "tag": "gamma"}}


def el(role: str, name: str = "", text: str = "") -> Element:
    return Element(role=role, name=name, text=text or name, ref=f"{role}:{name or text}")


class CatalogSurface:
    def __init__(self, items: dict | None = None):
        self.items = items or {name: dict(facts) for name, facts in ITEMS.items()}
        self.url, self.screen = "about:blank", "blank"
        self.history: list[tuple[str, str]] = []
        self.actions: list[tuple] = []

    # ---------- screens ----------

    def elements(self) -> list[Element]:
        if self.screen == "results":
            return [el("heading", text="Results"), el("text", text=f"{len(self.items)} items")] + [
                el("link", name) for name in self.items]
        if self.screen.startswith("detail:"):
            facts = self.items[self.screen.split(":", 1)[1]]
            shown = [el("heading", text="Details"), el("text", text=f"Score: {facts['score']}")]
            if facts.get("tag"):
                shown.append(el("text", text=f"Tag: {facts['tag']}"))
            if facts.get("code"):
                shown.append(el("text", text=f"Code: {facts['code']}"))
            return shown
        return []

    # ---------- Surface protocol ----------

    def observe(self) -> Observation:
        return Observation(url=self.url, elements=self.elements())

    def resolve(self, locator: Locator, exclude: frozenset = frozenset()) -> Element | None:
        for strategy in locator.strategies:
            if strategy["kind"] in ("role", "text"):
                found_all = self.matches(strategy)
                found = found_all[0] if len(found_all) == 1 else None
            else:
                found = next((e for e in self.elements() if strategy["kind"] == "css" and e.ref == strategy["selector"]), None)
            if found is None or found.ref in exclude:
                continue
            return found
        return None

    def matches(self, strategy: dict) -> list[Element]:
        found = []
        for element in self.elements():
            if strategy["kind"] == "role" and (element.role, element.name) == (strategy["role"], strategy["name"]):
                found.append(element)
            elif strategy["kind"] == "text" and element.text.startswith(strategy["text"]):
                found.append(element)
        return found

    def leave(self) -> None:
        if self.screen != "blank":
            self.history.append((self.screen, self.url))

    def navigate(self, url: str) -> None:
        self.actions.append(("navigate", url))
        self.leave()
        self.url, self.screen = url, "results"

    def back(self) -> None:
        self.actions.append(("back",))
        if not self.history:
            raise ActionError("back was not performed: there is no previous page in the browser history",
                              performed="no", url=self.url)
        self.screen, self.url = self.history.pop()

    def click(self, target: Element) -> None:
        self.actions.append(("click", target.name))
        if target.role == "link" and target.name in self.items:
            self.leave()
            self.screen = f"detail:{target.name}"
            self.url = CATALOG + "items/" + target.name.split()[-1].lower()

    def type(self, target: Element, value: str) -> None:
        raise ActionError("nothing to type into", performed="no", url=self.url)

    def screenshot(self, path: str) -> str:
        with open(path, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n")
        return path
