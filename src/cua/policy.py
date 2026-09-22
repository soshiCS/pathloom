"""Safety policy: allowlists, risk classification, and redaction.

The same Policy object guards both discovery (before the planner's action runs) and
replay (before each recorded step runs), so a tampered artifact cannot escape the
allowlist either.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .models import Action

# Action kinds that touch the surface. "done", "stuck" and "reuse_candidate" are planner decisions the agent
# handles itself; a chosen segment's actions are checked one by one when the segment replays.
SURFACE_ACTIONS = {"navigate", "back", "click", "type", "select", "extract", "extract_many"}

# Control names that suggest an irreversible or money-moving operation. Matching is
# deliberately broad: a false "risky" only costs a confirmation, a false "safe" could
# cost real money.
DEFAULT_RISKY_PATTERNS = [
    r"\bfinish\b", r"\bplace order\b", r"\bcomplete (?:order|purchase)\b", r"\bpay(?: now)?\b", r"\bpayment\b",
    r"\bbuy\b", r"\bpurchase\b", r"\bsubmit\b", r"\bapprove\b", r"\bconfirm\b", r"\btransfer\b", r"\bwire\b",
    r"\bwithdraw\b", r"\bdelete\b", r"\bremove\b", r"\bpost\b", r"\bsend\b", r"\bupgrade\b",
]

# "Close" is contextual: dismissing a piece of the interface is harmless, closing a persistent
# resource is not. The wording is judged first (what follows the word); a bare dismissal name is
# then judged by browser-derived evidence about the container the control sits in.
CLOSE_WORD = re.compile(r"\bcloses?\b", re.IGNORECASE)   # the verb "close"/"closes"; "Closed Loop" is not an action
DISMISSAL_NAMES = {"close", "dismiss", "x", "×", "✕", "✖", "⨯", "close button", "dismiss button"}
# Container roles the platform itself marks as dismissible chrome.
DISMISSIBLE_LANDMARKS = {"dialog", "alertdialog", "menu", "listbox", "tooltip", "popover", "details", "complementary",
                         "navigation", "region"}
# Risky patterns that stay risky wherever the control sits: moving money or destroying data is never a query.
# Only a submission verb ("submit", "apply", "go", "post", "send") can be excused by query evidence.
COMMITTING_PATTERNS = [r"\bfinish\b", r"\bplace order\b", r"\bcomplete (?:order|purchase)\b", r"\bpay(?: now)?\b",
                       r"\bpayment\b", r"\bbuy\b", r"\bpurchase\b", r"\bapprove\b", r"\bconfirm\b",
                       r"\btransfer\b", r"\bwire\b", r"\bwithdraw\b", r"\bdelete\b", r"\bremove\b",
                       r"\bupgrade\b"]
# Accessible purposes that prove a control only runs a query: it re-reads data, it commits nothing.
QUERY_PURPOSES = [r"\bsearch\b", r"\bfind\b", r"\blook ?up\b", r"\bquery\b", r"\bfilter\b", r"\bapply filters?\b",
                  r"\bshow results?\b", r"\bview results?\b", r"\bgo\b"]
# Container roles whose own semantics say the form around a control is a query, not a transaction.
QUERY_LANDMARKS = {"search"}
# Container names that describe a piece of the interface rather than a resource.
PANEL_WORDS = {"settings", "options", "preferences", "configuration", "legend", "layers", "info", "information",
               "about", "controls", "toolbar", "properties", "inspector", "console", "log"}
CLOSE_ARTICLES = {"the", "this", "that", "these", "those", "a", "an", "my", "your", "all", "current"}
# Things a "close" verb may name that are persistent resources: closing them is a business operation.
RESOURCE_WORDS = {
    "account", "accounts", "membership", "subscription", "subscriptions", "position", "positions", "order", "orders",
    "case", "cases", "ticket", "tickets", "contract", "contracts", "policy", "policies", "loan", "loans", "claim",
    "claims", "deal", "deals", "trade", "trades", "invoice", "invoices", "session", "sessions", "incident",
    "incidents", "issue", "issues", "request", "requests", "application", "applications", "job", "jobs", "listing",
    "listings", "auction", "auctions", "shift", "shifts", "batch", "batches", "period", "periods", "books", "register",
    "till", "cart", "checkout", "sale", "sales", "escrow", "thread", "threads", "conversation"
}
INTERFACE_WORDS = {
    "dialog", "dialogue", "window", "popup", "pop-up", "modal", "notice", "notification", "notifications",
    "message", "messages", "alert", "alerts", "toast", "menu", "panel", "overlay", "banner", "sidebar", "drawer",
    "tab", "tabs", "preview", "help", "tooltip", "tip", "tips", "hint", "welcome", "wizard", "tour", "guide", "chat",
    "search", "filter", "filters", "details", "editor", "viewer", "lightbox", "image", "gallery", "video", "player",
    "prompt", "it", "x", "glossary", "reference", "definitions", "faq", "instructions", "documentation", "docs",
    "toggle", "toggles", "accordion", "disclosure", "expander", "collapse", "dropdown", "flyout", "popover",
    "pane", "section", "summary", "description", "descriptions", "list", "map", "legend", "key",
}

# Controls the agent must never operate, regardless of goal. Denied outright, no confirmation
# offered. Kept short on purpose: logging in with supplied credentials is a legitimate goal.
DEFAULT_BLOCKED_PATTERNS = [
    r"\bsign ?up\b", r"\bregister\b", r"\bcreate (?:an )?account\b", r"\bdelete (?:my )?account\b",
]

# Keys whose values must never be written to disk, matched case-insensitively.
SENSITIVE_KEY_PATTERN = re.compile(
    r"password|passwd|secret|token|api[_-]?key|authorization|credential|ssn|social|"
    r"card[_-]?number|pin\b", re.IGNORECASE)

# Value shapes that are sensitive regardless of key name. A run of 13 to 19 digits (spaces or hyphens
# between them) is a card number only when it passes the Luhn check; ordinary long identifiers stay.
CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
SENSITIVE_VALUE_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED-SSN]"),
    (CARD_CANDIDATE, lambda match: "[REDACTED-CARD]" if luhn_valid(match.group(0)) else match.group(0)),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "[REDACTED-KEY]"),
]


def luhn_valid(candidate: str) -> bool:
    """The Luhn checksum over the digits of `candidate` (spaces and hyphens removed), as card numbers carry."""
    digits = [int(ch) for ch in candidate if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0

REDACTED = "[REDACTED]"


@dataclass
class Verdict:
    """Result of checking whether an action may run."""

    decision: str  # allow | deny | confirm
    reason: str = ""


@dataclass
class Policy:
    """Explicit, configurable allowlist plus risk classification."""

    allowed_hosts: list[str] = field(default_factory=list)
    allowed_actions: set[str] = field(default_factory=lambda: set(SURFACE_ACTIONS))
    risky_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_RISKY_PATTERNS))
    blocked_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_BLOCKED_PATTERNS))

    def check(self, action: Action, url: str) -> Verdict:
        """Decide allow / deny / confirm for one concrete action on the current page."""
        reach = self.check_reach(action, url)
        if reach.decision != "allow":
            return reach
        return self.check_control(action)

    def check_reach(self, action: Action, url: str) -> Verdict:
        """The part of the decision that needs no control: the action kind and where it happens.

        Always checked before anything touches the page, including before a locator is resolved.
        """
        if action.kind in ("done", "stuck", "reuse_candidate"):
            return Verdict("allow", "planner signal, no surface interaction")
        if action.kind not in self.allowed_actions:
            return Verdict("deny", f"action '{action.kind}' is not in the allowed action list")
        if action.kind == "navigate":
            # A navigation is judged by where it goes (the browser may still be on a blank page).
            target_host = urlparse(action.value or "").hostname or ""
            if not self.host_allowed(target_host):
                return Verdict("deny", f"navigation to host '{target_host}' is outside the allowlist")
        else:
            # Every other action operates on the page we are on, so that page must be allowlisted.
            current_host = urlparse(url).hostname or ""
            if not self.host_allowed(current_host):
                return Verdict("deny", f"current host '{current_host}' is outside the allowlist")
        return Verdict("allow")

    def check_control(self, action: Action) -> Verdict:
        """The part that judges the control itself, from whatever the caller knows about it.

        Replay calls this with the element it just resolved on the live page, so the runtime-only
        evidence a control carries (its native kind, enclosing landmark and dismissal metadata) is
        available; a serialized locator hint alone would hide it and force a needless confirmation.
        """
        label = control_label(action)
        if action.kind == "click" and matches_any(self.blocked_patterns, label):
            return Verdict("deny", f"'{label}' is a blocked control")
        risk = self.risk_of(action)
        if risk == "risky":
            return Verdict("confirm", f"'{label}' looks irreversible; a human must confirm")
        if risk == "unknown":
            return Verdict("confirm", f"'{label}' may dismiss the interface or close a resource and nothing on the "
                                      f"screen proves which; a human must confirm")
        return Verdict("allow")

    def host_allowed(self, host: str) -> bool:
        return host in self.allowed_hosts

    def risk_of(self, action: Action) -> str:
        """Classify a click as safe, risky (irreversible) or unknown (ambiguous: a person decides).

        Deterministic order, from the normalized control label and the browser-derived evidence on the
        target only (never the planner's reason or the goal): a control the platform proves is a
        disclosure toggle is safe; a label any risky pattern matches is risky unless the platform proves
        the control only runs a query;
        wording that closes a named resource is risky; wording that names the piece of interface it closes
        is safe; a bare "Close" (with or without an icon glyph) is judged by its container: a container
        that names a resource closure is risky, one that names a risky operation is unknown, a proven
        dismissal context (dialog, menu, panel, an explicit dismissal relationship, a panel-like heading;
        never a form's submit control) is safe, and anything else is unknown.
        """
        if action.kind != "click" or action.target is None:
            return "safe"  # typing and reading never commit anything by themselves
        target = action.target
        label = control_label(action)
        if proven_disclosure(target):
            return "safe"          # the platform says this control only expands or collapses content
        if matches_any(self.risky_patterns, label) and not proven_query(target, label):
            return "risky"
        if not CLOSE_WORD.search(label):
            return "safe"          # "Dismiss", "×", "Cancel": nothing in the wording closes anything
        if closes_a_resource(label):
            return "risky"
        if names_interface_piece(label):
            return "safe"
        context = nearby_context(target)
        if closes_a_resource(context):
            return "risky"
        if matches_any(self.risky_patterns, context):
            return "unknown"
        named = closure_object(label)
        if named and named in context_words(context):
            return "safe"          # "Close glossary" inside the panel named Glossary: the container itself
        return "safe" if proven_dismissal(target, context) else "unknown"


    def dismisses_interface(self, action: Action) -> bool:
        """Whether browser evidence proves this click only dismisses a piece of interface chrome.

        Narrower than a `safe` verdict on purpose. `safe` also covers controls that commit nothing for
        quite different reasons (a disclosure toggle, a proven query, a control whose wording closes
        nothing at all), and for those the disappearance of the control says nothing useful. This answers
        only the one question the disappearance rule may rely on: does the platform itself say this
        control dismisses the interface around it?

        A control whose wording closes a named resource, whose container names a risky operation, or that
        is a form's submit control is never a dismissal, whatever else is true, so a destructive or
        ambiguous close can never be proven by vanishing.
        """
        if action.kind != "click" or action.target is None:
            return False
        target = action.target
        label = control_label(action)
        if matches_any(self.risky_patterns, label) and not proven_query(target, label):
            return False
        if not CLOSE_WORD.search(label) and not target.dismisses:
            return False               # nothing in the wording or the platform says it dismisses anything
        if closes_a_resource(label):
            return False
        context = nearby_context(target)
        if closes_a_resource(context) or matches_any(self.risky_patterns, context):
            return False
        if target.dismisses or names_interface_piece(label):
            return True
        named = closure_object(label)
        if named and named in context_words(context):
            return True                # "Close glossary" inside the panel named Glossary
        return proven_dismissal(target, context)


def control_label(action: Action) -> str:
    """The control's wording once: its accessible name, plus its visible text only when that adds words."""
    if action.target is None:
        return action.value or ""
    name, text = " ".join(action.target.name.split()), " ".join(action.target.text.split())
    if not text or not re.search(r"\w", text) or text.lower() in name.lower():
        return name or text
    if name.lower() in text.lower():
        return text
    return f"{name} {text}".strip()


def is_dismissal_name(label: str) -> bool:
    """An exact dismissal wording, with decorations (an icon glyph) stripped: "Close", "Dismiss", "×"."""
    words = [w for w in re.findall(r"[\w×✕✖⨯-]+", label.lower())]
    return bool(words) and " ".join(words) in DISMISSAL_NAMES


def names_interface_piece(label: str) -> bool:
    """The wording itself says what is closed, and it is a piece of the interface ("Close menu", "Close it")."""
    for match in CLOSE_WORD.finditer(label):
        words = re.findall(r"[\w-]+", label[match.end():].lower())
        while words and words[0] in CLOSE_ARTICLES:
            words = words[1:]
        if words and words[0] in INTERFACE_WORDS | PANEL_WORDS:
            return True
    return False


def nearby_context(target) -> str:
    """The container evidence around a control: the enclosing landmark's name and the item it belongs to."""
    return " ".join(part for part in (target.landmark_name, target.context) if part).strip()


def proven_disclosure(target) -> bool:
    """Browser-derived proof that a control only expands or collapses interface content: an explicit
    `aria-expanded` state, or a native `<summary>` disclosure. Such a control commits nothing, whatever
    its wording says; a control without that evidence is judged by every other rule as before."""
    if "expanded" in (target.states or {}):
        return True
    return (target.native or "").lower() == "summary"


def proven_query(target, label: str) -> bool:
    """Browser-derived proof that a control runs a query rather than a transaction: it sits in a search
    landmark or a form whose name says it searches, or its own accessible purpose is to search, find or
    apply filters. A label that moves money or destroys data is never a query, wherever it sits, and the
    planner's reason and the goal are never consulted; a bare "Submit" proves nothing by itself."""
    if matches_any(COMMITTING_PATTERNS, label):
        return False
    if target.landmark in QUERY_LANDMARKS:
        return True
    if matches_any(QUERY_PURPOSES, label):
        return True
    return matches_any(QUERY_PURPOSES, target.landmark_name or "")


def proven_dismissal(target, context: str) -> bool:
    """Browser-derived proof that a bare dismissal name dismisses interface chrome: the platform's own
    dismissal metadata, an enclosing dialog, menu, panel or region, or a panel-like container name; a
    form's submit control is never taken for a dismissal."""
    native = (target.native or "").lower()
    if native.endswith(":submit") and not target.dismisses:
        return False
    if target.dismisses or target.landmark in DISMISSIBLE_LANDMARKS:
        return True
    words = set(re.findall(r"[\w-]+", context.lower()))
    return bool(words & (INTERFACE_WORDS | PANEL_WORDS))


def closure_object(label: str) -> str | None:
    """The first thing the wording says is closed ("account" in "Close my account"), or None for a bare close."""
    for match in CLOSE_WORD.finditer(label):
        words = re.findall(r"[\w-]+", label[match.end():].lower())
        while words and words[0] in CLOSE_ARTICLES:
            words = words[1:]
        if words:
            return words[0]
    return None


def closes_a_resource(label: str) -> bool:
    """Whether "close" in this wording closes something persistent: the named thing is a known resource
    ("Close account", "Close membership", "Close case permanently", "Close the ticket"). Any other noun is
    judged by its container (`Policy.risk_of`), never by wording alone."""
    return any(closure_object(label[match.start():]) in RESOURCE_WORDS for match in CLOSE_WORD.finditer(label))


def context_words(context: str) -> set[str]:
    return set(re.findall(r"[\w-]+", context.lower()))


def matches_any(patterns: list[str], label: str) -> bool:
    return any(re.search(pattern, label, re.IGNORECASE) for pattern in patterns)


def redact(value, secrets: tuple[str, ...] = ()):
    """Recursively scrub secrets and sensitive data from anything about to be persisted.

    - dict keys that look sensitive have their values replaced wholesale,
    - strings matching SSN / card / API-key shapes are masked,
    - any literal secret value passed by the caller (e.g. a sensitive input) is masked.
    """
    if isinstance(value, dict):
        # A sensitive key hides its scalar value; a nested structure under it (such as the
        # artifact's declared "password" input spec) is walked instead of being erased.
        return {key: (REDACTED if SENSITIVE_KEY_PATTERN.search(str(key)) and not isinstance(item, (dict, list, tuple))
                      else redact(item, secrets))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret and secret in value:
                value = value.replace(secret, REDACTED)
        for pattern, replacement in SENSITIVE_VALUE_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    return value
