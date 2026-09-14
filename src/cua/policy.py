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

# Action kinds that touch the surface. "done" and "stuck" are planner signals, not actions.
SURFACE_ACTIONS = {"navigate", "click", "type", "extract"}

# Control names that suggest an irreversible or money-moving operation. Matching is
# deliberately broad: a false "risky" only costs a confirmation, a false "safe" could
# cost real money.
DEFAULT_RISKY_PATTERNS = [
    r"\bfinish\b", r"\bplace order\b", r"\bcomplete (?:order|purchase)\b", r"\bpay(?: now)?\b", r"\bpayment\b",
    r"\bbuy\b", r"\bpurchase\b", r"\bsubmit\b", r"\bapprove\b", r"\bconfirm\b", r"\btransfer\b", r"\bwire\b",
    r"\bwithdraw\b", r"\bdelete\b", r"\bremove\b", r"\bclose\b", r"\bpost\b", r"\bsend\b", r"\bupgrade\b",
]

# Controls the agent must never operate, regardless of goal. Denied outright, no confirmation
# offered. Kept short on purpose: logging in with supplied credentials is a legitimate goal.
DEFAULT_BLOCKED_PATTERNS = [
    r"\bsign ?up\b", r"\bregister\b", r"\bcreate (?:an )?account\b", r"\bdelete (?:my )?account\b",
]

# Keys whose values must never be written to disk, matched case-insensitively.
SENSITIVE_KEY_PATTERN = re.compile(
    r"password|passwd|secret|token|api[_-]?key|authorization|credential|ssn|social|"
    r"card[_-]?number|pin\b", re.IGNORECASE)

# Value shapes that are sensitive regardless of key name.
SENSITIVE_VALUE_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED-SSN]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[REDACTED-CARD]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "[REDACTED-KEY]"),
]

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
        if action.kind in ("done", "stuck"):
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
        label = control_label(action)
        if action.kind == "click" and matches_any(self.blocked_patterns, label):
            return Verdict("deny", f"'{label}' is a blocked control")
        if self.risk_of(action) == "risky":
            return Verdict("confirm", f"'{label}' looks irreversible; a human must confirm")
        return Verdict("allow")

    def host_allowed(self, host: str) -> bool:
        return host in self.allowed_hosts

    def risk_of(self, action: Action) -> str:
        """Classify an action as safe (reversible read/navigation) or risky (irreversible)."""
        if action.kind != "click" or action.target is None:
            return "safe"  # typing and reading never commit anything by themselves
        return "risky" if matches_any(self.risky_patterns, control_label(action)) else "safe"


def control_label(action: Action) -> str:
    if action.target is None:
        return action.value or ""
    return f"{action.target.name} {action.target.text}".strip()


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
