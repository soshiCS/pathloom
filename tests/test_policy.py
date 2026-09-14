"""Allowlists, blocked controls, risk classification, and redaction."""
from src.cua.policy import Policy, redact
from tests.context import Action, Element

HOME = "https://www.saucedemo.com/"


def policy() -> Policy:
    return Policy(allowed_hosts=["www.saucedemo.com"])


def click(name: str) -> Action:
    return Action(kind="click", target=Element(role="button", name=name))


def test_navigation_outside_allowlist_is_denied():
    verdict = policy().check(Action(kind="navigate", value="https://evil.example.com/"), HOME)
    assert verdict.decision == "deny"
    assert "evil.example.com" in verdict.reason


def test_navigation_from_blank_page_to_allowed_host_is_allowed():
    # The browser starts on about:blank; the first navigate is judged by its destination.
    assert policy().check(Action(kind="navigate", value=HOME), "about:blank").decision == "allow"


def test_acting_on_a_page_outside_the_allowlist_is_denied():
    assert policy().check(click("Checkout"), "https://other.example.com/page").decision == "deny"


def test_action_kind_outside_allowlist_is_denied():
    read_only = Policy(allowed_hosts=["www.saucedemo.com"], allowed_actions={"navigate", "click", "extract"})
    typing = Action(kind="type", target=Element(role="textbox", name="Username"), value="x")
    assert read_only.check(typing, HOME).decision == "deny"


def test_finish_is_risky_and_requires_confirmation():
    for label in ("Finish", "Place order", "Pay now", "Buy", "Complete purchase"):
        assert policy().risk_of(click(label)) == "risky", label
        verdict = policy().check(click(label), HOME)
        assert verdict.decision == "confirm" and "human must confirm" in verdict.reason, label


def test_everyday_flow_controls_are_safe():
    for label in ("Login", "Add to cart", "Checkout", "Continue", "Open Menu"):
        assert policy().check(click(label), HOME).decision == "allow", label


def test_account_creation_and_deletion_are_blocked_outright():
    for label in ("Sign Up", "Register", "Create an account", "Delete my account"):
        verdict = policy().check(click(label), HOME)
        assert verdict.decision == "deny" and "blocked control" in verdict.reason, label


def test_typing_is_never_classified_risky_by_itself():
    typing = Action(kind="type", target=Element(role="textbox", name="Payment amount"), value="100")
    assert policy().risk_of(typing) == "safe"


def test_redact_masks_sensitive_keys_and_value_shapes():
    data = {"password": "secret_sauce", "note": "ssn 123-45-6789 card 4111 1111 1111 1111", "username": "standard_user",
            "nested": [{"api_key": "abc"}, "sk-abcdefghijklmnop"]}
    out = redact(data)
    assert out["password"] == "[REDACTED]"
    assert out["nested"][0]["api_key"] == "[REDACTED]"
    assert "123-45-6789" not in out["note"] and "[REDACTED-SSN]" in out["note"]
    assert "4111" not in out["note"] and "[REDACTED-CARD]" in out["note"]
    assert out["nested"][1] == "[REDACTED-KEY]"
    assert out["username"] == "standard_user"  # ordinary identifiers survive


def test_redact_masks_caller_supplied_secret_values_anywhere():
    out = redact({"observed": "typed secret_sauce into the box"}, secrets=("secret_sauce",))
    assert "secret_sauce" not in out["observed"]
