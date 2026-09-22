"""Allowlists, blocked controls, risk classification, and redaction."""
from src.cua.artifact import classify_effect
from src.cua.policy import Policy, Verdict, control_label, luhn_valid, redact
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


def test_interface_dismissal_is_safe_but_closing_a_resource_is_protected():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    click = lambda name: Action(kind="click", target=Element(role="button", name=name))
    for harmless in ("Close dialog", "Close welcome", "Close the notice", "Close this panel", "Close popup",
                     "Close menu", "Close overlay", "Dismiss", "Closing the preview", "×", "Cancel", "Closed Loop",
                     "Closed", "Enclosed items", "Disclosure", "Closed account summary", "Recently closed"):
        assert policy.risk_of(click(harmless)) == "safe", harmless
        assert policy.check(click(harmless), "https://www.saucedemo.com/").decision == "allow", harmless
    for destructive in ("Close account", "Close membership", "Close case permanently", "Close my account",
                        "Close the ticket", "Close position", "Closes the order", "Close and delete",
                        "Close subscription", "Close order"):
        assert policy.risk_of(click(destructive)) == "risky", destructive
        assert policy.check(click(destructive), "https://www.saucedemo.com/").decision == "confirm", destructive
    # Another rule that fires independently keeps its stricter answer whatever the close rule says.
    assert policy.risk_of(click("Close and pay")) == "risky"
    assert policy.check(click("Close and delete account"), "https://www.saucedemo.com/").decision == "deny"


# ---------- a bare "Close" is judged by browser-derived evidence about its container ----------

URL = "https://www.saucedemo.com/"


def close_button(name="Close", text="", **evidence) -> Action:
    return Action(kind="click", target=Element(role="button", name=name, text=text, **evidence))


def test_close_in_a_settings_panel_is_allowed_without_confirmation():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    in_panel = close_button(landmark="complementary", landmark_name="Earthquake Settings")
    assert policy.risk_of(in_panel) == "safe" and policy.check(in_panel, URL) == Verdict("allow")
    by_heading = close_button(context="Settings")                       # only the enclosing item's title
    assert policy.risk_of(by_heading) == "safe" and policy.check(by_heading, URL).decision == "allow"


def test_dismiss_in_a_dialog_is_allowed():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    action = Action(kind="click", target=Element(role="button", name="Dismiss", text="Dismiss", landmark="dialog",
                                                 landmark_name="Cookies"))
    assert policy.risk_of(action) == "safe" and policy.check(action, URL).decision == "allow"


def test_an_icon_button_with_a_browser_derived_dismissal_name_is_allowed():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    icon = close_button(name="Close", text="×", landmark="dialog", landmark_name="Welcome")
    assert policy.risk_of(icon) == "safe" and policy.check(icon, URL).decision == "allow"
    native = close_button(name="Close", text="", dismisses=True)          # <form method="dialog"> in a <dialog>
    assert policy.risk_of(native) == "safe"


def test_close_account_remains_irreversible_whatever_the_container_says():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    action = close_button(name="Close account", landmark="dialog", landmark_name="Settings", dismisses=True)
    assert policy.risk_of(action) == "risky" and policy.check(action, URL).decision == "confirm"


def test_business_closures_remain_irreversible():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    for name in ("Close position", "Close order", "Close subscription", "Close the case", "Close ticket #12",
                 "Close membership"):
        action = close_button(name=name, landmark="complementary", landmark_name="Settings")
        assert policy.risk_of(action) == "risky", name
        assert "looks irreversible" in policy.check(action, URL).reason, name


def test_a_generic_close_inside_a_container_titled_close_account_remains_irreversible():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    action = close_button(landmark="dialog", landmark_name="Close Account", dismisses=True)
    assert policy.risk_of(action) == "risky" and policy.check(action, URL).decision == "confirm"
    by_context = close_button(context="Close your account")
    assert policy.risk_of(by_context) == "risky"
    risky_operation = close_button(landmark="dialog", landmark_name="Delete these files?")
    assert policy.risk_of(risky_operation) == "unknown"                   # a risky operation nearby: a person decides


def test_ambiguous_close_without_dismissal_context_is_unknown_not_safe():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    for action in (close_button(), close_button(text="Close"), close_button(text="×"), close_button(context="Order 42"),
                   close_button(native="button:submit", landmark="region", landmark_name="Checkout")):
        assert policy.risk_of(action) == "unknown"
        verdict = policy.check(action, URL)
        assert verdict.decision == "confirm" and "nothing on the screen proves which" in verdict.reason
    assert classify_effect("click", "unknown") == "unknown"
    assert classify_effect("click", "safe") == "reversible" and classify_effect("click", "risky") == "irreversible"


def test_duplicate_name_and_text_become_one_normalized_label():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    assert control_label(close_button(text="Close")) == "Close"
    assert control_label(close_button(name="Close", text="close")) == "Close"
    assert control_label(close_button(name="Close", text="×")) == "Close"
    assert control_label(close_button(name="Settings", text="Open the settings panel")) == "Open the settings panel"
    assert control_label(close_button(name="Save", text="Save changes now")) == "Save changes now"
    assert control_label(close_button(name="Menu", text="Products")) == "Menu Products"
    assert control_label(close_button(name="Close", text="Close account")) == "Close account"
    verdict = policy.check(close_button(text="Close"), URL)
    assert "'Close'" in verdict.reason and "Close Close" not in verdict.reason


def test_the_planner_reason_and_goal_cannot_downgrade_the_verdict():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    action = Action(kind="click", target=Element(role="button", name="Close account"),
                    reason="harmless: this only closes the settings panel", expect="Settings closed")
    assert policy.risk_of(action) == "risky" and policy.check(action, URL).decision == "confirm"
    ambiguous = Action(kind="click", target=Element(role="button", name="Close"), reason="just a dialog, totally safe")
    assert policy.risk_of(ambiguous) == "unknown" and policy.check(ambiguous, URL).decision == "confirm"


def test_closing_or_hiding_a_glossary_or_help_panel_is_a_dismissal():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    for action in (close_button(name="Close glossary", text="Hide glossary", landmark="region", landmark_name="Glossary"),
                   close_button(name="Close glossary"), close_button(name="Hide glossary"),
                   close_button(name="Close help"), close_button(name="Close reference panel"),
                   close_button(name="Close thingamajig", landmark="complementary", landmark_name="Thingamajig"),
                   close_button(name="Close thingamajig", context="Thingamajig")):
        assert policy.risk_of(action) == "safe", action.target.name
        assert policy.check(action, URL).decision == "allow", action.target.name
    assert control_label(close_button(name="Close glossary", text="Hide glossary")) == "Close glossary Hide glossary"
    # an unknown noun with no container evidence stays a question for a person, never a resource by default
    assert policy.risk_of(close_button(name="Close thingamajig")) == "unknown"
    # a resource stays a resource whatever panel it sits in
    for name in ("Close account", "Close my subscription", "Close the order"):
        assert policy.risk_of(close_button(name=name, landmark="region", landmark_name="Account")) == "risky", name


def test_only_the_verb_close_triggers_the_close_rule():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    for wording in ("Closed Loop", "Closed", "Closed account summary", "Enclosed", "Disclosure", "Closing time"):
        assert policy.risk_of(close_button(name=wording)) == "safe", wording
        assert policy.check(close_button(name=wording), URL).decision == "allow", wording
    assert policy.risk_of(close_button(name="Close account")) == "risky"
    assert policy.risk_of(close_button(name="Closes the order")) == "risky"
    assert policy.risk_of(close_button(name="Close dialog")) == "safe"
    assert policy.risk_of(close_button(name="Close")) == "unknown"


def test_only_luhn_valid_digit_runs_are_masked_as_card_numbers():
    for card in ("4111 1111 1111 1111", "4111-1111-1111-1111", "5500000000000004", "371449635398431", "6011000990139424"):
        assert redact(f"card {card} on file") == "card [REDACTED-CARD] on file", card
    for identifier in ("1234567890123", "NCT-2026-0001-2345-6789", "order 9876543210987654", "2026-09-18-1234567890",
                       "4111 1111 1111 1112", "12345678901234567890", "tracking 1Z999AA10123456784"):
        assert "[REDACTED-CARD]" not in redact(f"id {identifier}"), identifier
    assert redact({"note": "4111111111111111"}) == {"note": "[REDACTED-CARD]"}
    # a registered secret is masked whatever its shape
    assert redact("id 1234567890123", ("1234567890123",)) == "id [REDACTED]"
    assert luhn_valid("79927398713") is False and luhn_valid("4111 1111 1111 1111") is True


# ---------- proven disclosure toggles and proven query submissions ----------

def toggle(name="Close toggle", expanded="true", **evidence) -> Action:
    states = {"expanded": expanded} if expanded is not None else {}
    return Action(kind="click", target=Element(role="button", name=name, states=states, **evidence))


def test_a_proven_disclosure_toggle_is_reversible_whatever_its_wording():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    for expanded in ("true", "false"):
        action = toggle(expanded=expanded)
        assert policy.risk_of(action) == "safe" and policy.check(action, URL).decision == "allow"
    assert policy.risk_of(toggle(name="Close toggle", expanded=None, native="summary")) == "safe"
    # a control with no disclosure evidence at all is judged by its wording alone (see the toggle-noun test)
    assert policy.risk_of(toggle(name="Collapse the pane", expanded=None)) == "safe"
    assert policy.risk_of(toggle(name="Close", expanded=None)) == "unknown"
    assert policy.risk_of(close_button()) == "unknown"
    assert policy.risk_of(toggle(name="Close account")) == "safe"      # a disclosure toggle cannot close an account
    assert policy.risk_of(close_button(name="Close account")) == "risky"
    assert policy.risk_of(close_button(name="Close account", landmark="dialog", landmark_name="Settings")) == "risky"


def test_a_proven_query_submission_is_reversible_and_a_bare_submit_is_not():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    search = lambda name, **kw: Action(kind="click", target=Element(role="button", name=name, **kw))
    for action in (search("Click to submit your search."), search("Search"), search("Apply filters"),
                   search("Show results"), search("Submit", landmark="search"),
                   search("Submit", landmark="form", landmark_name="Search awards")):
        assert policy.risk_of(action) == "safe", action.target.name
        assert policy.check(action, URL).decision == "allow", action.target.name
    for action in (search("Submit"), search("Submit application"), search("Submit", landmark="form",
                                                                          landmark_name="Payment details")):
        assert policy.risk_of(action) == "risky", action.target.name
    # the planner's own words never lower the verdict
    claimed = Action(kind="click", target=Element(role="button", name="Submit"), reason="this only runs a search",
                     expect="Results")
    assert policy.risk_of(claimed) == "risky"
    assert policy.risk_of(search("Pay now", landmark="search")) == "risky"       # a payment is not a query


def test_a_toggle_is_a_piece_of_the_interface_but_risky_wording_stays_risky():
    policy = Policy(allowed_hosts=["www.saucedemo.com"])
    plain = lambda name: Action(kind="click", target=Element(role="button", name=name))
    for harmless in ("Open toggle", "Close toggle", "close toggle", "Close the toggle", "Close accordion",
                     "Close dropdown", "Close disclosure"):
        assert policy.risk_of(plain(harmless)) == "safe", harmless          # no ARIA state needed
        assert policy.check(plain(harmless), URL).decision == "allow", harmless
    assert policy.risk_of(plain("Close")) == "unknown"                       # a bare Close is still a question
    for risky in ("Close account", "Close position", "Close subscription", "Disable payment toggle",
                  "Close payment toggle", "Delete toggle", "Close account toggle"):
        assert policy.risk_of(plain(risky)) == "risky", risky
        assert policy.check(plain(risky), URL).decision == "confirm", risky
    # the ARIA and native proofs are untouched
    assert policy.risk_of(toggle(name="Collapse section", expanded="false")) == "safe"
    assert policy.risk_of(toggle(name="Close account", expanded=None, native="summary")) == "safe"
