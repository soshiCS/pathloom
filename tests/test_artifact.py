"""Artifact parameterization, validation, versioning, save/load."""
import json
from dataclasses import replace

import pytest

from src.cua import artifact as artifact_module
from src.cua.artifact import (ArtifactError, load, parameterize, parameterize_locator, save, substitute,
                              substitute_locator, validate)
from tests.context import PARAMS, Locator, Step, checkout_artifact


def test_parameterize_and_substitute_round_trip():
    assert parameterize("Add Sauce Labs Backpack for Test User at 10001", PARAMS) == \
        "Add {{product_name}} for {{first_name}} {{last_name}} at {{postal_code}}"
    assert substitute("{{first_name}} {{last_name}}", PARAMS) == "Test User"
    assert parameterize(None, PARAMS) is None


def test_parameterize_leaves_existing_placeholders_alone():
    # A selector whose value equals another input's name must not rewrite that input's placeholder.
    params = {"lookup_method": "member_id", "member_id": "1001"}
    assert parameterize("{{member_id}}", params) == "{{member_id}}"
    assert parameterize("lookup by member_id: 1001", params) == "lookup by {{lookup_method}}: {{member_id}}"


def test_parameterize_only_replaces_whole_tokens():
    assert parameterize("standard_user", {"last_name": "user"}) == "standard_user"   # inside a longer token
    assert parameterize("Testing", {"first_name": "Test"}) == "Testing"
    assert parameterize("x1y", {"one": "1"}) == "x1y"                                  # too short to be safe


def test_locators_are_parameterized_and_substituted():
    locator = Locator(strategies=[{"kind": "role", "role": "button", "name": "Add to cart", "context": "Sauce Labs Backpack"},
                                  {"kind": "css", "selector": "button:Add to cart"}])
    generic = parameterize_locator(locator, PARAMS)
    assert generic.strategies == [{"kind": "role", "role": "button", "name": "Add to cart", "context": "{{product_name}}"}]
    # The structural rung is dropped: it would point at the discovery-time product for any input.
    concrete = substitute_locator(generic, {**PARAMS, "product_name": "Sauce Labs Bike Light"})
    assert concrete.strategies[0]["context"] == "Sauce Labs Bike Light"


def test_unparameterized_locators_keep_their_whole_ladder():
    locator = Locator(strategies=[{"kind": "role", "role": "button", "name": "Login"},
                                  {"kind": "css", "selector": "form > input"}, {"kind": "coords", "x": 1, "y": 2}])
    assert len(parameterize_locator(locator, PARAMS).strategies) == 3


def test_save_and_load_round_trip_preserves_the_contract():
    built = checkout_artifact()
    path = save(built, secrets=("secret_sauce",))
    loaded = load(path)
    assert loaded.name == built.name and loaded.version == 1
    assert len(loaded.steps) == 15
    assert loaded.steps[4].target.strategies[0] == {"kind": "role", "role": "button", "name": "Add to cart",
                                                    "context": "{{product_name}}"}
    assert loaded.inputs["password"] == {"type": "string", "required": True, "sensitive": True,
                                         "description": "Value typed where 'password' was used during discovery"}
    assert loaded.outputs["total"]["type"] == "number"
    assert loaded.success == {"text_contains": "Checkout: Overview"}
    assert {o["code"] for o in loaded.outcomes} == {"dismiss_cookie_notice", "user_locked_out", "invalid_credentials",
                                                    "product_not_found", "checkout_info_missing"}


def test_versions_increment_instead_of_overwriting():
    save(checkout_artifact())
    second = checkout_artifact()
    assert second.version == 2
    path = save(second)
    assert path.name.endswith(".v2.json")
    assert sorted(p.name for p in artifact_module.ARTIFACTS_DIR.iterdir()) == [
        "checkout_review.v1.json", "checkout_review.v2.json"]


def test_unsupported_schema_version_is_rejected():
    with pytest.raises(ArtifactError, match="schema_version"):
        validate(replace(checkout_artifact(), schema_version="0.9"))


def test_undeclared_placeholder_in_a_value_is_rejected():
    broken = checkout_artifact()
    broken.steps[1].value = "{{user}}"
    with pytest.raises(ArtifactError, match="user"):
        validate(broken)


def test_undeclared_placeholder_in_a_locator_is_rejected():
    broken = checkout_artifact()
    broken.steps[4].target.strategies[0]["context"] = "{{item}}"
    with pytest.raises(ArtifactError, match="locator placeholders"):
        validate(broken)


def test_extract_to_undeclared_output_is_rejected():
    broken = checkout_artifact()
    broken.steps[11].value = "shipping"
    with pytest.raises(ArtifactError, match="undeclared output"):
        validate(broken)


def test_outcome_needs_a_detect_condition():
    broken = checkout_artifact()
    broken.outcomes.append({"code": "x", "kind": "business", "detect": {}})
    with pytest.raises(ArtifactError, match="detect needs"):
        validate(broken)


def test_recoverable_outcome_without_recovery_is_rejected():
    broken = checkout_artifact()
    broken.outcomes.append({"code": "x", "kind": "recoverable", "detect": {"dialog_contains": "y"}})
    with pytest.raises(ArtifactError, match="recover"):
        validate(broken)


def test_step_without_target_is_rejected():
    broken = checkout_artifact()
    broken.steps.append(Step(id="s99", action="click", target=None))
    with pytest.raises(ArtifactError, match="needs a target"):
        validate(broken)


def test_load_reports_malformed_json_clearly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ArtifactError, match="cannot read"):
        load(bad)


def test_save_redacts_sensitive_keys_and_refuses_leaked_secrets():
    built = checkout_artifact()
    built.provenance["token"] = "abc123"
    text = save(built).read_text()
    assert "abc123" not in text and '"token": "[REDACTED]"' in text

    leaky = checkout_artifact()
    leaky.steps[2].value = "secret_sauce"  # the password was never parameterized
    with pytest.raises(ArtifactError, match="not parameterized"):
        save(leaky, secrets=("secret_sauce",))


def test_saved_artifact_is_plain_reviewable_json_without_run_literals():
    data = json.loads(save(checkout_artifact(), secrets=("secret_sauce",)).read_text())
    assert set(data) == {"schema_version", "name", "version", "status", "description", "surface", "inputs",
                         "outputs", "steps", "outcomes", "success", "provenance"}
    assert data["status"] == "draft"
    text = json.dumps(data)
    for literal in ("secret_sauce", "standard_user", "Sauce Labs Backpack", "10001"):
        assert literal not in text, literal
    assert "transcript" not in text
