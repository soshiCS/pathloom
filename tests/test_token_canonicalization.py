"""One logical selected token per removal control, however deeply the page nests it.

A chip is rarely marked up semantically, so perception's candidate walk accepts broad container tags and
several nested ancestors around one chip all look like tokens. Collapsing them during the walk cannot work:
an outer wrapper is reached before the inner element it wraps. `canonical_tokens` collapses the collected
candidates instead, keyed by what makes each one removable or selected rather than by its text. Offline;
no model, no browser, no live site.
"""
import pytest

from src.cua.surface import canonical_tokens, ref_encloses, token_key

CHIPS = "form:nth-of-type(1) > ul:nth-of-type(1)"


def candidate(ref, text, remove_ref="", selected=False):
    return {"ref": ref, "text": text, "name": "", "role": "listitem", "remove_ref": remove_ref,
            "removable": bool(remove_ref), "selected": selected, "remove_name": ""}


def texts(tokens):
    return [token["text"] for token in tokens]


def nested_chip(index, text, depth=3):
    """One chip wrapped in `depth` nested containers, all of which find the same removal control.

    This is the shape perception really collects: the outermost wrapper is walked first, so the canonical
    innermost element is not yet known when the wrapper is seen.
    """
    base = f"{CHIPS} > li:nth-of-type({index})"
    refs = [base] + [base + "".join(f" > div:nth-of-type(1)" for _ in range(n)) for n in range(1, depth)]
    remove = refs[-1] + " > button:nth-of-type(1)"
    return [candidate(ref, text, remove_ref=remove) for ref in refs]


# ---------- 1, 2. nesting collapses to the canonical inner token ----------

def test_a_nested_wrapper_structure_around_one_chip_produces_exactly_one_token():
    tokens = canonical_tokens(nested_chip(1, "renewable energy"))
    assert len(tokens) == 1 and texts(tokens) == ["renewable energy"]


def test_parent_and_child_candidates_sharing_one_removal_control_collapse_to_the_inner_token():
    chip = nested_chip(1, "renewable energy", depth=3)
    [token] = canonical_tokens(chip)
    assert token["ref"] == chip[-1]["ref"]                 # the innermost candidate, not the outer wrapper
    assert all(ref_encloses(other["ref"], token["ref"]) for other in chip[:-1])


def test_the_walk_order_does_not_decide_which_candidate_survives():
    chip = nested_chip(1, "renewable energy", depth=3)
    inner_first = canonical_tokens(list(reversed(chip)))
    assert len(inner_first) == 1 and inner_first[0]["ref"] == chip[-1]["ref"]


def test_an_ancestor_that_merely_contains_another_token_is_not_a_token_of_its_own():
    # A list wrapper that has its own removal control but also encloses a real chip.
    chip = nested_chip(2, "solar")
    wrapper = candidate(CHIPS, "solar", remove_ref=f"{CHIPS} > button:nth-of-type(1)")
    tokens = canonical_tokens([wrapper, *chip])
    assert texts(tokens) == ["solar"] and tokens[0]["ref"] == chip[-1]["ref"]


# ---------- 5, 6. genuinely separate chips stay separate ----------

def test_two_chips_with_different_removal_controls_remain_two_tokens():
    tokens = canonical_tokens([*nested_chip(1, "solar"), *nested_chip(2, "wind")])
    assert texts(tokens) == ["solar", "wind"]


def test_two_chips_with_identical_text_remain_distinct():
    tokens = canonical_tokens([*nested_chip(1, "solar"), *nested_chip(2, "solar")])
    assert len(tokens) == 2 and {t["ref"] for t in tokens} == {t["ref"] for t in tokens}
    assert texts(tokens) == ["solar", "solar"]            # never merged by their words


def test_document_order_is_preserved_after_canonicalization():
    tokens = canonical_tokens([*nested_chip(1, "alpha"), *nested_chip(2, "beta"), *nested_chip(3, "gamma")])
    assert texts(tokens) == ["alpha", "beta", "gamma"]


# ---------- 7. a natively selected option without a removal control ----------

def test_a_native_selected_option_without_a_removal_control_stays_keyed_by_its_own_element():
    option = candidate(f"{CHIPS} > li:nth-of-type(1)", "FY 2026", selected=True)
    [token] = canonical_tokens([option])
    assert token_key(token) == f"self:{option['ref']}" and token["selected"] is True


def test_a_selected_option_and_a_removable_chip_are_two_tokens():
    option = candidate(f"{CHIPS} > li:nth-of-type(1)", "FY 2026", selected=True)
    tokens = canonical_tokens([option, *nested_chip(2, "solar")])
    assert texts(tokens) == ["FY 2026", "solar"]


def test_two_selected_options_stay_separate_even_with_the_same_text():
    first = candidate(f"{CHIPS} > li:nth-of-type(1)", "FY 2026", selected=True)
    second = candidate(f"{CHIPS} > li:nth-of-type(2)", "FY 2026", selected=True)
    assert len(canonical_tokens([first, second])) == 2


# ---------- the identity rule itself ----------

def test_a_token_is_identified_by_its_removal_control_not_by_its_text():
    remove = f"{CHIPS} > li:nth-of-type(1) > button:nth-of-type(1)"
    outer = candidate(f"{CHIPS} > li:nth-of-type(1)", "solar", remove_ref=remove)
    inner = candidate(f"{CHIPS} > li:nth-of-type(1) > span:nth-of-type(1)", "solar", remove_ref=remove)
    assert token_key(outer) == token_key(inner) == f"remove:{remove}"


def test_a_structural_reference_is_never_read_as_the_ancestor_of_a_sibling():
    # "li:nth-of-type(1)" is a character-prefix of "li:nth-of-type(11)" but encloses nothing of it.
    assert not ref_encloses(f"{CHIPS} > li:nth-of-type(1)", f"{CHIPS} > li:nth-of-type(11)")
    assert ref_encloses(f"{CHIPS} > li:nth-of-type(1)", f"{CHIPS} > li:nth-of-type(1) > span:nth-of-type(1)")
    assert not ref_encloses("", "anything") and not ref_encloses("same", "same")


def test_candidates_without_references_are_left_alone_rather_than_merged():
    first, second = candidate("", "solar", remove_ref="r1"), candidate("", "wind", remove_ref="r2")
    assert len(canonical_tokens([first, second])) == 2


# ---------- replay agrees about what a token is ----------

def test_replay_recognises_a_removal_control_that_is_the_labels_sibling():
    # The usual chip: <container><span>label</span><button>x</button></container>. Perception lists the
    # label and the button, never the container, so the control is beside the text and not inside it.
    from src.cua.models import Element, Observation
    from src.cua.replay import selection_present

    box = f"{CHIPS} > li:nth-of-type(1) > div:nth-of-type(1)"
    label = Element(role="text", name="", text="renewable energy", ref=f"{box} > span:nth-of-type(1)")
    remove = Element(role="button", name="Remove renewable energy", text="x", ref=f"{box} > button:nth-of-type(1)")
    assert selection_present("renewable energy", Observation(url="x", elements=[label, remove]))


def test_replay_does_not_read_page_copy_beside_an_unrelated_button_as_a_token():
    from src.cua.models import Element, Observation
    from src.cua.replay import selection_present

    copy = Element(role="text", name="", text="renewable energy", ref="body:nth-of-type(1) > p:nth-of-type(1)")
    button = Element(role="button", name="Submit", text="Submit", ref="body:nth-of-type(1) > form:nth-of-type(1)")
    assert not selection_present("renewable energy", Observation(url="x", elements=[copy, button]))


def test_replay_counts_a_wrapper_and_its_inner_token_as_one():
    from src.cua.models import Element, Observation
    from src.cua.replay import selected_tokens

    box = f"{CHIPS} > li:nth-of-type(1)"
    outer = Element(role="listitem", name="", text="solar", ref=box, states={"selected": "true"})
    inner = Element(role="option", name="", text="solar", ref=f"{box} > span:nth-of-type(1)",
                    states={"selected": "true"})
    assert len(selected_tokens("solar", Observation(url="x", elements=[outer, inner]))) == 1


# ---------- 12. the rule stays site-agnostic ----------

def test_the_canonicalization_rule_names_no_site_capability_or_selector():
    import ast
    from pathlib import Path
    banned = ["usaspending", "saucedemo", "renewable energy", "applied filters", "search using keywords",
              "advanced search", "prime awards", "filter by keyword"]
    for path in sorted(Path("src/cua").glob("*.py")):
        tree = ast.parse(path.read_text())
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                      and getattr(node, "body", None) and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                for word in banned:
                    assert word not in node.value.lower(), f"{path} carries {word!r} in a live string"
