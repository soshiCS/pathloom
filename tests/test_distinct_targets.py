"""extract_many targets resolve to distinct page elements, in declared order, or the list is not emitted at all."""
import json

import pytest

from src.cua.escalation import NoOperator
from src.cua.models import Element, GraphAction, Locator
from src.cua.policy import Policy
from src.cua.replay import replay
from src.cua.artifact import build_linear
from tests.context import (ENTRY, HOSTS, PARAMS, Escalator, RunLog, SessionControl, checkout_artifact, checkout_nodes,
                           linear_node)
from tests.fake_surface import FakeSurface


def cell(text: str, path: str) -> Element:
    return Element(role="cell", name="Item Value", text=text, box=(10, 10, 50, 20), ref=path)


def ladder(path: str) -> Locator:
    return Locator(strategies=[{"kind": "role", "role": "cell", "name": "Item Value"}, {"kind": "css", "selector": path}])


def values_artifact(paths=("td:1", "td:2", "td:3")):
    """The checkout flow up to the overview, then one list read from three same-named cells."""
    read = linear_node("s16", GraphAction(action="extract_many", target=None, value="values",
                                          targets=[ladder(path) for path in paths]))
    base = checkout_artifact()
    outputs = {**base.outputs, "values": {"type": "list", "required": True, "items": {"type": "integer", "pattern": None},
                                          "description": "", "example": ["1", "2", "3"]}}
    return build_linear(name=base.name, goal=base.description, surface_meta=dict(base.surface), params=dict(PARAMS),
                        nodes=checkout_nodes() + [read], outputs=outputs, outcomes=list(base.outcomes),
                        success=dict(base.success), run_id="discovery-test", sensitive={"password"},
                        planner_name="scripted")


def run(artifact, extra):
    surface = FakeSurface(extra_elements=extra)
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(PARAMS), surface, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    return result, log


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def test_three_same_named_cells_resolve_to_three_distinct_elements_in_order():
    result, log = run(values_artifact(), [cell("1", "td:1"), cell("2", "td:2"), cell("3", "td:3")])
    assert result.status == "success" and result.outputs["values"] == [1, 2, 3]
    resolved = [e for e in events(log, "target_resolved") if e["step_id"] == "s16"]
    assert [(e["target_index"], e["rung"]) for e in resolved] == [(0, 1), (1, 1), (2, 1)]   # the shared rung is ambiguous
    assert [e["matches"] for e in events(log, "rung_ambiguous") if e["step_id"] == "s16"] == [3, 3, 3]


def test_the_output_follows_the_declared_target_order():
    artifact = values_artifact(paths=("td:1", "td:3", "td:2"))          # targets declared in this order
    result, _ = run(artifact, [cell("1", "td:1"), cell("2", "td:2"), cell("3", "td:3")])
    # item 0 takes the shared rung's own result (the first same-named cell), the others their structural rungs
    assert result.status == "success" and result.outputs["values"] == [1, 3, 2]


def test_when_one_element_must_serve_several_targets_the_list_is_not_emitted():
    result, log = run(values_artifact(), [cell("1", "td:1")])
    assert result.status == "failure" and result.outcome_code == "ambiguous_target" and result.step_id == "s16"
    assert result.expected == "a distinct control for item 1 of 'values'" and result.outputs == {}
    [ambiguous] = events(log, "ambiguous_target")
    assert (ambiguous["target_index"], ambiguous["taken"]) == (1, ["td:1"])
    assert [e for e in events(log, "output_extracted") if e["step_id"] == "s16"] == []   # nothing partial for the list


def test_a_missing_element_is_still_a_missing_target_not_an_ambiguity():
    optional = values_artifact()
    optional.outputs["values"]["required"] = False
    third = optional.nodes[-2].action.targets[2] if optional.nodes[-2].action else None
    node = next(n for n in optional.nodes if n.action and n.action.action == "extract_many")
    node.action.targets[2] = Locator(strategies=[{"kind": "role", "role": "cell", "name": "Nothing"},
                                                 {"kind": "css", "selector": "td:3"}])
    result, log = run(optional, [cell("1", "td:1"), cell("2", "td:2"), Element(role="cell", name="Other", text="9", ref="td:9")])
    assert result.status == "success" and "values" not in result.outputs
    assert events(log, "optional_output_absent")[0]["target_index"] == 2 and events(log, "ambiguous_target") == []
