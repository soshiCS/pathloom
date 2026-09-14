"""Capability graphs (schema 2.0): round trip, validation, and the version-1 adapter."""
import json
from dataclasses import replace

import pytest

from src.cua import artifact as artifact_module
from src.cua import graph as graph_module
from src.cua.artifact import ArtifactError, load, save
from src.cua.escalation import NoOperator
from src.cua.graph import (GRAPH_SCHEMA_VERSION, from_dict, from_linear, load_graph, nodes_by_id, outgoing,
                           save_graph, to_dict, validate_graph)
from src.cua.models import ArtifactV2, GraphAction, Guard
from src.cua.replay import replay
from tests.context import (ENTRY, HOSTS, PARAMS, Escalator, Policy, RunLog, SessionControl, Step,
                           action_node as action, checkout_artifact, decision_node as decision, edge, ladder,
                           terminal_node as terminal)
from tests.fake_surface import FakeSurface


def branched_graph() -> ArtifactV2:
    """Login, then branch on what the screen shows: locked out, bad credentials, products, or nothing known.

    Uses every guard kind once. The metadata is the checkout capability's.
    """
    base = checkout_artifact()
    nodes = [
        action("s1", GraphAction(action="navigate", target=None, value=ENTRY,
                                 checkpoint={"text_contains": "Swag Labs"}), effect="none", retry="safe"),
        decision("d1"),
        action("s2", GraphAction(action="type", target=ladder("textbox", "Username"), value="{{username}}")),
        action("s3", GraphAction(action="type", target=ladder("textbox", "Password"), value="{{password}}")),
        action("s4", GraphAction(action="click", target=ladder("button", "Login"))),
        decision("d2"),
        terminal("no_user", "failure", "username_blank"),
        terminal("locked", "business_outcome", "user_locked_out"),
        terminal("bad_creds", "business_outcome", "invalid_credentials"),
        terminal("ok", "success"),
        terminal("unknown_screen", "failure", "login_unverified"),
    ]
    edges = [
        edge("s1", "d1"),
        edge("d1", "no_user", Guard(kind="input_equals", input="username", value=""), priority=0),
        edge("d1", "s2", priority=1),
        edge("s2", "s3"),
        edge("s3", "s4"),
        edge("s4", "d2"),
        edge("d2", "locked", Guard(kind="text_visible", value="locked out"), priority=0),
        edge("d2", "bad_creds", Guard(kind="dialog_contains", value="do not match"), priority=1),
        edge("d2", "ok", Guard(kind="url_matches", pattern=r"/inventory"),
             Guard(kind="element_present", target=ladder("button", "Add to cart", "{{product_name}}")), priority=2),
        edge("d2", "unknown_screen", priority=3),
    ]
    return ArtifactV2(
        schema_version=GRAPH_SCHEMA_VERSION, name="login_check", version=1, status="draft",
        description="log in and classify the screen that follows", surface=base.surface, inputs=base.inputs,
        outputs={}, entry_node="s1", nodes=nodes, edges=edges, outcomes=base.outcomes,
        success={"text_contains": "Products"}, provenance={"run_id": "test-run", "planner": "hand-written"},
    )


def rejects(graph: ArtifactV2, message: str) -> None:
    with pytest.raises(ArtifactError, match=message):
        validate_graph(graph)


def with_node(mutate) -> ArtifactV2:
    """A fresh branched graph with one node edited in place."""
    graph = branched_graph()
    mutate(nodes_by_id(graph))
    return graph


# ---------- round trip ----------

def test_branched_graph_is_valid_and_every_guard_kind_is_used():
    graph = branched_graph()
    validate_graph(graph)
    kinds = {guard.kind for e in graph.edges for guard in e.guards}
    assert kinds == {"always", "input_equals", "text_visible", "url_matches", "element_present", "dialog_contains"}
    assert [e.target for e in outgoing(graph, "d2")] == ["locked", "bad_creds", "ok", "unknown_screen"]
    assert nodes_by_id(graph)["ok"].status == "success"


def test_dict_round_trip_loses_nothing():
    graph = branched_graph()
    data = json.loads(json.dumps(to_dict(graph)))
    assert from_dict(data) == graph
    assert data["schema_version"] == "2.0"
    assert list(data) == ["schema_version", "name", "version", "status", "description", "surface", "inputs",
                          "outputs", "entry_node", "nodes", "edges", "outcomes", "success", "provenance"]


def test_action_nodes_carry_id_and_effect_once():
    data = to_dict(branched_graph())
    node = next(n for n in data["nodes"] if n["id"] == "s4")
    assert list(node) == ["id", "kind", "action", "effect", "retry_safety", "status", "outcome_code"]
    assert list(node["action"]) == ["action", "target", "value", "checkpoint"]
    assert "id" not in node["action"] and "risk" not in node["action"]
    assert '"risk"' not in json.dumps(data)
    assert [n["action"] for n in data["nodes"] if n["kind"] != "action"] == [None] * 7


def test_serialized_guards_carry_only_the_fields_their_kind_uses():
    data = to_dict(branched_graph())
    by_target = {e["target"]: e for e in data["edges"] if e["source"] == "d2"}
    assert by_target["locked"]["guards"] == [{"kind": "text_visible", "value": "locked out"}]
    assert by_target["unknown_screen"] == {"source": "d2", "target": "unknown_screen",
                                           "guards": [{"kind": "always"}], "priority": 3}
    assert by_target["ok"]["guards"][1] == {"kind": "element_present", "target": {"strategies": [
        {"kind": "role", "role": "button", "name": "Add to cart", "context": "{{product_name}}"},
        {"kind": "css", "selector": "button:Add to cart:{{product_name}}"}]}}


def test_file_round_trip_is_stable_and_redacted():
    graph = branched_graph()
    graph.provenance["token"] = "abc123"
    path = save_graph(graph, secrets=("secret_sauce",))
    assert path.name == "login_check.v1.json"
    text = path.read_text()
    assert "abc123" not in text and '"token": "[REDACTED]"' in text
    assert text == graph_module.dumps(graph, secrets=("secret_sauce",))   # byte-stable for diffs
    loaded = load_graph(path)
    assert loaded == replace(graph, provenance={**graph.provenance, "token": "[REDACTED]"})


def test_save_refuses_secrets_and_never_overwrites(tmp_path):
    leaky = with_node(lambda n: setattr(n["s3"].action, "value", "secret_sauce"))
    with pytest.raises(ArtifactError, match="not parameterized"):
        save_graph(leaky, secrets=("secret_sauce",))

    path = save_graph(branched_graph())
    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        save_graph(branched_graph())
    elsewhere = save_graph(branched_graph(), path=tmp_path / "copy.json")
    assert elsewhere.read_text() == path.read_text()


# ---------- validation ----------

def test_duplicate_node_ids_are_rejected():
    graph = branched_graph()
    graph.nodes.append(terminal("ok", "success"))
    rejects(graph, "node id 'ok' is missing or duplicated")


def test_missing_entry_node_is_rejected():
    rejects(replace(branched_graph(), entry_node="s0"), "entry_node 's0' is not a node")
    rejects(replace(branched_graph(), entry_node=""), "entry_node '' is not a node")
    rejects(replace(branched_graph(), entry_node="ok"), "is a terminal node")


def test_edges_to_missing_nodes_are_rejected():
    graph = branched_graph()
    graph.edges.append(edge("s4", "s99"))
    rejects(graph, "edge s4->s99: 's99' is not a node")
    graph = branched_graph()
    graph.edges.append(edge("s98", "ok"))
    rejects(graph, "edge s98->ok: 's98' is not a node")


def test_action_nodes_need_a_valid_action():
    rejects(with_node(lambda n: setattr(n["s4"], "action", None)), "node s4: an action node needs an action")
    rejects(with_node(lambda n: setattr(n["s4"].action, "target", None)), "node s4: click needs a target locator")
    rejects(with_node(lambda n: setattr(n["s4"].action, "action", "hover")), "node s4: unknown action 'hover'")
    rejects(with_node(lambda n: setattr(n["s2"].action, "value", "{{user}}")),
            "node s2: placeholders \\['user'\\] are not declared inputs")
    rejects(with_node(lambda n: setattr(n["s4"].action, "checkpoint", {"text_contains": "{{x}}"})),
            "node s4 checkpoint: placeholders \\['x'\\]")
    rejects(with_node(lambda n: setattr(n["s4"], "status", "success")), "only terminal nodes carry a status")
    rejects(with_node(lambda n: setattr(n["s4"], "kind", "loop")), "unknown kind 'loop'")
    rejects(with_node(lambda n: setattr(n["s4"], "effect", "maybe")), "effect must be one of")
    rejects(with_node(lambda n: setattr(n["s4"], "retry_safety", "yolo")), "retry_safety must be one of")


def test_irreversible_and_unknown_actions_are_never_retried_automatically():
    for effect in ("irreversible", "unknown"):
        for retry in ("safe", "verify_before_retry"):
            graph = with_node(lambda n: (setattr(n["s4"], "effect", effect), setattr(n["s4"], "retry_safety", retry)))
            rejects(graph, f"node s4: an action with effect '{effect}' must have retry_safety 'never_retry', "
                           f"got '{retry}'")
        validate_graph(with_node(lambda n: (setattr(n["s4"], "effect", effect),
                                            setattr(n["s4"], "retry_safety", "never_retry"))))
    validate_graph(with_node(lambda n: setattr(n["s4"], "retry_safety", "safe")))   # reversible may retry


def test_decision_and_terminal_nodes_carry_no_action_metadata():
    login = nodes_by_id(branched_graph())["s4"].action
    rejects(with_node(lambda n: setattr(n["d1"], "action", login)), "node d1: a decision node does not carry an action")
    rejects(with_node(lambda n: setattr(n["ok"], "action", login)), "node ok: a terminal node does not carry an action")
    rejects(with_node(lambda n: setattr(n["d1"], "status", "success")), "only terminal nodes carry a status")
    rejects(with_node(lambda n: setattr(n["d1"], "outcome_code", "x")), "only terminal nodes carry a status")


def test_decision_and_terminal_nodes_reject_effect_and_retry_metadata():
    rejects(with_node(lambda n: setattr(n["d1"], "effect", "reversible")),
            "node d1: a decision node has effect 'none', got 'reversible'")
    rejects(with_node(lambda n: setattr(n["d1"], "retry_safety", "never_retry")),
            "node d1: a decision node has retry_safety 'safe', got 'never_retry'")
    rejects(with_node(lambda n: setattr(n["ok"], "effect", "unknown")),
            "node ok: a terminal node has effect 'none', got 'unknown'")
    rejects(with_node(lambda n: setattr(n["ok"], "retry_safety", "verify_before_retry")),
            "node ok: a terminal node has retry_safety 'safe', got 'verify_before_retry'")


def test_terminal_status_and_outcome_code_are_checked():
    rejects(with_node(lambda n: setattr(n["ok"], "status", None)), "node ok: a terminal node needs a status")
    rejects(with_node(lambda n: setattr(n["ok"], "outcome_code", "x")), "a success terminal has no outcome_code")
    rejects(with_node(lambda n: setattr(n["locked"], "outcome_code", None)),
            "a business_outcome terminal needs an outcome_code")
    rejects(with_node(lambda n: setattr(n["locked"], "outcome_code", "never_declared")),
            "outcome_code 'never_declared' is not a declared outcome")


def test_unsupported_and_malformed_guards_are_rejected():
    def with_guards(*guards: Guard) -> ArtifactV2:
        graph = branched_graph()
        graph.edges[1].guards = list(guards)
        return graph

    rejects(with_guards(Guard(kind="python", value="1 == 1")), "unsupported guard kind 'python'")
    rejects(with_guards(Guard(kind="input_equals", input="username")), "input_equals guard needs value")
    rejects(with_guards(Guard(kind="always", value="x")), "always guard does not take value")
    rejects(with_guards(Guard(kind="input_equals", input="user", value="")), "undeclared input 'user'")
    rejects(with_guards(Guard(kind="url_matches", pattern="(")), "not a valid regex")
    rejects(with_guards(Guard(kind="text_visible", value="{{item}}")),
            "placeholders \\['item'\\] are not declared inputs")
    rejects(with_guards(), "needs at least one guard")


def test_always_must_be_the_only_guard_on_its_edge():
    graph = branched_graph()
    graph.edges[6].guards.append(Guard(kind="always"))
    rejects(graph, "edge d2->locked: an 'always' guard must be the only guard on its edge")
    graph = branched_graph()
    graph.edges[0].guards = [Guard(kind="always"), Guard(kind="always")]
    rejects(graph, "edge s1->d1: an 'always' guard must be the only guard on its edge")


def test_edge_structure_is_checked():
    graph = branched_graph()
    graph.edges.append(edge("ok", "s1"))
    rejects(graph, "a terminal node has no outgoing edges")

    graph = branched_graph()
    graph.edges.append(edge("d1", "s3", priority=1))
    rejects(graph, "priority 1 is used twice on edges leaving d1")

    graph = branched_graph()
    graph.edges[0].priority = True
    rejects(graph, "priority must be an integer")

    graph = branched_graph()                                   # cut the flow off after s3: a dead end
    dropped = {"s4", "d2", "locked", "bad_creds", "unknown_screen"}
    graph.nodes = [n for n in graph.nodes if n.id not in dropped]
    graph.edges = [e for e in graph.edges if e.source not in dropped and e.target not in dropped]
    rejects(graph, "node s3: action nodes need at least one outgoing edge")


def test_unreachable_nodes_are_rejected():
    graph = branched_graph()
    graph.nodes.append(action("orphan", GraphAction(action="click", target=ladder("button", "Back"))))
    graph.edges.append(edge("orphan", "ok"))
    rejects(graph, "nodes \\['orphan'\\] are unreachable from entry_node 's1'")


def test_cycles_are_rejected():
    graph = branched_graph()
    graph.edges.append(edge("d2", "s2", Guard(kind="text_visible", value="Try again"), priority=-1))
    rejects(graph, "graph has a cycle: s2 -> s3 -> s4 -> d2 -> s2")


def test_an_always_edge_is_the_single_last_resort_of_its_node():
    graph = branched_graph()
    graph.edges.append(edge("d2", "ok", priority=4))
    rejects(graph, "edge d2->ok: node d2 already has an 'always' edge to unknown_screen; only one fallback edge")

    graph = branched_graph()
    graph.edges[9].priority = -1                                # the fallback would win before any guard is judged
    rejects(graph, "edge d2->unknown_screen: an 'always' edge must have the greatest priority among the edges "
                   "leaving d2")

    graph = branched_graph()
    graph.edges[2].priority = -5                                # d1's fallback below its guarded edge
    rejects(graph, "edge d1->s2: an 'always' edge must have the greatest priority")


def test_wrong_schema_version_is_rejected():
    rejects(replace(branched_graph(), schema_version="1.0"), "unsupported schema_version '1.0'")


def test_malformed_json_is_rejected_with_the_location(tmp_path):
    good = to_dict(branched_graph())

    def broken(mutate):
        data = json.loads(json.dumps(good))
        mutate(data)
        return data

    with pytest.raises(ArtifactError, match="artifact is missing keys: \\['entry_node'\\]"):
        from_dict(broken(lambda d: d.pop("entry_node")))
    with pytest.raises(ArtifactError, match="node 2 \\('s2'\\) has unknown keys: \\['retry'\\]"):
        from_dict(broken(lambda d: d["nodes"][2].update(retry="safe")))
    with pytest.raises(ArtifactError, match="node 0 is missing keys: \\['id'\\]"):
        from_dict(broken(lambda d: d["nodes"][0].pop("id")))
    with pytest.raises(ArtifactError, match="node 2 \\('s2'\\) action has unknown keys: \\['id', 'risk'\\]"):
        from_dict(broken(lambda d: d["nodes"][2]["action"].update(id="s2", risk="safe")))   # version-1 leftovers
    with pytest.raises(ArtifactError, match="node 4 \\('s4'\\) action is missing keys: \\['target'\\]"):
        from_dict(broken(lambda d: d["nodes"][4]["action"].pop("target")))
    with pytest.raises(ArtifactError, match="node 4 \\('s4'\\) action target needs a strategies list"):
        from_dict(broken(lambda d: d["nodes"][4]["action"].update(target={})))
    with pytest.raises(ArtifactError, match="edge 1 \\('d1'->'no_user'\\) guard 0 has unknown keys: \\['expr'\\]"):
        from_dict(broken(lambda d: d["edges"][1]["guards"][0].update(expr="x")))
    with pytest.raises(ArtifactError, match="edges must be a JSON list"):
        from_dict(broken(lambda d: d.update(edges={})))
    with pytest.raises(ArtifactError, match="edge 0 \\('s1'->'d1'\\) guards must be a JSON list"):
        from_dict(broken(lambda d: d["edges"][0].update(guards="always")))
    with pytest.raises(ArtifactError, match="artifact must be a JSON object"):
        from_dict([])

    path = tmp_path / "bad.json"
    path.write_text(json.dumps(broken(lambda d: d.update(schema_version="3.0"))))
    with pytest.raises(ArtifactError, match="unsupported schema_version '3.0', expected '1.0' or '2.0'"):
        load_graph(path)


# ---------- version-1 adapter ----------

def action_data(step: Step) -> tuple:
    return step.action, step.target, step.value, step.checkpoint


def test_linear_artifact_becomes_a_chain_of_action_nodes():
    linear = checkout_artifact()
    graph = from_linear(linear)
    validate_graph(graph)
    assert graph.schema_version == "2.0" and graph.entry_node == "s1"
    assert [n.id for n in graph.nodes] == [s.id for s in linear.steps] + ["success"]
    assert [n.kind for n in graph.nodes] == ["action"] * 15 + ["terminal"]
    assert [action_data(n.action) for n in graph.nodes[:-1]] == [action_data(s) for s in linear.steps]
    assert graph.nodes[-1] == terminal("success", "success")
    assert [(e.source, e.target, e.priority) for e in graph.edges] == \
        [(f"s{i}", f"s{i + 1}", 0) for i in range(1, 15)] + [("s15", "success", 0)]
    assert all(e.guards == [Guard(kind="always")] for e in graph.edges)


def test_conversion_maps_risk_and_action_to_effect_and_retry_safety():
    finish = Step(id="s16", action="click", target=ladder("button", "Finish"),
                  checkpoint={"text_contains": "Thank you"}, risk="risky")
    graph = from_linear(checkout_artifact(extra_step=finish))
    nodes = nodes_by_id(graph)
    assert (nodes["s1"].effect, nodes["s1"].retry_safety) == ("none", "verify_before_retry")       # safe navigate
    assert (nodes["s2"].effect, nodes["s2"].retry_safety) == ("reversible", "verify_before_retry")  # safe type
    assert (nodes["s4"].effect, nodes["s4"].retry_safety) == ("reversible", "verify_before_retry")  # safe click
    assert (nodes["s12"].effect, nodes["s12"].retry_safety) == ("none", "verify_before_retry")      # safe extract
    assert (nodes["s16"].effect, nodes["s16"].retry_safety) == ("irreversible", "never_retry")     # risky click
    assert action_data(nodes["s16"].action) == action_data(finish)
    assert not hasattr(nodes["s16"].action, "risk") and not hasattr(nodes["s16"].action, "id")


def test_conversion_preserves_metadata_and_leaves_the_original_alone():
    linear = checkout_artifact()
    before = artifact_module.asdict(linear)
    graph = from_linear(linear)
    for field in ("name", "version", "status", "description", "surface", "inputs", "outputs", "outcomes", "success"):
        assert getattr(graph, field) == getattr(linear, field), field
    assert graph.provenance == {**linear.provenance, "converted_from_schema": "1.0"}
    assert "converted_from_schema" not in linear.provenance

    # The graph owns copies: editing it cannot reach back into the version-1 artifact.
    graph.nodes[0].action.value = "https://elsewhere.example/"
    graph.nodes[4].action.target.strategies.clear()
    graph.nodes[3].action.checkpoint["text_contains"] = "changed"
    graph.inputs["username"]["required"] = False
    graph.surface["entry_url"] = "changed"
    assert artifact_module.asdict(linear) == before


def test_conversion_refuses_an_invalid_or_colliding_artifact():
    broken = checkout_artifact()
    broken.steps[1].value = "{{user}}"
    with pytest.raises(ArtifactError, match="user"):
        from_linear(broken)
    colliding = checkout_artifact()
    colliding.steps[0].id = "success"
    with pytest.raises(ArtifactError, match="collides with the success terminal"):
        from_linear(colliding)


def test_load_graph_converts_a_version_1_file_without_touching_it():
    path = save(checkout_artifact(), secrets=("secret_sauce",))
    original = path.read_bytes()
    graph = load_graph(path)
    assert graph == from_linear(load(path))
    assert path.read_bytes() == original
    assert sorted(p.name for p in artifact_module.ARTIFACTS_DIR.iterdir()) == ["checkout_review.v1.json"]

    # Saving the converted graph never replaces the version-1 file it came from.
    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        save_graph(graph)
    assert path.read_bytes() == original


def test_converted_graph_round_trips_through_json(tmp_path):
    graph = from_linear(checkout_artifact())
    path = save_graph(graph, secrets=("secret_sauce",), path=tmp_path / "checkout_review.graph.json")
    assert load_graph(path) == graph
    assert path.read_text() == graph_module.dumps(graph, secrets=("secret_sauce",))


# ---------- version-1 behavior is unchanged ----------

def test_version_1_loading_and_replay_are_unchanged():
    path = save(checkout_artifact(), secrets=("secret_sauce",))
    loaded = load(path)
    assert loaded.schema_version == "1.0" and len(loaded.steps) == 15 and not hasattr(loaded, "nodes")
    assert loaded.steps[0].id == "s1" and loaded.steps[0].risk == "safe"    # the version-1 step keeps id and risk

    surface = FakeSurface()
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(loaded, dict(PARAMS), surface, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success"
    assert result.outputs == {"product_name": "Sauce Labs Backpack", "subtotal": 29.99, "tax": 2.4, "total": 32.39}
    assert surface.actions[:2] == [("navigate", ENTRY), ("type", "Username", "standard_user")]


def test_version_1_loader_points_a_graph_file_at_the_graph_loader(tmp_path):
    path = save_graph(branched_graph(), path=tmp_path / "login_check.v1.json")
    with pytest.raises(ArtifactError, match="unsupported schema_version '2.0'.*graph.load_graph"):
        load(path)
