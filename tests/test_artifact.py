"""The artifact schema: parameterization, the linear builder and its defaults, validation, serialization,
save/load, versioning, and the refusal of other schema versions."""
import json
from dataclasses import replace

import pytest

from src.cua import artifact as artifact_module
from src.cua.artifact import (SCHEMA_VERSION, ArtifactError, build_linear, check_path_names_artifact,
                              classify_effect, classify_retry_safety, from_dict, linear_path, load_artifact,
                              nodes_by_id, outgoing, parameterize, parameterize_locator, save_artifact, substitute,
                              substitute_locator, to_dict, validate)
from src.cua.models import Artifact, GraphAction, Guard
from tests.context import (ENTRY, PARAMS, Locator, action_node as action, checkout_artifact, decision_node as decision,
                           edge, finish_node, ladder, terminal_node as terminal)


# ---------- parameters ----------

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
    assert generic.strategies == [
        {"kind": "role", "role": "button", "name": "Add to cart", "context": "{{product_name}}"},
        {"kind": "css", "selector": "button:Add to cart"},
    ]
    # Replay may use the structural rung only to disambiguate matches of the semantic rung.
    concrete = substitute_locator(generic, {**PARAMS, "product_name": "Sauce Labs Bike Light"})
    assert concrete.strategies[0]["context"] == "Sauce Labs Bike Light"


def test_unparameterized_locators_keep_their_whole_ladder():
    locator = Locator(strategies=[{"kind": "role", "role": "button", "name": "Login"},
                                  {"kind": "css", "selector": "form > input"}, {"kind": "coords", "x": 1, "y": 2}])
    assert len(parameterize_locator(locator, PARAMS).strategies) == 3


# ---------- the linear builder and its classification defaults ----------

def test_build_linear_emits_a_valid_chain_ending_in_the_success_terminal():
    artifact = checkout_artifact()
    validate(artifact)
    assert artifact.schema_version == SCHEMA_VERSION == "2.0" and artifact.status == "draft" and artifact.version == 1
    assert artifact.entry_node == "s1"
    assert [n.id for n in artifact.nodes] == [f"s{i}" for i in range(1, 16)] + ["success"]
    assert [n.kind for n in artifact.nodes] == ["action"] * 15 + ["terminal"]
    assert artifact.nodes[-1] == terminal("success", "success")
    assert [(e.source, e.target, e.priority) for e in artifact.edges] == \
        [(f"s{i}", f"s{i + 1}", 0) for i in range(1, 15)] + [("s15", "success", 0)]
    assert all(e.guards == [Guard(kind="always")] for e in artifact.edges)
    assert [n.id for n in linear_path(artifact)] == [f"s{i}" for i in range(1, 16)]
    assert artifact.provenance["node_count"] == 16


def test_effect_defaults_follow_the_action_kind_and_the_policy_verdict():
    assert classify_effect("navigate", risk=False) == "none" and classify_effect("extract", risk=False) == "none"
    assert classify_effect("click", risk=False) == "reversible" and classify_effect("type", risk=False) == "reversible"
    assert classify_effect("click", risk=True) == "irreversible"
    assert classify_effect("hover", risk=False) == "unknown"                    # unclassifiable


def test_retry_defaults_are_conservative():
    checkpoint = {"text_contains": "Products"}
    assert classify_retry_safety("extract", "none", None) == "safe"
    assert classify_retry_safety("navigate", "none", checkpoint) == "verify_before_retry"
    assert classify_retry_safety("navigate", "none", None) == "safe"              # reloading a page changes nothing
    assert classify_retry_safety("click", "reversible", checkpoint) == "verify_before_retry"
    assert classify_retry_safety("type", "reversible", checkpoint) == "verify_before_retry"
    assert classify_retry_safety("click", "reversible", None) == "never_retry"    # nothing could verify it
    assert classify_retry_safety("type", "reversible", None) == "never_retry"
    assert classify_retry_safety("click", "irreversible", checkpoint) == "never_retry"
    assert classify_retry_safety("click", "unknown", checkpoint) == "never_retry"


def test_checkout_nodes_carry_the_defaults_discovery_would_record():
    nodes = nodes_by_id(checkout_artifact(extra_node=finish_node()))
    assert (nodes["s1"].effect, nodes["s1"].retry_safety) == ("none", "verify_before_retry")       # navigate + checkpoint
    assert (nodes["s2"].effect, nodes["s2"].retry_safety) == ("reversible", "never_retry")          # type, no checkpoint
    assert (nodes["s4"].effect, nodes["s4"].retry_safety) == ("reversible", "verify_before_retry")  # click + checkpoint
    assert (nodes["s12"].effect, nodes["s12"].retry_safety) == ("none", "safe")                     # extract
    assert (nodes["s16"].effect, nodes["s16"].retry_safety) == ("irreversible", "never_retry")     # risky click
    assert not hasattr(nodes["s16"].action, "risk") and not hasattr(nodes["s16"].action, "id")


def test_build_linear_needs_at_least_one_node():
    with pytest.raises(ArtifactError, match="at least one action node"):
        build_linear(name="x", goal="g", surface_meta={"entry_url": ENTRY}, params={}, nodes=[], outputs={},
                     outcomes=[], success={"url_contains": ENTRY}, run_id="r")


# ---------- a branching graph: every node kind and every guard kind ----------

def branched_graph() -> Artifact:
    """Login, then branch on what the screen shows: locked out, bad credentials, products, or nothing known."""
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
    return Artifact(
        schema_version=SCHEMA_VERSION, name="login_check", version=1, status="draft",
        description="log in and classify the screen that follows", surface=base.surface, inputs=base.inputs,
        outputs={}, entry_node="s1", nodes=nodes, edges=edges, outcomes=base.outcomes,
        success={"text_contains": "Products"}, provenance={"run_id": "test-run", "planner": "hand-written"},
    )


def rejects(graph: Artifact, message: str) -> None:
    with pytest.raises(ArtifactError, match=message):
        validate(graph)


def with_node(mutate) -> Artifact:
    """A fresh branched graph with one node edited in place."""
    graph = branched_graph()
    mutate(nodes_by_id(graph))
    return graph


def test_branched_graph_is_valid_and_every_guard_kind_is_used():
    graph = branched_graph()
    validate(graph)
    kinds = {guard.kind for e in graph.edges for guard in e.guards}
    assert kinds == {"always", "input_equals", "text_visible", "url_matches", "element_present", "dialog_contains"}
    assert [e.target for e in outgoing(graph, "d2")] == ["locked", "bad_creds", "ok", "unknown_screen"]
    assert nodes_by_id(graph)["ok"].status == "success"
    with pytest.raises(ArtifactError, match="not a linear artifact at node 'd1'"):
        linear_path(graph)


# ---------- serialization ----------

def test_dict_round_trip_loses_nothing():
    for graph in (branched_graph(), checkout_artifact()):
        data = json.loads(json.dumps(to_dict(graph)))
        assert from_dict(data) == graph
        assert data["schema_version"] == "2.0"
        assert list(data) == ["schema_version", "name", "version", "status", "description", "surface", "inputs",
                              "outputs", "entry_node", "nodes", "edges", "outcomes", "success", "provenance"]


def test_action_nodes_carry_id_effect_and_retry_once():
    data = to_dict(checkout_artifact(extra_node=finish_node()))
    node = next(n for n in data["nodes"] if n["id"] == "s16")
    assert list(node) == ["id", "kind", "action", "effect", "retry_safety", "status", "outcome_code"]
    assert list(node["action"]) == ["action", "target", "value", "checkpoint", "targets", "mode"]
    assert node["effect"] == "irreversible" and node["retry_safety"] == "never_retry"
    assert "id" not in node["action"] and "risk" not in node["action"]
    assert '"risk"' not in json.dumps(data)
    assert [n["action"] for n in data["nodes"] if n["kind"] != "action"] == [None]


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
    path = save_artifact(graph, secrets=("secret_sauce",))
    assert path.name == "login_check.v1.json"
    text = path.read_text()
    assert "abc123" not in text and '"token": "[REDACTED]"' in text
    assert text == artifact_module.dumps(graph, secrets=("secret_sauce",))   # byte-stable for diffs
    loaded = load_artifact(path)
    assert loaded == replace(graph, provenance={**graph.provenance, "token": "[REDACTED]"})


def test_save_and_load_round_trip_preserves_the_contract():
    built = checkout_artifact()
    path = save_artifact(built, secrets=("secret_sauce",))
    loaded = load_artifact(path)
    assert loaded == built and loaded.version == 1
    assert loaded.nodes[4].action.target.strategies[0] == {"kind": "role", "role": "button", "name": "Add to cart",
                                                           "context": "{{product_name}}"}
    assert loaded.inputs["password"] == {"type": "string", "required": True, "sensitive": True,
                                         "description": "Value typed where 'password' was used during discovery"}
    assert loaded.outputs["total"]["type"] == "number"
    assert loaded.success == {"text_contains": "Checkout: Overview"}
    assert {o["code"] for o in loaded.outcomes} == {"dismiss_cookie_notice", "user_locked_out", "invalid_credentials",
                                                    "product_not_found", "checkout_info_missing"}
    assert json.loads(path.read_text())["schema_version"] == "2.0"


def test_versions_increment_instead_of_overwriting():
    save_artifact(checkout_artifact())
    second = checkout_artifact()
    assert second.version == 2
    path = save_artifact(second)
    assert path.name.endswith(".v2.json")
    assert sorted(p.name for p in artifact_module.ARTIFACTS_DIR.iterdir()) == [
        "checkout_review.v1.json", "checkout_review.v2.json"]


def test_save_refuses_secrets_and_never_overwrites(tmp_path):
    leaky = with_node(lambda n: setattr(n["s3"].action, "value", "secret_sauce"))
    with pytest.raises(ArtifactError, match="not parameterized"):
        save_artifact(leaky, secrets=("secret_sauce",))

    path = save_artifact(branched_graph())
    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        save_artifact(branched_graph())
    elsewhere = save_artifact(branched_graph(), path=tmp_path / "copy.json")
    assert elsewhere.read_text() == path.read_text()


def test_saved_artifact_is_plain_reviewable_json_without_run_literals():
    data = json.loads(save_artifact(checkout_artifact(), secrets=("secret_sauce",)).read_text())
    assert set(data) == {"schema_version", "name", "version", "status", "description", "surface", "inputs",
                         "outputs", "entry_node", "nodes", "edges", "outcomes", "success", "provenance"}
    assert data["status"] == "draft"
    text = json.dumps(data)
    for literal in ("secret_sauce", "standard_user", "Sauce Labs Backpack", "10001"):
        assert literal not in text, literal
    assert "transcript" not in text


# ---------- other schema versions are refused ----------

def test_unsupported_schema_versions_are_refused_with_a_clear_error(tmp_path):
    path = tmp_path / "checkout_review.v1.json"
    path.write_text(json.dumps({**to_dict(checkout_artifact()), "schema_version": "9.9"}))
    with pytest.raises(ArtifactError, match="unsupported schema_version '9.9'.*only '2.0' capability graphs"):
        load_artifact(path)
    path.write_text(json.dumps({k: v for k, v in to_dict(checkout_artifact()).items() if k != "schema_version"}))
    with pytest.raises(ArtifactError, match="unsupported schema_version None"):
        load_artifact(path)
    rejects(replace(checkout_artifact(), schema_version="9.9"), "unsupported schema_version '9.9', expected '2.0'")


def test_file_name_must_agree_with_the_capability_and_revision_inside(tmp_path):
    artifact = checkout_artifact()                                       # checkout_review, revision 1
    check_path_names_artifact(tmp_path / "checkout_review.v1.json", artifact)
    check_path_names_artifact(tmp_path / "anything.json", artifact)      # unversioned names are not checked
    with pytest.raises(ArtifactError, match="holds checkout_review v1, not checkout_review v2 as its name says"):
        check_path_names_artifact(tmp_path / "checkout_review.v2.json", artifact)
    with pytest.raises(ArtifactError, match="holds checkout_review v1, not shop_login v1"):
        check_path_names_artifact(tmp_path / "shop_login.v1.json", artifact)


def test_load_reports_malformed_json_clearly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ArtifactError, match="cannot read"):
        load_artifact(bad)


def test_malformed_json_is_rejected_with_the_location():
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
    with pytest.raises(ArtifactError, match="node 2 \\('s2'\\) action has unknown keys: \\['id', 'note'\\]"):
        from_dict(broken(lambda d: d["nodes"][2]["action"].update(id="s2", note="x")))   # unknown keys inside an action
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
    with pytest.raises(ArtifactError, match="artifact has unknown keys: \\['extra'\\]"):
        from_dict(broken(lambda d: d.update(extra=[])))


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


def test_linear_artifact_contract_errors_name_the_node():
    broken = checkout_artifact()
    broken.nodes[4].action.target.strategies[0]["context"] = "{{item}}"
    rejects(broken, "node s5: locator placeholders \\['item'\\] are not declared inputs")
    broken = checkout_artifact()
    broken.nodes[11].action.value = "shipping"
    rejects(broken, "node s12: extract writes undeclared output 'shipping'")
    broken = checkout_artifact()
    broken.outcomes.append({"code": "x", "kind": "business", "detect": {}})
    rejects(broken, "detect needs")
    broken = checkout_artifact()
    broken.outcomes.append({"code": "x", "kind": "recoverable", "detect": {"dialog_contains": "y"}})
    rejects(broken, "recover")


def test_irreversible_and_unknown_actions_are_never_retried_automatically():
    for effect in ("irreversible", "unknown"):
        for retry in ("safe", "verify_before_retry"):
            graph = with_node(lambda n: (setattr(n["s4"], "effect", effect), setattr(n["s4"], "retry_safety", retry)))
            rejects(graph, f"node s4: an action with effect '{effect}' must have retry_safety 'never_retry', "
                           f"got '{retry}'")
        validate(with_node(lambda n: (setattr(n["s4"], "effect", effect),
                                      setattr(n["s4"], "retry_safety", "never_retry"))))
    validate(with_node(lambda n: setattr(n["s4"], "retry_safety", "safe")))   # reversible may retry


def test_decision_and_terminal_nodes_carry_no_action_metadata():
    login = nodes_by_id(branched_graph())["s4"].action
    rejects(with_node(lambda n: setattr(n["d1"], "action", login)), "node d1: a decision node does not carry an action")
    rejects(with_node(lambda n: setattr(n["ok"], "action", login)), "node ok: a terminal node does not carry an action")
    rejects(with_node(lambda n: setattr(n["d1"], "status", "success")), "only terminal nodes carry a status")
    rejects(with_node(lambda n: setattr(n["d1"], "outcome_code", "x")), "only terminal nodes carry a status")
    rejects(with_node(lambda n: setattr(n["d1"], "effect", "reversible")),
            "node d1: a decision node has effect 'none', got 'reversible'")
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
    def with_guards(*guards: Guard) -> Artifact:
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


def test_always_must_be_the_only_guard_on_its_edge_and_its_nodes_last_resort():
    graph = branched_graph()
    graph.edges[6].guards.append(Guard(kind="always"))
    rejects(graph, "edge d2->locked: an 'always' guard must be the only guard on its edge")
    graph = branched_graph()
    graph.edges.append(edge("d2", "ok", priority=4))
    rejects(graph, "edge d2->ok: node d2 already has an 'always' edge to unknown_screen; only one fallback edge")
    graph = branched_graph()
    graph.edges[9].priority = -1                                # the fallback would win before any guard is judged
    rejects(graph, "edge d2->unknown_screen: an 'always' edge must have the greatest priority among the edges "
                   "leaving d2")


def test_edge_structure_reachability_and_cycles_are_checked():
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
    graph = branched_graph()
    graph.nodes.append(action("orphan", GraphAction(action="click", target=ladder("button", "Back"))))
    graph.edges.append(edge("orphan", "ok"))
    rejects(graph, "nodes \\['orphan'\\] are unreachable from entry_node 's1'")
    graph = branched_graph()
    graph.edges.append(edge("d2", "s2", Guard(kind="text_visible", value="Try again"), priority=-1))
    rejects(graph, "graph has a cycle: s2 -> s3 -> s4 -> d2 -> s2")
