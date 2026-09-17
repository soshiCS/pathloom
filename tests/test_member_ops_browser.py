"""MemberOps through Pathloom's real browser adapter, no network and no model.

A scripted planner discovers the three prepare paths and the separate side-effecting capability on the
local sandbox; the graphs then replay in every runtime mode: business outcomes, a recovered notice, a
transient load, the persistent unexpected dialog resolved through the console-operator seam, a session
expiry resolved by `restart` alone, and the effect policy on the irreversible action. Skips when Chromium
is unavailable. This is the check worth running before spending API credits.
"""
import json

import pytest

from src.cua.campaign import load_spec, run_campaign
from src.cua.artifact import outgoing
from src.cua.escalation import NoOperator
from src.cua.lifecycle import run_replay
from src.cua.policy import Policy
from tests.context import Escalator, RecordingOperator, RunLog, SessionControl
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]
HOSTS = ["127.0.0.1"]
CREDENTIALS = {"username": "operator", "password": "training_only"}
PREPARE = {**CREDENTIALS, "lookup_method": "member_id", "member_id": "1001", "account_type": "savings",
           "opening_deposit": "500"}
OPEN = {**CREDENTIALS, "member_id": "1001", "opening_deposit": "500"}


@pytest.fixture(scope="module")
def sandbox():
    """One server whose mode the tests switch; state is reset before every replay."""
    from examples.member_ops.app import serve_in_thread
    server, base = serve_in_thread("normal")
    yield server.app, base
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def chromium():
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        PlaywrightSurface(headless=True).close()
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    return PlaywrightSurface


@pytest.fixture(scope="module")
def isolated_dirs(tmp_path_factory):
    """Module-scoped isolation: the function-scoped conftest fixture runs too late for a module fixture."""
    from src.cua import artifact, evidence
    root = tmp_path_factory.mktemp("member_ops_campaign")
    patch = pytest.MonkeyPatch()
    patch.setattr(artifact, "ARTIFACTS_DIR", root / "artifacts")
    patch.setattr(evidence, "LOGS_DIR", root / "logs")
    patch.setattr(evidence, "EVIDENCE_DIR", root / "evidence")
    yield root
    patch.undo()


def prepare_script(lookup_method: str, account_type: str) -> list[ScriptedStep]:
    """The decisions a model would make, bound to real perceived controls at decision time."""
    lookup = ([ScriptedStep("click", "link", "By phone", expect="Phone number"),
               ScriptedStep("type", "textbox", "Phone number", value="{{phone}}")]
              if lookup_method == "phone" else
              [ScriptedStep("type", "textbox", "Member ID", value="{{member_id}}")])
    label = "Savings" if account_type == "savings" else "Checking"
    return [
        ScriptedStep("type", "textbox", "Username", value="{{username}}"),
        ScriptedStep("type", "textbox", "Password", value="{{password}}"),
        ScriptedStep("click", "button", "Sign in", expect="Member lookup"),
        *lookup,
        ScriptedStep("click", "button", "Search", expect="Member profile"),
        ScriptedStep("click", "link", "New sub-account", expect="Choose the account type"),
        ScriptedStep("click", "link", label, expect="Opening deposit"),
        ScriptedStep("type", "textbox", "Opening deposit (USD)", value="{{opening_deposit}}"),
        ScriptedStep("click", "button", "Continue to review", expect="Review new sub-account"),
        ScriptedStep("extract", "text", text="Member name:", output_name="member_name",
                     pattern=r"Member name:\s*(.+)"),
        ScriptedStep("extract", "text", text="Account type:", output_name="account_type",
                     pattern=r"Account type:\s*(\w+)"),
        ScriptedStep("extract", "text", text="Opening deposit:", output_name="opening_deposit",
                     pattern=r"Opening deposit:\s*\$\s*([\d.]+)"),
        ScriptedStep("done", expect="Review new sub-account"),
    ]


def open_script() -> list[ScriptedStep]:
    """The side-effecting capability: the same route, then the final click (approved by the operator)."""
    return prepare_script("member_id", "savings")[:-4] + [
        ScriptedStep("click", "button", "Confirm and Open Account", expect="Sub-account opened"),
        ScriptedStep("extract", "text", text="Account ", output_name="account_number",
                     pattern=r"Account ([A-Z]+-\d+) \("),
        ScriptedStep("extract", "text", text="Account ", output_name="opening_deposit",
                     pattern=r"opening deposit of \$\s*(\d+\.\d{2})"),
        ScriptedStep("done", expect="Sub-account opened"),
    ]


def discover_campaign(spec_path: str, base: str, chromium, scripts, operator=None):
    spec = load_spec(spec_path)
    spec.url = base + "/"

    def surface_factory(secrets):
        return chromium(headless=True, secrets=secrets)

    return run_campaign(spec, surface_factory, lambda scenario: ScriptedPlanner(scripts(scenario)),
                        operator or NoOperator(), HOSTS, spec_path=spec_path)


@pytest.fixture(scope="module")
def prepare(sandbox, chromium, isolated_dirs):
    app, base = sandbox
    app.reset()
    result = discover_campaign("scenarios/member_account_prepare.json", base, chromium,
                               lambda s: prepare_script(s.params["lookup_method"], s.params["account_type"]))
    assert str(isolated_dirs) in result.artifact_path            # nothing may land in the real artifacts/
    assert app.created == []                                      # the safe capability creates nothing
    return result.graph


@pytest.fixture(scope="module")
def opener(sandbox, chromium, isolated_dirs):
    """The side-effecting capability, discovered supervised: the operator approves the final click."""
    app, base = sandbox
    app.mode = "normal"
    app.reset()
    approving = RecordingOperator(disposition="approve")
    result = discover_campaign("scenarios/member_account_open.json", base, chromium, lambda s: open_script(),
                               operator=approving)
    assert approving.requests[0].kind == "confirm" and "Confirm and Open Account" in approving.requests[0].reason
    assert len(app.created) == 1                                  # discovery really opened one account
    app.reset()
    return result.graph


def replay(app, chromium, graph, params, mode="normal", operator=None, irreversible="confirm"):
    app.mode = mode
    app.reset()
    surface = chromium(headless=True, secrets=("training_only",))
    log = RunLog("replay", secrets=("training_only",))
    try:
        result = run_replay(graph, dict(params), surface, Policy(allowed_hosts=HOSTS),
                            Escalator(operator or NoOperator(), SessionControl(), log), log, purpose="supervised",
                            irreversible_policy=irreversible)
    finally:
        surface.close()
        app.mode = "normal"
    return result, log


# ---------- the safe prepare capability ----------

def test_scripted_campaign_discovers_all_three_paths_with_the_declared_contract(prepare):
    graph = prepare
    assert [s["name"] for s in graph.provenance["scenarios"]] == ["member_id_savings", "phone_savings",
                                                                  "member_id_checking"]
    assert graph.outputs["opening_deposit"]["pattern"] == r"Opening deposit:\s*\$\s*([\d.]+)"
    assert [len(outgoing(graph, graph.entry_node))] == [3]
    assert graph.inputs["phone"]["required_when"] == [{"lookup_method": "phone", "account_type": "savings"}]
    assert graph.inputs["member_id"]["required"] is False and graph.inputs["password"]["sensitive"] is True
    assert [o["code"] for o in graph.outcomes] == ["invalid_credentials", "member_not_found", "member_access_denied",
                                                   "invalid_opening_deposit", "dismiss_maintenance_notice"]
    assert not any(n.kind == "action" and n.action.target and "Confirm" in json.dumps(n.action.target.strategies)
                   for n in graph.nodes)


@pytest.mark.parametrize("params, expected_type, expected_deposit", [
    ({"lookup_method": "member_id", "member_id": "1001", "account_type": "savings", "opening_deposit": "500"},
     "Savings", 500.0),
    ({"lookup_method": "phone", "phone": "5550101", "account_type": "savings", "opening_deposit": "500"},
     "Savings", 500.0),
    ({"lookup_method": "member_id", "member_id": "1001", "account_type": "checking", "opening_deposit": "250"},
     "Checking", 250.0),
])
def test_each_declared_path_replays_and_creates_nothing(sandbox, chromium, prepare, params, expected_type,
                                                        expected_deposit):
    app, _ = sandbox
    result, log = replay(app, chromium, prepare, {**CREDENTIALS, **params})
    assert result.status == "success", result
    assert result.outputs == {"member_name": "Alex Morgan", "account_type": expected_type,
                              "opening_deposit": expected_deposit}
    assert app.created == [] and "training_only" not in log.path.read_text()


def test_undeclared_phone_checking_stops_at_the_gate(sandbox, chromium, prepare):
    app, _ = sandbox
    params = {**CREDENTIALS, "lookup_method": "phone", "phone": "5550101", "account_type": "checking",
              "opening_deposit": "250"}
    result, log = replay(app, chromium, prepare, params)
    assert result.status == "failure" and result.outcome_code == "no_matching_edge"
    assert result.step_id == prepare.entry_node and '"acted"' not in log.path.read_text()


@pytest.mark.parametrize("change, code", [
    ({"password": "wrong"}, "invalid_credentials"),
    ({"member_id": "4040"}, "member_not_found"),
    ({"member_id": "1002"}, "member_access_denied"),
    ({"opening_deposit": "10"}, "invalid_opening_deposit"),
])
def test_declared_business_outcomes_classify_correctly(sandbox, chromium, prepare, change, code):
    app, _ = sandbox
    result, _ = replay(app, chromium, prepare, {**PREPARE, **change})
    assert (result.status, result.outcome_code) == ("business_outcome", code), result
    assert result.interventions == [] and app.created == []


def test_known_maintenance_notice_is_recovered_not_reported(sandbox, chromium, prepare):
    app, _ = sandbox
    result, log = replay(app, chromium, prepare, PREPARE, mode="known_interstitial")
    assert result.status == "success" and result.outputs["opening_deposit"] == 500.0
    assert len(result.recoveries) == 1 and result.recoveries[0].endswith(": dismiss_maintenance_notice")
    assert result.interventions == [] and '"dismiss_maintenance_notice"' in log.path.read_text()


def test_slow_once_is_one_transient_recovery(sandbox, chromium, prepare):
    app, _ = sandbox
    result, _ = replay(app, chromium, prepare, PREPARE, mode="slow_once")
    assert result.status == "success" and result.interventions == []
    assert len(result.recoveries) == 1 and "transient error" in result.recoveries[0]


def test_unexpected_dialog_needs_the_operator_who_acknowledges_on_the_same_session(sandbox, chromium, prepare):
    app, _ = sandbox
    unattended, _ = replay(app, chromium, prepare, PREPARE, mode="unexpected_dialog")
    assert unattended.status == "failure" and unattended.outcome_code == "unknown_dialog"   # nobody to help

    operator = RecordingOperator(disposition="resume", manual=[("click", "Acknowledge")])
    result, log = replay(app, chromium, prepare, PREPARE, mode="unexpected_dialog", operator=operator)
    assert result.status == "success" and result.outputs["member_name"] == "Alex Morgan"
    [intervention] = result.interventions
    assert intervention["kind"] == "stuck" and intervention["disposition"] == "resume"
    assert intervention["human_actions"] == [{"action": "click", "target": "Acknowledge"}]
    text = log.path.read_text()
    assert '"to": "human"' in text and '"to": "automation"' in text and '"step_already_satisfied"' in text
    assert operator.requests[0].goal.startswith("Sign in to MemberOps")


def test_session_expiry_is_resolved_by_restart_alone(sandbox, chromium, prepare):
    app, _ = sandbox
    operator = RecordingOperator(disposition="restart")                # no manual sign-in first
    result, log = replay(app, chromium, prepare, PREPARE, mode="session_expired_once", operator=operator)
    assert result.status == "success" and result.outputs["opening_deposit"] == 500.0
    assert [i["disposition"] for i in result.interventions] == ["restart"]
    assert '"flow_restarted"' in log.path.read_text() and app.created == []


# ---------- the separate side-effecting capability ----------

def test_open_capability_records_the_final_click_as_irreversible(opener):
    final = next(n for n in opener.nodes if n.kind == "action" and n.action.target
                 and n.action.target.strategies[0].get("name") == "Confirm and Open Account")
    assert (final.effect, final.retry_safety) == ("irreversible", "never_retry")
    assert opener.outputs["account_number"]["type"] == "string" and opener.entry_node == "s1"


def test_effect_policy_on_the_irreversible_action(sandbox, chromium, opener):
    app, _ = sandbox
    denied, _ = replay(app, chromium, opener, OPEN, irreversible="deny")
    assert denied.outcome_code == "irreversible_denied" and app.created == []

    nobody, _ = replay(app, chromium, opener, OPEN, irreversible="confirm")
    assert nobody.outcome_code == "irreversible_not_confirmed" and app.created == []

    refused, _ = replay(app, chromium, opener, OPEN, irreversible="confirm", operator=RecordingOperator("deny"))
    assert refused.outcome_code == "irreversible_not_confirmed" and app.created == []

    approved, _ = replay(app, chromium, opener, OPEN, irreversible="confirm", operator=RecordingOperator("approve"))
    assert approved.status == "success" and approved.outputs["account_number"] == "SAV-9001"
    assert approved.outputs["opening_deposit"] == 500.0 and len(app.created) == 1

    allowed, _ = replay(app, chromium, opener, OPEN, irreversible="allow")
    assert allowed.status == "success" and allowed.interventions == [] and len(app.created) == 1


def test_uncertain_result_after_the_irreversible_action_is_never_repeated(sandbox, chromium, opener):
    app, _ = sandbox
    result, log = replay(app, chromium, opener, OPEN, mode="slow_after_open_once", irreversible="allow")
    assert result.status == "failure" and result.outcome_code == "action_result_uncertain"
    assert len(app.created) == 1                                        # opened exactly once, never repeated
    assert '"action_repeated"' not in log.path.read_text() and result.interventions[0]["kind"] == "stuck"


def test_open_capability_classifies_a_restricted_member_without_acting(sandbox, chromium, opener):
    app, _ = sandbox
    result, log = replay(app, chromium, opener, {**OPEN, "member_id": "1002"}, irreversible="allow")
    assert (result.status, result.outcome_code) == ("business_outcome", "member_access_denied"), result
    assert app.created == [] and "Confirm and Open Account" not in log.path.read_text().split('"acted"')[-1]
    assert all("Confirm" not in line for line in log.path.read_text().splitlines() if '"acted"' in line)


def test_open_capability_recovers_the_maintenance_notice_and_opens_exactly_once(sandbox, chromium, opener):
    app, _ = sandbox
    result, _ = replay(app, chromium, opener, OPEN, mode="known_interstitial", irreversible="allow")
    assert result.status == "success" and result.outputs["account_number"] == "SAV-9001"
    assert len(result.recoveries) == 1 and result.recoveries[0].endswith(": dismiss_maintenance_notice")
    assert result.interventions == [] and len(app.created) == 1
