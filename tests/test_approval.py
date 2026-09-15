"""Approval and the unattended gate: draft -> stability -> approved -> unattended replay."""
import json
import shutil

import pytest

from src.cua import __main__ as main_module
from src.cua import artifact as artifact_module
from src.cua import evidence as evidence_module
from src.cua.__main__ import main
from src.cua.approval import ApprovalError, approve
from src.cua.artifact import save
from src.cua.escalation import NoOperator
from src.cua.graph import from_linear, save_graph
from src.cua.graph import nodes_by_id
from src.cua.lifecycle import load_any_version, replay_any, sha256_of, write_bundle
from src.cua.merge import merge_traces
from src.cua.models import Guard, ReplayResult
from src.cua.stability import run_stability
from tests.context import HOSTS, PARAMS, Escalator, Policy, RunLog, SessionControl, checkout_artifact
from tests.fake_surface import FakeSurface
from tests.test_merge import LOGIN, link_trace, url_trace

ARGS = [f"--param={k}={v}" for k, v in PARAMS.items()] + ["--sensitive", "password", "--quiet"]
CART_PARAMS = {**LOGIN, "product_name": "Sauce Labs Backpack"}


class Session(FakeSurface):
    def __init__(self, **options):
        super().__init__(**options)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def sessions(options=None):
    return lambda secrets: Session(**(options or {}))


def draft(tmp_path, schema="1.0"):
    if schema == "1.0":
        return save(checkout_artifact(), secrets=("secret_sauce",))
    return save_graph(from_linear(checkout_artifact()), secrets=("secret_sauce",), path=tmp_path / "graph.v1.json")


def report(path, params=None, runs=3, options=None):
    return run_stability(path, params or dict(PARAMS), ["password"], runs, sessions(options), HOSTS)


def campaign_draft(tmp_path):
    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], "campaign-test")
    return save_graph(graph, secrets=("secret_sauce",), path=tmp_path / "paths.v1.json")


LINK = {**CART_PARAMS, "cart_route": "cart_link"}
URL = {**CART_PARAMS, "cart_route": "cart_url", "cart_url": "https://www.saucedemo.com/cart.html"}


# ---------- approval ----------

@pytest.mark.parametrize("schema", ["1.0", "2.0"])
def test_approval_creates_the_next_immutable_version_and_leaves_the_draft_alone(tmp_path, schema):
    path = draft(tmp_path, schema)
    before = path.read_bytes()
    report_path = report(path)
    approved_path = approve(path, [report_path], "soroush")

    assert path.read_bytes() == before                                        # the draft is untouched
    assert approved_path == artifact_module.ARTIFACTS_DIR / "checkout_review.v2.json"
    approved = load_any_version(approved_path)                                # loads and validates
    source = load_any_version(path)
    assert approved.status == "approved" and approved.version == 2 and approved.schema_version == schema
    assert approved.inputs == source.inputs and approved.outputs == source.outputs
    assert approved.success == source.success and approved.outcomes == source.outcomes
    if schema == "1.0":
        assert approved.steps == source.steps
    else:
        assert approved.nodes == source.nodes and approved.edges == source.edges
    approval = approved.provenance["approval"]
    assert approval["reviewer"] == "soroush" and approval["source_version"] == 1
    assert approval["source_sha256"] == sha256_of(path) and approval["source_path"] == str(path)
    assert approval["reports"] == [{"path": str(report_path), "sha256": sha256_of(report_path),
                                    "stability_id": report_path.parent.name, "runs_completed": 3,
                                    "selector_assignment": None}]
    assert approval["tested_selector_assignments"] == [None] and "approved_at" in approval
    assert {k: v for k, v in approved.provenance.items() if k != "approval"} == source.provenance
    assert "secret_sauce" not in approved_path.read_text()

    surface = Session()                                                       # the approved copy behaves the same
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay_any(approved, dict(PARAMS), surface, Policy(allowed_hosts=HOSTS),
                        Escalator(NoOperator(), SessionControl(), log), log, purpose="unattended")
    assert result.status == "success" and result.outputs["total"] == 32.39


def test_digest_mismatch_is_rejected(tmp_path):
    path = draft(tmp_path)
    report_path = report(path)
    other = tmp_path / "edited.v1.json"                                       # the same capability, one byte apart
    other.write_text(path.read_text().replace('"status": "draft"', '"status":  "draft"'))
    with pytest.raises(ApprovalError, match="stale or about another artifact"):
        approve(other, [report_path], "soroush")
    assert not (artifact_module.ARTIFACTS_DIR / "checkout_review.v2.json").exists()


def test_ineligible_and_insufficient_reports_are_rejected(tmp_path):
    path = draft(tmp_path)
    failing = report(path, options={"faults": ["verification"]})
    with pytest.raises(ApprovalError, match=r"not eligible for approval \(not every run returned success; a run "
                                            r"required human intervention\)"):
        approve(path, [failing], "soroush")
    short = report(path, runs=2)
    with pytest.raises(ApprovalError, match="only 2 of the required 3 runs completed"):
        approve(path, [short], "soroush")
    assert not (artifact_module.ARTIFACTS_DIR / "checkout_review.v2.json").exists()


def test_malformed_missing_and_foreign_reports_are_rejected(tmp_path):
    path = draft(tmp_path)
    good = json.loads(report(path).read_text())

    def broken(mutate, name):
        data = json.loads(json.dumps(good))
        mutate(data)
        target = tmp_path / name
        target.write_text(json.dumps(data))
        return target

    with pytest.raises(ApprovalError, match="cannot read report"):
        approve(path, [tmp_path / "nowhere.json"], "soroush")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ApprovalError, match="cannot read report"):
        approve(path, [bad], "soroush")
    with pytest.raises(ApprovalError, match=r"missing keys \['runs'\]"):
        approve(path, [broken(lambda d: d.pop("runs"), "keys.json")], "soroush")
    with pytest.raises(ApprovalError, match="unsupported report_schema_version"):
        approve(path, [broken(lambda d: d.update(report_schema_version="0.1"), "ver.json")], "soroush")
    with pytest.raises(ApprovalError, match="is about checkout_review v7, not checkout_review v1"):
        approve(path, [broken(lambda d: d["artifact"].update(version=7), "v7.json")], "soroush")
    with pytest.raises(ApprovalError, match="at least one stability report"):
        approve(path, [], "soroush")
    with pytest.raises(ApprovalError, match="reviewer name is required"):
        approve(path, [tmp_path / "x"], "  ")


def test_only_a_draft_can_be_approved(tmp_path):
    path = draft(tmp_path)
    approved_path = approve(path, [report(path)], "soroush")
    with pytest.raises(ApprovalError, match="has status 'approved'; only a draft can be approved"):
        approve(approved_path, [report(approved_path)], "soroush")


def test_campaign_graphs_need_one_eligible_report_per_selector_assignment(tmp_path):
    path = campaign_draft(tmp_path)
    link_report, url_report = report(path, LINK), report(path, URL)
    with pytest.raises(ApprovalError, match=r"no stability report for selector assignment\(s\) "
                                            r"\[\{'cart_route': 'cart_url'\}\]"):
        approve(path, [link_report], "soroush")
    with pytest.raises(ApprovalError, match="two reports cover the same selector assignment"):
        approve(path, [link_report, link_report, url_report], "soroush")
    foreign = tmp_path / "foreign.json"
    data = json.loads(url_report.read_text())
    data["selector_assignment"] = {"cart_route": "teleport"}
    foreign.write_text(json.dumps(data))
    with pytest.raises(ApprovalError, match=r"undeclared selector assignment \{'cart_route': 'teleport'\}"):
        approve(path, [link_report, foreign], "soroush")

    approved_path = approve(path, [link_report, url_report], "soroush")
    approved = load_any_version(approved_path)
    assert approved.status == "approved" and approved.version == 2     # always later than its source
    assert approved.provenance["approval"]["tested_selector_assignments"] == [{"cart_route": "cart_link"},
                                                                              {"cart_route": "cart_url"}]
    assert approved.provenance["scenarios"] == load_any_version(path).provenance["scenarios"]
    for params in (LINK, URL):
        log = RunLog("replay", secrets=("secret_sauce",))
        result = replay_any(approved, dict(params), Session(), Policy(allowed_hosts=HOSTS),
                            Escalator(NoOperator(), SessionControl(), log), log, purpose="unattended")
        assert result.status == "success"


# ---------- reports are recomputed, not trusted ----------

def edited_report(tmp_path, path, mutate, name="edited.json"):
    data = json.loads(path.read_text())
    mutate(data)
    target = tmp_path / name
    target.write_text(json.dumps(data))
    return target


def test_flipping_the_eligibility_flag_is_caught(tmp_path):
    path = draft(tmp_path)
    failing = report(path, options={"faults": ["verification"]})
    flipped = edited_report(tmp_path, failing, lambda d: d.update(eligible_for_approval=True, ineligible_reasons=[]))
    with pytest.raises(ApprovalError, match="inconsistent and cannot be trusted: eligible_for_approval does not "
                                            "follow from the run records: True vs False"):
        approve(path, [flipped], "soroush")


def test_editing_run_records_without_the_totals_is_caught(tmp_path):
    path = draft(tmp_path)
    failing = report(path, options={"faults": ["verification"]})

    def promote_run(d):                     # a failed run rewritten as a success, totals left alone
        d["runs"][0].update(status="success", outcome_code=None, interventions=0, clean=True)
        d["eligible_for_approval"], d["ineligible_reasons"] = True, []
    with pytest.raises(ApprovalError, match="status_counts does not follow from the run records"):
        approve(path, [edited_report(tmp_path, failing, promote_run)], "soroush")

    good = report(path)

    def hide_intervention(d):               # the intervention count of one run lowered by hand
        d["runs"][1]["interventions"] = 3
    with pytest.raises(ApprovalError, match="run record 2: clean flag does not follow"):
        approve(path, [edited_report(tmp_path, good, hide_intervention)], "soroush")

    def other_digest(d):
        d["runs"][2]["artifact_sha256"] = "0" * 64
    with pytest.raises(ApprovalError, match="run record 3: artifact digest '0000"):
        approve(path, [edited_report(tmp_path, good, other_digest)], "soroush")

    def inflate_rate(d):
        d["clean_run_rate"] = 0.5
    with pytest.raises(ApprovalError, match="clean_run_rate does not follow from the run records: 0.5 vs 1.0"):
        approve(path, [edited_report(tmp_path, good, inflate_rate)], "soroush")

    def wrong_summary(d):
        d["outcome_counts"] = {"success": 2}
    with pytest.raises(ApprovalError, match="outcome_counts does not follow"):
        approve(path, [edited_report(tmp_path, good, wrong_summary)], "soroush")

    def fewer_requested(d):
        d["runs_requested"] = 2
    with pytest.raises(ApprovalError, match="runs_requested 2 does not match 3 run records"):
        approve(path, [edited_report(tmp_path, good, fewer_requested)], "soroush")

    def wrong_policy(d):
        d["irreversible_policy"] = "allow"
    with pytest.raises(ApprovalError, match="irreversible_policy must be 'deny'"):
        approve(path, [edited_report(tmp_path, good, wrong_policy)], "soroush")

    def bad_record(d):
        d["runs"][0].pop("drift_signals")
    with pytest.raises(ApprovalError, match="run record 1 must carry exactly the keys"):
        approve(path, [edited_report(tmp_path, good, bad_record)], "soroush")

    assert not (artifact_module.ARTIFACTS_DIR / "checkout_review.v2.json").exists()
    approve(path, [good], "soroush")                                          # untouched: still approves
    assert (artifact_module.ARTIFACTS_DIR / "checkout_review.v2.json").exists()


# ---------- coverage comes from the graph ----------

def saved(tmp_path, graph, name):
    from src.cua.graph import save_graph
    return save_graph(graph, secrets=("secret_sauce",), path=tmp_path / name)


def test_stripping_provenance_does_not_lower_required_coverage(tmp_path):
    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], "campaign-test")
    graph.provenance = {"run_id": "hand-edited"}
    path = saved(tmp_path, graph, "stripped.json")
    link_report, url_report = report(path, LINK), report(path, URL)
    with pytest.raises(ApprovalError, match=r"no stability report for selector assignment\(s\) "
                                            r"\[\{'cart_route': 'cart_url'\}\]"):
        approve(path, [link_report], "soroush")
    approved = load_any_version(approve(path, [link_report, url_report], "soroush"))
    assert approved.provenance["approval"]["tested_selector_assignments"] == [{"cart_route": "cart_link"},
                                                                              {"cart_route": "cart_url"}]


def test_provenance_that_disagrees_with_the_graph_is_rejected(tmp_path):
    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], "campaign-test")
    graph.provenance["scenarios"][1]["selectors"] = {"cart_route": "teleport"}
    path = saved(tmp_path, graph, "lying.json")
    with pytest.raises(ApprovalError, match="provenance records selector assignments .* but the entry gate admits"):
        approve(path, [report(path, LINK), report(path, URL)], "soroush")

    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], "campaign-test")
    graph.provenance["selectors"] = ["cart_route", "zip_source"]
    path = saved(tmp_path, graph, "lying2.json")
    with pytest.raises(ApprovalError, match=r"provenance lists selectors \['cart_route', 'zip_source'\]"):
        approve(path, [report(path, LINK), report(path, URL)], "soroush")


def test_a_malformed_entry_gate_is_rejected(tmp_path):
    def gated(mutate, name):
        graph = merge_traces([link_trace(), url_trace()], ["cart_route"], "campaign-test")
        graph.provenance = {"run_id": "hand-edited"}
        mutate(graph)
        return saved(tmp_path, graph, name)

    def text_guard(graph):
        graph.edges[0].guards = [Guard(kind="text_visible", value="Swag Labs")]
    path = gated(text_guard, "text.json")
    with pytest.raises(ApprovalError, match=r"entry gate edge d1->s1 \(priority 0\): only input_equals guards"):
        approve(path, [report(path, LINK), report(path, URL)], "soroush")

    def doubled(graph):
        graph.edges[0].guards.append(Guard(kind="input_equals", input="cart_route", value="cart_link"))
    path = gated(doubled, "double.json")
    with pytest.raises(ApprovalError, match="a selector is guarded twice"):
        approve(path, [report(path, LINK), report(path, URL)], "soroush")

    def extra_selector(graph):
        graph.inputs["username"]["selector"] = True
    path = gated(extra_selector, "extra.json")
    with pytest.raises(ApprovalError, match=r"do not cover exactly the selectors \['username', 'cart_route'\]"):
        approve(path, [report(path, LINK), report(path, URL)], "soroush")

    linear = from_linear(checkout_artifact())
    linear.inputs["username"]["selector"] = True
    path = saved(tmp_path, linear, "nogate.json")
    with pytest.raises(ApprovalError, match="entry node 's1' is not a decision gate"):
        approve(path, [report(path)], "soroush")


# ---------- the lifecycle boundary ----------

def run_direct(loaded, purpose, params=None):
    surface = Session()
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay_any(loaded, dict(params or PARAMS), surface, Policy(allowed_hosts=HOSTS),
                        Escalator(NoOperator(), SessionControl(), log), log, purpose=purpose)
    return result, surface, log


@pytest.mark.parametrize("schema", ["1.0", "2.0"])
def test_replay_any_enforces_the_purpose(tmp_path, schema):
    path = draft(tmp_path, schema)
    loaded = load_any_version(path)
    result, surface, log = run_direct(loaded, "unattended")
    assert result.status == "failure" and result.outcome_code == "artifact_not_approved"
    assert surface.actions == [] and '"replay_blocked"' in log.path.read_text()
    for purpose in ("supervised", "stability"):
        result, surface, _ = run_direct(loaded, purpose)
        assert result.status == "success" and surface.screen == "overview", purpose
    approved = load_any_version(approve(path, [report(path)], "soroush"))
    result, surface, _ = run_direct(approved, "unattended")
    assert result.status == "success" and surface.screen == "overview"
    with pytest.raises(ValueError, match="purpose must be one of"):
        run_direct(loaded, "production")
    with pytest.raises(TypeError):
        replay_any(loaded, dict(PARAMS), Session(), Policy(allowed_hosts=HOSTS),
                   Escalator(NoOperator(), SessionControl(), RunLog("replay")), RunLog("replay"))


def test_stability_purpose_forces_deny(tmp_path):
    from tests.test_stability import finish_step
    finish = from_linear(checkout_artifact(extra_step=finish_step()))
    finish.success = {"text_contains": "Thank you"}
    surface = Session()
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay_any(finish, dict(PARAMS), surface, Policy(allowed_hosts=HOSTS),
                        Escalator(NoOperator(), SessionControl(), log), log, purpose="stability",
                        irreversible_policy="allow")
    assert result.outcome_code == "irreversible_denied" and ("click", "Finish", "") not in surface.actions


# ---------- persisted results are redacted ----------

def test_persisted_result_is_redacted_but_the_caller_keeps_the_real_one(tmp_path):
    path = draft(tmp_path)
    loaded = load_any_version(path)
    result = ReplayResult(status="failure", outputs={"total": 32.39, "note": "typed secret_sauce"},
                          outcome_code="checkpoint_not_met", step_id="s4", expected="text secret_sauce gone",
                          observed="screen shows secret_sauce", recoveries=["s1: typed secret_sauce again"],
                          interventions=[{"kind": "stuck", "note": "operator retyped secret_sauce",
                                          "human_actions": [{"action": "type", "value": "secret_sauce"}]}])
    before = json.dumps(result.__dict__, sort_keys=True)
    log = RunLog("replay", secrets=("secret_sauce",))
    log.event("probe")
    bundle = write_bundle(log, loaded, path, result, dict(PARAMS), ["password"])
    persisted = (bundle / "result.json").read_text()
    assert "secret_sauce" not in persisted and "[REDACTED]" in persisted
    data = json.loads(persisted)
    assert data["outputs"]["total"] == 32.39 and data["outcome_code"] == "checkpoint_not_met"
    assert data["interventions"][0]["human_actions"][0]["value"] == "[REDACTED]"
    assert json.dumps(result.__dict__, sort_keys=True) == before                # in memory: untouched
    assert result.observed == "screen shows secret_sauce"
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["sha256"] == sha256_of(path) and manifest["result_status"] == "failure"


def test_approval_never_launches_a_surface(monkeypatch, tmp_path, capsys):
    def no_browser(**_):
        raise AssertionError("approval must not open a browser")

    path = draft(tmp_path)
    report_path = report(path)
    monkeypatch.setattr(main_module, "PlaywrightSurface", no_browser)
    assert main(["approve", "--artifact", str(path), "--report", str(report_path), "--reviewer", "soroush"]) == 0
    assert "Approved:" in capsys.readouterr().out
    assert main(["approve", "--artifact", str(path), "--report", str(report_path), "--reviewer", "soroush"]) == 0
    assert (artifact_module.ARTIFACTS_DIR / "checkout_review.v3.json").exists()     # each approval is a new version
    assert main(["approve", "--artifact", str(tmp_path / "none.json"), "--report", str(report_path),
                 "--reviewer", "soroush"]) == 2
    assert "Approval refused" in capsys.readouterr().out


# ---------- the unattended gate ----------

def bundles():
    return sorted(evidence_module.EVIDENCE_DIR.glob("replay-*"))


@pytest.mark.parametrize("schema", ["1.0", "2.0"])
def test_unattended_draft_replay_is_blocked_before_any_surface_exists(monkeypatch, tmp_path, capsys, schema):
    created = []
    monkeypatch.setattr(main_module, "PlaywrightSurface", lambda **_: created.append(Session()) or created[-1])
    path = draft(tmp_path, schema)
    assert main(["replay", "--artifact", str(path), "--operator", "none", *ARGS]) == 2
    out = capsys.readouterr().out
    assert "Refusing unattended replay" in out and '"outcome_code": "artifact_not_approved"' in out
    assert created == []                                                      # no browser, no action

    [bundle] = bundles()
    result = json.loads((bundle / "result.json").read_text())
    assert result["status"] == "failure" and result["outcome_code"] == "artifact_not_approved"
    assert "--operator console" in result["observed"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest == {**manifest, "artifact_path": str(path), "artifact_file": path.name, "sha256": sha256_of(path),
                        "capability": "checkout_review", "capability_version": 1, "schema_version": schema,
                        "status": "draft", "result_status": "failure", "outcome_code": "artifact_not_approved",
                        "param_names": sorted(PARAMS), "sensitive_params": ["password"]}
    assert "secret_sauce" not in (bundle / "manifest.json").read_text() and "standard_user" not in json.dumps(manifest)
    assert (bundle / path.name).read_bytes() == path.read_bytes()
    assert '"replay_blocked"' in (bundle / "run.jsonl").read_text()


@pytest.mark.parametrize("schema", ["1.0", "2.0"])
def test_supervised_draft_replay_runs_with_a_warning(monkeypatch, tmp_path, capsys, schema):
    created = []
    monkeypatch.setattr(main_module, "PlaywrightSurface", lambda **_: created.append(Session()) or created[-1])
    path = draft(tmp_path, schema)
    assert main(["replay", "--artifact", str(path), "--operator", "console", *ARGS]) == 0
    out = capsys.readouterr().out
    assert "WARNING: checkout_review v1 has status 'draft'" in out and '"status": "success"' in out
    assert len(created) == 1 and created[0].closed and created[0].screen == "overview"
    [bundle] = bundles()
    assert '"draft_warning"' in (bundle / "run.jsonl").read_text()
    assert json.loads((bundle / "manifest.json").read_text())["status"] == "draft"


@pytest.mark.parametrize("schema", ["1.0", "2.0"])
def test_approved_artifact_replays_unattended(monkeypatch, tmp_path, capsys, schema):
    created = []
    monkeypatch.setattr(main_module, "PlaywrightSurface", lambda **_: created.append(Session()) or created[-1])
    path = draft(tmp_path, schema)
    approved_path = approve(path, [report(path)], "soroush")
    assert main(["replay", "--artifact", str(approved_path), "--operator", "none", *ARGS]) == 0
    out = capsys.readouterr().out
    assert "WARNING" not in out and '"status": "success"' in out
    assert len(created) == 1 and created[0].closed
    bundle = [b for b in bundles() if (b / "manifest.json").exists()
              and json.loads((b / "manifest.json").read_text())["status"] == "approved"]
    assert len(bundle) == 1 and json.loads((bundle[0] / "manifest.json").read_text())["capability_version"] == 2
    assert (bundle[0] / approved_path.name).read_bytes() == approved_path.read_bytes()
