"""The MemberOps demonstration of verified prefix reuse, on local Chromium with no model:

    approved reusable prefix:  sign in -> find member -> open profile
    new discovery:             reuse that prefix -> prepare a new sub-account

The prefix executes without any planner, the scripted planner's first look is the member profile, the new
artifact contains the whole path, and it replays after the reusable artifact is deleted."""
import json
from dataclasses import replace

import pytest

from src.cua import agent as agent_module
from src.cua.artifact import linear_path, save_artifact
from src.cua.escalation import NoOperator
from src.cua.lifecycle import run_replay
from src.cua.policy import Policy
from src.cua.reuse import discover_with_reuse, resolve_reuse
from tests.context import Escalator, RunLog, SessionControl
from tests.scripted_planner import ScriptedPlanner, ScriptedStep
from tests.test_member_ops_browser import prepare_script

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]
HOSTS = ["127.0.0.1"]
LOOKUP = {"username": "operator", "password": "training_only", "member_id": "1001"}
PREPARE = {**LOOKUP, "opening_deposit": "500"}


@pytest.fixture(scope="module")
def sandbox():
    from examples.member_ops.app import serve_in_thread
    server, base = serve_in_thread("normal")
    yield server.app, base + "/"
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


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 2.0)


class Spy(ScriptedPlanner):
    def __init__(self, script):
        super().__init__(script)
        self.first_screen = None

    def decide(self, goal, params, observation, history, candidates=()):
        if self.first_screen is None:
            self.first_screen = " ".join(e.text or e.name for e in observation.elements)
        return super().decide(goal, params, observation, history, candidates)


def test_member_lookup_prefix_opens_the_sub_account_discovery(sandbox, chromium, tmp_path):
    app, base = sandbox
    app.reset()
    lookup_script = prepare_script("member_id", "savings")[:5] + [ScriptedStep("done", expect="Member profile")]
    surface = chromium(headless=True, secrets=("training_only",))
    log = RunLog("discovery", secrets=("training_only",))
    try:
        lookup = agent_module.discover(goal="find a member and open the profile", name="member_profile_lookup",
                                       params=dict(LOOKUP), surface=surface, planner=ScriptedPlanner(lookup_script),
                                       policy=Policy(allowed_hosts=HOSTS),
                                       escalator=Escalator(NoOperator(), SessionControl(), log), log=log,
                                       entry_url=base, sensitive={"password"})
    finally:
        surface.close()
    assert [n.action.action for n in linear_path(lookup)] == ["navigate", "type", "type", "click", "type", "click"]
    prefix_path = save_artifact(replace(lookup, status="approved"), secrets=("training_only",))   # approved by data

    reuse = resolve_reuse("member_profile_lookup", None)
    assert reuse is not None and reuse.version == 1
    sessions = []

    def surface_factory():
        sessions.append(chromium(headless=True, secrets=("training_only",)))
        return sessions[-1]

    planner = Spy(prepare_script("member_id", "savings")[5:])        # from "New sub-account" onward
    log = RunLog("discovery", secrets=("training_only",))
    artifact = discover_with_reuse(surface_factory, reuse, goal="prepare a savings sub-account",
                                   name="member_savings_prepare", params=dict(PREPARE), planner=planner,
                                   policy=Policy(allowed_hosts=HOSTS),
                                   escalator=Escalator(NoOperator(), SessionControl(), log), log=log, entry_url=base,
                                   sensitive={"password"})
    assert len(sessions) == 1                                          # the prefix and the discovery shared it
    assert "Member profile" in planner.first_screen and "Alex Morgan" in planner.first_screen
    text = log.path.read_text()
    events = [json.loads(l) for l in text.splitlines()]
    reuse_steps = [e for e in events if e["event"] == "reuse_step_executed"]
    first_decision = next(i for i, e in enumerate(events) if e["event"] == "planner_decided")
    assert len(reuse_steps) == 6 and all(events.index(e) < first_decision for e in reuse_steps)
    assert "training_only" not in text
    assert [n.action.action for n in linear_path(artifact)] == [
        "navigate", "type", "type", "click", "type", "click", "click", "click", "type", "click", "extract", "extract",
        "extract"]
    [forced] = artifact.provenance["reuses"]
    assert forced["mode"] == "forced_prefix" and forced["source_name"] == "member_profile_lookup"
    assert forced["imported_nodes"] == 6 and artifact.provenance["planner_decisions"] == 8
    assert artifact.outputs["opening_deposit"]["pattern"] and app.created == []

    prefix_path.unlink()                                               # the reusable artifact is gone
    surface = chromium(headless=True, secrets=("training_only",))
    replay_log = RunLog("replay", secrets=("training_only",))
    try:
        result = run_replay(artifact, dict(PREPARE), surface, Policy(allowed_hosts=HOSTS),
                            Escalator(NoOperator(), SessionControl(), replay_log), replay_log, purpose="supervised")
    finally:
        surface.close()
    assert result.status == "success" and result.outputs == {"member_name": "Alex Morgan", "account_type": "Savings",
                                                             "opening_deposit": 500.0}
    assert [e["id"] for e in result.executed_path] == [f"s{i}" for i in range(1, 14)]
    assert result.performed_attempts == 10                             # navigate, 4 types, 5 clicks; 3 reads
    assert app.created == []
