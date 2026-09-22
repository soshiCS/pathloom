"""The MemberOps demonstration of automatic verified composition, on local Chromium with no model:

    approved artifact A:  sign in                         (member_login)
    approved artifact B:  sign in -> find member -> open profile  (member_profile_lookup)
    new discovery:        prepare a Savings sub-account

Nobody names A or B. On the sign-in screen the library offers A; after A the screen matches B's internal
checkpoint, so only B's suffix (find member, open profile) is offered and run; the planner's first
ordinary decision is the genuinely new part. The produced artifact replays after A and B are deleted."""
import json
from dataclasses import replace

import pytest

from src.cua import agent as agent_module
from src.cua.artifact import linear_path, save_artifact
from src.cua.escalation import NoOperator
from src.cua.lifecycle import run_replay
from src.cua.policy import Policy
from src.cua.reuse import discover_with_reuse
from tests.context import Escalator, RunLog, SessionControl
from tests.scripted_planner import ScriptedPlanner, ScriptedStep
from tests.test_member_ops_browser import prepare_script

pytestmark = [pytest.mark.browser, pytest.mark.filterwarnings("ignore::ResourceWarning")]
HOSTS = ["127.0.0.1"]
LOGIN = {"username": "operator", "password": "training_only"}
LOOKUP = {**LOGIN, "member_id": "1001"}
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
        self.decisions: list[tuple] = []

    def decide(self, goal, params, observation, history, candidates=()):
        action = super().decide(goal, params, observation, history, candidates)
        self.decisions.append((action.kind, action.candidate_id, [c.candidate_id for c in candidates],
                               " ".join(e.text or e.name for e in observation.elements)[:200]))
        return action


def discover_approved(chromium, base, name, goal, params, script):
    surface = chromium(headless=True, secrets=("training_only",))
    log = RunLog("discovery", secrets=("training_only",))
    try:
        built = agent_module.discover(goal=goal, name=name, params=dict(params), surface=surface,
                                      planner=ScriptedPlanner(script), policy=Policy(allowed_hosts=HOSTS),
                                      escalator=Escalator(NoOperator(), SessionControl(), log), log=log,
                                      entry_url=base, sensitive={"password"})
    finally:
        surface.close()
    return save_artifact(replace(built, status="approved"), secrets=("training_only",))   # approved by data


def test_login_and_member_lookup_are_composed_automatically_before_the_new_part(sandbox, chromium):
    app, base = sandbox
    app.reset()
    steps = prepare_script("member_id", "savings")
    login_path = discover_approved(chromium, base, "member_login", "sign in to MemberOps", LOGIN,
                                   steps[:3] + [ScriptedStep("done", expect="Member lookup")])
    lookup_path = discover_approved(chromium, base, "member_profile_lookup", "find a member and open the profile",
                                    LOOKUP, steps[:5] + [ScriptedStep("done", expect="Member profile")])
    sessions = []

    def surface_factory():
        sessions.append(chromium(headless=True, secrets=("training_only",)))
        return sessions[-1]

    planner = Spy([ScriptedStep("reuse", name="member_login")] * 3                 # type, type, click Sign in
                  + [ScriptedStep("reuse", name="member_profile_lookup")] * 2       # type Member ID, click Search
                  + steps[5:])
    log = RunLog("discovery", secrets=("training_only",))
    artifact = discover_with_reuse(surface_factory, None, goal="prepare a savings sub-account",
                                   name="member_savings_prepare", params=dict(PREPARE), planner=planner,
                                   policy=Policy(allowed_hosts=HOSTS),
                                   escalator=Escalator(NoOperator(), SessionControl(), log), log=log, entry_url=base,
                                   sensitive={"password"})
    assert len(sessions) == 1                                            # everything happened on one live session

    # Both artifacts were found without being named; each verified action ran on its own, the planner asked
    # again after every one, and none of it before the planner invented anything new.
    kinds = [d[0] for d in planner.decisions]
    assert kinds[:6] == ["reuse_candidate"] * 5 + ["click"]
    first, fourth, sixth = planner.decisions[0], planner.decisions[3], planner.decisions[5]
    assert first[1] == "member_login.v1@s2" and "Sign in" in first[3]
    assert sorted(first[2]) == [f"member_login.v1@s{i}" for i in (2, 3, 4)] + [f"member_profile_lookup.v1@s{i}"
                                                                               for i in (2, 3, 4)]
    assert fourth[1] == "member_profile_lookup.v1@s5"
    assert fourth[2] == ["member_profile_lookup.v1@s5", "member_profile_lookup.v1@s6"]   # each action on its own
    assert "Member lookup" in fourth[3] and "Member profile" in sixth[3] and "Alex Morgan" in sixth[3]
    records = artifact.provenance["reuses"]
    assert [(r["source_name"], r["start_node"], r["executed_path"]) for r in records] == [
        ("member_login", "s2", ["s2"]), ("member_login", "s3", ["s3"]), ("member_login", "s4", ["s4"]),
        ("member_profile_lookup", "s5", ["s5"]), ("member_profile_lookup", "s6", ["s6"])]
    events = [json.loads(l) for l in log.path.read_text().splitlines()]
    first_ordinary = next(i for i, e in enumerate(events)
                          if e["event"] == "planner_decided" and e["kind"] != "reuse_candidate")
    assert all(i < first_ordinary for i, e in enumerate(events) if e["event"] == "reuse_step_executed")
    assert sum(e["event"] == "reuse_step_executed" for e in events) == 5           # 3 for login, 2 for the lookup
    assert sum(e["event"] == "reuse_succeeded" for e in events) == 5 and "training_only" not in log.path.read_text()

    assert [n.action.action for n in linear_path(artifact)] == [
        "navigate", "type", "type", "click", "type", "click", "click", "click", "type", "click", "extract", "extract",
        "extract"]
    assert app.created == []                                             # nothing was opened

    login_path.unlink()                                                  # the library is gone ...
    lookup_path.unlink()
    surface = chromium(headless=True, secrets=("training_only",))
    replay_log = RunLog("replay", secrets=("training_only",))
    try:
        result = run_replay(artifact, dict(PREPARE), surface, Policy(allowed_hosts=HOSTS),
                            Escalator(NoOperator(), SessionControl(), replay_log), replay_log, purpose="supervised")
    finally:
        surface.close()
    assert result.status == "success" and result.outputs == {"member_name": "Alex Morgan", "account_type": "Savings",
                                                             "opening_deposit": 500.0}                # ... and it replays
    assert [e["id"] for e in result.executed_path] == [f"s{i}" for i in range(1, 14)] and app.created == []
