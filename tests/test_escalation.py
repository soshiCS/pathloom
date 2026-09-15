"""Human handoff: control transfer, same-session operation, recording, resume/restart."""
import io

import pytest

from src.cua.escalation import AUTOMATION, HUMAN, ConsoleOperator
from src.cua.replay import replay
from tests.context import (HOSTS, PARAMS, Escalator, InterventionRequest, InterventionResult, Policy,
                           RecordingOperator, RunLog, SessionControl, checkout_artifact)
from tests.fake_surface import ENTRY, FakeSurface


def request(step="s4", kind="stuck") -> InterventionRequest:
    return InterventionRequest(run_id="r", capability="cap", step_id=step, kind=kind, reason="why",
                               observed="what", screenshot=None)


def test_session_control_enforces_a_single_owner():
    control = SessionControl()
    assert control.owner == AUTOMATION
    control.transfer(HUMAN)
    with pytest.raises(RuntimeError):
        control.require(AUTOMATION)
    with pytest.raises(RuntimeError):
        control.transfer(HUMAN)  # already the owner
    control.transfer(AUTOMATION)
    assert [h["to"] for h in control.history] == [HUMAN, AUTOMATION]


def test_escalator_hands_the_same_live_surface_to_the_human_and_takes_it_back():
    control = SessionControl()
    seen = {}

    class Operator:
        def handle(self, req, surface):
            seen["owner_during"] = control.owner
            seen["surface"] = surface
            return InterventionResult(True, [{"action": "click", "target": "x"}], disposition="resume")

    log = RunLog("replay")
    surface = FakeSurface()
    escalator = Escalator(Operator(), control, log)
    result = escalator.request(request(), surface)
    assert seen["owner_during"] == HUMAN and seen["surface"] is surface
    assert control.owner == AUTOMATION
    assert result.disposition == "resume"
    assert escalator.interventions[0]["human_actions"] == [{"action": "click", "target": "x"}]
    events = log.path.read_text()
    assert '"intervention_requested"' in events and '"control_transferred"' in events


def test_console_operator_drives_the_surface_and_redacts_the_password():
    surface = FakeSurface()
    surface.navigate(ENTRY)
    commands = io.StringIO("observe\ntype 1 standard_user\ntype 2 secret_sauce\nclick 3\nresume\n")
    output = io.StringIO()
    result = ConsoleOperator(commands, output).handle(request(), surface)
    assert result.disposition == "resume" and result.resolved
    assert surface.logged_in and surface.screen == "inventory"          # really acted on the live session
    assert result.human_actions[0] == {"action": "type", "target": "textbox 'Username'", "value": "standard_user"}
    assert result.human_actions[1] == {"action": "type", "target": "textbox 'Password'", "value": "[REDACTED]"}
    assert result.human_actions[2] == {"action": "click", "target": "button 'Login'"}
    assert "[2] textbox: Password" in output.getvalue() and "secret_sauce" not in output.getvalue()


def test_console_operator_aborts_when_input_closes():
    result = ConsoleOperator(io.StringIO(""), io.StringIO()).handle(request(), FakeSurface())
    assert result.disposition == "abort" and not result.resolved


def test_console_operator_answers_confirmation_requests():
    result = ConsoleOperator(io.StringIO("deny\n"), io.StringIO()).handle(request(kind="confirm"), FakeSurface())
    assert result.disposition == "deny" and not result.resolved


def test_replay_restarts_after_the_human_clears_a_verification_challenge():
    # After login the site shows an unknown "verify you are human" dialog that automation must
    # not bypass; the human clears it on the same session and asks for a restart.
    operator = RecordingOperator(disposition="restart", manual=[("click", "Verify")])
    surface = FakeSurface(faults=["verification"])
    log = RunLog("replay")
    result = replay(checkout_artifact(), PARAMS, surface, Policy(allowed_hosts=HOSTS),
                    Escalator(operator, SessionControl(), log), log)
    assert result.status == "success" and result.outputs["total"] == 32.39
    assert operator.surfaces[0] is surface
    assert result.interventions[0]["disposition"] == "restart"
    assert [a["target"] for a in result.interventions[0]["human_actions"]] == ["Verify"]
    assert surface.actions.count(("navigate", ENTRY)) == 2  # flow really started over


def test_replay_resume_rechecks_the_step_before_redoing_it():
    # The human clears the challenge and lands on the product list, then says "resume": the
    # login step is verified against its checkpoint ("Products"), not re-run.
    operator = RecordingOperator(disposition="resume", manual=[("click", "Verify"), ("click", "Login")])
    surface = FakeSurface(faults=["verification"])
    log = RunLog("replay")
    result = replay(checkout_artifact(), PARAMS, surface, Policy(allowed_hosts=HOSTS),
                    Escalator(operator, SessionControl(), log), log)
    assert result.status == "success"
    assert '"step_already_satisfied"' in log.path.read_text()
    assert surface.actions.count(("click", "Login", "")) == 2   # once by automation, once by the human


# ---------- intervention context ----------

def test_replay_handoffs_carry_the_capability_goal():
    from src.cua.graph import from_linear
    from src.cua.graph_replay import replay_graph
    from tests.context import Step, ladder

    goal = checkout_artifact().description
    stuck = RecordingOperator(disposition="abort")
    log = RunLog("replay")
    replay(checkout_artifact(), PARAMS, FakeSurface(faults=["verification"]), Policy(allowed_hosts=HOSTS),
           Escalator(stuck, SessionControl(), log), log)
    assert stuck.requests[0].goal == goal and stuck.requests[0].capability == "checkout_review"

    finish = Step(id="s16", action="click", target=ladder("button", "Finish"),
                  checkpoint={"text_contains": "Thank you"}, risk="risky")
    confirm = RecordingOperator(disposition="deny")
    log = RunLog("replay")
    replay(checkout_artifact(extra_step=finish), PARAMS, FakeSurface(), Policy(allowed_hosts=HOSTS),
           Escalator(confirm, SessionControl(), log), log)
    assert confirm.requests[0].kind == "confirm" and confirm.requests[0].goal == goal

    graph_stuck, graph_confirm = RecordingOperator(disposition="abort"), RecordingOperator(disposition="deny")
    log = RunLog("replay")
    replay_graph(from_linear(checkout_artifact()), PARAMS, FakeSurface(faults=["verification"]),
                 Policy(allowed_hosts=HOSTS), Escalator(graph_stuck, SessionControl(), log), log)
    log = RunLog("replay")
    replay_graph(from_linear(checkout_artifact(extra_step=finish)), PARAMS, FakeSurface(),
                 Policy(allowed_hosts=HOSTS), Escalator(graph_confirm, SessionControl(), log), log)
    assert graph_stuck.requests[0].goal == goal and graph_confirm.requests[0].goal == goal
    assert graph_confirm.requests[0].kind == "confirm"
    assert '"goal": "add a product and read the checkout overview"' in log.path.read_text()


def test_discovery_handoffs_carry_the_goal():
    from src.cua.agent import discover
    from tests.scripted_planner import ScriptedPlanner, ScriptedStep, checkout_script

    stuck = RecordingOperator(disposition="abort")
    log = RunLog("discovery")
    with pytest.raises(Exception, match="aborted"):
        discover(goal="read the totals", name="checkout_review", params=dict(PARAMS), surface=FakeSurface(),
                 planner=ScriptedPlanner([ScriptedStep("stuck")]), policy=Policy(allowed_hosts=HOSTS),
                 escalator=Escalator(stuck, SessionControl(), log), log=log, entry_url=ENTRY)
    assert stuck.requests[0].goal == "read the totals" and stuck.requests[0].kind == "stuck"

    confirm = RecordingOperator(disposition="deny")
    script = checkout_script()
    script.insert(-1, ScriptedStep("click", "button", "Finish", expect="Thank you"))
    log = RunLog("discovery")
    discover(goal="read the totals", name="checkout_review", params=dict(PARAMS), surface=FakeSurface(),
             planner=ScriptedPlanner(script), policy=Policy(allowed_hosts=HOSTS),
             escalator=Escalator(confirm, SessionControl(), log), log=log, entry_url=ENTRY, max_steps=25)
    assert confirm.requests[0].kind == "confirm" and confirm.requests[0].goal == "read the totals"


def test_console_operator_shows_the_goal():
    output = io.StringIO()
    shown = request()
    shown.goal = "add a product and read the checkout overview"
    ConsoleOperator(io.StringIO("abort\n"), output).handle(shown, FakeSurface())
    assert "goal: add a product and read the checkout overview" in output.getvalue()
