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
