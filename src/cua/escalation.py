"""Human-in-the-loop escalation and control transfer.

The control-transfer model:
  - `SessionControl` is the single source of truth for who owns the live session.
    Automation only acts while it is the owner; the Escalator flips ownership to the
    human for the duration of an intervention and back afterwards.
  - An `Operator` is any channel through which a person can act on the *same* Surface
    object the automation was using (same browser, same cookies, same page).
  - `ConsoleOperator` is a minimal but real operator surface: it drives the live
    Surface from terminal commands and records every action the human took.
"""
from __future__ import annotations

import sys
import time
from typing import Protocol, TextIO

from .models import ActionError, HumanAction, InterventionRequest, InterventionResult, TransientError
from .policy import redact
from .surface import Surface

AUTOMATION = "automation"
HUMAN = "human"


class SessionControl:
    """Tracks and enforces whether automation or a human owns the session."""

    owner: str

    def __init__(self):
        self.owner = AUTOMATION
        self.history: list[dict] = []

    def transfer(self, to: str) -> None:
        """Transfer exclusive control; refuses no-op transfers so bugs surface loudly."""
        if to not in (AUTOMATION, HUMAN):
            raise ValueError(f"unknown session owner {to!r}")
        if to == self.owner:
            raise RuntimeError(f"session is already controlled by {to}")
        self.history.append({"from": self.owner, "to": to, "at": time.time()})
        self.owner = to

    def require(self, owner: str) -> None:
        """Guard called before acting: raises if the caller does not own the session."""
        if self.owner != owner:
            raise RuntimeError(f"{owner} tried to act but {self.owner} controls the session")


class Operator(Protocol):
    """Interface implemented by any human-intervention channel."""

    def handle(
        self,
        request: InterventionRequest,
        surface: Surface,
    ) -> InterventionResult: ...


class NoOperator:
    """Unattended mode: every intervention request comes back unresolved."""

    def handle(self, request: InterventionRequest, surface: Surface) -> InterventionResult:
        return InterventionResult(resolved=False, human_actions=[], note="no operator available",
                                  disposition="abort")


HELP = """Commands (you are driving the SAME live session the automation was using):
  observe              list the numbered controls currently on screen
  click <n>            click control n
  type <n> <text>      type text into control n
  navigate <url>       go to a url
  resume               hand control back; automation re-checks and continues the current step
  restart              hand control back; automation starts the flow over from step 1
  abort                hand control back; automation stops and reports a failure
  approve / deny       answer a risky-action confirmation request"""


class ConsoleOperator:
    """A person controls the live session through the terminal (stdin/stdout)."""

    def __init__(self, input_stream: TextIO | None = None, output_stream: TextIO | None = None):
        self.input = input_stream or sys.stdin
        self.output = output_stream or sys.stdout
        self.performed: list[HumanAction] = []

    def say(self, text: str) -> None:
        print(text, file=self.output, flush=True)

    def handle(self, request: InterventionRequest, surface: Surface) -> InterventionResult:
        self.say("\n=== HUMAN INTERVENTION REQUESTED ===")
        self.say(f"capability: {request.capability}   run: {request.run_id}   step: {request.step_id}")
        if request.goal:
            self.say(f"goal: {request.goal}")
        self.say(f"kind: {request.kind}\nreason: {request.reason}")
        self.say(f"observed: {request.observed[:400]}")
        if request.screenshot:
            self.say(f"screenshot: {request.screenshot}")
        if request.kind == "confirm":
            self.say("Reply 'approve' or 'deny'.")
        else:
            self.say(HELP)
            self.show_controls(surface)
        actions: list[dict] = []
        self.performed = []
        while True:
            self.say("operator> ")
            line = self.input.readline()
            if not line:  # stdin closed: treat as abort so the run never hangs
                return InterventionResult(False, actions, "operator input closed", disposition="abort",
                                          performed=list(self.performed))
            command, _, argument = line.strip().partition(" ")
            if command in ("resume", "restart", "abort", "approve", "deny"):
                resolved = command in ("resume", "restart", "approve")
                return InterventionResult(resolved, actions, f"operator chose {command}", disposition=command,
                                          performed=list(self.performed))
            try:
                record = self.run_command(command, argument, surface)
            except Exception as error:  # a bad command must not end the handoff
                self.say(f"error: {error}")
                continue
            if record:
                actions.append(record)

    def show_controls(self, surface: Surface) -> None:
        observation = surface.observe()
        self.say(f"url: {observation.url}")
        if observation.dialog:
            self.say(f"dialog: {observation.dialog}")
        for index, element in enumerate(observation.elements):
            label = element.name or element.text
            self.say(f"  [{index}] {element.role}: {label[:80]}")

    def run_command(self, command: str, argument: str, surface: Surface) -> dict | None:
        """Execute one operator command on the live surface; return a redacted record of it. The structured
        record of a performed action is kept in `self.performed` for the discovery loop to record."""
        if command == "observe":
            self.show_controls(surface)
            return None
        if command == "navigate":
            done = perform_human_action(surface, "navigate", None, argument)
            self.performed.append(done)
            if done.performed != "yes":
                self.say("warning: the navigation may not have completed; discovery cannot record it")
            return {"action": "navigate", "value": argument}
        if command in ("click", "type"):
            index_text, _, text = argument.partition(" ")
            element = surface.observe().elements[int(index_text)]
            done = perform_human_action(surface, command, element, text if command == "type" else None)
            self.performed.append(done)
            if done.performed != "yes":
                self.say(f"warning: the {command} may not have completed; discovery cannot record it")
            if command == "click":
                return {"action": "click", "target": f"{element.role} '{element.name or element.text}'"}
            # Redact by field name so a password typed by the operator never reaches the log.
            value = redact({element.name: text})[element.name]
            return {"action": "type", "target": f"{element.role} '{element.name}'", "value": value}
        self.say(HELP)
        return None


def perform_human_action(surface: Surface, kind: str, element, value: str | None = None) -> HumanAction:
    """Perform one operator action and describe it as the runtime saw it (see `HumanAction`).

    A failure the surface proves happened before anything was dispatched is raised as is: nothing to record.
    A failure after dispatch, or a load failure, is a performed action with an unknown outcome.
    """
    before = surface.observe()
    try:
        if kind == "navigate":
            surface.navigate(value or "")
        elif kind == "click":
            surface.click(element)
        elif kind == "type":
            surface.type(element, value or "")
        else:
            raise ValueError(f"unsupported human action {kind!r}")
    except ActionError as error:
        if error.performed == "no":
            raise
        return HumanAction(kind=kind, target=element, value=value, before=before, after=None, performed="unknown")
    except TransientError:
        return HumanAction(kind=kind, target=element, value=value, before=before, after=None, performed="unknown")
    try:
        after = surface.observe()
    except TransientError:
        after = None
    return HumanAction(kind=kind, target=element, value=value, before=before, after=after)


class Escalator:
    """Coordinates pause, evidence capture, handoff, and resume."""

    def __init__(self, operator: Operator, control: SessionControl, log):
        self.operator = operator
        self.control = control
        self.log = log
        self.interventions: list[dict] = []

    def request(
        self,
        request: InterventionRequest,
        surface: Surface,
    ) -> InterventionResult:
        """Request help on the same live surface, then return control to automation."""
        # 1. Pause: capture evidence of the state the automation is stuck in.
        request.screenshot = self.log.screenshot(surface, f"escalation-{request.kind}")
        self.log.event("intervention_requested", kind=request.kind, step_id=request.step_id,
                       capability=request.capability, goal=request.goal, reason=request.reason,
                       observed=request.observed[:400], screenshot=request.screenshot)
        # 2. Cede control. From here on only the operator may act on the surface.
        self.control.transfer(HUMAN)
        self.log.event("control_transferred", to=HUMAN)
        try:
            result = self.operator.handle(request, surface)
        finally:
            # 3. Hand control back, whatever happened, so automation can resume or stop cleanly.
            self.control.transfer(AUTOMATION)
            self.log.event("control_transferred", to=AUTOMATION)
        self.log.screenshot(surface, "after-intervention")
        record = {"step_id": request.step_id, "kind": request.kind, "reason": request.reason,
                  "resolved": result.resolved, "disposition": result.disposition,
                  "human_actions": result.human_actions, "note": result.note}
        self.interventions.append(record)
        self.log.event("intervention_finished", **record)
        return result
