"""One-time LLM-driven discovery.

The loop is:  observe -> find verified reusable segments for this screen -> ask the planner for ONE
              decision (a segment, or an ordinary action) -> policy check -> act -> verify -> record

What gets recorded is not the transcript but the reusable flow as a linear capability graph:
every performed action becomes an action node (with a checkpoint only when the planner's
expectation was actually seen, an effect from the policy's risk verdict, and a retry policy
derived from both), the nodes are chained by unconditional edges into a success terminal,
dialog dismissals become recoverable outcomes, and every literal input value is replaced by
a named placeholder.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import replace
from urllib.parse import parse_qsl, urlparse

from .artifact import (CARDINALITY_KEYS, READ_ACTIONS, build_linear, classify_effect, classify_retry_safety, output_problems,
                       parameterize, parameterize_locator,
                       placeholders_in, substitute)
from .escalation import AUTOMATION, Escalator, NoOperator
from .evidence import RunLog
from .library import (Library, candidate_summary, classify, execute_candidate, find_candidates, importable_nodes,
                      screen_key, verified_entries)
from .models import (MAX_EXTRACT_TARGETS, MAX_TOKEN_TEXT, Action, ActionError, Artifact, Element, GraphAction, GraphNode, VisualDecision,
                     InterventionRequest, Locator, Observation, ReuseCandidate, ReusePrefix, ScreenshotFrame,
                     TransientError)
from .planner import Planner, VisionPlanner, drain_retries
from .policy import SENSITIVE_KEY_PATTERN, Policy, redact
from .surface import Surface, is_ambiguous, locator_for, token_key, visible_text

DEFAULT_MAX_STEPS = 15
DEFAULT_MAX_VISION_ATTEMPTS = 2
MAX_CONSECUTIVE_DENIALS = 3
MAX_SAME_ACTION_FAILURES = 2   # the same action failing before acting on the same screen, then a person decides
MAX_UNVERIFIED_ATTEMPTS = 1    # a performed action nothing can prove is never dispatched a second time
SNAPSHOT_RECHECK_MS = 400      # how long a result list must hold still before its order is trusted
MAX_SNAPSHOT_ATTEMPTS = 3      # re-observations allowed while a list keeps changing under us
LOOP_WINDOW = 12               # recent (action, screen) entries the loop detector keeps
LOOP_PERIODS = (1, 2, 3)       # cycle lengths recognized: the same step, two states alternating, three states
LOOP_WARN_REPEATS = 2          # a cycle seen this many times in a row without progress: one warning
LOOP_STOP_REPEATS = 3          # and this many times: the structured stuck handoff, cause planner_loop
# Stuck causes the vision fallback may act on: a control perception cannot list, or a listed control that the
# surface proved cannot take the click and whose enclosing controls are not actionable either.
VISION_CAUSES = ("perception", "actionability", "missing_data")
EXPECT_TIMEOUT_S = 6.0
POLL_INTERVAL_S = 0.3


class DiscoveryFailed(Exception):
    """Discovery hit a stopping condition without reaching the goal."""


class RestartDiscovery(DiscoveryFailed):
    """The live session is in an unverified state after a reused segment: close it and start over
    on a fresh one with that segment's source artifact excluded. The session owner
    (reuse.discover_with_reuse) handles it; `excluded` holds source digests."""

    def __init__(self, reason: str, excluded: set):
        super().__init__(reason)
        self.excluded = set(excluded)


class ProgressTracker:
    """Detects a planner going in circles: the same action on the same screen again and again, or a cycle of
    two or three states alternating (Search -> Refine -> Search -> Refine), judged by what each step would
    actually do and where it would do it.

    An entry is (action signature, starting screen). The signature is executable behaviour only: kind,
    target identity, the parameterized value, the output name, the list targets and the chosen reuse
    candidate. The planner's prose, its expected text and the checkpoint that expectation became are never
    part of it, so rewording an action does not make it look new. The starting screen is the structured
    fingerprint of the page the action runs on, so a cycle whose two screens each pass their own checkpoint
    and change the URL still reads as a cycle.

    Only real progress clears the window (`reset`): a new or grown output, a human recovery or restart, or
    an action that moved the run somewhere it has not been going round in circles. A passed checkpoint and a
    changed URL are not progress by themselves: both are exactly what a two-screen cycle produces on every
    turn.
    """

    def __init__(self, params: dict):
        self.params = params
        self.entries: list[tuple] = []
        self.warned = False

    def signature(self, action: Action) -> tuple:
        """What this action would do, ignoring how the planner described it."""
        target = action.target
        return (action.kind, target.role if target else None, target.name if target else None,
                target.context if target else None, target.ref if target else None,
                parameterize(action.value, self.params) if action.value else None,
                action.output_name, tuple(f"{t.role}:{t.name or t.text}" for t in action.targets), action.candidate_id)

    def check(self, action: Action, observation: Observation) -> tuple[str | None, int, int]:
        """(verdict, cycle length, repeats) for the action about to run: "stop", "warn" or None."""
        sequence = self.entries + [(self.signature(action), screen_key(observation))]
        for period in LOOP_PERIODS:
            repeats = cycles_at_tail(sequence, period)
            if repeats >= LOOP_STOP_REPEATS:
                return "stop", period, repeats
            if repeats >= LOOP_WARN_REPEATS and not self.warned:
                return "warn", period, repeats
        return None, 0, 0

    def record(self, action: Action, observation: Observation) -> None:
        self.entries.append((self.signature(action), screen_key(observation)))
        del self.entries[:-LOOP_WINDOW]

    def reset(self, why: str, log: RunLog) -> None:
        if self.entries or self.warned:
            log.event("progress_made", why=why, forgotten=len(self.entries))
        self.entries, self.warned = [], False


def stable_screen_key(observation: Observation) -> str:
    """A fingerprint of the controls outside any dialog: the page state a transient popup does not change.

    Native control state counts as page state: a control that is now checked, selected, expanded or pressed
    is a screen that changed, even when every label on it reads the same.
    """
    digest = hashlib.sha256()
    for element in observation.elements:
        if element.landmark in ("dialog", "alertdialog"):
            continue
        native = "".join(f"|{key}={element.states[key]}" for key in ("checked", "selected", "expanded", "pressed")
                         if key in element.states)
        digest.update(f"|{element.role}|{element.name}|{element.text}{native}".encode())
    return digest.hexdigest()[:16]


def cycles_at_tail(sequence: list, period: int) -> int:
    """How many times the last `period` entries repeat consecutively at the end of the sequence."""
    if len(sequence) < period or period < 1:
        return 0
    cycle = sequence[-period:]
    repeats = 0
    end = len(sequence)
    while end >= period and sequence[end - period:end] == cycle:
        repeats += 1
        end -= period
    return repeats


class Recorder:
    """Accumulates the pieces of the artifact while discovery runs: the action nodes of a linear graph."""

    def __init__(self, params: dict, output_contract: dict | None = None, policy: Policy | None = None):
        self.params = params
        self.output_contract = output_contract or {}   # declared outputs: their spec wins over the planner's
        self.policy = policy or Policy()                # classifies the effect of what gets recorded
        self.nodes: list[GraphNode] = []
        self.outputs: dict = {}
        self.outcomes: list[dict] = []
        self.success: dict | None = None  # set when the planner's "done" claim is verified
        self.human_steps: list[str] = []  # ids of nodes a person performed during a handoff
        self.trail: dict[str, dict] = {}  # node id -> where it was performed (url, stable screen key), for pruning
        self.snapshot: list[str] = []     # the ordered result identities this run committed to, if any
        self.snapshot_output: str | None = None   # the declared output those identities came from

    def next_id(self) -> str:
        return f"s{len(self.nodes) + 1}"

    def record_step(self, action: Action, risk: str, seen: Observation, locator: Locator | None = None,
                    ladders: list[Locator] | None = None) -> GraphNode:
        """Record one performed action as the next action node; returns it.

        The checkpoint is the proof verification attached to the action (see `checkpoint_for`): an
        extraction verifies itself (the pattern must match) and carries none. A caller may supply the
        ladder itself (`locator`), or one per selected element for extract_many (`ladders`).
        """
        checkpoint = None
        if action.checkpoint and action.kind not in ("extract", "extract_many"):
            checkpoint = parameterize_checkpoint(action.checkpoint, self.params)
        target, targets = None, None
        if action.kind == "extract_many":
            # One parameterized ladder per selected element, in the order the planner asked for.
            # Several items of one list can share the same wording; the structural rung is what keeps them
            # distinct, so it survives parameterization for list targets (see `parameterize_locator`).
            targets = [parameterize_locator(ladder, self.params, keep_structure=True) for ladder in
                       (ladders or [locator_for(element, is_ambiguous(element, seen)) for element in action.targets])]
        elif action.target:
            # Qualify the locator by its item only when the name alone was ambiguous on screen,
            # then parameterize it: "Add to cart" in "{{product_name}}". A caller may supply the
            # ladder itself (a vision target records exact coordinates instead).
            control = activated_control(action, seen)
            ladder = locator or locator_for(control, is_ambiguous(control, seen))
            target = parameterize_locator(ladder, self.params)
        value = action.output_name if action.kind in ("extract", "extract_many") else parameterize(action.value,
                                                                                                   self.params)
        effect = classify_effect(action.kind, risk)
        mode = "append" if action.kind in ("extract", "extract_many") and action.output_mode == "append" else None
        node = GraphNode(id=self.next_id(), kind="action",
                         action=GraphAction(action=action.kind, target=target, value=value, checkpoint=checkpoint,
                                            targets=targets, mode=mode),
                         effect=effect, retry_safety=classify_retry_safety(action.kind, effect, checkpoint))
        self.nodes.append(node)
        self.trail[node.id] = {"before_url": seen.url, "before_key": stable_screen_key(seen),
                               "abandoned": bool(action.expectation_contradicted)}
        return node

    def mark_abandoned(self, node_id: str) -> None:
        """The loop detector saw this step repeat without progress: evidence the planner was going nowhere."""
        if node_id in self.trail:
            self.trail[node_id]["abandoned"] = True

    def prune_abandoned_suffix(self, handoff: Observation, log: RunLog) -> list[str]:
        """Drop trailing automated actions that a person's navigation supersedes, when that is provable.

        The candidate suffix is the trailing run of nodes that were all performed on the page the person left
        from, each with effect `none` or `reversible`, none reading an output, none a boundary (the entry
        navigation, an irreversible or unknown effect, a recorded output, an imported or human step). It is
        pruned only when the screen the person left from is the screen the suffix started on (same URL and the
        same controls outside any dialog): the suffix then changed nothing that later steps could need. When the
        screen differs, actions that carry abandonment evidence (a contradicted expectation, a detected loop)
        cannot be proven harmless and fail discovery rather than mislead a replay; actions without such evidence
        are state the person continued from and are kept.
        """
        suffix: list[GraphNode] = []
        for node in reversed(self.nodes):
            where = self.trail.get(node.id)
            if (where is None or node.effect not in ("none", "reversible") or node.action.action in READ_ACTIONS
                    or node.id in self.human_steps or where["before_url"] != handoff.url):
                break
            suffix.append(node)
        suffix.reverse()
        if not suffix:
            return []
        ids = [node.id for node in suffix]
        if self.trail[suffix[0].id]["before_key"] != stable_screen_key(handoff):
            # The page changed: either the actions built state the person continued from (kept), or they were
            # going nowhere (a contradicted expectation, a detected loop) and cannot be proven harmless (fail).
            if any(self.trail[node.id]["abandoned"] for node in suffix):
                raise DiscoveryFailed(f"abandoned_actions_unprovable: the operator's navigation supersedes {ids}, "
                                      f"which repeated without progress or contradicted their expectations, but the "
                                      f"screen at handoff differs from the screen before them, so removing them cannot "
                                      f"be proven safe and keeping them would mislead replay")
            log.event("speculative_suffix_kept", node_ids=ids, reason="the page changed after them and nothing "
                      "marks them as abandoned: they are state the operator continued from")
            return []
        self.nodes = [node for node in self.nodes if node.id not in ids]
        for node_id in ids:
            self.trail.pop(node_id, None)
        log.event("speculative_suffix_pruned", node_ids=ids, superseded_by="human navigate",
                  reason="the actions changed nothing lasting: same page and same controls before them and at the "
                         "handoff, effects none or reversible, no outputs, no boundary crossed")
        return ids

    def aligned_lists(self) -> list[str]:
        """The declared list outputs of the same length: the ones that must stay aligned position by position.

        A task that asks for the first N results as several parallel lists declares them with the same
        cardinality; those are the outputs an ordered snapshot governs. A contract with one list, or none,
        describes ordinary work and is left alone.
        """
        sized = [(name, spec) for name, spec in self.output_contract.items()
                 if spec.get("type") == "list" and spec.get("max_items")]
        sizes = {spec["max_items"] for _, spec in sized}
        if len(sized) < 2 or len(sizes) != 1:
            return []                       # one list, or lists of different lengths: ordinary work
        every_list = [name for name, spec in self.output_contract.items() if spec.get("type") == "list"]
        if len(every_list) != len(sized):
            return []                       # an unsized list alongside them: not one aligned set
        return sorted(name for name, _ in sized)

    def snapshot_needed(self, action: Action) -> bool:
        """Whether this extraction may only run once an ordered snapshot exists.

        Only for a task with several aligned list outputs, and only for a value read one at a time (an
        append). One `extract_many` over the list page reads a whole column in displayed order and is how a
        snapshot is taken in the first place, so it is never blocked.
        """
        return bool(self.aligned_lists()) and action.output_mode == "append" and not self.snapshot

    def take_snapshot(self, action: Action, values: list[str]) -> None:
        """Remember the ordered identities an `extract_many` just read, so later pages follow them."""
        if self.snapshot or not self.aligned_lists() or action.kind != "extract_many":
            return
        self.snapshot, self.snapshot_output = list(values), action.output_name

    def output_conflict(self, action: Action) -> str | None:
        """Why an extraction may not land in the outputs, or None.

        `set` assigns an output exactly once: a second set to the same name would silently replace what an
        earlier step read, so it is refused and the planner is told how to add instead. `append` grows a list:
        the output must not already be a scalar. Under a declared contract the name must be one of the declared
        outputs, the action's shape must fit the declared type, and an append may not push a list past its
        max_items.
        """
        if action.kind not in ("extract", "extract_many") or not action.output_name:
            return None
        name, current = action.output_name, self.outputs.get(action.output_name)
        declared = self.output_contract.get(name)
        if self.snapshot_needed(action):
            aligned = self.aligned_lists()
            size = self.output_contract[aligned[0]].get("max_items")
            return (f"the goal asks for {size} results as the aligned lists {aligned}, and nothing fixes which "
                    f"results they are yet: read one whole column from the list page first with extract_many (the "
                    f"identity the goal names, such as a title, name or id, in displayed order), then open each "
                    f"result and append the rest; a list that changes while you are away cannot be repaired")
        if self.output_contract and declared is None:
            return (f"output {name!r} is not declared in the output contract; the only valid output names are "
                    f"{sorted(self.output_contract)}")
        if action.output_mode == "append":
            if current is not None and current.get("type") != "list":
                return (f"output {name!r} is already recorded as a single {current.get('type')} value; "
                        f"append only adds to a list output, so use a different output name")
            if declared is not None and declared.get("type") != "list":
                return f"output {name!r} is declared as a single {declared.get('type')} value, so nothing can be appended"
            return self.capacity_conflict(name, current, declared, len(action.targets) if action.kind == "extract_many" else 1)
        if current is not None:
            return (f"output {name!r} is already recorded and a set would overwrite it; use "
                    f"output_mode 'append' to add more items to that list, or a new output name")
        if declared is not None and declared.get("type") == "list" and action.kind == "extract":
            return (f"output {name!r} is declared as a list; read it with extract_many, or add one item at a time "
                    f"with output_mode 'append'")
        if declared is not None and declared.get("type") != "list" and action.kind == "extract_many":
            return f"output {name!r} is declared as a single {declared.get('type')} value, not a list"
        return None

    @staticmethod
    def capacity_conflict(name: str, current: dict | None, declared: dict | None, adding: int) -> str | None:
        """An append that would exceed the declared max_items is refused; a full list is reported complete."""
        limit = (declared or {}).get("max_items")
        if limit is None:
            return None
        have = len(current["example"]) if current else 0
        if have >= limit:
            return f"output {name!r} is already complete ({have} of at most {limit} items); do not append to it again"
        if have + adding > limit:
            return (f"output {name!r} holds {have} of at most {limit} items; appending {adding} more would exceed the "
                    f"contract, so nothing was appended")
        return None

    def record_output(self, action: Action, sample: str) -> None:
        """A scalar output assigned once; an append records one item into the named list output."""
        if action.output_mode == "append":
            self.append_items(action, [sample], f"Values read one at a time from {action.target.role} controls")
            return
        declared = self.output_contract.get(action.output_name, {})
        self.outputs[action.output_name] = {
            "type": declared.get("type", infer_type(sample)),
            "required": declared.get("required", not action.optional),   # optional outputs come back as null
            "pattern": self.output_pattern(action),
            "description": f"Text read from {action.target.role} '{action.target.name}'",
            "example": parameterize(sample, self.params),
        }

    def record_list_output(self, action: Action, samples: list[str]) -> None:
        """The canonical list contract: type list, an items rule (scalar type and pattern), requiredness."""
        if action.output_mode == "append":
            self.append_items(action, samples, f"Ordered values collected from {len(action.targets)} controls at a time")
            return
        self.outputs[action.output_name] = self.list_spec(action, samples, f"Ordered values read from "
                                                                            f"{len(action.targets)} controls")

    def append_items(self, action: Action, samples: list[str], description: str) -> None:
        """Grow the named list output: create it from the first items, then extend its example in order."""
        current = self.outputs.get(action.output_name)
        if current is None:
            self.outputs[action.output_name] = self.list_spec(action, samples, description)
            return
        current["example"] = list(current["example"]) + [parameterize(sample, self.params) for sample in samples]

    def list_spec(self, action: Action, samples: list[str], description: str) -> dict:
        declared = self.output_contract.get(action.output_name, {})
        declared_items = declared.get("items") or {}
        types = {infer_type(sample) for sample in samples}
        spec = {
            "type": "list",
            "required": declared.get("required", not action.optional),
            "items": {"type": declared_items.get("type", types.pop() if len(types) == 1 else "string"),
                      "pattern": declared_items.get("pattern", action.pattern) if declared else action.pattern},
            "description": description,
            "example": [parameterize(sample, self.params) for sample in samples],
        }
        spec.update({key: declared[key] for key in CARDINALITY_KEYS if key in declared})
        return spec

    def missing_data_problem(self, action: Action) -> str | None:
        """Why a "missing data" stuck state may not go to visual extraction: the output must be named,
        declared in the contract and not recorded yet."""
        name = action.output_name
        if not name:
            return "no output_name was named"
        if name not in self.output_contract:
            return f"output {name!r} is not declared in the output contract"
        if name in self.outputs:
            return f"output {name!r} is already recorded"
        return None

    def contract_problems(self) -> dict[str, str]:
        """Declared outputs the recording does not yet satisfy (see artifact.output_problems); empty without a
        contract, when discovery infers its own output definitions."""
        if not self.output_contract:
            return {}
        recorded = {name: spec.get("example") for name, spec in self.outputs.items()}
        return output_problems(self.output_contract, recorded)

    def output_pattern(self, action: Action) -> str | None:
        """The declared pattern for a contracted output, else what the planner proposed."""
        declared = self.output_contract.get(action.output_name)
        if declared and declared.get("type") == "list":
            return (declared.get("items") or {}).get("pattern")
        return declared["pattern"] if declared else action.pattern

    def record_recoverable_dialog(self, dialog_text: str, action: Action) -> None:
        """A dialog the model dismissed may or may not appear next time: record how to clear it."""
        snippet = dialog_text[:60].strip()
        code = "dismiss_" + re.sub(r"[^a-z0-9]+", "_", snippet.lower()).strip("_")[:40]
        if any(outcome["code"] == code for outcome in self.outcomes):
            return
        self.outcomes.append({
            "code": code, "kind": "recoverable", "source": "observed",
            "detect": {"dialog_contains": snippet},
            "recover": {"action": "click", "target": {"strategies": locator_for(action.target).strategies}},
        })

    def record_business_outcomes(self, declared: list[dict]) -> None:
        """Outcomes the planner declares were not necessarily seen; the source tag says so to reviewers."""
        for outcome in declared:
            detect = {}
            if outcome.get("text_contains"):
                detect["text_contains"] = outcome["text_contains"]
            if outcome.get("text_missing"):
                detect["text_missing"] = parameterize(outcome["text_missing"], self.params)
            if detect:
                self.outcomes.append({"code": outcome["code"], "kind": "business", "source": "planner",
                                      "detect": detect})

    def last_checkpoint(self) -> dict | None:
        for node in reversed(self.nodes):
            if node.action.checkpoint:
                return node.action.checkpoint
        return None

    def import_nodes(self, nodes: list[GraphNode], outputs: dict, outcomes: list[dict]) -> list[str]:
        """Inline verified nodes from another artifact, renumbered; returns the ids they got."""
        ids = []
        for node in nodes:
            node.id = self.next_id()
            self.nodes.append(node)
            ids.append(node.id)
        self.outputs.update(outputs)
        known = {o["code"] for o in self.outcomes}
        self.outcomes.extend(dict(o) for o in outcomes if o.get("kind") == "recoverable" and o["code"] not in known)
        return ids


def discover(
    goal: str,
    name: str,
    params: dict,
    surface: Surface,
    planner: Planner,
    policy: Policy,
    escalator: Escalator,
    log: RunLog,
    entry_url: str,
    sensitive: set[str] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    extra_outcomes: list[dict] | None = None,
    output_contract: dict | None = None,
    selectors: set[str] | None = None,
    vision: VisionPlanner | None = None,
    max_vision_attempts: int = DEFAULT_MAX_VISION_ATTEMPTS,
    prefix: ReusePrefix | None = None,
    library: Library | None = None,
) -> Artifact:
    """Discover a capability by driving the live surface with the planner; return its artifact.

    `output_contract` (name -> {type, required, pattern}) fixes how declared outputs are read,
    whatever regex the planner proposes, so independent runs record the same contract.
    `selectors` are inputs that choose a path rather than data the flow uses: their values are
    recorded literally, never as placeholders, so a route called "details" cannot rewrite the
    "View details" link a path happens to click.
    `vision`, when given, is the bounded screenshot fallback (see `Vision`): tried only when the
    planner is stuck, at most `max_vision_attempts` times per run and once per unchanged screen.
    `prefix`, when given, is a verified reusable path that already ran on this surface (reuse.py):
    its action nodes open the recording and the surface is already positioned, so the entry
    navigation is skipped and the planner's first look is the screen the prefix ended on.
    `library`, when given, is the catalog of approved artifacts (library.py): before every planner
    decision the segments whose entry state holds on the current screen are offered, and a chosen
    one is replayed deterministically and inlined. A session left in an unverified state by such a
    segment raises RestartDiscovery for the session owner.
    """
    sensitive = sensitive or set()
    # Sensitive values are never shown to the model; it sees (and types) the placeholder.
    visible_params = {key: ("{{" + key + "}}" if key in sensitive else value) for key, value in params.items()}
    recorder = Recorder({key: value for key, value in params.items() if key not in (selectors or set())},
                        output_contract, policy)
    history: list[Action] = []
    log.event("discovery_started", goal=goal, capability=name, params=visible_params, entry_url=entry_url,
              planner=planner.name, allowed_hosts=policy.allowed_hosts)

    escalator.control.require(AUTOMATION)
    if prefix is None:
        navigate_with_retry(surface, entry_url, log)
        observation = observe_with_retry(surface, log)
        app_name = getattr(surface, "title", "") or "unknown"  # captured at the entry screen, before navigating away
        checkpoint = entry_checkpoint(observation)
        recorder.nodes.append(GraphNode(
            id="s1", kind="action", action=GraphAction(action="navigate", target=None, value=entry_url,
                                                        checkpoint=checkpoint),
            effect="none", retry_safety=classify_retry_safety("navigate", "none", checkpoint)))
        log.screenshot(surface, "entry")
    else:
        recorder.nodes.extend(prefix.nodes)
        recorder.outcomes.extend(prefix.outcomes)
        recorder.outputs.update(prefix.outputs)
        observation = observe_with_retry(surface, log)
        app_name = getattr(surface, "title", "") or "unknown"
        log.event("discovery_resumed_after_reuse", imported_nodes=len(prefix.nodes),
                  source=prefix.provenance.get("source_name"), source_version=prefix.provenance.get("source_version"))
        log.screenshot(surface, "after-reuse")

    reuses: list[dict] = [dict(prefix.provenance)] if prefix is not None else []
    declared_codes = {o["code"] for o in extra_outcomes or []}
    failures: dict[tuple, int] = {}        # (action identity, screen) -> failures before acting
    unverified: dict[tuple, int] = {}      # (action identity, screen) -> performed attempts nothing could prove
    consecutive_denials = 0
    fallback = Vision(vision, max_vision_attempts, goal, visible_params, params, surface, log) if vision else None
    pending: Action | None = None          # a stuck state the runtime raised itself, taken next turn instead of a decision
    progress = ProgressTracker(params)
    last_outputs = json.dumps(recorder.outputs, sort_keys=True, default=str)
    turn = 0
    while turn < max_steps:
        turn += 1
        observation = observe_with_retry(surface, log)
        if pending is not None:
            action, pending = pending, None
            log.event("runtime_stuck", turn=turn, stuck_cause=action.stuck_cause, reason=action.reason)
        else:
            candidates = find_candidates(library, observation, params, surface, log, turn) if library else []
            action = planner.decide(goal, visible_params, observation, history, candidates)
            for retry in drain_retries(planner):
                log.event("planner_retry", turn=turn, **retry)
            log.event("planner_decided", turn=turn, kind=action.kind, target=describe(action.target),
                      targets=[describe(t) for t in action.targets] or None, value=action.value, expect=action.expect,
                      output_name=action.output_name, optional=action.optional, reason=action.reason,
                      candidate_id=action.candidate_id)

        if action.kind == "reuse_candidate":
            chosen = next((c for c in candidates if c.candidate_id == action.candidate_id), None)
            if chosen is None:
                log.event("reuse_candidate_rejected", candidate_id=action.candidate_id, turn=turn,
                          reason="not among the candidates offered this turn")
                action.result = f"candidate {action.candidate_id!r} was not offered; choose an ordinary action"
                history.append(action)
                continue
            record = reuse_segment(chosen, observation, library, recorder, params, surface, policy, escalator, log,
                                   turn, output_contract, declared_codes)
            if record is not None:
                reuses.append(record)
            action.result = (f"reused segment {chosen.candidate_id} completed: {record['imported_nodes']} verified "
                             f"action(s) recorded; outputs recorded: {record['outputs'] or 'none'}"
                             if record is not None else "the segment did not complete; re-observe the screen")
            history.append(action)
            continue

        if action.kind == "done":
            if finish(action, observation, params, recorder, log):
                log.screenshot(surface, "goal-reached")
                break
            history.append(action)
            continue
        if action.kind == "stuck":
            # The structured planner cannot see a control it needs: the bounded vision fallback may
            # propose one visual action; everything else about it (policy, acting, verification) is
            # the normal path below.
            visual = None
            if fallback and action.stuck_cause == "missing_data":
                problem = recorder.missing_data_problem(action)
                if problem:
                    log.event("vision_fallback_skipped", reason=problem)
                else:
                    visual = fallback.propose(action, observation, history,
                                              spec=recorder.output_contract[action.output_name])
            elif fallback:
                visual = fallback.propose(action, observation, history)
            if visual is None:
                handle_stuck(action, observation, name, goal, escalator, surface, log, recorder)
                history.append(action)
                continue
            action = visual

        # Policy check happens before anything touches the surface, on the concrete value the surface
        # would receive: a navigation to "{{cart_url}}" is judged by the host it really goes to.
        verdict = policy.check(replace(action, value=substitute(action.value, params)), observation.url)
        log.event("policy_checked", decision=verdict.decision, reason=verdict.reason)
        if verdict.decision == "deny":
            action.result = f"denied: {verdict.reason}"
            history.append(action)
            consecutive_denials += 1
            if consecutive_denials >= MAX_CONSECUTIVE_DENIALS:
                raise DiscoveryFailed(f"planner kept choosing disallowed actions ({consecutive_denials} in a row)")
            continue
        consecutive_denials = 0

        # A planner going in circles is stopped before it burns the step budget, asks a person to confirm the
        # same thing again, or repeats an action whose effect may not be reversible: one warning in its
        # history, then the structured stuck handoff.
        loop, period, repeats = progress.check(action, observation)
        if loop == "stop" or (loop == "warn" and verdict.decision == "confirm"):
            plural = "action" if period == 1 else f"{period} actions"
            stuck = Action(kind="stuck", stuck_cause="planner_loop",
                           reason=f"no progress: the same {plural} repeated {repeats} times from the same screens "
                                  f"with no new output; latest: {action.kind} on {describe(action.target)}")
            log.event("planner_loop_detected", turn=turn, period=period, repeats=repeats, decision="stop")
            handle_stuck(stuck, observation, name, goal, escalator, surface, log, recorder)
            progress.reset("human intervened", log)
            history.append(stuck)
            continue
        if loop == "warn":
            log.event("planner_loop_detected", turn=turn, period=period, repeats=repeats, decision="warn")
            progress.warned = True
        if verdict.decision == "confirm":
            verdict = confirm_with_human(action, observation, name, goal, escalator, surface, log, verdict.reason,
                                         recorder)
            if verdict.decision == "deny":
                action.result = f"denied: {verdict.reason}"
                history.append(action)
                continue

        # The same action on the same control at the same address, already performed with no provable effect,
        # is not dispatched again. Refusing before `perform` is what keeps the second attempt off the page:
        # escalating only afterwards would let the planner re-send the same blind click after every handoff.
        blind = unverified_key(action, observation)
        if unverified.get(blind):
            action.result = (f"refused: this {action.kind} was already performed here and nothing showed it had any "
                             f"effect; it is not repeated until the run makes measurable progress or a person resolves it")
            log.event("unverified_repeat_refused", kind=action.kind, target=describe(action.target),
                      attempts=unverified[blind], dispatched=False, cause="action_effect_unverified")
            stuck = Action(kind="stuck", stuck_cause="planner", reason=action.result)
            handle_stuck(stuck, observation, name, goal, escalator, surface, log, recorder)
            history.append(action)
            continue

        try:
            conflict = recorder.output_conflict(action)
            if conflict:
                raise ActionError(f"{action.kind} was not performed: {conflict}", performed="no")
            perform(action, params, surface, log)
        except TransientError as error:
            # The surface could not complete the action. What is safe next depends on whether anything
            # physical happened; the planner is never crashed out of and never blindly repeats. A target
            # proven unactionable may go to the bounded vision fallback on the next turn.
            pending = recover_from_action_failure(action, error, observation, params, failures, recorder, surface,
                                                  escalator, name, goal, log, vision_enabled=fallback is not None)
            history.append(action)
            continue
        if vision_sourced(action):
            verify_vision_action(action, observation, params, policy, recorder, surface, log, fallback, name, goal,
                                 escalator)
        else:
            verify_and_record(action, observation, params, policy, recorder, surface, log)
        if action.unverified:
            # Performed, unproven, unrecorded. The same action again would be a second blind attempt, so it
            # is refused until something measurable changes; a person decides if the planner insists.
            key = unverified_key(action, observation)
            unverified[key] = unverified.get(key, 0) + 1
            if unverified[key] >= MAX_UNVERIFIED_ATTEMPTS:
                stuck = Action(kind="stuck", stuck_cause="planner",
                               reason=f"the {action.kind} on {describe(action.target)} was performed "
                                      f"{unverified[key]} times and nothing on the screen shows it had any "
                                      f"effect; it must not be repeated blindly")
                log.event("unverified_repeat_refused", kind=action.kind, attempts=unverified[key], dispatched=True,
                          cause="action_effect_unverified")
                handle_stuck(stuck, observation, name, goal, escalator, surface, log, recorder)
                # the count stays: the same action from the same screen is refused before it is dispatched again
            history.append(action)
            continue
        # Every performed action joins the window: a cycle is only visible when its steps are all remembered.
        # A new or grown output is the one thing that clears it here; a passed checkpoint or a changed URL is
        # what a two-screen cycle produces every turn, so neither counts as progress by itself.
        progress.record(action, observation)
        outputs_now = json.dumps(recorder.outputs, sort_keys=True, default=str)
        if outputs_now != last_outputs:
            progress.reset("output recorded", log)
            unverified.clear()       # measurable progress: an earlier blind attempt may legitimately be tried again
            last_outputs = outputs_now
        elif loop == "warn":
            if recorder.nodes:
                recorder.mark_abandoned(recorder.nodes[-1].id)
            action.result = (f"{action.result}; WARNING: this repeats the last {period} action(s) for the "
                             f"{repeats}nd time from the same screen without any progress (no new output); doing it "
                             f"again will stop discovery, so choose a different approach")
        history.append(action)
    else:
        log.screenshot(surface, "max-steps")
        raise DiscoveryFailed(f"goal not reached within {max_steps} steps")

    recorder.outcomes.extend(extra_outcomes or [])
    success = recorder.success or recorder.last_checkpoint() or {"url_contains": entry_url}
    built = build_linear(
        name=name, goal=goal,
        surface_meta={"kind": "web", "app": app_name, "entry_url": entry_url,
                      "allowed_hosts": list(policy.allowed_hosts)},
        params=params, nodes=recorder.nodes, outputs=recorder.outputs, outcomes=recorder.outcomes,
        success=success, run_id=log.run_id, sensitive=sensitive, planner_name=planner.name,
    )
    built.provenance["interventions"] = len(escalator.interventions)
    if recorder.human_steps:
        built.provenance["human_steps"] = list(recorder.human_steps)
    if fallback and fallback.attempts:
        built.provenance["vision_fallback"] = fallback.provenance()
    if reuses:
        built.provenance["reuses"] = reuses
        built.provenance["planner_decisions"] = turn
    log.event("artifact_built", capability=built.name, version=built.version, nodes=len(built.nodes),
              edges=len(built.edges), entry_node=built.entry_node, outputs=list(built.outputs),
              outcomes=[o["code"] for o in built.outcomes])
    return built


# ---------- automatic reuse of a verified segment ----------

def reuse_segment(candidate: ReuseCandidate, before: Observation, library: Library, recorder: Recorder,
                  params: dict, surface: Surface, policy: Policy, escalator: Escalator, log: RunLog, turn: int,
                  output_contract: dict | None, declared_codes: set[str]) -> dict | None:
    """Run the chosen segment deterministically and inline what it proved; returns the provenance record
    of a reuse that imported something, else None.

    A failure before any physical action, or after only checkpointed actions, leaves the session in
    a known state: discovery continues on it. An unverified physical action does not: a person may
    vouch for the screen (resume) or the session is restarted with the segment excluded.
    """
    key = (candidate.source_digest, candidate.start_node, screen_key(before))
    library.attempted.add(key)
    log.event("reuse_candidate_selected", turn=turn, **candidate_summary(candidate))
    execution = execute_candidate(library, candidate, params, surface, policy, log, turn)
    if execution.rejected is not None:
        log.event("reuse_candidate_rejected", candidate_id=candidate.candidate_id, turn=turn, reason=execution.rejected)
        return None
    result = execution.result
    verdict = classify(result)
    common = {"candidate_id": candidate.candidate_id, "turn": turn, "status": result.status,
              "outcome_code": result.outcome_code, "step_id": result.step_id,
              "executed_path": [e["id"] for e in result.executed_path], "performed_attempts": result.performed_attempts}
    if verdict == "success":
        after = observe_with_retry(surface, log)
        nodes, outputs = importable_nodes(verified_entries(result, complete=True), execution.segment, output_contract,
                                          result.outputs, params)
        reused = execution.segment.nodes[0].action
        judged = bool(reused.checkpoint) or reused.action in ("extract", "extract_many")   # its effect is observable
        if judged and screen_key(after) == screen_key(before) and not outputs:
            log.event("reuse_candidate_rejected", candidate_id=candidate.candidate_id, turn=turn,
                      reason="no progress: the screen is unchanged and nothing was extracted")
            return None
        return import_reuse(candidate, result, nodes, outputs, execution, recorder, log, turn, "succeeded")
    if verdict == "before_acting":
        log.event("reuse_failed", classification=verdict, **common)
        return None
    if verdict == "partial_verified":
        log.event("reuse_failed", classification=verdict, **common)
        nodes, outputs = importable_nodes(verified_entries(result, complete=False), execution.segment, output_contract,
                                          result.outputs, params)
        return import_reuse(candidate, result, nodes, outputs, execution, recorder, log, turn, "partial")
    if verdict == "business_outcome":
        log.event("reuse_failed", classification=verdict, declared=result.outcome_code in declared_codes, **common)
        if result.outcome_code in declared_codes:
            nodes, outputs = importable_nodes(verified_entries(result, complete=False), execution.segment,
                                              output_contract, result.outputs, params)
            return import_reuse(candidate, result, nodes, outputs, execution, recorder, log, turn, "business_outcome")
        raise RestartDiscovery(f"reused segment {candidate.candidate_id} ended in the undeclared business outcome "
                               f"{result.outcome_code!r}", {candidate.source_digest})
    log.event("reuse_failed", classification="uncertain", **common)
    return resolve_uncertain(candidate, result, execution, recorder, surface, escalator, log, turn, output_contract)


def import_reuse(candidate: ReuseCandidate, result, nodes: list[GraphNode], outputs: dict, execution, recorder: Recorder,
                 log: RunLog, turn: int, how: str) -> dict | None:
    if not nodes:
        return None
    ids = recorder.import_nodes(nodes, outputs, execution.segment.outcomes)
    record = {"mode": "automatic", "candidate_id": candidate.candidate_id, "source_name": candidate.source_name,
              "source_version": candidate.source_version, "source_digest": candidate.source_digest,
              "source_path": candidate.source_path, "start_node": candidate.start_node,
              "executed_path": [e["id"] for e in result.executed_path], "imported_nodes": len(ids),
              "imported_as": ids, "outputs": sorted(outputs), "reuse_run_id": log.run_id, "planner_turn": turn,
              "entry_condition": candidate.entry_condition, "ending_checkpoint": candidate.ending_condition,
              "result": how}
    log.event("reuse_succeeded" if how == "succeeded" else "reuse_imported", **{k: v for k, v in record.items()
                                                                              if k != "mode"})
    return record


def resolve_uncertain(candidate: ReuseCandidate, result, execution, recorder: Recorder, surface: Surface,
                      escalator: Escalator, log: RunLog, turn: int, output_contract: dict | None) -> dict | None:
    """A physical action of unproven effect: never repeat it, never continue silently."""
    exclusion = {candidate.source_digest}                 # the whole source: any of its segments could repeat the act
    why = (f"reused segment {candidate.candidate_id} left the session unverified at {result.step_id}: "
           f"{result.outcome_code}")
    if isinstance(escalator.operator, NoOperator):
        raise RestartDiscovery(why, exclusion)
    request = InterventionRequest(run_id=log.run_id, capability=candidate.source_name, goal=candidate.description,
                                  step_id=result.step_id, kind="stuck", reason=why,
                                  observed=str(result.observed)[:400], screenshot=None)
    outcome = escalator.request(request, surface)
    if outcome.disposition == "resume":
        nodes, outputs = importable_nodes(verified_entries(result, complete=False), execution.segment, output_contract,
                                          result.outputs, recorder.params)
        log.event("reuse_resumed_by_operator", candidate_id=candidate.candidate_id, turn=turn)
        return import_reuse(candidate, result, nodes, outputs, execution, recorder, log, turn, "partial")
    if outcome.disposition == "restart":
        raise RestartDiscovery(why, exclusion)
    raise DiscoveryFailed(f"human aborted discovery after an uncertain reuse: {outcome.note}")


# ---------- action failures during discovery ----------

def unverified_key(action: Action, observation: Observation) -> tuple:
    """What makes two attempts "the same blind attempt": the same action on the same control, same address.

    The full screen fingerprint is deliberately not used. A page that merely logged the click, moved a
    spinner or re-rendered a list would count as a different screen and let the identical action through a
    second time, which is exactly what this rule exists to prevent. The address is the coarse boundary that
    a real transition crosses, and measurable progress clears the record explicitly.
    """
    return (action_identity(action), observation.url)


def action_identity(action: Action) -> tuple:
    target = action.target
    return (action.kind, target.role if target else None, target.name if target else None,
            target.ref if target else None, action.value)


def recover_from_action_failure(action: Action, error: TransientError, before: Observation, params: dict,
                                failures: dict, recorder: Recorder, surface: Surface, escalator: Escalator,
                                name: str, goal: str, log: RunLog, vision_enabled: bool = False) -> Action | None:
    """Handle a click, type or navigation the surface could not complete.

    Not performed: the failure and its reason go into the history so the planner can choose
    differently; the same action failing again on the same screen is bounded, after which a
    person decides through the ordinary stuck handoff. A target the surface proved unactionable
    (cause `unactionable_target`) is, when the vision fallback is enabled, returned as a stuck
    action with cause "actionability" for the caller to offer to that bounded fallback instead.
    Performed or unknown: the postcondition the planner named is checked first, and a proven one
    records the action exactly once; otherwise nothing is repeated automatically (an irreversible
    or unknown effect above all) and the ordinary handoff decides. Returns the stuck action to
    hand to vision, or None.
    """
    performed = getattr(error, "performed", "unknown")
    cause = getattr(error, "cause", None)
    reason = str(error)
    log.event("action_failed", kind=action.kind, target=describe(action.target), performed=performed, reason=reason,
              cause=cause)
    if performed == "no":
        key = (action_identity(action), screen_key(before))
        failures[key] = failures.get(key, 0) + 1
        action.result = (f"failed before acting: {reason}; the screen is unchanged, choose a different control or "
                         f"approach (this exact action has failed {failures[key]} time(s) here)")
        if cause == "unactionable_target" and vision_enabled:
            failures.pop(key, None)
            return Action(kind="stuck", stuck_cause="actionability", expect=action.expect,
                          reason=f"{action.kind} on {describe(action.target)} is not actionable and no enclosing "
                                 f"control is: {reason}")
        if failures[key] >= MAX_SAME_ACTION_FAILURES:
            stuck = Action(kind="stuck", reason=f"the same {action.kind} on {describe(action.target)} failed "
                                                f"{failures[key]} times on this screen: {reason}", stuck_cause="planner")
            handle_stuck(stuck, before, name, goal, escalator, surface, log, recorder)
            failures.pop(key, None)
        return None
    if cause == "ambiguous_selection":
        # Typed text may sit in the field, but nothing was chosen: the planner can name the value more precisely
        # or choose differently; the same ambiguity on the same screen is bounded like any pre-action failure.
        key = (action_identity(action), screen_key(before))
        failures[key] = failures.get(key, 0) + 1
        action.result = (f"not selected: {reason}; give a value that matches exactly one suggestion, or choose a "
                         f"different control (this exact selection has been ambiguous {failures[key]} time(s) here)")
        if failures[key] >= MAX_SAME_ACTION_FAILURES:
            stuck = Action(kind="stuck", reason=f"the same {action.kind} on {describe(action.target)} was ambiguous "
                                                f"{failures[key]} times on this screen: {reason}", stuck_cause="planner")
            handle_stuck(stuck, before, name, goal, escalator, surface, log, recorder)
            failures.pop(key, None)
        return None
    expected = substitute(action.expect, params)
    proof = proof_after_failure(action, expected, before, surface, log, params)
    if proof:
        # The action took effect after all: record it once, exactly as a clean run would have.
        log.event("action_proven_after_failure", kind=action.kind, target=describe(action.target), expect=expected,
                  proof=proof)
        verify_and_record(action, before, params, None, recorder, surface, log)
        return None
    action.result = f"the {action.kind} may or may not have happened ({reason}); a person was asked to look"
    stuck = Action(kind="stuck", reason=f"{action.kind} on {describe(action.target)} ended in an unknown state and "
                                        f"must not be repeated blindly: {reason}", stuck_cause="planner")
    handle_stuck(stuck, before, name, goal, escalator, surface, log, recorder)
    return None


def proof_after_failure(action: Action, expected: str | None, before: Observation, surface: Surface,
                        log: RunLog, params: dict | None = None) -> str | None:
    """What proves that an action the surface lost track of did happen: "text", when the expected text was
    absent before the action and appeared after it; "url", when the page's URL changed after it (the
    expectation is then dropped so the recorded checkpoint is the URL change). Text that was already on
    the screen before the action proves nothing and is never taken as proof.

    A selection proves nothing this way at all: `select` types the value into the control itself, so the
    value, a suggestion showing it, or any other matching text on screen would all "appear" whether or
    not an option was ever accepted. Only the adapter's own acceptance check can confirm one, so a failed
    select is never recorded (`unverifiable_by_screen`).
    """
    if action.kind == "select":
        log.event("checkpoint_not_evidence", kind="select", expected=expected,
                  reason="a selection cannot be proven from the screen: the value typed into the control looks the "
                         "same whether or not an option was accepted")
        return None
    if expected and expected in visible_text(before):
        log.event("checkpoint_not_evidence", expected=expected, reason="the text was already on screen before acting")
    elif expected and wait_for_text(surface, expected):
        return "text"
    try:
        after = surface.observe()
    except TransientError:
        return None
    # The same invariant as ordinary recording: a URL predicate that was already true proves nothing, so a
    # lost action is proven by the URL only when a safe part of it is false before and true afterwards.
    if url_checkpoint(before.url, after.url, params) is not None:
        action.expect = None          # the URL change is the proof; a text guess must not become the checkpoint
        return "url"
    return None


# ---------- loop pieces ----------

def describe(element) -> str | None:
    return f"{element.role} '{element.name or element.text}'" if element else None


def activated_control(action: Action, seen: Observation) -> Element:
    """The control the artifact must locate: the enclosing control the surface activated when the target
    itself could not take the click (as perception listed it, when it did), else the target."""
    if action.performed_on is None:
        return action.target
    for element in seen.elements:
        if element.ref and element.ref == action.performed_on.ref:
            return element
    return action.performed_on


def entry_checkpoint(observation: Observation) -> dict | None:
    """The first heading on the entry page is a cheap 'we are on the right app' check."""
    for element in observation.elements:
        if element.role == "heading" and element.text:
            return {"text_contains": element.text}
    return None


def navigate_with_retry(surface: Surface, url: str, log: RunLog, attempts: int = 3) -> None:
    """The entry page can be slow like any other; a transient load failure is retried, not fatal."""
    for attempt in range(1, attempts + 1):
        try:
            surface.navigate(url)
            return
        except TransientError as error:
            log.event("transient_error", attempt=attempt, error=str(error))
            if attempt == attempts:
                raise DiscoveryFailed(f"entry page kept failing to load: {error}") from error
            time.sleep(attempt)


def observe_with_retry(surface: Surface, log: RunLog, attempts: int = 3) -> Observation:
    """Discovery tolerates a slow/failed load by waiting, the same way replay will."""
    for attempt in range(1, attempts + 1):
        try:
            return surface.observe()
        except TransientError as error:
            log.event("transient_error", attempt=attempt, error=str(error))
            if attempt == attempts:
                raise DiscoveryFailed(f"surface kept failing to load: {error}") from error
            time.sleep(attempt)
            if error.url:
                surface.navigate(error.url)
    raise AssertionError("unreachable")


def perform(action: Action, params: dict, surface: Surface, log: RunLog) -> None:
    """Execute one allowed action. Extract reads; the others drive the surface.

    A malformed extraction is refused here as a not-performed action failure, whatever built it, so it
    is never logged as acted and never records an empty output.
    """
    problem = extraction_problem(action)
    if problem:
        raise ActionError(f"{action.kind} was not performed: {problem}", performed="no")
    concrete_value = substitute(action.value, params)
    # The committed state of the field group this action runs in, read before it runs. It is the only
    # baseline a form-local commit can be judged against, and it must be taken while the screen is still
    # the one the planner chose from.
    action.committed_before = committed_state(surface, action.target)
    # Whether the control already read as selected. A control that publishes the state only once it is
    # selected has nothing to compare against afterwards unless the "not selected" reading is kept now.
    action.selected_before = selected_state(surface, action.target)
    if action.kind == "navigate":
        surface.navigate(concrete_value or "")
    elif action.kind == "back":
        surface.back()
    elif action.kind == "click":
        action.performed_on = surface.click(action.target)
    elif action.kind == "type":
        surface.type(action.target, concrete_value or "")
    elif action.kind == "select":
        surface.select(action.target, concrete_value or "")
        log.event("selection_accepted", how=getattr(surface, "last_selection_proof", ""))
    # "extract" touches nothing; its value is read during verification below.
    log.event("acted", kind=action.kind, target=describe(action.target),
              value="[typed]" if action.kind in ("type", "select") else concrete_value,
              performed_on=describe(action.performed_on))


def extraction_problem(action: Action) -> str | None:
    """Why an extraction cannot run, or None: exactly one target and a name for extract; a non-empty, bounded,
    duplicate-free ordered target list and a name for extract_many."""
    if action.kind not in ("extract", "extract_many"):
        return None
    if not action.output_name or not str(action.output_name).strip():
        return "no output_name was given"
    if action.kind == "extract":
        return None if action.target is not None else "no target control was given"
    if not action.targets:
        return "no target controls were given"
    if len(action.targets) > MAX_EXTRACT_TARGETS:
        return f"{len(action.targets)} targets exceed the limit of {MAX_EXTRACT_TARGETS}"
    keys = [(t.ref, t.role, t.name, t.text, t.context, tuple(t.box)) for t in action.targets]
    if len(set(keys)) != len(keys):
        return "the same control was selected more than once"
    return None


def verify_and_record(action: Action, before: Observation, params: dict, policy: Policy | None,
                      recorder: Recorder, surface: Surface, log: RunLog) -> None:
    """Record what was actually done, with the proof the action left as its checkpoint (`checkpoint_for`).

    An action nobody asked anything of is recorded as before: its own effect and retry safety describe it,
    and the edges judge the screen. An action whose explicit expectation was contradicted is different: the
    one thing that was supposed to prove it did not happen. Such an action is recorded only when some other
    deterministic proof exists (`independent_proof`: a safe before/after URL change, which is also how an
    adopted new page shows up, or a native state transition on the control). With no proof at all it is not
    a graph node: the planner is told through the structured cause `action_effect_unverified`, and it may not
    simply try the same thing again (see `unverified` in the discovery loop).
    """
    if action.kind == "extract":
        record_extraction(action, recorder, log, before)
        return
    if action.kind == "extract_many":
        record_list_extraction(action, recorder, log, before, surface=surface)
        return
    expected = substitute(action.expect, params)
    met = not expected or wait_for_text(surface, expected)
    if not met:
        observed = visible_text(surface.observe())[:300]
        log.event("expectation_failed", expected=expected, observed=observed)
        log.screenshot(surface, "expectation-failed")
        after = surface.observe()
        proof = independent_proof(action, before, after, params, log, surface,
                                  policy or recorder.policy)
        if proof is None:
            # Nothing shows this action did anything. Recording it would put a blind step in the artifact.
            action.result = (f"unverified: {expected!r} never appeared and nothing else on the screen shows the "
                             f"{action.kind} had any effect, so it was not recorded; the screen shows: {observed}")
            action.unverified = True
            log.event("action_effect_unverified", kind=action.kind, target=describe(action.target),
                      cause="action_effect_unverified")
            return
        action.expect = None               # the expectation was wrong; the independent proof stands instead
        action.expectation_contradicted = True     # evidence the planner was guessing, for the abandonment check
        action.result = f"done; {expected!r} never appeared, but {proof['why']} shows the {action.kind} took effect"
        log.event("action_effect_proven", kind=action.kind, target=describe(action.target), proof=proof["why"])
        action.checkpoint = proof["checkpoint"]
    else:
        action.result = "ok"
        action.checkpoint = checkpoint_for(action, expected, met, before, surface.observe(), log, params, surface,
                                           policy or recorder.policy)
    if before.dialog and action.kind == "click":
        # Clicking while a modal is up is a dismissal, not part of the main flow.
        recorder.record_recoverable_dialog(before.dialog, action)
        log.event("recorded_recoverable_outcome", dialog=before.dialog[:60])
    else:
        node = recorder.record_step(action, risk=(policy or recorder.policy).risk_of(action), seen=before)
        log.event("recorded_step", step_id=node.id, action=action.kind, checkpoint=node.action.checkpoint,
                  effect=node.effect, retry_safety=node.retry_safety)


def independent_proof(action: Action, before: Observation, after: Observation, params: dict,
                      log: RunLog, surface: Surface | None = None, policy: Policy | None = None) -> dict | None:
    """Deterministic evidence that an action took effect although its expectation was contradicted.

    Only deterministic browser evidence, tried in order: a safe before/after URL change (`url_checkpoint`,
    which is also how a page the click opened shows up), the acted control itself becoming selected
    (`selection_transition`), a native state the control changed as perception reports it
    (`state_transition`), and a commit inside the control's own field group (`commit_transition`). Each
    returns the checkpoint that proves it, or None when no such checkpoint can be written. Text is not
    consulted here, because the text that was supposed to appear is precisely what did not, and styling is
    never consulted at all.
    """
    if action.kind in ("navigate", "back", "click"):
        checkpoint = url_checkpoint(before.url, after.url, params)
        if checkpoint is not None:
            checkpoint.pop("_part", None)
            return {"checkpoint": checkpoint, "why": "the page address changed"}
    # A click the surface reports as dispatched is not evidence that it did anything, and an enclosing
    # control activated in its place is less so: only an observable change counts. A page the click opened
    # is adopted by the surface, so it shows up as the URL change above.
    #
    # Disappearance is checked before any state read: a control that is gone cannot report its state, and
    # asking would spend the whole locator timeout discovering that. The URL rung stays ahead of it, so a
    # navigation that replaced the whole document is never read as this control dismissing something.
    gone = dismissal_transition(action, before, after, policy, log)
    if gone is not None:
        return gone
    if surface is not None:
        became = selection_transition(action, surface, log)
        if became is not None:
            return became
    changed = state_transition(action, before, after)
    if changed is not None:
        return {"checkpoint": None, "why": f"the control is now {changed}"}
    if surface is not None:
        committed = commit_transition(action, surface, params, log)
        if committed is not None:
            return committed
    return None


def dismissal_transition(action: Action, before: Observation, after: Observation, policy: Policy | None,
                         log: RunLog) -> dict | None:
    """A proven interface dismissal whose own control is gone: the dismissal happened.

    Only for a control the policy already proves is a UI dismissal (`Policy.dismisses_interface`), judged
    from the same browser-derived metadata the safety rules use. For any other control disappearance
    proves nothing: an ordinary button, link or submission may vanish because the page re-rendered, and a
    destructive or unknown control must never be proven by vanishing at all.

    The false-before/true-after invariant applies as everywhere else: the exact target must have been in
    the pre-action observation and must be absent from the one after. "The same control" means the same
    structural reference; a different control that merely shares its name is not the one that was clicked,
    so a panel closing while an identically named control remains elsewhere is still proven, and a control
    that is still there is not.
    """
    if action.kind != "click" or action.target is None or policy is None:
        return None
    ref = action.target.ref
    if not ref:
        return None
    if not policy.dismisses_interface(action):
        return None
    if not any(element.ref == ref for element in before.elements):
        return None                    # it was not there to begin with: nothing to have disappeared
    if any(element.ref == ref for element in after.elements):
        log.event("dismissal_not_proven", reason="the control is still on screen")
        return None
    return {"checkpoint": {"target_absent": "true"}, "why": "the dismissal's own control is gone"}


def selection_transition(action: Action, surface: Surface, log: RunLog) -> dict | None:
    """The acted control went from not selected to selected, by browser state alone.

    The reading is taken from the control itself, or from a control deterministically associated with it
    (a label's input, a single wrapped native control, a hidden control it drives). Only a false-to-true
    transition counts: a control that was already selected proves nothing about this click, and one whose
    selectedness the browser cannot report at all yields no proof rather than a false negative. Styling is
    never consulted, so a control that merely changed colour is not proven by this rule.

    Asked once per action and remembered: both the expectation-met and the expectation-failed paths need
    the answer, and reading the live control twice would both cost a round trip and log the same refusal
    twice. `_selection_proof` holds the memo; `False` means "asked, and there was no proof".
    """
    if action.selection_proof is not False:
        return action.selection_proof
    proof = read_selection_transition(action, surface, log)
    action.selection_proof = proof
    return proof


def read_selection_transition(action: Action, surface: Surface, log: RunLog) -> dict | None:
    """The uncached reading: see `selection_transition`."""
    was = action.selected_before
    if not was:
        return None
    now = selected_state(surface, action.target)
    if not now:
        log.event("selection_not_proven", reason="the control no longer reports a selected state")
        return None
    # A control that publishes no state at all still had its class list read, and a state word appearing
    # there is a transition from "no state word" to "selected". That is the only case where a reading the
    # browser called unknown may take part, and only against a later reading that is a class token.
    if not was.get("known") and not (was.get("class_token") is False and now.get("source") == "class-token"):
        return None
    if not now.get("known"):
        log.event("selection_not_proven", reason="the control no longer reports a selected state")
        return None
    if was.get("known") and was.get("source") != now.get("source"):
        # The two readings came from different evidence, so they are not comparable as a transition.
        log.event("selection_not_proven", reason="the control's selected state is reported differently now")
        return None
    if was.get("ref") != now.get("ref"):
        log.event("selection_not_proven", reason="the selected state now comes from a different control")
        return None
    if was.get("selected") or not now.get("selected"):
        log.event("selection_not_proven",
                  reason="the control was already selected" if was.get("selected") else "the control is not selected")
        return None
    return {"checkpoint": {"target_selected": "true"}, "why": f"the control is now selected ({now['source']})"}


def commit_transition(action: Action, surface: Surface, params: dict, log: RunLog) -> dict | None:
    """A form-local commit: an editable field gave up its value and a new selected token carries it.

    Both halves are required, and both are read from the acted control's own field group, so a token that
    appears elsewhere on the page, a toast, arbitrary new copy and a screen fingerprint change are never
    proof. The token must be new (a pre-existing one proves nothing), must carry its own removal control or
    a native selected state, and exactly one new token may qualify: two are ambiguous and refused. The value
    left in the field is not proof by itself, and a value the redaction rules rewrite is never compared by
    literal or written down.
    """
    before = action.committed_before
    if not before or not before.get("scoped"):
        return None
    after = committed_state(surface, action.target)
    if not after or not after.get("scoped"):
        return None

    cleared = [was for was in before.get("fields", [])
               if str(was.get("value") or "").strip()
               and str(field_value(after, was.get("ref"))).strip() != str(was.get("value")).strip()]
    if not cleared:
        log.event("commit_not_proven", reason="no editable field in the group gave up its value")
        return None

    # Compared by what makes each token removable or selected, never by its text: two chips may
    # legitimately read the same, and the canonical element may sit at a different depth run to run.
    known = {token_key(token) for token in before.get("tokens", [])}
    fresh = [token for token in after.get("tokens", [])
             if token_key(token) not in known and (token.get("removable") or token.get("selected"))]
    if not fresh:
        stored = new_stored(before, after)
        if stored is None:
            log.event("commit_not_proven", reason="no new selected token or stored value appeared in the group")
            return None
        return {"checkpoint": None, "why": "the field was committed into a new stored value"}
    if len(fresh) > 1:
        log.event("commit_not_proven", reason=f"{len(fresh)} new tokens appeared and none of them can be chosen")
        return None

    token = fresh[0]
    checkpoint = token_checkpoint(token, params, log.secrets)
    if checkpoint is None:
        # The token exists but cannot be written down safely (a sensitive or unrepresentable value).
        log.event("commit_not_proven", reason="the committed token cannot be recorded safely")
        return None
    return {"checkpoint": checkpoint, "why": "the field was committed into a new removable token"}


def field_value(state: dict, ref: str | None) -> str:
    """What the field with this structural reference holds now; "" when it is gone."""
    return next((str(f.get("value") or "") for f in state.get("fields", []) if f.get("ref") == ref), "")


def new_stored(before: dict, after: dict) -> str | None:
    """A value a hidden input or select in the group newly holds, or None."""
    known = {(v.get("ref"), v.get("value")) for v in before.get("stored", [])}
    fresh = [v for v in after.get("stored", []) if (v.get("ref"), v.get("value")) not in known]
    return fresh[0].get("value") if len(fresh) == 1 else None


def token_checkpoint(token: dict, params: dict, secrets: tuple = ()) -> dict | None:
    """A replay-verifiable checkpoint for a committed token, or None when it cannot be written safely.

    The token's own visible text is what replay looks for, stored as a placeholder when it is a declared
    input's value, exactly as every other checkpoint is. A value the redaction rules would rewrite is never
    written: such a commit keeps the unverified handoff instead.
    """
    text = " ".join(str(token.get("text") or "").split())
    if not text or len(text) > MAX_TOKEN_TEXT:
        return None
    if unsafe_token_text(text, secrets):
        return None
    return {"selection_present": parameterize(text, params)}


def unsafe_token_text(text: str, secrets: tuple = ()) -> bool:
    """Whether a token's own text must never be written into an artifact.

    A registered sensitive literal (compared here, never logged or serialized), anything the redaction
    rules rewrite, and anything opaque enough that no reader could recognise it as page state. Unlike a
    URL fragment there is no key to inspect, so the text is judged whole. A sensitive value is refused
    even though `parameterize` would replace it: a placeholder would make the commit look assertable when
    replay cannot compare the literal it stands for without putting it back on screen.
    """
    if any(secret and secret in text for secret in secrets):
        return True
    if redact(text) != text:
        return True
    return bool(OPAQUE_VALUE.match(text)) and not text.replace(" ", "").isalpha()


def committed_state(surface: Surface, target: Element | None) -> dict | None:
    """The acted control's field-group state, when the surface can report it; None otherwise."""
    return read_state(surface, "committed_state", target)


def selected_state(surface: Surface, target: Element | None) -> dict | None:
    """Whether the acted control reads as selected, when the surface can report it; None otherwise."""
    return read_state(surface, "selected_state", target)


def read_state(surface: Surface, name: str, target: Element | None) -> dict | None:
    """One optional read-only perception call, never allowed to break the action it precedes."""
    reader = getattr(surface, name, None)
    if reader is None or target is None:
        return None
    try:
        return reader(target)
    except Exception:                       # perception must never break an action that is about to run
        return None


def state_transition(action: Action, before: Observation, after: Observation) -> str | None:
    """A native state the control itself changed: "checked", "expanded=false" and so on, else None."""
    target = action.target
    if target is None or not target.ref:
        return None
    was = next((e for e in before.elements if e.ref == target.ref), None)
    now = next((e for e in after.elements if e.ref == target.ref), None)
    if was is None or now is None:
        return None
    for state in ("checked", "selected", "expanded", "pressed"):
        old, new = was.states.get(state), now.states.get(state)
        if new is not None and old is not None and new != old:
            return f"{state}={new}"
    return None


def checkpoint_for(action: Action, expected: str | None, met: bool, before: Observation, after: Observation,
                   log: RunLog, params: dict | None = None, surface: Surface | None = None,
                   policy: Policy | None = None) -> dict | None:
    """The proof a performed action left behind, or None when nothing observable can be attributed to it.

    Expected text that was absent before the action and present after it is evidence of the action; text that
    was already on the screen before (a site-wide heading) proves nothing and is never recorded, whatever the
    planner expected. A page change proven by the URL is the fallback for a navigate, back or click: the new
    path becomes a url_contains checkpoint. A typed value is proven by text that newly shows it; without that,
    typing is recorded without a checkpoint. An expectation the screen contradicted leaves no checkpoint at all.

    A control that became selected is proof whatever the planner expected, so it is considered here too:
    genuinely new text still wins (it describes the screen the next step will act on), but where the
    expectation was already on screen, was absent altogether, or the URL says nothing, the selection stands
    rather than recording nothing at all.
    """
    params = params or {}
    if not met:
        return None
    if expected and expected not in visible_text(before):
        return {"text_contains": expected}
    if expected:
        log.event("checkpoint_not_evidence", expected=expected, reason="the text was already on screen before acting")
    if action.kind in ("navigate", "back", "click"):
        checkpoint = url_checkpoint(before.url, after.url, params)
        if checkpoint is not None:
            part = checkpoint.pop("_part")
            log.event("checkpoint_from_url", kind=action.kind, part=part, url_contains=checkpoint["url_contains"])
            return checkpoint
        if after.url != before.url:
            log.event("checkpoint_not_evidence", kind=action.kind,
                      reason="the url changed but no part of it can be asserted safely and was false before")
    # The same rungs a contradicted expectation would use, in the same order: a vanished dismissal is
    # checked before any state read, since a control that is gone cannot report its state.
    for proof in (dismissal_transition(action, before, after, policy, log),
                  selection_transition(action, surface, log) if surface is not None else None):
        if proof is None:
            continue
        log.event("action_effect_proven", kind=action.kind, target=describe(action.target), proof=proof["why"])
        if expected:
            action.result = (f"done; {expected!r} was already on screen before the action, so it is not proof "
                             f"the action worked; {proof['why']}, and that was recorded instead")
        return proof["checkpoint"]
    if expected:
        action.result = (f"done; {expected!r} was already on screen before the action, so it is not proof the "
                         f"action worked and no checkpoint was recorded; next time expect text that only appears "
                         f"afterwards")
    return None


def url_path(url: str) -> str:
    """The path of a URL, without host, query or fragment: what a url_contains checkpoint can safely assert."""
    path = urlparse(url).path
    return path if path and path != "/" else ""


# Query keys whose values identify a session or a campaign rather than the state an action reached: asserting
# one would tie a capability to a single visit. Matched case-insensitively against the whole key.
UNSTABLE_QUERY_KEYS = re.compile(r"^(?:utm_.*|ga_.*|fb.*|gcl.*|msclk.*|mc_.*|ref|referrer|session|sid|token|nonce|"
                                 r"cache|_|timestamp|ts|rand|v)$|id$|token$|hash$|nonce$|time$", re.IGNORECASE)


def url_checkpoint(before_url: str, after_url: str, params: dict) -> dict | None:
    """A `url_contains` checkpoint that is false for `before_url` and true for `after_url`, or None.

    The whole point of a checkpoint is to prove the action happened, so a predicate that already held before
    the action is worthless however true it is afterwards. The path is preferred when it genuinely changed;
    otherwise a query or fragment change is asserted through the smallest safe fragment of it, with literal
    input values replaced by their placeholders. Nothing sensitive, nothing unstable, and never a raw URL:
    when no part qualifies, there is no checkpoint and the caller records the action without one.
    """
    if before_url == after_url:
        return None
    candidates: list[tuple[str, str]] = []
    path = url_path(after_url)
    if path and path != url_path(before_url):
        candidates.append(("path", path))
    after, prior = urlparse(after_url), urlparse(before_url)
    if after.query != prior.query:
        old_pairs = set(parse_qsl(prior.query, keep_blank_values=True))
        for key, value in parse_qsl(after.query, keep_blank_values=True):
            if (key, value) in old_pairs or not value or UNSTABLE_QUERY_KEYS.search(key):
                continue
            candidates.append(("query", f"{key}={value}"))
    if after.fragment and after.fragment != prior.fragment:
        candidates.append(("fragment", f"#{after.fragment}"))
    for part, raw in candidates:
        if unsafe_for_checkpoint(raw):
            continue
        asserted = parameterize(raw, params)
        # The invariant, checked on the values a replay would compare: false before, true after.
        if asserted is None or substitute(asserted, params) in before_url:
            continue
        if substitute(asserted, params) not in after_url:
            continue
        return {"url_contains": asserted, "_part": part}
    return None


# A value long and random-looking enough to be a credential or an opaque handle rather than readable state.
OPAQUE_VALUE = re.compile(r"^[A-Za-z0-9._~-]{16,}$")


def unsafe_for_checkpoint(raw: str) -> bool:
    """Whether a URL fragment must never be written into an artifact.

    A sensitive key name, a value any redaction rule rewrites, a value that carries a registered secret, or
    an opaque high-entropy value that no reader could recognise as page state.
    """
    key, _, value = raw.partition("=")
    if SENSITIVE_KEY_PATTERN.search(key):
        return True
    if redact(raw) != raw:
        return True
    return bool(value) and bool(OPAQUE_VALUE.match(value)) and not value.isalpha()


def parameterize_checkpoint(checkpoint: dict, params: dict) -> dict:
    return {key: parameterize(value, params) for key, value in checkpoint.items()}


def record_extraction(action: Action, recorder: Recorder, log: RunLog, before: Observation,
                      locator: Locator | None = None) -> None:
    if not action.output_name or action.target is None:
        action.result = "extract needs output_name and a target control"
        return
    pattern = recorder.output_pattern(action)
    value = apply_pattern(action.target.text or action.target.name, pattern)
    if not value:
        action.result = f"pattern {pattern!r} matched nothing in {action.target.text!r}"
        log.event("extraction_failed", output_name=action.output_name, text=action.target.text)
        return
    action.result = f"ok, extracted {value!r}"
    recorder.record_output(action, value)
    node = recorder.record_step(action, risk="safe", seen=before, locator=locator)
    log.event("recorded_output", output_name=action.output_name, value=value, step_id=node.id)


def result_identity(element: Element) -> str:
    """The identity a result is known by: its visible text or accessible name, whitespace and case normalized.

    Deliberately nothing structural. A dynamic page re-renders the same rows with new element references,
    paths, geometry and enclosing items, and none of that changes which results are on screen.
    """
    return " ".join((element.text or element.name or "").split()).casefold()


def snapshot_instability(wanted: list[str], elements: list[Element]) -> str | None:
    """Why the identities `wanted` cannot be trusted on this observation, or None.

    They must still appear, each exactly once, and in the same displayed order relative to one another.
    Anything else on the page is irrelevant: unrelated content may come and go freely. A duplicated or
    missing identity is refused rather than resolved by document order, since neither copy can be shown
    to be the result that was read.
    """
    seen = [result_identity(element) for element in elements]
    for identity in dict.fromkeys(wanted):
        count = seen.count(identity)
        if count == 0:
            return "missing"
        if count > 1:
            return "ambiguous"
    if len(set(wanted)) != len(wanted):
        return "ambiguous"                       # the read itself named one result twice
    return None if [identity for identity in seen if identity in set(wanted)] == wanted else "reordered"


def stable_result_list(action: Action, surface: Surface, log: RunLog) -> str | None:
    """Why the list under an ordered snapshot cannot be trusted yet, or None.

    The identities just read must still be on screen, each exactly once and in the same order, after a
    bounded re-observation: a list that reshuffles or re-renders different rows while it is being read
    would fix the wrong order for every later page. Identities are compared as normalized text
    (`result_identity`), never as structural references, so a page that merely re-renders is stable.
    Only the extraction that takes the snapshot is checked this way, and no identity reaches the log.
    """
    wanted = [result_identity(element) for element in action.targets]
    problem = "missing"
    for attempt in range(1, MAX_SNAPSHOT_ATTEMPTS + 1):
        time.sleep(SNAPSHOT_RECHECK_MS / 1000)
        try:
            elements = surface.observe().elements
        except TransientError:
            continue
        problem = snapshot_instability(wanted, elements)
        if problem is None:
            log.event("result_list_stable", results=len(wanted), attempts=attempt)
            return None
        log.event("result_list_unstable", results=len(wanted), attempt=attempt, problem=problem)
    return (f"the result list kept changing while its order was being read ({MAX_SNAPSHOT_ATTEMPTS} attempts, "
            f"last seen {problem}): the first {len(wanted)} results cannot be fixed, so nothing was recorded")


def record_list_extraction(action: Action, recorder: Recorder, log: RunLog, before: Observation,
                           ladders: list[Locator] | None = None, surface: Surface | None = None) -> None:
    """Every selected element is read in the requested order through one item rule, all or nothing: one item
    that does not parse means nothing is recorded (no node, no partial example) and the planner is told; for an
    optional list the message says the whole output would be absent, so the planner may select differently."""
    pattern = recorder.output_pattern(action)
    values: list[str] = []
    for index, element in enumerate(action.targets):
        value = apply_pattern(element.text or element.name, pattern)
        if not value:
            kind = "optional" if action.optional else "required"
            action.result = (f"item {index} ({describe(element)}): pattern {pattern!r} matched nothing in "
                             f"{element.text!r}; the {kind} list {action.output_name!r} was not recorded (a list is "
                             f"all or nothing: select only items that carry the value)")
            log.event("extraction_failed", output_name=action.output_name, target_index=index, optional=action.optional,
                      text=element.text)
            return
        values.append(value)
    if surface is not None and recorder.aligned_lists() and not recorder.snapshot:
        unstable = stable_result_list(action, surface, log)
        if unstable:
            action.result = unstable
            log.event("extraction_failed", output_name=action.output_name, reason="unstable result list")
            return
    action.result = f"ok, extracted {len(values)} item(s)"
    recorder.record_list_output(action, values)
    recorder.take_snapshot(action, values)
    node = recorder.record_step(action, risk="safe", seen=before, ladders=ladders)
    log.event("recorded_output", output_name=action.output_name, value=values, items=len(values), step_id=node.id)
    if recorder.snapshot and recorder.snapshot_output == action.output_name and len(recorder.snapshot) == len(values):
        log.event("result_snapshot_taken", output_name=action.output_name, results=len(values))


def infer_type(sample: str) -> str:
    """Declare the output type from the discovery-time sample so callers get numbers as numbers.

    Currency signs and thousands separators are tolerated ("$ 1,299.99" is a number).
    """
    plain = re.sub(r"^[$€£]\s*", "", sample.strip()).replace(",", "")
    if plain.isdigit():
        return "integer"
    if re.fullmatch(r"\d+\.\d+", plain):
        return "number"
    return "string"


def apply_pattern(text: str, pattern: str | None) -> str:
    if not pattern:
        return text.strip()
    match = re.search(pattern, text)
    if not match:
        return ""
    return (match.group(1) if match.groups() else match.group(0)).strip()


def wait_for_text(surface: Surface, expected: str, timeout: float | None = None) -> bool:
    """Poll perception until the expected text shows up. This is the discovery-time checkpoint.

    The default is read at call time rather than bound at import, so the module's timeout is what every
    caller actually waits; a default captured in the signature would freeze the value a test or a caller
    later changes.
    """
    deadline = time.time() + (EXPECT_TIMEOUT_S if timeout is None else timeout)
    while True:
        try:
            if expected in visible_text(surface.observe()):
                return True
        except TransientError:
            pass
        if time.time() >= deadline:
            return False
        time.sleep(POLL_INTERVAL_S)


def finish(action: Action, observation: Observation, params: dict, recorder: Recorder, log: RunLog) -> bool:
    """The planner says the goal is met. Trust but verify against the screen."""
    expected = substitute(action.expect, params)
    if expected and expected not in visible_text(observation):
        action.result = f"claimed done but {expected!r} is not on screen"
        log.event("done_rejected", expected=expected)
        return False
    problems = recorder.contract_problems()
    if problems:
        action.result = (f"not done: the declared outputs are not complete yet: " +
                         ", ".join(f"{name} ({why})" for name, why in problems.items()) +
                         "; record the missing or incomplete outputs before finishing")
        log.event("done_rejected", outputs=problems)
        return False
    recorder.record_business_outcomes(action.outcomes)
    if expected and not depends_on_extracted_values(expected, recorder):
        recorder.success = {"text_contains": parameterize(expected, recorder.params)}
    else:
        log.event("success_checkpoint_fallback", reason="done text was empty or was an extracted value")
    log.event("goal_reached", outcomes=[o["code"] for o in action.outcomes])
    return True


def depends_on_extracted_values(expected: str, recorder: Recorder) -> bool:
    """A success check must hold for every input; a value read from this run (a balance) does not."""
    for spec in recorder.outputs.values():
        example = spec.get("example")
        samples = example if isinstance(example, list) else [example]
        if any(sample and str(sample) in expected for sample in samples):
            return True
    return False


def handle_stuck(action: Action, observation: Observation, name: str, goal: str, escalator: Escalator,
                 surface: Surface, log: RunLog, recorder: "Recorder | None" = None) -> None:
    """The planner cannot proceed: bring a human in on the same session, then let the planner look again.

    With a recorder (discovery), the person's performed navigate/click/type actions become ordinary action
    nodes in execution order when the handoff ends with resume; restart abandons the attempt and its human
    actions; abort ends discovery. Without a recorder nothing is recorded.
    """
    request = InterventionRequest(run_id=log.run_id, capability=name, goal=goal, step_id=None, kind="stuck",
                                  reason=f"planner is stuck: {action.reason}",
                                  observed=visible_text(observation)[:400], screenshot=None)
    result = escalator.request(request, surface)
    if result.disposition == "abort":
        raise DiscoveryFailed(f"human aborted discovery: {result.note}")
    if result.disposition == "restart" and recorder is not None:
        raise RestartDiscovery(f"operator chose restart after {len(result.performed)} manual action(s); the attempt "
                               f"and its human actions are abandoned", excluded=set())
    recorded = record_human_actions(result, recorder, log) if recorder is not None else []
    action.result = (f"human intervened ({len(result.human_actions)} manual actions"
                     f"{', recorded as steps ' + ', '.join(recorded) if recorded else ''}); re-observe the screen")


def record_human_actions(result, recorder: "Recorder", log: RunLog) -> list[str]:
    """Record the person's performed actions as action nodes, exactly as automated ones are recorded.

    Each action goes through the same policy check, effect and retry classification, locator ladder,
    placeholder substitution and checkpoint rule (a URL change the action caused). An action whose outcome
    the surface lost, one the policy would deny at replay, or a typed value that is sensitive by field name
    and not a declared parameter cannot be represented safely: discovery fails with an
    `unrecordable_human_action` error rather than building an artifact that silently depends on it.
    """
    ids: list[str] = []
    policy = recorder.policy
    first = result.performed[0] if result.performed else None
    if first is not None and first.kind == "navigate" and first.performed == "yes" and first.after is not None \
            and first.after.url != first.before.url:
        recorder.prune_abandoned_suffix(first.before, log)
    for human in result.performed:
        what = f"the operator's {human.kind} on {describe(human.target) or human.value}"
        if human.performed != "yes":
            raise DiscoveryFailed(f"unrecordable_human_action: {what} may not have completed; an artifact cannot "
                                  f"depend on it")
        action = Action(kind=human.kind, target=human.target, value=human.value, reason="performed by the operator")
        url = human.before.url
        verdict = policy.check(action, url)
        if verdict.decision == "deny":
            raise DiscoveryFailed(f"unrecordable_human_action: {what} would be denied at replay ({verdict.reason})")
        if human.kind == "type" and human.value and human.value not in map(str, recorder.params.values()) \
                and redact({human.target.name: human.value})[human.target.name] != human.value:
            raise DiscoveryFailed(f"unrecordable_human_action: the value typed into {describe(human.target)} is "
                                  f"sensitive by its field name and is not a declared parameter")
        if human.after is not None:
            action.checkpoint = checkpoint_for(action, None, True, human.before, human.after, log, recorder.params)
        node = recorder.record_step(action, risk=policy.risk_of(action), seen=human.before)
        recorder.human_steps.append(node.id)
        log.event("recorded_human_step", step_id=node.id, action=human.kind, target=describe(human.target),
                  checkpoint=node.action.checkpoint, effect=node.effect, retry_safety=node.retry_safety)
        ids.append(node.id)
    return ids


def confirm_with_human(action: Action, observation: Observation, name: str, goal: str, escalator: Escalator,
                       surface: Surface, log: RunLog, reason: str, recorder: "Recorder | None" = None):
    """Risky actions are never taken on the model's say-so; a person approves or denies."""
    from .policy import Verdict

    request = InterventionRequest(run_id=log.run_id, capability=name, goal=goal, step_id=None, kind="confirm",
                                  reason=f"confirmation required: {reason}",
                                  observed=f"planner wants to {action.kind} {describe(action.target)}; {action.reason}",
                                  screenshot=None)
    result = escalator.request(request, surface)
    if result.disposition == "abort":
        raise DiscoveryFailed(f"human aborted discovery: {result.note}")
    if recorder is not None and result.disposition != "restart":
        record_human_actions(result, recorder, log)          # anything the person did meanwhile is part of the flow
    if result.disposition == "approve":
        return Verdict("allow", "approved by human operator")
    return Verdict("deny", f"human operator did not approve ({result.disposition})")


# ---------- the vision fallback ----------

class Vision:
    """Bounded screenshot fallback for discovery: one visual proposal when the planner is stuck.

    Budget: at most `max_attempts` attempts per run, and never twice for the same screen (a
    fingerprint of the structured observation plus the masked screenshot bytes; the bytes are
    hashed, never stored). A proposal is turned into an ordinary click or type Action whose
    target carries `source="vision"` and no structural reference, so it goes through the normal
    policy check and is acted on by coordinates.
    """

    def __init__(self, planner: VisionPlanner, max_attempts: int, goal: str, visible_params: dict, params: dict,
                 surface: Surface, log: RunLog):
        self.planner, self.max_attempts = planner, max_attempts
        self.goal, self.visible_params, self.params, self.surface, self.log = goal, visible_params, params, surface, log
        self.attempts = 0
        self.last_fingerprint: str | None = None
        self.exhausted_logged = False
        self.recorded_steps: list[str] = []
        self.grounded: list[dict] = []          # visual extractions recorded: step, output, boxes, ladder kinds
        self.frame: ScreenshotFrame | None = None

    def propose(self, stuck: Action, observation: Observation, history: list[Action],
                spec: dict | None = None) -> Action | None:
        if stuck.stuck_cause not in VISION_CAUSES:
            # Provider failures and ordinary uncertainty are not perception gaps: straight to a person.
            self.log.event("vision_fallback_skipped", reason=f"stuck cause is {stuck.stuck_cause or 'unknown'!r}, "
                                                             f"not a missing or unactionable control")
            return None
        capture = getattr(self.surface, "viewport_screenshot", None)
        decide = getattr(self.planner, "decide_visually", None)
        if capture is None or decide is None:
            self.log.event("vision_fallback_unavailable", reason="surface or planner cannot provide vision")
            return None
        if self.attempts >= self.max_attempts:
            if not self.exhausted_logged:
                self.log.event("vision_budget_exhausted", attempts=self.attempts, max_attempts=self.max_attempts)
                self.exhausted_logged = True
            return None
        self.log.shots_dir.mkdir(parents=True, exist_ok=True)
        path = self.log.shots_dir / f"{self.log.seq:03d}-vision-attempt-{self.attempts + 1}.png"
        try:
            frame = capture(str(path))
        except Exception as error:   # an unusable capture (scale mismatch, browser gone) is not a proposal
            self.log.event("vision_fallback_unavailable", reason=f"{type(error).__name__}: {str(error)[:200]}")
            return None
        fingerprint = screen_fingerprint(observation, frame.png)
        if fingerprint == self.last_fingerprint:
            self.log.event("vision_fallback_skipped", reason="screen unchanged since the last visual attempt",
                           fingerprint=fingerprint)
            return None
        self.attempts += 1
        self.last_fingerprint = fingerprint
        self.frame = frame
        remaining = self.max_attempts - self.attempts
        self.log.event("vision_fallback_requested", attempt=self.attempts, reason=stuck.reason, fingerprint=fingerprint,
                       screenshot=str(path), width=frame.width, height=frame.height, scroll=[frame.scroll_x,
                       frame.scroll_y], remaining=remaining)
        goal = self.goal
        if stuck.stuck_cause == "missing_data":
            goal += (f"\n\nMISSING VALUE: the output {stuck.output_name!r} is on screen but not in the list of "
                     f"controls; propose visual_extract with one box per value, in order.")
        decision = decide(goal, self.visible_params, observation, history, frame, remaining)
        target = decision.target
        self.log.event("vision_fallback_decided", attempt=self.attempts, kind=decision.kind,
                       target=describe_visual(target), confidence=target.confidence if target else None,
                       expect=target.expect if target else None, reason=decision.reason[:200],
                       output_name=decision.output_name, boxes=[[b.x, b.y, b.width, b.height] for b in decision.boxes])
        if decision.kind == "unavailable":
            self.log.event("vision_fallback_unavailable", reason=decision.reason[:200])
            return None
        if decision.rejected:
            self.log.event("vision_target_rejected", attempt=self.attempts, reason=decision.rejected)
            return None
        if (decision.kind == "visual_extract") != (stuck.stuck_cause == "missing_data"):
            self.log.event("vision_target_rejected", attempt=self.attempts,
                           reason=f"{decision.kind} does not answer a {stuck.stuck_cause or 'unknown'} stuck state")
            return None
        if decision.kind == "visual_extract":
            return self.ground_extraction(decision, stuck, spec, frame)
        if decision.kind == "no_target" or target is None:
            return None
        rejection = self.unsafe_to_act(decision.value, target.expect, observation)
        if rejection:
            self.log.event("vision_target_rejected", attempt=self.attempts, reason=rejection)
            return None
        element = Element(role=target.role, name=target.name, text="", box=(target.x, target.y, target.width,
                                                                              target.height), source="vision")
        kind = "type" if decision.kind == "visual_type" else "click"
        return Action(kind=kind, target=element, value=decision.value, expect=target.expect,
                      reason=f"vision fallback: {target.reason}")

    def ground_extraction(self, decision: VisualDecision, stuck: Action, spec: dict | None,
                          frame: ScreenshotFrame) -> Action | None:
        """Turn screenshot boxes into an ordinary extraction of page elements, or nothing.

        The trust boundary: the model only says where to look. Every box must be answered by a page element
        with visible text of its own under it, read through the surface at the exact scroll position and
        viewport of the capture; a single box with no page-backed text (a canvas, a picture) rejects the
        whole proposal, so a list stays all or nothing. The declared contract decides the shape: one box for
        a scalar, an ordered list of boxes for a list output.
        """
        name = decision.output_name
        if spec is None or name != stuck.output_name:
            return self.reject(f"visual_extract names {name!r}, not the missing output {stuck.output_name!r}")
        text_under = getattr(self.surface, "text_under", None)
        if text_under is None:
            self.log.event("vision_fallback_unavailable", reason="surface cannot read text under a box")
            return None
        if self.surface.viewport_size() != (frame.width, frame.height) or \
                self.surface.scroll_position() != (frame.scroll_x, frame.scroll_y):
            return self.reject("the viewport or scroll position changed since the capture")
        many = spec.get("type") == "list"
        if not many and len(decision.boxes) != 1:
            return self.reject(f"output {name!r} is a single value but {len(decision.boxes)} boxes were proposed")
        elements: list[Element] = []
        for index, box in enumerate(decision.boxes):
            element = text_under((box.x, box.y, box.width, box.height))
            if element is None or not element.text.strip():
                return self.reject(f"box {index} has no page element with visible text under it")
            elements.append(replace(element, source="vision"))
        self.log.event("vision_extraction_grounded", attempt=self.attempts, output_name=name, boxes=len(elements),
                       grounded=[bool(e.ref) for e in elements])
        return Action(kind="extract_many" if many else "extract", target=None if many else elements[0],
                      targets=elements if many else [], output_name=name, optional=not spec.get("required", True),
                      reason=f"vision fallback: {decision.reason}")

    def reject(self, reason: str) -> None:
        self.log.event("vision_target_rejected", attempt=self.attempts, reason=reason)
        return None

    def unsafe_to_act(self, value: str | None, expect: str, observation: Observation) -> str | None:
        """Why an accepted-looking proposal must still not run: an undeclared placeholder in what would be typed or
        checked, or an expected text that is already on screen (the action could then never be verified)."""
        unknown = (placeholders_in(value) | placeholders_in(expect)) - set(self.params)
        if unknown:
            return f"undeclared placeholders {sorted(unknown)} in the proposed value or expected text"
        expected = substitute(expect, self.params)
        if expected in visible_text(observation):
            return f"expected text {expected!r} is already on screen before acting; the action could not be verified"
        return None

    def provenance(self) -> dict:
        record = {"attempts": self.attempts, "max_attempts": self.max_attempts, "planner": self.planner.name,
                  "steps": list(self.recorded_steps)}
        if self.grounded:
            record["grounded_extractions"] = list(self.grounded)
        return record


def screen_fingerprint(observation: Observation, png: bytes) -> str:
    """A hash of what is on screen: structured controls plus the masked image bytes. Nothing sensitive is kept."""
    digest = hashlib.sha256()
    digest.update(observation.url.encode())
    for element in observation.elements:
        digest.update(f"{element.role}|{element.name}|{element.text}|{element.states}".encode())
    digest.update(png)
    return digest.hexdigest()[:16]


def describe_visual(target) -> dict | None:
    if target is None:
        return None
    return {"role": target.role, "name": target.name[:80], "box": [target.x, target.y, target.width, target.height]}


def verify_vision_action(action: Action, before: Observation, params: dict, policy: Policy, recorder: Recorder,
                         surface: Surface, log: RunLog, fallback: "Vision", name: str, goal: str,
                         escalator: Escalator) -> None:
    """A visual action is kept only when structured perception proves its expected text appeared.

    Otherwise it is neither repeated nor recorded: a person is asked, with the screenshot in the log.
    A visual extraction is kept only when every grounded page element parses under the contract.
    """
    if action.kind in ("extract", "extract_many"):
        record_grounded_extraction(action, before, recorder, log, fallback, name, goal, escalator, surface)
        return
    expected = substitute(action.expect, params)
    verified = wait_for_text(surface, expected)
    log.event("vision_action_verified", verified=verified, kind=action.kind, target=describe_visual_element(action),
              expect=action.expect)
    if not verified:
        # The visual action may already have reached the page: it is never proposed or repeated again.
        action.result = f"visual {action.kind} did not lead to {expected!r}"
        action.unverified = True
        fallback.attempts = fallback.max_attempts          # no further visual proposal this run
        log.event("action_effect_unverified", kind=action.kind, target=describe_visual_element(action),
                  cause="action_effect_unverified", source="vision")
        stuck = Action(kind="stuck", reason=f"visual action could not be verified and must not be repeated: "
                                            f"expected {expected!r}")
        handle_stuck(stuck, surface.observe(), name, goal, escalator, surface, log, recorder)
        return
    action.result = "ok"
    action.checkpoint = {"text_contains": expected}    # absent before acting by construction (Vision.unsafe_to_act)
    node = recorder.record_step(action, risk=policy.risk_of(action), seen=before,
                                locator=vision_locator(action.target, before, fallback.frame))
    fallback.recorded_steps.append(node.id)
    log.event("recorded_step", step_id=node.id, action=action.kind, checkpoint=action.expect,
              effect=node.effect, retry_safety=node.retry_safety, targeting="vision")


def vision_sourced(action: Action) -> bool:
    return (action.target is not None and action.target.source == "vision") or \
        any(element.source == "vision" for element in action.targets)


def record_grounded_extraction(action: Action, before: Observation, recorder: Recorder, log: RunLog, fallback: "Vision",
                               name: str, goal: str, escalator: Escalator, surface: Surface) -> None:
    """Record a visually grounded extraction exactly like an ordinary one, with grounded ladders: the element's
    structural reference when it has one, and the exact capture point as the last rung. A value that does not
    parse under the contract records nothing and goes to a person, never to the model again."""
    ladders = [grounded_locator(element, fallback.frame) for element in (action.targets or [action.target])]
    count = len(recorder.nodes)
    if action.kind == "extract":
        record_extraction(action, recorder, log, before, locator=ladders[0])
    else:
        record_list_extraction(action, recorder, log, before, ladders=ladders, surface=surface)
    if len(recorder.nodes) == count:
        log.event("vision_extraction_failed", output_name=action.output_name, reason=action.result)
        stuck = Action(kind="stuck", reason=f"visual extraction of {action.output_name!r} could not be parsed: "
                                            f"{action.result}")
        handle_stuck(stuck, surface.observe(), name, goal, escalator, surface, log, recorder)
        return
    node = recorder.nodes[-1]
    fallback.recorded_steps.append(node.id)
    fallback.grounded.append({"step_id": node.id, "output": action.output_name, "boxes": len(ladders),
                              "ladders": [[rung["kind"] for rung in ladder.strategies] for ladder in ladders]})
    log.event("recorded_step", step_id=node.id, action=action.kind, checkpoint=None, effect=node.effect,
              retry_safety=node.retry_safety, targeting="vision")


def grounded_locator(element: Element, frame: ScreenshotFrame) -> Locator:
    """The ladder of a visually grounded value: its label or structural rung first, then the exact capture
    point, bound to the viewport and scroll position, which replay reads page text at (never pixels)."""
    x, y, w, h = element.box
    rungs = [rung for rung in locator_for(element).strategies if rung["kind"] != "coords"]
    rungs.append({"kind": "coords", "x": x + w // 2, "y": y + h // 2, "exact": True, "read": True,
                  "viewport": {"width": frame.width, "height": frame.height},
                  "scroll": {"x": frame.scroll_x, "y": frame.scroll_y}})
    return Locator(strategies=rungs)


def vision_locator(target: Element, seen: Observation, frame: ScreenshotFrame) -> Locator:
    """The ladder recorded for a visual target: exact coordinates, bound to the viewport and scroll position
    they were captured at, and a role/name rung only when a structured element with that identity really
    sits under the box (grounded); a guessed name alone must never let replay click a different control."""
    x, y, w, h = target.box
    rungs: list[dict] = []
    for element in seen.elements:
        if (element.role, element.name) == (target.role, target.name) and boxes_overlap(element.box, target.box):
            rungs.append({"kind": "role", "role": element.role, "name": element.name})
            break
    rungs.append({"kind": "coords", "x": x + w // 2, "y": y + h // 2, "exact": True,
                  "viewport": {"width": frame.width, "height": frame.height},
                  "scroll": {"x": frame.scroll_x, "y": frame.scroll_y}})
    return Locator(strategies=rungs)


def boxes_overlap(a: tuple, b: tuple) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return aw > 0 and ah > 0 and ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def describe_visual_element(action: Action) -> dict | None:
    target = action.target
    if target is None:
        return None
    return {"role": target.role, "name": target.name[:80], "box": list(target.box)}
