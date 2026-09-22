"""Shared data types. Everything the seams exchange is defined here, nowhere else."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------- perception / action (what a Surface exposes) ----------

@dataclass
class Element:
    """One control as a human would perceive it (accessibility-tree style, not DOM)."""
    role: str                     # "textbox" | "button" | "text" | ...
    name: str                     # accessible name / label
    text: str = ""                # visible text content
    box: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h — for screenshot-coordinate fallback
    context: str = ""             # label of the item this control belongs to (e.g. the product card's title)
    ref: str = ""                 # opaque handle owned by the Surface that observed it; never stored in artifacts
    states: dict = field(default_factory=dict)  # computed states when known: {"checked": "true", "expanded": "false"}
    source: str = "dom"           # which perception source(s) produced it: dom | ax | dom+ax; runtime only
    # Runtime-only evidence for the safety policy, never written into an artifact:
    native: str = ""              # the platform control kind: "button:submit", "button:button", "a", "input:radio", ...
    landmark: str = ""            # role of the nearest enclosing dialog, menu, panel or region, if any
    landmark_name: str = ""       # that container's bounded heading or label, masked like any text
    dismisses: bool = False       # browser metadata says this control closes its dialog or popover


@dataclass
class Observation:
    url: str
    elements: list[Element]
    dialog: str | None = None     # a modal is covering the screen (interstitial, error, notice)

# Every kind of decision the runtime can carry: surface actions (navigate, back, click, type, extract,
# extract_many), the choice of a verified library segment (reuse_candidate), and the planner's two
# signals (done, stuck). The artifact records only the surface actions; the policy allows only those.
ActionKind = Literal["navigate", "back", "click", "type", "select", "extract", "extract_many", "reuse_candidate",
                     "done", "stuck"]
# How an extraction lands in the outputs: "set" assigns a scalar output once; "append" adds to a list output.
OUTPUT_MODES = ("set", "append")

@dataclass
class Action:
    """What the planner (LLM) decided to do next."""
    kind: ActionKind
    target: Element | None = None
    value: str | None = None      # text to type / url to navigate
    output_name: str | None = None  # for extract / extract_many
    targets: list[Element] = field(default_factory=list)   # for extract_many: the selected elements, in order
    pattern: str | None = None    # for extract: regex whose first group (or whole match) is the value
    optional: bool = False        # for extract: the value may legitimately be absent for some inputs
    expect: str | None = None     # text the planner expects to see after acting (becomes the checkpoint)
    reason: str = ""              # model's rationale — logged, never stored in the artifact
    outcomes: list[dict] = field(default_factory=list)  # declared on "done": known non-happy-path states
    result: str = ""              # filled in by the agent after acting ("ok", "denied: ...", "expected X not seen")
    stuck_cause: str = ""         # for "stuck": perception (a needed control is not in the list) | planner | provider
    candidate_id: str | None = None   # for "reuse_candidate": the verified segment chosen from those offered
    output_mode: str = "set"      # for extract / extract_many: set (assign once) | append (add to a list output)
    checkpoint: dict | None = None  # filled in by verification: the proof this action left, if any
    performed_on: Element | None = None   # the enclosing control the surface activated instead of the target, if any
    unverified: bool = False
    # an explicit expectation was contradicted, though other evidence proved the action took effect
    expectation_contradicted: bool = False
    # the acted control's field-group state read immediately before the action; runtime only, never stored
    committed_before: dict | None = None
    # whether the acted control read as selected before the action; runtime only, never stored
    selected_before: dict | None = None
    # the selection-transition verdict, asked once after the action: False until asked, then the proof or None
    selection_proof: dict | None | bool = False      # performed, its explicit expectation contradicted, and nothing proves an effect


@dataclass
class ScreenshotFrame:
    """A masked viewport capture handed to a vision-capable planner. Bytes stay in memory; only the
    masked image file (evidence) touches disk."""
    png: bytes
    width: int                    # CSS pixels: the coordinate space of the mouse
    height: int
    path: str = ""                # where the masked image was saved as evidence
    scroll_x: int = 0             # scroll position at capture; exact coordinates are only valid there
    scroll_y: int = 0


@dataclass
class VisualTarget:
    """A control the vision planner points at, in the coordinate space of the screenshot it saw."""
    role: str                     # best semantic guess (button, link, ...)
    name: str                     # short human description
    x: int
    y: int
    width: int
    height: int
    expect: str | None            # text that must appear after acting; the only proof replay can check
    reason: str = ""
    confidence: float = 0.0


@dataclass
class VisualDecision:
    """One visual proposal: visual_click | visual_type | visual_extract | no_target, or unavailable when the
    provider cannot see. A visual_extract names a declared output and the screenshot boxes to read it from, in
    output order; what the model itself read there is never carried, the page supplies the value."""
    kind: str
    target: VisualTarget | None = None
    value: str | None = None      # for visual_type; may contain {{param}} placeholders
    reason: str = ""
    rejected: str | None = None   # why a proposal was refused by validation (out of range, low confidence, ...)
    output_name: str | None = None            # for visual_extract
    boxes: list[VisualTarget] = field(default_factory=list)   # for visual_extract: one box per value, in order


class TransientError(Exception):
    """Slow/failed load. Recoverable by waiting and retrying."""

    def __init__(self, message: str, url: str | None = None):
        super().__init__(message)
        self.url = url  # page that failed to load, so replay can reload it before retrying


PERFORMED_STATES = ("no", "yes", "unknown")
MAX_EXTRACT_TARGETS = 20      # the most elements one extract_many may read into one ordered list
MAX_TOKEN_TEXT = 120          # a selected token longer than this is page copy, not a committed value
# Structured causes an adapter may attach to an action failure: the target cannot take the action and no
# enclosing control can either; or a selection could not be made because more than one option (or popup) fits.
ACTION_FAILURE_CAUSES = ("unactionable_target", "ambiguous_selection", "stale_target")


class ActionError(TransientError):
    """A surface action (click, type, navigate) did not complete.

    `performed` says what the surface knows about the physical action: "no" (it was never
    dispatched: an actionability check or a pre-check failed first), "yes" (it was dispatched but
    did not have the intended effect) or "unknown" (it may or may not have reached the page).
    Callers decide what is safe to repeat from that, never from the message. `url` is set only
    when reloading the page is a sensible recovery. `cause` names a structured reason when the
    adapter has one: "unactionable_target" means the trial proved the control cannot take a
    pointer action and no enclosing control is provably actionable either.
    """

    def __init__(self, message: str, performed: str = "unknown", url: str | None = None, cause: str | None = None):
        if performed not in PERFORMED_STATES:
            raise ValueError(f"performed must be one of {PERFORMED_STATES}, got {performed!r}")
        if cause is not None and cause not in ACTION_FAILURE_CAUSES:
            raise ValueError(f"cause must be one of {ACTION_FAILURE_CAUSES}, got {cause!r}")
        super().__init__(message, url=url)
        self.performed = performed
        self.cause = cause


# ---------- artifact (the capability contract) ----------

@dataclass
class Locator:
    """Ordered ladder of strategies; replay tries each until one resolves.
    Most stable first (role+name), most brittle last (screen coordinates)."""
    strategies: list[dict]

# ---------- the capability graph: the one artifact schema ("2.0") ----------

NodeKind = Literal["action", "decision", "terminal"]
Effect = Literal["none", "reversible", "irreversible", "unknown"]
RetrySafety = Literal["safe", "verify_before_retry", "never_retry"]
GuardKind = Literal["always", "input_equals", "text_visible", "url_matches", "element_present", "dialog_contains"]


@dataclass
class Guard:
    """One deterministic condition on an edge. Which fields a kind carries is fixed (artifact.GUARD_FIELDS):

    always           no fields
    input_equals     input (a declared input name), value (the literal it must equal)
    text_visible     value (text that must be on screen; may contain {{param}})
    url_matches      pattern (regular expression searched in the current url)
    element_present  target (a locator ladder that must resolve)
    dialog_contains  value (text the open dialog must contain)
    """
    kind: str
    input: str | None = None
    value: str | None = None
    pattern: str | None = None
    target: Locator | None = None


@dataclass
class GraphEdge:
    source: str                   # node id
    target: str                   # node id
    guards: list[Guard]           # all must hold; an unconditional edge carries one "always" guard
    priority: int = 0             # lower is tried first; unique among the edges leaving one node


@dataclass
class GraphAction:
    """What an action node does on the surface; its identity, effect and retry policy live on the node."""
    action: str                   # click | type | select | navigate | back | extract | extract_many
    target: Locator | None
    value: str | None = None      # may contain {{param}} placeholders; the output name for extract(_many)
    checkpoint: dict | None = None  # {"text_contains": "..."} or {"url_contains": "..."} — asserted after the action
    targets: list[Locator] | None = None   # extract_many: one ladder per selected element, in output order
    mode: str | None = None       # extract(_many): "append" adds to a list output; None or "set" assigns once


@dataclass
class GraphNode:
    id: str
    kind: str                     # action | decision | terminal
    action: GraphAction | None = None  # action nodes only
    effect: str = "unknown"       # none | reversible | irreversible | unknown; decision/terminal: none
    retry_safety: str = "never_retry"  # safe | verify_before_retry | never_retry; decision/terminal: safe
    status: str | None = None     # terminal: success | business_outcome | failure
    outcome_code: str | None = None  # terminal: a declared outcome code (business_outcome) or a failure code


@dataclass
class Artifact:
    """A capability: typed inputs and outputs, declared outcomes, a success checkpoint, and a guarded,
    acyclic graph of action, decision and terminal nodes. `version` is the capability revision
    (`name.v<version>.json`); `schema_version` is the file format, always "2.0"."""
    schema_version: str
    name: str
    version: int
    status: str                   # draft | approved
    description: str
    surface: dict
    inputs: dict
    outputs: dict
    entry_node: str               # id of the node replay starts at
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    outcomes: list[dict]
    success: dict                 # final checkpoint, asserted on reaching a success terminal
    provenance: dict

# ---------- multi-scenario discovery (campaigns) ----------

@dataclass
class Scenario:
    """One human-declared path through a capability: a full selector assignment plus its own inputs."""
    name: str
    params: dict                  # name -> value; includes a value for every selector
    sensitive: list[str] = field(default_factory=list)


@dataclass
class CampaignSpec:
    name: str                     # capability name
    goal: str
    url: str
    selectors: list[str]          # the declared inputs whose values pick the path
    scenarios: list[Scenario]
    outputs: dict = field(default_factory=dict)  # optional declared contract: name -> {type, required, pattern}
    outcomes: list[dict] = field(default_factory=list)  # reviewer-declared outcomes every scenario records
    reuse_capability: str | None = None          # newest approved artifact with this name opens every scenario
    reuse_artifact: str | None = None            # or this exact approved artifact file


@dataclass
class ScenarioTrace:
    """The verified linear graph one scenario's discovery run recorded."""
    scenario: Scenario
    artifact: Artifact
    run_id: str
    planner: str


@dataclass
class ReusePlan:
    """A resolved, approved capability whose executed path may open a new discovery."""
    path: str
    digest: str
    artifact: Artifact
    name: str
    version: int
    schema_version: str


@dataclass
class ReuseCandidate:
    """A verified segment of an approved artifact whose entry state holds on the current screen.

    What the planner sees when choosing between reuse and a novel action. Inputs and outputs are
    named only; no parameter value ever appears here. `node_ids` are the source's action nodes
    the segment would run, in order.
    """
    candidate_id: str
    source_name: str
    source_version: int
    source_digest: str
    source_path: str
    start_node: str
    description: str
    required_inputs: list[str]
    outputs: list[str]
    outline: list[str]
    entry_condition: dict
    ending_condition: dict
    effects: list[str]
    action_count: int
    node_ids: list[str]


@dataclass
class ReusePrefix:
    """What a successful reuse hands to discovery: the executed action nodes renumbered, their recoverable
    outcomes and outputs, and the provenance to record. Parameter values never appear here."""
    nodes: list[GraphNode]
    outcomes: list[dict]
    outputs: dict
    provenance: dict


@dataclass
class CampaignResult:
    campaign_id: str
    artifact_path: str
    summary_path: str
    graph: Artifact
    scenarios: list[dict]         # one record per scenario: evidence, trace, node path

# ---------- replay result contract ----------

ReplayStatus = Literal["success", "business_outcome", "failure"]

@dataclass
class ReplayResult:
    status: ReplayStatus
    outputs: dict = field(default_factory=dict)
    outcome_code: str | None = None      # stable business-outcome or failure code
    step_id: str | None = None
    expected: str | None = None
    observed: str | None = None
    recoveries: list[str] = field(default_factory=list)  # what was auto-recovered along the way
    interventions: list[dict] = field(default_factory=list)
    executed_path: list[dict] = field(default_factory=list)  # action nodes completed and traversed, once each, in order
    performed_attempts: int = 0                          # physical UI actions attempted (diagnostic; never imported)


# ---------- escalation ----------

@dataclass
class InterventionRequest:
    run_id: str
    capability: str
    step_id: str | None
    reason: str
    observed: str
    screenshot: str | None
    kind: str = "stuck"           # stuck (needs manual steps) | confirm (needs a yes/no on a risky action)
    goal: str = ""                # the capability's natural-language goal, so the operator knows the job


@dataclass
class HumanAction:
    """One surface action a person performed during a handoff, as the runtime saw it: what was done, on which
    perceived control, the screen before and after, and whether it certainly completed. Discovery records it
    as an ordinary action node; replay keeps it as intervention evidence only. Never serialized as such."""
    kind: str                     # navigate | click | type
    target: Element | None
    value: str | None             # the url or the typed text, concrete; parameterized only when recorded
    before: Observation
    after: Observation | None     # None when the screen could not be read afterwards
    performed: str = "yes"        # "yes" | "unknown": a dispatched action whose outcome the surface lost


@dataclass
class InterventionResult:
    resolved: bool
    human_actions: list[dict]     # redacted evidence records, logged
    note: str = ""
    disposition: str = "resume"   # resume | restart | abort | approve | deny — what automation should do next
    performed: list = field(default_factory=list)   # HumanAction records in execution order; runtime only
