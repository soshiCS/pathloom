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


@dataclass
class Observation:
    url: str
    elements: list[Element]
    dialog: str | None = None     # a modal is covering the screen (interstitial, error, notice)

ActionKind = Literal["navigate", "click", "type", "extract", "done", "stuck"]

@dataclass
class Action:
    """What the planner (LLM) decided to do next."""
    kind: ActionKind
    target: Element | None = None
    value: str | None = None      # text to type / url to navigate
    output_name: str | None = None  # for extract
    pattern: str | None = None    # for extract: regex whose first group (or whole match) is the value
    optional: bool = False        # for extract: the value may legitimately be absent for some inputs
    expect: str | None = None     # text the planner expects to see after acting (becomes the checkpoint)
    reason: str = ""              # model's rationale — logged, never stored in the artifact
    outcomes: list[dict] = field(default_factory=list)  # declared on "done": known non-happy-path states
    result: str = ""              # filled in by the agent after acting ("ok", "denied: ...", "expected X not seen")
    stuck_cause: str = ""         # for "stuck": perception (a needed control is not in the list) | planner | provider


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
    """One visual proposal: visual_click | visual_type | no_target, or unavailable when the provider cannot see."""
    kind: str
    target: VisualTarget | None = None
    value: str | None = None      # for visual_type; may contain {{param}} placeholders
    reason: str = ""
    rejected: str | None = None   # why a proposal was refused by validation (out of range, low confidence, ...)


class TransientError(Exception):
    """Slow/failed load. Recoverable by waiting and retrying."""

    def __init__(self, message: str, url: str | None = None):
        super().__init__(message)
        self.url = url  # page that failed to load, so replay can reload it before retrying


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
    action: str                   # click | type | navigate | extract
    target: Locator | None
    value: str | None = None      # may contain {{param}} placeholders; the output name for extract
    checkpoint: dict | None = None  # {"text_contains": "..."} — asserted after the action


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
class InterventionResult:
    resolved: bool
    human_actions: list[dict]
    note: str = ""
    disposition: str = "resume"   # resume | restart | abort | approve | deny — what automation should do next
