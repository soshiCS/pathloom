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

@dataclass
class Step:
    id: str
    action: str                   # click | type | navigate
    target: Locator | None
    value: str | None = None      # may contain {{param}} placeholders
    checkpoint: dict | None = None  # {"text_contains": "..."} — asserted after the action
    risk: str = "safe"            # safe | risky (irreversible)


@dataclass
class Artifact:
    schema_version: str
    name: str
    version: int
    status: str                   # draft | approved
    description: str
    surface: dict                 # {"kind": "web", "app": ..., "entry_url": ...}
    inputs: dict                  # name -> {"type", "required", "sensitive"}
    outputs: dict                 # name -> {"type", "locator", "pattern"}
    steps: list[Step]
    outcomes: list[dict]          # declared non-success states with kind business|recoverable
    success: dict                 # final checkpoint
    provenance: dict              # run id, timestamp, model — NOT the transcript


# ---------- capability graph (schema 2.0) ----------

NodeKind = Literal["action", "decision", "terminal"]
Effect = Literal["none", "reversible", "irreversible", "unknown"]
RetrySafety = Literal["safe", "verify_before_retry", "never_retry"]
GuardKind = Literal["always", "input_equals", "text_visible", "url_matches", "element_present", "dialog_contains"]


@dataclass
class Guard:
    """One deterministic condition on an edge. Which fields a kind carries is fixed (graph.GUARD_FIELDS):

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
    """What an action node does: a version-1 Step without its id and risk, which the node carries
    as GraphNode.id and GraphNode.effect."""
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
class ArtifactV2:
    """The version-1 contract with the ordered step list replaced by a guarded, acyclic graph."""
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


@dataclass
class ScenarioTrace:
    """The verified linear artifact one scenario's discovery run recorded."""
    scenario: Scenario
    artifact: Artifact
    run_id: str
    planner: str


@dataclass
class CampaignResult:
    campaign_id: str
    artifact_path: str
    summary_path: str
    graph: ArtifactV2
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
