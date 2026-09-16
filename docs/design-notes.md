# Pathloom design notes

Implementation detail that backs `REPORT.md`. The report stays short; this file is the reference.

## 1. Artifact schemas

**Schema 1.0 (linear).** `artifacts/<name>.v<N>.json`, one immutable file per version, validated on
build and on load. Top level: `schema_version`, `name`, `version`, `status` (`draft` |
`approved`), `description` (the natural-language goal), `surface` (`kind`, `app`, `entry_url`,
`allowed_hosts`), `inputs` (`name -> {type, required, sensitive, description}`), `outputs`
(`name -> {type, required, pattern, description, example}`), ordered `steps`
(`{id, action, target, value, checkpoint, risk}`), `outcomes` (`{code, kind: business |
recoverable, source: observed | planner | reviewer, detect, recover?}`), `success` (final
checkpoint) and `provenance`. Targets are locator ladders ordered most stable first: `role` +
accessible `name` (+ enclosing item `context` when the name alone was ambiguous), visible
`text`, structural `css` path, screen `coords`. Ladders are parameterized like values; a ladder
that depends on an input drops its structural and coordinate rungs so it can never fall back
onto the discovery-time item.

**Schema 2.0 (capability graph).** The same contract with `steps` replaced by `entry_node`,
`nodes` and `edges`. An action node is `{id, kind: action, action: {action, target, value,
checkpoint}, effect, retry_safety}`; the step's id and risk live once, as the node id and its
`effect` (`none` | `reversible` | `irreversible` | `unknown`). `retry_safety` is `safe` |
`verify_before_retry` | `never_retry`; an `irreversible` or `unknown` effect must be
`never_retry`. Decision nodes carry no action, `effect: none`, `retry_safety: safe`. Terminal
nodes carry `status` (`success` | `business_outcome` | `failure`) and an `outcome_code` for
the latter two. Edges are `{source, target, guards, priority}` with typed guards: `always`
(no fields), `input_equals` (`input`, `value`), `text_visible` (`value`), `url_matches`
(`pattern`, a regex), `element_present` (`target`, a ladder), `dialog_contains` (`value`).
All guards on an edge must hold; edges leave a node in ascending priority; an `always` edge
must be the single highest-priority edge of its node. Validation rejects duplicate ids, a
missing or terminal entry node, edges to missing nodes, node data that does not fit the
kind, unsupported or malformed guards, undeclared placeholders, dead ends, unreachable nodes
and cycles. Version-1 artifacts convert in memory with `graph.from_linear` (a chain of action
nodes joined by `always` edges ending in a `success` terminal); the file is never rewritten.

**Inputs with conditions.** A schema 2.0 input may carry `selector: true` (always required; its
value picks the path) and `required_when: [{selector: value, ...}, ...]` (AND inside an object,
OR across objects, selectors only). Replay's missing-input check evaluates the conditions
against the supplied parameters before any surface action.

## 2. Discovery and recording

Discovery is `observe -> decide (LLM, one strict tool call, stateless) -> policy -> act ->
verify -> record`. Every performed action becomes a step; the planner's expected text becomes
the checkpoint only if it was actually seen. Literal input values are replaced by `{{name}}`
placeholders in values, ladders and checkpoints, whole-token only, longest value first, and
never inside an existing placeholder. Sensitive inputs are shown to the model only as the
placeholder, typed for real, masked in logs and screenshots, and refused by `save` if they
survive into the artifact. Two rules were added after live campaign runs: selector values are
recorded literally (a route called `details` must not rewrite the `View details` link the path
clicks), and a declared output contract (`outputs` in the campaign spec) fixes the type,
requiredness and regex of each output whatever the planner proposes, so independent runs
record one contract.

Perception (`surface.py`) is an operator's view, not the DOM: role, accessible name, visible
text, enclosing item, box, and computed states. Rules added after live runs: a native control's
tag role wins over a conflicting ARIA role, while ARIA supplies the role when no native control
role exists; an icon-only control is named by its test id or element id rather than by a numeric
badge; text nested inside a control is the control's, not a separate element.

**Accessibility merge (Chromium).** Each observation runs the page projection, then reads
`DOM.getDocument` and `Accessibility.getFullAXTree` over one CDP session (opened lazily, reused).
`dom_paths` recomputes the projection's `tag:nth-of-type(n)` path for every element from the DOM
tree, so an accessibility node's `backendDOMNodeId` maps to the same `ref` the projection uses:
no per-node protocol call. Nodes are kept when not ignored and their computed role is in
`AX_ROLES` (checkbox, radio, switch, combobox, listbox, option, menuitem, tab, slider,
spinbutton, button, link, textbox, treeitem), bounded to `MAX_AX_NODES`. States kept:
`checked`, `expanded`, `selected`, `pressed` with any value, and `disabled`, `required`,
`readonly` only when true. Merge rules by `ref`: projected element present with a native role
-> keep it, add states, source `dom+ax`; projected element is generic `text` -> upgrade to the
computed role and name, source `ax`; not projected -> one page call enriches all such nodes with
visibility, box, text (never a password's value, never a checkbox's `on`) and enclosing item,
invisible ones are dropped, source `ax`. The result is sorted by document order. The merge works
on copies, so a failure at any stage (tree, mapping, enrichment) returns the caller's projection
untouched, object for object, and sets `last_accessibility_error`; the next observation recovers.
Registered secrets are masked in names, texts, contexts and dialog text. Screenshots mask them
too: text nodes (which covers option labels) and the current values of text-like inputs and
textareas are replaced for the capture and restored exactly in `finally`; values are assigned
directly, so no input or change event fires, and password fields already render as dots. `states` and `source`
are runtime-only fields on `Element`; artifacts are unchanged. Limits: Chromium only; option
nodes inside a closed `<select>` are invisible and therefore not listed; shadow DOM and iframes
are not walked by either source.

**Vision fallback (discovery only, opt-in).** `agent.Vision` runs only when the structured
planner returns `stuck` with `stuck_cause == "perception"`, which the tool contract produces
solely from an explicit `stuck_cause: missing_control`; refusals, malformed output, bad indexes
(`provider`) and ordinary uncertainty (`planner`) go straight to a person, as do denials,
confirmations, effect handling, transient errors, outcomes, bad parameters and completed goals.
Each attempt: `surface.viewport_screenshot` (masked like `screenshot`, viewport only,
`scale="css"` so pixels are mouse coordinates, the PNG header checked against the viewport,
scroll position recorded, saved as the only image file, as evidence), a fingerprint of the structured observation plus the masked bytes (hashed,
never stored), then `planner.decide_visually(goal, redacted params, observation, history, frame,
remaining)`. Budget: `max_vision_attempts` per run, at most one attempt per fingerprint, no retry
after `no_target`, `vision_budget_exhausted` logged once. The provider adapters send the PNG as
an image block (Anthropic) or `input_image` data URL (OpenAI) with the `choose_visual_action`
tool; `visual_decision_from_tool_input` refuses unknown kinds, non-finite or out-of-viewport
boxes, confidence under 0.6, a missing expected text, and a `visual_type` without a value; any
API failure yields `unavailable`. The agent then refuses a proposal whose value or expected text
carries an undeclared placeholder, or whose expected text is already on screen (it could never
prove the click). An accepted proposal becomes an ordinary `Action` whose target is an `Element`
with `source="vision"`, the box, and no structural ref, so the normal policy check applies and
the click lands at the box centre. After acting, `verify_vision_action` waits for the expected
text to newly appear through structured perception; success records the step with
`vision_locator`: a `coords` rung with `exact: true`, the capture `viewport` and `scroll`, preceded
by a role/name rung only when a structured element with that identity overlaps the box; failure
escalates to a person and records nothing. Replay is untouched except in `_resolve_one`: an exact
rung is honoured only in the recorded viewport and scroll position and then resolves to the
recorded point itself (never to a containing element); legacy `coords` rungs keep their old
behaviour. Events:
`vision_fallback_requested`, `vision_fallback_decided`, `vision_target_rejected`,
`vision_action_verified`, `vision_budget_exhausted`, `vision_fallback_skipped`,
`vision_fallback_unavailable`; never image bytes, secrets or raw responses.

## 3. Graph replay

`graph_replay.py` walks `entry_node -> ... -> terminal` with no model. Action nodes reuse the
version-1 helpers through a `Step` view: dialog recovery, policy check, ladder resolution,
acting, checkpoint wait (with the version-1 business-outcome scan while a checkpoint is
pending, so converted artifacts classify outcomes exactly as before), output extraction. An
action without a checkpoint passes at once and its edges judge the screen. Edge selection
observes, evaluates guards in priority order (each evaluation logged), takes the first match,
polls guarded edges for a bounded time, takes the `always` fallback only after the wait, clears
known dialogs meanwhile and hands off to a human when nothing matches. Terminals verify the
top-level `success` checkpoint or return the declared business or failure code.

**Effect policy** (`--irreversible-policy deny|confirm|allow`, default `confirm`). The global
allowlist and blocked-control policy runs first and is never overridable. `none`/`reversible`
run without effect-based confirmation; `irreversible` is refused under `deny`, confirmed by a
human under `confirm`, run under `allow`; `unknown` is refused under `deny` and otherwise
always needs a human. When the policy classifies a control as risky but the graph calls it
reversible, the stricter view wins and a human decides (or `deny` refuses).

**Retry safety.** A transient error before the node acted is a plain bounded retry for every
mode. After acting: `safe` repeats within the budget; `verify_before_retry` skips when the
checkpoint holds, repeats otherwise, and asks a human when there is no checkpoint;
`never_retry` skips on proof and otherwise asks. Resume after such a handoff proceeds to the
node's edges without repeating the action. `restart` returns to the entry node on the same
session, so a one-time expired session is resolved by choosing `restart` immediately: the
artifact signs in again itself, and signing in by hand first would leave the entry checkpoint
(the sign-in page) unmet.

## 4. Campaigns and the prefix-tree merge

A campaign spec (`scenarios/*.json`) declares `name`, `goal`, `url`, `selectors`, `scenarios`
(`name`, `params` including every selector, `sensitive`) and optionally `outputs`. Selector
combinations must be unique and selectors cannot be sensitive (their values are written into
guards). Each scenario is discovered on a fresh session with its own log and evidence, in
order; any failure (a stopped discovery or an unexpected exception) ends the campaign with the
summary written, the session closed, secrets redacted and nothing saved.

A spec may also declare `outcomes`: reviewer-declared business or recoverable outcomes in the
artifact's own structure (`code`, `kind`, `detect`, `recover` for recoverable ones), validated by
the artifact's `validate_outcome`, with `source: reviewer`. Every scenario's discovery receives
them through `extra_outcomes`; for a code the reviewer declared, the reviewer's definition wins:
planner copies are dropped, an identical one silently and a different one with a
`reviewer_outcome_overrode_planner` event (both definitions) in the scenario's run log. Planner
outcomes with other codes are kept. The merged graph therefore classifies states discovery
never visited.

Traces are merged only when their contract agrees: name, goal, surface kind, entry URL,
allowed hosts, success checkpoint, and for every output the same name, type, requiredness and
pattern. Outcomes are unioned by code; a code defined differently is a conflict. Merging is a
prefix tree: two steps share a node only when action, parameterized value, full ladder,
checkpoint, effect and retry safety are identical (ids are ignored). The entry node is a
decision gate with one edge per declared scenario, guarded by the complete selector
assignment, so an undeclared combination stops before any action. At the first difference a
decision node branches, again on complete assignments; merging continues inside branches;
nothing is shared after divergence; a scenario that ends where another continues gets a guarded
edge to the success terminal. Node ids follow depth-first creation order, priorities follow
specification order, and edges are sorted, so the same traces give the same JSON. Provenance
records the campaign id, merge strategy, planners, and per scenario the selector assignment,
discovery run id, planner, step count and node path.

## 5. Lifecycle

`draft -> supervised replay or stability runs -> approve -> approved -> unattended replay`.

`stability` replays one invocation N times, each on a fresh browser, session control, log and
context, unattended, with the irreversible policy forced to `deny`, and closes every session.
`report.json` (schema 1.0) records the artifact identity and SHA-256, parameter names only,
the non-sensitive selector assignment, counts by status and outcome, success and clean-run
rates, totals of recoveries, interventions and drift signals, a record per run with its
evidence, and `eligible_for_approval`: at least three completed runs, all `success`, no
intervention, one digest throughout. Every summary field is derived from the run records by one
function that approval re-runs.

`approve` accepts only a draft, verifies each report (readable, right schema, about this name and
version, the draft's exact digest, internally consistent, eligible by recomputation), derives the
required selector coverage from the graph's entry gate (exactly one `input_equals` guard per
selector on every gate edge), cross-checks campaign provenance, requires one report per
assignment, and saves the next version with `status: approved` and approval provenance
(reviewer, timestamp, source version and digest, report paths and digests, tested assignments).
The draft is never modified. `replay --operator none` refuses a draft with
`artifact_not_approved` before opening a browser; `lifecycle.replay_any` enforces the same rule
for library callers through an explicit purpose. Every replay bundle holds `run.jsonl`,
screenshots, a redacted `result.json`, the exact artifact that ran and `manifest.json`.

This is a local review record. There is no signing: anyone who can write to `artifacts/` can
approve, and a report rewritten consistently end to end passes the consistency checks.
