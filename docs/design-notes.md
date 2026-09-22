# Pathloom design notes

Implementation detail that backs `REPORT.md`. The report stays short; this file is the reference.

## 1. The artifact schema

**The capability graph.** `artifacts/<name>.v<N>.json`, one immutable file per
revision, validated on build and on load. `schema_version` is the file format and is always
`"2.0"` (the only value `artifact.load_artifact` accepts; any other is refused with
`unsupported schema_version`). `version` is the
capability revision: `checkout_paths.v3.json` is the third revision of `checkout_paths`, and
`lifecycle.load_artifact` refuses a file whose name disagrees with the `name` and `version`
inside it. Top level: `schema_version`, `name`, `version`, `status` (`draft` | `approved`),
`description` (the natural-language goal), `surface` (`kind`, `app`, `entry_url`,
`allowed_hosts`), `inputs` (`name -> {type, required, sensitive, description}`), `outputs`
(`name -> {type, required, pattern, description, example}`), `entry_node`, `nodes`, `edges`,
`outcomes` (`{code, kind: business | recoverable, source: observed | planner | reviewer, detect,
recover?}`), `success` (final checkpoint) and `provenance`. Targets are locator ladders ordered
most stable first: `role` + accessible `name` (+ enclosing item `context` when the name alone
was ambiguous), visible `text`, structural `css` path, screen `coords`. Ladders are parameterized
like values; a ladder that depends on an input drops its structural and coordinate rungs so it
can never fall back onto the discovery-time item.

**Nodes and edges.** An action node is `{id, kind: action, action: {action, target, value,
checkpoint}, effect, retry_safety}` with `effect` `none` | `reversible` | `irreversible` |
`unknown`. `retry_safety` is `safe` |
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
and cycles. The loader rejects unknown keys at every level, so a node's identity and effect
live only on the node, never inside its `action`.

**Linear graphs.** A single discovery records a linear graph: action nodes `s1 .. sN` in the
order performed, joined by `always` edges, ending in the one `success` terminal
(`artifact.build_linear`). `artifact.linear_path` returns the action nodes of such a graph in
flow order and refuses a branching one; the merge, and nothing else, depends on that shape.

**Inputs with conditions.** An input may carry `selector: true` (always required; its
value picks the path) and `required_when: [{selector: value, ...}, ...]` (AND inside an object,
OR across objects, selectors only). Replay's missing-input check evaluates the conditions
against the supplied parameters before any surface action.

## 2. Discovery and recording

Discovery is `observe -> decide (LLM, one strict tool call, stateless) -> policy -> act ->
verify -> record`. Every performed action becomes an action node; its checkpoint is the proof the
action left (`agent.checkpoint_for`, below), never merely the planner's guess. Each node's `effect` comes from the policy's
risk verdict and the action kind (`artifact.classify_effect`: navigate, back and extract `none`; an
allowed click or type `reversible`; a control the policy classifies as risky `irreversible`;
anything else `unknown`) and its `retry_safety` from both (`artifact.classify_retry_safety`:
irreversible or unknown `never_retry`; extract `safe`; a checkpointed `none` or `reversible`
action `verify_before_retry`; a click, type or back without a checkpoint `never_retry`; a navigation
without a checkpoint `safe`). The entry navigation is node `s1`; `build_linear` chains the nodes
with `always` edges into the `success` terminal and validates the result before it is saved.
Literal input values are replaced by `{{name}}`
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
**Action phases (`PlaywrightSurface.click` / `type`).** Every action has an inspection phase (an
explicit `click(trial=True)`, a state read, a label lookup, a fill preflight of attached, visible,
enabled and editable) and a dispatch phase (the real click, fill or key press). The phase that fails
decides `ActionError.performed`: inspection `"no"` (nothing dispatched, no `url`, so replay does not
reload), dispatch `"unknown"`, and a readable post-state that did not change `"yes"`. Playwright's
error family is caught (`playwright.sync_api.Error`, which covers timeouts and detached or replaced
elements); other exceptions are programming errors and propagate. The error text is reduced to its
first line as a diagnostic (`diagnostic`) and never consulted for the classification. Replay counts an
attempt only when `performed` is not `"no"`.

**Choice controls.** A click through a structural reference takes an element handle and asks it
whether it is a native `<input type=radio|checkbox>` (`CHOICE_STATE_JS`; the perceived role is not
trusted); the same handle is used for every later step, so a control the page replaces is seen as
detached or as having left the document rather than re-resolved to a look-alike.
`_activate_choice`: refuse a disabled control; a checked radio is already satisfied (no click); then
`_activate_by_pointer`: `HTMLInputElement.labels` (`for=` and wrapping), the first visible label that
passes the trial is clicked, else the input after its own trial; if neither is actionable,
`_keyboard_activate`: `KEYBOARD_SAFETY_JS` decides from DOM semantics only that the target is a native
enabled radio or checkbox, connected, visible, not inside `[inert]` or `aria-hidden`, not outside an
open `<dialog>` or `aria-modal` dialog, and that what `elementFromPoint` returns at the control's centre
is the control or one of its own labels, or, for a radio only, a sibling of its native group (same
name, same form owner, same type) or one of that sibling's labels; an independent checkbox that merely
shares a name, and any unrelated overlay, refuse the path; then `focus()` and `document.activeElement === el`
must hold; only then
`Space` is pressed. Finally the state is re-read: a checked radio or a toggled checkbox is success,
unchanged is `"yes"`, unreadable or no longer in the document is `"unknown"`. Never `force`, never
script `.click()`, never property assignment, never coordinates in place of a reference.

**Named pointer controls (`namedPointerControl` in `OBSERVE_JS`).** After the native and ARIA branches, an
element qualifies as a `button` when it matches neither `CONTROL_SELECTOR` nor `[tabindex]`, has a non-empty
`aria-label`/`title`, has computed `cursor: pointer` and not `pointer-events: none`, is not `aria-disabled`,
passes `availability` (so hidden, clipped, disabled and covered are out), contains no `CONTROL_SELECTOR`
descendant and no nested named pointer candidate, so the nearest one wins. `pointerBudget` (40) bounds it per
observation. It is pushed through the same `push` helper as every other control, so it deduplicates by
structural reference, carries context and geometry, and is masked like any name. Nothing framework-specific
is read.

**Stale targets (`PlaywrightSurface.still_present`).** `click`, `type` and `select` ask once, without waiting,
whether the structural reference perception recorded is still in the page; when it is not, the action fails
with `ActionError(performed="no", cause="stale_target")` before anything is dispatched. Discovery's ordinary
not-performed path re-observes and returns to the planner; `replay.attempt_action` turns the cause into a
`stale_target` failure with zero actions rather than a transient retry. Nothing parses an exception message,
and a control that was never observed keeps the full actionability wait.

**Availability (`availability` in `HELPERS_JS`).** For every non-text element the projection
pushes: `disabled` (native, `aria-disabled`, a disabled fieldset), `hidden` (`aria-hidden`/inert
subtree), `clipped` (`clippedAway`), else a hit test at the centre: the hit may be the element, a
descendant, an ancestor, one of its labels or a label around it, or anything inside the same
control (`CONTROL_SELECTOR`); otherwise `covered`; a centre outside the viewport is not judged.
Hidden and clipped elements are dropped; disabled and covered ones carry `states`. `ENRICH_JS`
returns the same verdict for accessibility-only elements and `_enrich` applies it; the DOM states
are merged with the accessibility states rather than replaced.

**`select` (`PlaywrightSurface.select`).** `SELECT_KIND_JS` classifies the control (native
`select`; `combobox` by role, `aria-autocomplete` or `list`; else an editable `textbox`).
`match_option` decides for both paths: levels exact, exact after `TRAILING_BRACKET` strips a
parenthesised gloss from the displayed text (never from the wanted value, and never mid-text),
prefix, containing over normalized text,
disabled options excluded; one match at a level is chosen, more than one is `ambiguous` and ends
the search, none moves on; `pick_option` is the single-match view. `_select_native` matches over
`SELECT_OPTIONS_JS`, calls `select_option` and verifies `el.value`; duplicate labels are
`ActionError(performed="no", cause="ambiguous_selection")`. `_select_suggestion` trials the
control, clicks it, clears it with select-all and Delete, types the value key by key
(`KEYSTROKE_DELAY_MS`) so key-driven filters run, and polls `SUGGESTIONS_JS` up to `SUGGESTION_TIMEOUT_MS`: a control with
`aria-controls`/`aria-owns` is searched only in those referenced popups that exist and are
visible; an unlinked control collects every container that passes `usable` (visible, holding rows, not an
ancestor of the control) and `associated` (horizontal overlap with the control and directly below,
above or around it), keeping only the outermost of each nested group, and `SUGGESTIONS_JS` returns
their candidates as separate `groups`. `matching_suggestion` then matches inside each group:
containers with no match are ignored, one container with one match proceeds, an ambiguity inside
the matching container is `ambiguous`, and matches in several containers are `ambiguous_popups`
(`performed="yes"`, the text was typed). A linked control still matches over the popup it names. `rowsIn` collects semantic `[role=option]`/`[role=menuitem]`
/`option` rows, else the innermost actionable descendants (`li`, `a[href]`, `button`, roles,
`[tabindex]`, `[onclick]`, `[data-value]`), else the container's own children with text, so a
custom div widget yields candidates. The matching option is trialed, clicked and checked with
`ACCEPTED_JS` and `acceptance_proof`: the control's value reading as the option (refused when it
is still exactly the typed text and not the option), the active descendant, a newly appearing
stored value from a hidden input or select, or a new chip carrying its own removal control;
`isConnected` failing is `unknown`. Nothing visible on the page counts. `replay.attempt_action` turns an `ambiguous_selection` error into the structured
`ambiguous_selection` failure without a retry; `agent.recover_from_action_failure` tells the
planner and applies the per-screen bound before the stuck handoff. `agent.proof_after_failure`
refuses to prove a `select` from the screen at all (`checkpoint_not_evidence` with
`kind="select"`): the typed value, a suggestion showing it and unrelated matching text are
indistinguishable, so a lost selection goes to the ordinary handoff and records nothing. The
action is in `ActionKind`, `ACTIONS`, `SURFACE_ACTIONS`, the planner tool and prompt, `perform`,
`replay.act` and `describe_node`; the artifact value keeps placeholders and logs show `[typed]`.
The fake surface implements it through an `options` map and a `select_unknown:<name>` fault.

**Human actions in discovery (`escalation.perform_human_action`, `agent.record_human_actions`).**
An operator command (`ConsoleOperator.run_command`, and the test `RecordingOperator`) runs through
`perform_human_action`, which observes the screen, performs the navigate/click/type, observes again
and returns a runtime-only `HumanAction` (kind, perceived target, concrete value, before, after,
`performed` yes/unknown); a failure the surface proves happened before dispatch is raised and the
operator just sees the error. `InterventionResult.performed` carries these records (the redacted
`human_actions` evidence is unchanged). In discovery, `handle_stuck` and `confirm_with_human`
receive the `Recorder`: `abort` raises, `restart` raises `RestartDiscovery` (the session owner
starts a fresh attempt; the human actions are gone with it), anything else calls
`record_human_actions`, which for each action in order refuses an unknown outcome, a policy denial
and a sensitive-by-field-name value that is no declared parameter (`DiscoveryFailed`
`unrecordable_human_action: ...`), computes the checkpoint with `checkpoint_for` (a URL change),
and records through `Recorder.record_step` with `policy.risk_of` (`recorded_human_step`);
`Recorder.human_steps` lands in `provenance.human_steps`. Replay's `escalate` never receives a
recorder, so replay handoffs cannot touch the artifact; a `resume` after the person performed any
action marks the step as acted, so the retry re-checks the node's checkpoint first and repeats
nothing already satisfied, as the handoff rehearsal in the evidence index describes.

**Suffix pruning (`Recorder.prune_abandoned_suffix`).** `record_step` keeps a `trail` per node:
the URL and `stable_screen_key` (controls outside dialog landmarks) of the screen before it, plus
an `abandoned` flag set by a contradicted expectation or by `mark_abandoned` when the loop
detector warned on that step. `record_human_actions` calls the pruner when the first performed
human action is a navigate that changed the URL: the candidate suffix is the trailing run of
nodes on the handoff URL with effect none/reversible, not a read action, not a human step, with a
trail entry (the entry node and imported nodes have none, so they bound it). Same stable key as
the handoff screen: pruned (`speculative_suffix_pruned`, ids and reason only). Different key:
`DiscoveryFailed abandoned_actions_unprovable` when any node is flagged abandoned, else kept
(`speculative_suffix_kept`). Ids are reassigned by position, so the human step takes the first
freed id.

**New pages (`PlaywrightSurface._popups`, `_adopt_popup`).** Every click runs inside `_popups`, a
context manager that collects the browser context's `page` events. "No page yet" and "no more
pages" are different questions and are waited on separately, because a browser may emit the event
several ticks after the click call returns: an empty list is not evidence of a same-page click
until `POPUP_FIRST_MS` has fully elapsed, and only once a page has appeared does collection
continue for `POPUP_SETTLE_MS` so a script opening two windows on separate ticks is seen as two.
Neither loop stops at the first quiet poll; both run their window out. **The tradeoff is click
latency:** every ordinary same-page click pays the whole first window before it is known to be
same-page, so that constant is a floor on click cost and is deliberately kept small (600ms against
a 100ms poll). Raising it buys tolerance for slower openers at a cost paid by every click in every
run; lowering it re-opens the race this window exists to close. `_adopt_popup`: none means a
same-page click; more than one closes them all and raises `ActionError(performed="unknown",
cause="ambiguous_selection")`; one is waited for (`wait_for_load_state`, the bounded timeout, a
slow child left to `observe`), its host checked against `self.allowed_hosts` (passed by the CLI
from the same allowlist the policy enforces; `None` means the policy checks alone), and on refusal
closed with `performed="unknown"` so nothing is retried blindly. An adopted page becomes `_page`,
`_opener` keeps the parent, `_watch` attaches the document-status and navigation listeners, and
`_cdp` is dropped because the accessibility session belonged to the old page. `back` on an adopted
page closes it and restores `_opener` instead of using history. The fake surface models this with
`opens_page` (a `(screen, url)` pair, an int for several pages, a host string for a refusal) and
`opener`.

**Disclosure and query proofs (`policy.proven_disclosure`, `proven_query`).** `risk_of` starts with
`proven_disclosure` (an `expanded` state from the accessibility merge, or `native == "summary"`):
such a control is safe before any wording rule runs. A risky-pattern match is then risky unless
`proven_query` holds: the target's landmark is in `QUERY_LANDMARKS`, or its label or landmark name
matches `QUERY_PURPOSES` (search, find, look up, query, filter, apply filters, show results, go).
`COMMITTING_PATTERNS` (pay, buy, transfer, delete, approve, ...) can never be excused that way, so
`Pay now` inside a search form stays risky. Neither helper reads `Action.reason` or the goal.

**Unique rungs (`Surface.matches`, `replay.resolve_ladder_detail`).** Every surface answers
`matches(strategy)` with all perceived elements a role or text rung matches (`role_matches`
applies role, name and context); `_resolve_one` and the fakes resolve such a rung only for exactly
one match. `resolve_ladder_detail` walks a ladder: a role/text rung with several matches is logged
`rung_ambiguous` and skipped; a rung whose unique result is in `exclude` is `rung_taken` and
skipped; a css or coordinate rung goes through `resolve` unchanged (viewport and scroll rules
intact). The status is `ok`, `ambiguous` (nothing resolved and some rung was ambiguous or taken)
or `missing`. `attempt_action` turns `ambiguous` into the `ambiguous_target` failure before any
action; `extract_many_output` does the same per item, so a list is never partial.

**Parameterized structural disambiguation (`artifact.parameterize_locator`).** Parameterization
drops coordinate rungs but retains the css rung after the semantic rungs. For a single target,
`replay.resolve_ladder_detail` accepts that css result only when its structural reference belongs
to the controls already matched ambiguously by the parameterized role/text rung. A changed input
that matches nothing therefore cannot fall back to the discovery-time position. For list items,
the existing one-to-one resolution rule uses each retained structural rung to distinguish items
that share parameterized wording. Duplicate-target validation is unchanged; output values are
never deduplicated.

**Distinct list targets (`replay.extract_many_output`).** `Surface.resolve(locator, exclude)` skips
elements whose structural reference is in `exclude` (role, text and coordinate rungs by the
perceived `ref`; the css rung by its selector). The list loop passes the identities already taken
(`identity_of`: the ref, else the box); a target that resolves to nothing under exclusion but to
something without it is `ambiguous_target` (a structured failure, no partial output); a bare
coordinate hit with no identity or text is not a resolution (`readable`).

**Loop detection (`agent.ProgressTracker`).** Entries are `(signature, screen_key)`: the signature
is kind, target role/name/context/ref, the parameterized value, output name, list targets and
candidate id, and the screen is the fingerprint of the page the action starts from. Neither
`Action.reason` nor `Action.expect` is included, so the same step reworded is the same entry.
`check` runs before acting: `cycles_at_tail` over `LOOP_PERIODS` (1, 2, 3) so multi-state cycles
count; `LOOP_WARN_REPEATS` (2) warns once (`planner_loop_detected`, the warning appended to the
action's history result), `LOOP_STOP_REPEATS` (3) refuses the action and raises a `planner_loop`
stuck state through `handle_stuck` before `perform` is reached; an action the policy wants
confirmed is stopped at the warning stage instead of being confirmed and repeated. `record` runs
after every verified action, so the cycle's own steps accumulate. `reset` fires only on a changed
serialized outputs map or a human recovery/restart (`progress_made`); a passed checkpoint and a
changed URL do not reset, because a two-screen cycle produces both every turn, which is why the
live arXiv run reached 27 turns undetected. Failed actions are not recorded (the per-screen
failure bound covers them).

**Leaf text (`OBSERVE_JS` `leafText`).** The old rule listed own text only for `p`, `li`,
`span`, `div`, `label` and `th`, so a value in an `<aside>`, `<time>`, `<small>` or a custom
element was dropped, and the accessibility merge folded only control roles, never static text.
Now any visible element's own text is a `text` element unless its tag is structural noise
(`SKIP_TAGS`), an ancestor already emitted covers it (`REPRESENTED`: controls, headings, form
fields; a cell emitted as `cell`), it is inside `[aria-hidden]` or `[inert]`, or `clippedAway`:
no overlap with the document box or with an overflow-hiding ancestor's box. Generated content
(`::before`/`::after` string content with a letter or digit, at most 80 characters, at most 40
elements per observation) is appended to the element's text. `dom_paths` now maps text nodes to
their parent element's path; `_merge_accessibility` adds a `StaticText` node whose backing element
the projection did not list, bounded by `MAX_AX_TEXT_NODES` (40), through `ENRICH_JS`, which
measures the text node's own range box (so a `display: contents` wrapper counts as visible) and
refuses represented or hidden-away text; such elements have role `text`, no name and source `ax`.

**Grounded visual extraction (`Vision.ground_extraction`, `TEXT_UNDER_JS`).** The planner tool's
`stuck_cause` gained `missing_data` with `output_name` (`STUCK_CAUSES`, `VISION_CAUSES`). The loop
lets such a state reach vision only past `Recorder.missing_data_problem` (declared in the
contract, not recorded) and hands the declared spec to `Vision.propose`. The visual tool gained
kind `visual_extract` with `output_name` and `boxes` (`visual_extraction_from_tool_input`: finite,
inside the viewport, confidence at least `MIN_VISION_CONFIDENCE`, at most `MAX_VISUAL_BOXES`; any
reading the model sends is not carried). A `visual_extract` answers only a `missing_data` state
and vice versa. Grounding: the viewport and scroll position must equal the capture's; for each
box `PlaywrightSurface.text_under` hit-tests the box centre (piercing shadow roots), searches down
then up (bounded) for an element with own text whose text-node range box overlaps the box, and
returns it masked with its structural ref (empty inside a shadow tree); any box without one
rejects the whole proposal (`vision_target_rejected`) and the stuck handoff follows. The result is
an ordinary `extract`/`extract_many` action whose elements carry `source="vision"`;
`verify_vision_action` routes it to `record_grounded_extraction`, which records through the normal
`record_extraction`/`record_list_extraction` with `grounded_locator` ladders (the label/structural
rungs of `locator_for`, then `{"kind": "coords", "exact": true, "read": true, viewport, scroll}`);
a value that does not parse records nothing (`vision_extraction_failed`) and hands off. Replay:
`_resolve_one` honours a `read` exact rung only in the recorded viewport and scroll position and
returns `text_under` at that point, so a canvas or a moved layout yields nothing rather than a
wrong value. Provenance: `vision_fallback.grounded_extractions` (step, output, boxes, ladder kinds).
The fake surface grounds boxes through a `grounding` map for offline tests.

**Text nested in a clickable container (`PlaywrightSurface._click_actionable_ancestor`).** The
exact element's trial comes first (Playwright's own retargeting already accepts text inside
`button`, `a[href]`, `role=button` and `role=link`). A failed trial proves nothing was dispatched;
`ACTIONABLE_ANCESTOR_JS` then walks up from the element and returns the nearest ancestor whose
semantics make it a control (native button, input button, native choice control, `summary` in a
`details`, a `label` with a `control`, `select`, an anchor with `href`, or an explicit role from
`INTERACTIVE_ROLES`), refusing a disabled one; a container that only contains the text or only has
a script handler is never returned, and the script neither clicks, focuses nor mutates. That
ancestor gets its own trial and one real click; the returned `Element` (role, name, text, box,
structural ref) becomes `Action.performed_on`, and `Recorder.record_step` locates the node by
`agent.activated_control`: the same control as perception listed it when it did, so the ladder is
the control's ordinary one. Every failure before the ancestor's real click is `performed="no"` with
`ActionError.cause = "unactionable_target"` (a closed vocabulary, `models.ACTION_FAILURE_CAUSES`);
a failure after it is `unknown` as before. `agent.recover_from_action_failure` turns such a failure
into a runtime stuck action with `stuck_cause="actionability"` when a vision fallback is
configured, which the loop takes on the next turn instead of a planner decision (`runtime_stuck`);
`Vision.propose` accepts `VISION_CAUSES` (`perception`, `actionability`) under the unchanged
budgets, and the visual click then goes through the existing policy check, masked capture, the
"expected text absent before" rule, `verify_vision_action` and exact-coordinate recording. Without
vision the planner is told and the per-screen bound hands off as before. A performed or unknown
failure never takes this path.

**Action failures in discovery (`agent.recover_from_action_failure`).** Not performed: the failure
is logged (`action_failed`), the action's `result` carries the reason and a count for the planner,
the loop re-observes and asks again; the same action identity on the same screen key failing
`MAX_SAME_ACTION_FAILURES` (2) times raises the ordinary stuck handoff. Performed or unknown: the
planner's `expect` is checked (`action_proven_after_failure`) and a proven action is recorded once
through the normal path; otherwise the stuck handoff decides and nothing is repeated
automatically (an irreversible or unknown effect above all). `max_steps` counts every decision.
Vision is not involved: only the planner's explicit `missing_control` stuck cause can reach it.

**Extraction.** `extract` reads one element into one scalar output and must carry exactly one
target and an output name; `extract_many` reads an ordered list of elements into one `list` output:
`GraphAction.targets` is an ordered list of at most `MAX_EXTRACT_TARGETS` (20) distinct locator
ladders, `target` is null, `effect` is `none` and `retry_safety` is `safe`. A list output's contract is
`{"type": "list", "required": bool, "items": {"type": string|number|integer, "pattern": regex|null}}`;
the merge compares `items` like any other contract field and a campaign spec declares it the same way.
The planner tool gained kind `extract_many` and `element_indexes` (ordered, distinct, bounded);
`action_from_tool_input` turns a malformed extraction of either kind into a provider stuck state with
the reason, and `agent.extraction_problem` refuses one again at execution as a not-performed action
failure, so it is never logged as acted and never records an output. Discovery reads every selected
element in order through one item rule (a required item that does not parse fails the whole
extraction and tells the planner; an optional one is omitted) and records exactly one node
(`Recorder.record_list_output`). Replay (`replay.extract_many_output`) resolves each ladder on its own
with the usual rung and drift logging (`target_index` on the events), keeps the declared order, parses
each item, all or nothing: a required list whose item is missing or does not parse ends the node through
the existing outcome, failure or handoff paths; an optional list with such an item is absent as a whole
(`optional_output_absent` with the failing `target_index` and a reason that never quotes text, the output
name left out of the result) and the graph continues; a list is never shortened or emitted empty.
Discovery records nothing for a selection with an unparseable item, required or optional, and tells the
planner why. Reuse copies the ladders and the list contract verbatim; the library imports a list output
only under the identical contract.

**URL checkpoints (`agent.url_checkpoint`).** `checkpoint_for` and `proof_after_failure` both go through it.
Candidates, in order: the path when `url_path(after) != url_path(before)`; each query pair that is new and
whose key is not in `UNSTABLE_QUERY_KEYS` (utm/ga/fb/gcl/session/sid/token/nonce/timestamp and any key ending
id/token/hash/nonce/time); the fragment when it changed. Each is refused by `unsafe_for_checkpoint` when its
key matches `SENSITIVE_KEY_PATTERN`, when `redact` rewrites it, or when the value is an opaque high-entropy
token (`OPAQUE_VALUE`). The survivor is parameterized, then checked against the invariant on the values a
replay would compare: `substitute(...)` absent from the pre-action URL and present in the post-action one.
Nothing qualifying means no checkpoint, so the node keeps `never_retry` rather than carrying a predicate that
was already true. This is why a click that changed only the query once recorded the unchanged path.

**Checkpoints are evidence (`agent.verify_and_record`, `agent.checkpoint_for`).** The observation
taken before the action is kept. After a performed action whose expected text was seen: the text
is the checkpoint only when it was absent from the pre-action screen (`{"text_contains": ...}`,
parameterized); when it was already there (`checkpoint_not_evidence` logged) a `navigate`, `back`
or `click` that changed the URL gets `{"url_contains": <new path>}` (`checkpoint_from_url`; host,
query and fragment are never part of it), anything else gets no checkpoint and the planner's
history says why. A typed value is proven by the control showing it (new
text); the vision path keeps its text checkpoint because `Vision.unsafe_to_act` already refuses
text that is on screen before acting. The `done` success checkpoint is unchanged: it describes the
final screen, not an action.

**A contradicted expectation is not a completed action (`agent.independent_proof`,
`agent.unverified_key`).** An action with no requested expectation and an action whose explicit
expectation the screen contradicted are different things, and the second is never quietly recorded
as the first. When the expected text does not appear (`expectation_failed`), discovery looks for
independent deterministic proof that the action did something: a safe before/after address change
through `url_checkpoint`, or a native state the control itself flipped (`checked`, `selected`,
`expanded`, `pressed`, from `state_transition`). A click the surface merely reports as dispatched
is not proof, and neither is an enclosing control activated in its place. With proof, the wrong
guess is discarded, the proof becomes the node's checkpoint (or none, for a state flip, which is
evidence rather than a replayable predicate) and `action_effect_proven` is logged; the step is
still marked `expectation_contradicted`, which is what the abandoned-suffix rule reads. Without
proof, the action is classified `action_effect_unverified`, **no graph node is recorded**, and the
planner is told so in its history rather than being told it succeeded.

The same action on the same control at the same address is then refused *before* it is dispatched
again (`unverified_repeat_refused` with `dispatched: false`), so a second blind attempt never
reaches the page; a person is brought in instead. The key is deliberately the action identity plus
the address, not the full screen fingerprint: a page that merely logged the click or re-rendered a
list would otherwise read as a new screen and let the identical action straight through. A
recorded output clears the record, because that is measurable progress. This build takes the
conservative branch the rule permits and performs no automatic retry at all, in discovery or in
replay, so no control class can be retried by inference from `effect: reversible`. The vision
fallback follows the same rule: an unverified visual action spends the whole visual budget
(`source: "vision"`), so a possibly dispatched action is never proposed a second time.

**Dismissals (`agent.dismissal_transition`, `Policy.dismisses_interface`).** Closing a modal, a panel or
a sidebar often changes no text a planner could predict, changes no address, and leaves nothing selected.
The one thing it reliably does is remove what it closed, including the control that closed it. That is
proof, but only for a control the safety rules already prove is a UI dismissal.

`Policy.dismisses_interface` is deliberately narrower than a `safe` verdict. `safe` also covers controls
that commit nothing for unrelated reasons (a disclosure toggle, a proven query, wording that closes
nothing at all), and for those a disappearance says nothing useful. It answers only whether the platform
itself says this control dismisses the interface around it: an explicit dismissal relationship, a
dismissible landmark, wording that names the interface piece it closes, or a closure object matching the
container's own name. Wording that closes a named resource, a container naming a risky operation, and a
form's submit control are never dismissals, so a destructive or ambiguous close can never be proven by
vanishing.

The false-before/true-after invariant holds as everywhere else: the exact target must appear in the
pre-action observation and be absent afterwards, compared by structural reference. A different control
that merely shares its name is not the one clicked, so a panel closing while an identically named control
remains elsewhere is still proven, and a control still present is not. The rung runs **before** any state
read, because a control that is gone cannot report its state and asking would spend the whole locator
timeout finding that out. The URL rung stays ahead of it, so a navigation that replaced the document is
never read as this control dismissing something.

The checkpoint is `{"target_absent": "true"}` and carries nothing else: no selector, no container name, no
dismissal metadata beyond the locator the step already had. Replay resolves and clicks the step by the
ordinary path, then passes only when the recorded control no longer resolves (`replay.target_absent`). It
fails safely and for a stated reason when the control is still there, when the ladder now resolves
ambiguously, and when another control has taken its place. The coordinate rung is excluded from that
question: it answers "whatever is drawn at this point", which would report a control present as soon as
anything moved into its place.

**Selection transitions (`agent.selection_transition`, `surface.SELECTED_JS`).** A great many controls
neither navigate, nor commit a field, nor change any text: clicking one simply selects it. `perform`
records whether the acted control already read as selected, because a control that publishes the state
only once selected has nothing to compare against afterwards unless the "not selected" reading is kept
first.

`Surface.selected_state` answers from browser state alone, in the order a browser would: the element's own
native selectedness (a checkbox or radio's `checked`, an option's `selected`), the ARIA state it publishes
about itself (`aria-pressed`, `aria-checked`, `aria-selected`, `aria-current`), a boolean state attribute
a custom control sets alongside those (`data-selected`, `data-checked`), or a control deterministically
associated with it: the input a label drives through `for`, the single native control it wraps, or the
hidden control it names through `aria-controls`. Those are attribute *names*, not values, so no site's
class names, framework identifiers or markup conventions are involved.

When a control publishes none of that, one last rung reads its class list (`surface.STATE_TOKENS`). Many
custom controls carry their selectedness nowhere else. This is the weakest evidence here and is treated
accordingly. Whole tokens only, compared by exact membership against a closed vocabulary of eight words
that mean selectedness, in the plain and the `is-` spellings. `unselected`, `selected-item` and
`button-selected-style` are different tokens and never match. `active` is excluded on purpose, because it
equally means focused, running, enabled or hovered, and would turn an unrelated style change into false
proof. The token name is what is read, never the class string, and nothing about it is written down: the
checkpoint stays the abstract `target_selected`, which replay evaluates through the same runtime rule.

A control carrying no state word reports `known: false` with `class_token: false`, which records that the
class list was read and held nothing. That is the one case where a reading the browser called unknown may
take part in a transition, and only against a later reading that is itself a class token, so "no state
word" to "selected" reads as the transition it is without claiming the control had any state before.

Computed style is deliberately absent throughout: colour is how a selection is shown, never what makes it
true, so a control that merely changed a presentational class is not proven. A control that publishes no
state at all still reports `known: false`, and the caller then has no proof rather than a false negative.
Both readings must also come from the same resolved control, compared by structural reference.

Only a false-to-true transition counts. A control that was already selected proves nothing about this
click, and two readings taken from different evidence are not comparable as a transition at all. Each
refusal is logged as `selection_not_proven` with its reason.

The rule applies to every performed click, not only to one whose expectation failed. `checkpoint_for`
consults it too, so the order for a click that succeeded is: genuinely new expected text first (it
describes the screen the next step acts on), then a safe URL change, then the selection. Expected text
that was already on screen still proves nothing, and where it used to leave `checkpoint: null` the
selection now stands in its place; the same holds when the planner expected nothing at all. The verdict is
computed once per action and memoized on it (`Action.selection_proof`), so the live control is not read
twice and a refusal is not logged twice.

The checkpoint is `{"target_selected": "true"}`: it names no markup detail, and replay verifies it by
resolving the node's own recorded locator and asking the same question (`replay.target_selected`). It
fails safely and for a stated reason when the control is absent, when the ladder cannot resolve it
uniquely, when the surface cannot report selectedness, and when the control reports "not selected".
Nothing is clicked and no element is guessed at.

**Form-local commits (`agent.commit_transition`, `surface.COMMITTED_JS`).** A great many controls neither
navigate nor change their own state: they move a value out of a field and into the group as a selected
thing. That is deterministic and local, so it is the third kind of independent proof. Before an action
runs, `perform` records the acted control's field-group state (`Surface.committed_state`, optional on the
protocol): the editable fields' own values, the values stored in hidden inputs and selects, and the
selected tokens. The container is the control's nearest *meaningful* group (a form, search landmark,
fieldset, group or labelled region) and never a bare div or the document, so nothing elsewhere on the page
can be attributed to this action. A password field's contents are never read.

Both halves are required. An editable field in that group must have given up its value, **and** exactly one
new token must have appeared carrying its own removal control or a native selected state. A pre-existing
token proves nothing (tokens are compared by structural reference), two new tokens are ambiguous and
refused, a token in another group is invisible to the rule, and a toast, arbitrary new copy or a screen
fingerprint change is never consulted. A token's text is its own text minus the controls it contains, so a
chip reading "x" for its close button is not a different value. Each refusal is logged as
`commit_not_proven` with its reason. A group whose only change is a new stored value is proven but records
no checkpoint, because a hidden value is evidence rather than something replay can see.

**Canonical tokens (`surface.canonical_tokens`, `surface.token_key`).** A chip is rarely marked up
semantically, so the candidate walk deliberately accepts broad container tags. That means several nested
ancestors around one chip all find the same removal control and all look like tokens, and they cannot be
collapsed during the walk: an outer wrapper is reached before the inner element it wraps, so the canonical
one is not yet known when the wrapper is seen. Perception therefore reports every candidate with its
structural references, and `canonical_tokens` collapses them afterwards in one place shared with replay.

A token's identity is what makes it removable or selected, never its text: a candidate offering a removal
control is keyed by that control, one that is only natively selected by its own backing element. Two chips
may legitimately display the same words, and merging those would hide a real second selection. Within a
key the canonical element is the innermost candidate that still carries the label, the nearest meaningful
container around the removal control. A candidate that merely contains another logical token is a wrapper,
not a token of its own. Document order survives, because the candidates arrive in it. Two chips with
separate removal controls stay two tokens and so stay ambiguous, which is the refusal the rule intends.

Nesting is judged by structural reference (`ref_encloses`), and the path separator is required, so a path
is never read as the ancestor of a sibling whose own path merely begins with the same characters.

Replay applies the same canonical rule (`replay.selected_tokens`), so both sides agree about what one
token is. Its notion of a removal control has to be a little wider: perception lists the label and the
controls but never the container that holds them, so in the usual chip the removal control is the label's
sibling rather than its descendant. A control nested inside the token counts by containment alone; a
sibling counts only when its own accessible name reads as removing or deselecting something
(`REMOVAL_WORDS`), because without containment to scope it any button sharing a parent with any text would
otherwise qualify.

The checkpoint is `{"selection_present": <value>}`, parameterized like every other checkpoint. Replay
verifies it through ordinary perception (`replay.selection_present`): an element whose own visible text is
the value and which reports a selected state or contains a control of its own. That is strictly stronger
than `text_contains`, which page copy would satisfy. A value any redaction rule rewrites, one carrying a
registered sensitive literal, or an opaque one is never written down: such a commit keeps the unverified
handoff instead, so a sensitive value is compared without ever being logged or serialized.

**The checkpoint vocabulary is closed (`artifact.CHECKPOINT_KEYS`, `artifact.validate_checkpoint`).**
`text_contains`, `url_contains` and `selection_present` are the conditions a step checkpoint may assert.
`checkpoint_met` refuses a condition it does not recognise rather than skipping it, and validation refuses
one at save time, so a predicate this engine cannot verify can never sit in an artifact passing vacuously.
A condition name is also checked against the redaction vocabulary: a key reading as sensitive (one
containing "token", "secret" and so on) would have its value replaced wholesale in every log and artifact,
leaving a checkpoint that looks recorded and asserts nothing.

Native control state is part of the screen fingerprint (`agent.stable_screen_key`): a control that
is now pressed or checked is a changed screen even when every label reads the same.

**Replay says which it is (`replay`).** A step that carries no checkpoint logs
`no_checkpoint_recorded` with its action kind. `checkpoint_passed` is only ever written for a
checkpoint that was actually verified, so "no checkpoint exists" and "a checkpoint was verified"
can never be confused in a run log.

**Output accumulation (`Action.output_mode`, `GraphAction.mode`).** `set` (the default, and the
meaning of a node without the `mode` key) assigns an output once; `Recorder.output_conflict`
refuses a second `set` into a recorded name, and an `append` into a scalar, as a not-performed
action failure before anything runs, with the reason in the planner's history. `append` records
one item (`extract`) or every item (`extract_many`, all or nothing) into a `list` output whose
contract is created from the first read (or taken from the declared contract) and whose example
grows in order (`Recorder.append_items`). The node carries `"mode": "append"`; validation
(`artifact._validate_output_mode`) allows `append` only on extraction nodes writing a `list`
output, and `extract` in `set` mode still may not target a list. Replay (`replay.store_items`)
extends `ctx.outputs[name]` in node order, parses a scalar append through the list's `items` rule,
adds a node's items exactly once however often the node is attempted (`ReplayContext.appended`,
cleared with the outputs on a flow restart), and leaves what is already collected alone when an
optional append cannot read a page (`optional_output_absent`). Reuse and the merge carry the mode
with the action; the library describes such nodes as "append to".

**Declared output contract (`discover(output_contract=...)`, `--output-contract`).**
`campaign.load_output_contract` reads a JSON object keyed by output name through the campaign's own
`outputs_from_dict` (a `{"outputs": {...}}` wrapper is accepted), which now also carries `min_items`
and `max_items` and validates every spec with `artifact.validate_output_spec` so a spec and an
artifact obey one rule set (`validate_cardinality`: non-negative integers, `max_items >= 1`,
`min_items <= max_items`, list outputs only). In the `Recorder` the declared type, requiredness,
pattern, items rule and cardinality win over the action (`record_output`, `list_spec`);
`output_conflict` refuses, before anything runs, an output name the contract does not declare
(the history lists the valid names), a `set` into a recorded name, an `extract` in set
mode into a declared list, an `extract_many` or `append` into a declared scalar, and an append that
would exceed `max_items` (`capacity_conflict`: "already complete" when the list is full, otherwise
the whole append is refused, so a multi-item append stays all or nothing). `finish` calls
`Recorder.contract_problems`, which runs `artifact.output_problems` over the recorded examples:
a required output missing, a value that does not fit its type, or a list outside its bounds
rejects `done` (`done_rejected` with `outputs: {name: why}`) and the history entry names only those
outputs; without a contract nothing is checked and discovery infers definitions as before.
`replay.terminal_result` runs the same `output_problems` over the accumulated outputs once the
success checkpoint holds (`outputs_validated`) and returns `failure`/`outputs_incomplete` naming the
outputs and counts, never a value, when anything is off. Patterns are enforced where a value is
read (the stored value is the capture itself), so the final check covers presence, type and
cardinality. A library segment carries only the outputs its own node writes and no whole-flow
bounds (`library.segment_outputs`), since it replays one step of an accumulated list. The merge and
the library compare `min_items`/`max_items` like every other contract field.

**`back`.** A surface action with no target and no value: `Surface.back` moves through the
browser history (`page.go_back`, `wait_until="load"`), reports no history as a not-performed
`ActionError`, a timeout or browser error as `performed="unknown"` (the move may have happened),
and a 5xx as a load failure. It is in `ActionKind`, the planner tool, `policy.SURFACE_ACTIONS`
and `artifact.ACTIONS`; the observation transaction handles the document it lands on. The fake
surfaces keep a history stack the same way.

**Provider retries (`planner.call_provider`).** Both adapters run every request through one
boundary: `failure_category` classifies an exception as `rate_limited` (429), `server_error`
(5xx), `connection` (the SDKs' connection/timeout classes by name, or a built-in
`ConnectionError`/`TimeoutError`) or `permanent` (any other status or class), from attributes and
classes only, never from message text. Transient categories are retried after `PROVIDER_BACKOFF_S`
up to `MAX_PROVIDER_ATTEMPTS` (3) in all, each retry appended to the planner's `retries` list and
logged by the discovery loop as `planner_retry` with the turn; a permanent failure or an exhausted
budget raises `PlannerUnavailable` (provider, category, attempts, error type, status; no provider
text). `reuse.run_discovery` logs `discovery_failed` and the failure screenshot as for any
exception, `cmd_discover` catches it next to `DiscoveryFailed` (evidence copied, one line, exit 2),
and a campaign records the scenario failure and finishes its cleanup unchanged. The vision
fallback uses the same boundary and still reports the provider's own error as `unavailable`.

**The `Close` rule (`Policy.risk_of`).** Verdicts are `safe`, `risky` or `unknown`
(`classify_effect` maps unknown to an `unknown` effect: never retried, confirmed at replay). The
label is `control_label`: the accessible name, plus the visible text only when it adds words
(`Close`/`Close` is one `Close`; a glyph-only text is dropped). Order: a risky-pattern match is
risky; no close word is safe (`Dismiss`, `×`, `Hide glossary`); `closes_a_resource(label)` (the
`closure_object`, the word after articles, is in `RESOURCE_WORDS`) is risky;
`names_interface_piece(label)` (in `INTERFACE_WORDS | PANEL_WORDS`, which include glossary, help,
reference and the like) is safe; an unknown noun is safe when the container's words name it
(`Close thingamajig` inside the Thingamajig panel) and otherwise falls to the bare-close rule;
otherwise the bare close is judged by `nearby_context` (the enclosing landmark's name plus the
item title `contextOf` found): a context that closes a resource is risky, one matching a risky
pattern is unknown, `proven_dismissal` is safe (never a `button:submit` without dismissal
metadata; `Element.dismisses` from `<form method=dialog>` in a `<dialog>`, `formmethod=dialog`,
`popovertargetaction=hide` or a framework dismiss attribute; `Element.landmark` in
`DISMISSIBLE_LANDMARKS`; or a context word in `INTERFACE_WORDS | PANEL_WORDS`), else unknown.
The evidence is perceived by `OBSERVE_JS` (`nativeOf`, `landmarkOf`, `dismissesOf`) into
runtime-only `Element` fields (`native`, `landmark`, `landmark_name`, `dismisses`), masked like any
text and never written into an artifact; the planner prompt does not render them.

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
text to newly appear through structured perception; success records the node with
`vision_locator`: a `coords` rung with `exact: true`, the capture `viewport` and `scroll`, preceded
by a role/name rung only when a structured element with that identity overlaps the box; failure
escalates to a person and records nothing. Replay is untouched except in `_resolve_one`: an exact
rung is honoured only in the recorded viewport and scroll position and then resolves to the
recorded point itself (never to a containing element); a `coords` rung without viewport
metadata resolves anywhere. Events:
`vision_fallback_requested`, `vision_fallback_decided`, `vision_target_rejected`,
`vision_action_verified`, `vision_budget_exhausted`, `vision_fallback_skipped`,
`vision_fallback_unavailable`; never image bytes, secrets or raw responses.

**The verified artifact library (`library.py`).** `build_library` scans the artifact directory
once per discovery: a file enters the catalog only if `lifecycle.load_artifact` accepts it (the
schema, validation and the file-name check), it is approved, its `entry_url` equals the
discovery's, its `allowed_hosts` are within the discovery's, and every input it marks sensitive is
marked sensitive here; each rejection is logged with a value-free reason (`artifact_catalog_built`)
and never stops discovery. Segments are cut from graph structure: `boundaries` are the entry node
(proven by being at the entry url) and every node that follows an action with a checkpoint (proven
by that checkpoint holding on the current observation); `segment_path` walks from a boundary,
resolving decisions from the parameters anywhere and from the live screen only at the start,
stopping before an `irreversible` or `unknown` node, before a decision it cannot resolve and at a
terminal, then cuts back to the last checkpointed action so the end state is proven too. A
candidate carries the source name, revision, digest and path, the start node, the description,
required inputs and outputs by name, an action outline, the entry condition, the ending
checkpoint, the effects and the node ids; inputs missing from the parameters, a source excluded
after an uncertain failure, or a `(digest, start, screen key)` already attempted are rejected with
a logged reason, and when one applicable segment of an artifact starts inside another only the
later one is offered. `find_candidates` runs before every planner decision and logs
`reuse_candidates_found`; the planner tool gained kind `reuse_candidate` plus `candidate_id`, the
prompt lists the candidates and the rules (advance the remaining goal, never merely match the
screen, never repeat work, prefer verified over invented), and the agent accepts only an id offered
that turn. `execute_candidate` rechecks the digest and the entry condition, builds a self-contained
linear segment artifact (the source nodes copied verbatim, `always` edges, a success terminal
verifying the ending checkpoint, inputs narrowed to what the segment uses, the source outcomes),
and replays it through `lifecycle.run_replay` with purpose `discovery_reuse`, `NoOperator` and the
discovery's log; the budget counts at that point. `classify` reads the result: `success`,
`business_outcome`, `before_acting` (no physical attempt), `partial_verified` (every attempt belongs
to a completed node and the last completed node has a checkpoint) or `uncertain`. The agent then
imports (`Recorder.import_nodes`) the verified entries (all after success, else up to the last
checkpointed one; extractions only under the output-contract rule; recoverable outcomes), rejects a
success that changed nothing and extracted nothing as no progress, continues on the same session
after `before_acting` and `partial_verified`, continues after a business outcome only when the
discovery declares that code, and otherwise raises `RestartDiscovery`; an uncertain result asks the
operator (`resume` imports the proven nodes and continues, `restart` restarts, `abort` fails) and
with `NoOperator` restarts. `discover_with_reuse` owns the sessions: it reruns discovery on a fresh
surface with the source digest excluded, at most `MAX_DISCOVERY_RESTARTS` (2) times, replaying the
forced prefix again when there is one. The automatic budget (`Library.count` against
`max_reuses`), its single `reuse_exhausted` log and the exclusions are discovery-wide and survive
restarts; only the per-screen attempt memory is session-local, so a fresh session may rerun the
safe segments that rebuild its state. A forced prefix never counts against the budget. Provenance `reuses` is an ordered list of records (mode,
candidate id, source name, revision, digest and path, start node, executed path, imported count and
ids, reuse run id, planner turn, entry condition, ending checkpoint, result). Events:
`artifact_catalog_built`, `reuse_candidates_found`, `reuse_candidate_selected`,
`reuse_candidate_rejected`, `reuse_started`, `reuse_step_executed`, `reuse_succeeded`,
`reuse_imported`, `reuse_failed`, `reuse_exhausted`, `reuse_resumed_by_operator`,
`discovery_restart_requested`, `discovery_restarted`. CLI: `--no-auto-reuse`, `--max-auto-reuses`,
`--library-dir`, for `discover` and `discover-campaign`; each campaign scenario builds its own
library. Limits: entry conditions are text checkpoints, so a screen that repeats a checkpoint's
text elsewhere can make a segment applicable that then fails before acting (harmless, logged);
decisions with screen-dependent guards are only resolved at a segment's start.

**Forced prefix (`reuse.py`).** `resolve_reuse` takes a capability name (the newest
approved `name.vN.json` under `artifacts/`, else `reuse_not_found` and ordinary discovery) or an
exact path (must load and be approved, else `ReuseError`), never a similarity guess. `preflight`
runs before any surface: approved, digest unchanged since resolution, `missing_inputs` empty for
the new parameters, every input the prefix marks sensitive marked sensitive here too, the same
entry URL, allowed hosts within the discovery's, and no `irreversible` or `unknown` action node.
Resolution by name is deterministic: the candidates are the `name.vN.json` files that load, whose
contents really are capability `name` at revision `N`, and are approved; the greatest `N` wins.
`execute_reuse` replays through `lifecycle.run_replay` with purpose `discovery_reuse` (approved
only, irreversible policy forced to deny), `NoOperator`, and the discovery's own run log; a crash
is a failed reuse. The engine fills `ReplayResult.executed_path` with the action nodes it
completed and traversed, once each, in order (`replay.executed_entry`: id, action, effect, retry
safety), so the imported path needs no log parsing and is exactly the branch taken; decision and
terminal nodes never appear in it. `performed_attempts` counts every physical action attempted,
repeats included, and is logged on failure but never imported. `prefix_from` accepts only a
successful result and copies each entry into a renumbered action node with its action, effect
and retry safety unchanged. `discover_with_reuse` owns the sessions: reuse succeeds and imports
-> `discover(prefix=...)` on the same surface, where the recorder starts with the imported nodes,
the source's recoverable outcomes and any extraction whose output contract the new discovery
declares identically, the entry navigation is skipped and the planner's first observation is the
post-prefix screen; reuse fails, or its path cannot be imported (`reuse_import_failed`) -> that
surface is closed, `reuse_fallback_started`, a fresh surface, ordinary discovery. Every surface
opened here is closed exactly once. The forced prefix is the first record in `provenance.reuses`
(mode `forced_prefix`); automatic reuse continues after it unless turned off. Events:
`reuse_resolved`, `reuse_preflight_passed`, `reuse_started`, `reuse_step_executed`,
`reuse_succeeded`, `reuse_failed`, `reuse_import_failed`, `reuse_fallback_started`,
`discovery_resumed_after_reuse`.

## 3. Replay

`replay.py` is the one engine; `replay.replay` walks `entry_node -> ... -> terminal` with no
model, and every lifecycle operation (supervised or unattended replay, stability, discovery
reuse, the evidence bundle) goes through `lifecycle.run_replay` with an explicit purpose. An
action node goes through dialog recovery, the policy check, ladder resolution, acting, the
checkpoint wait (a declared business outcome on a changed screen ends the run while a checkpoint
is pending) and output extraction; a completed node is appended to `executed_path` once. An
action without a checkpoint passes at once and its edges judge the screen. `back` needs no
locator; an `append` extraction adds to the list built so far (once per node, see section 2). Edge selection
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
session and clears `executed_path` (the nodes will be completed again) without clearing
`performed_attempts`, so a one-time expired session is resolved by choosing `restart`
immediately: the artifact signs in again itself, and signing in by hand first would leave the
entry checkpoint (the sign-in page) unmet.

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
prefix tree over each trace's `linear_path`: two action nodes are shared only when action,
parameterized value, full ladder, checkpoint, effect and retry safety are identical (ids are
ignored), and the shared node is a copy of the source node with its effect and retry safety
unchanged. The entry node is a
decision gate with one edge per declared scenario, guarded by the complete selector
assignment, so an undeclared combination stops before any action. At the first difference a
decision node branches, again on complete assignments; merging continues inside branches;
nothing is shared after divergence; a scenario that ends where another continues gets a guarded
edge to the success terminal. Node ids follow depth-first creation order, priorities follow
specification order, and edges are sorted, so the same traces give the same JSON. Provenance
records the campaign id, merge strategy, planners, and per scenario the selector assignment,
discovery run id, planner, action count and node path.

## 5. Lifecycle

`draft -> supervised replay or stability runs -> approve -> approved -> unattended replay`.

`stability` replays one invocation N times, each on a fresh browser, session control, log and
context, unattended, with the irreversible policy forced to `deny`, and closes every session.
`report.json` (`report_schema_version` 1.0, unrelated to the artifact schema) records the
artifact identity and SHA-256, parameter names only,
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
`artifact_not_approved` before opening a browser; `lifecycle.run_replay` enforces the same rule
for library callers through an explicit purpose. Every replay bundle holds `run.jsonl`,
screenshots, a redacted `result.json`, the exact artifact that ran and `manifest.json`.

This is a local review record. There is no signing: anyone who can write to `artifacts/` can
approve, and a report rewritten consistently end to end passes the consistency checks.
