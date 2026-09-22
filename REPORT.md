# Pathloom design report

Pathloom: LLM-discovered UI workflows compiled into deterministic automation. A model discovers a
workflow once, per human-declared scenario; the result is a typed, versioned capability graph
that replays with no model, is verified by repeated live runs, approved by a person, and only then
allowed to run unattended. Detail lives in `docs/design-notes.md`; evidence in `evidence/INDEX.md`.

## 1. Architecture

One Python process and four seams: `Surface` (the UI adapter; one Playwright/Chromium
implementation), `Planner` (the LLM; Anthropic and OpenAI adapters, one strict tool call per
decision, stateless), `Operator` (a human channel on the same live session) and the artifact
itself, the contract between discovery and replay. Discovery is `observe -> decide -> policy ->
act -> verify -> record`. Perception is an operator's view rather than the DOM: controls with a
role, an accessible name, visible text, the item they belong to and, in the browser adapter,
computed states merged from Chromium's accessibility tree (the projection keeps native roles and
geometry; the tree adds states, upgrades generic text that is really a control, and adds
controls the projection missed; nodes are matched by a shared structural path in two protocol
calls, and the observation degrades to the projection if the tree cannot be read). What gets recorded is the
reusable flow, never the transcript: each performed action becomes an action node with a locator
ladder, a checkpoint that was actually seen, an effect and a retry policy; inputs become
`{{placeholders}}`; outcomes the model saw or declared are kept with their source. A single
discovery therefore records a linear capability graph, and a campaign merges such graphs.

A campaign runs one such discovery per declared scenario on a fresh session and merges the
verified traces by a prefix tree into a graph: shared identical prefixes, a decision node where
paths differ, guards on the complete selector assignment, no merging after divergence, and an
entry gate that admits only the declared assignments. Replay walks the graph deterministically.
The lifecycle around it is `draft -> stability -> approve -> approved -> unattended`.

Key trade-offs: text checkpoints are simple and surface-agnostic but coarse, so classification
order carries the weight (Section 3); the prefix tree never guesses that two screens reached by
different routes are the same, at the cost of duplicated suffixes (a live graph has 28 action
nodes for 37 recorded actions); and the model is asked only for one action at a time with no
transcript, which keeps artifacts provider-independent but makes discovery slower.

## 2. Artifact schema

The artifact is a capability graph (`schema_version: "2.0"`; the `N` in `name.vN.json` is the
capability revision). An artifact carries
the contract fields, `inputs` (typed, `sensitive`, `selector` and `required_when` conditions),
`outputs` (type, requiredness and the regex that reads each value; or `list` with an `items`
rule and optional `min_items`/`max_items` bounds, read by an `extract_many` node from an ordered
list of locator ladders, or grown one page at a time by extraction nodes marked `mode: "append"`),
declared
`outcomes`, a `success` checkpoint and `provenance`, and the flow as `entry_node`, `nodes` and
`edges`. Actions are `navigate`, `back` (the browser history), `click`, `type`, `extract` and
`extract_many`; a checkpoint is `text_contains` or `url_contains`. A
single discovery emits a linear graph (action nodes joined by `always` edges into the `success`
terminal); a campaign merges linear graphs into a branching one. An action node carries what it
does plus an `effect` (`none`, `reversible`, `irreversible`, `unknown`) and a `retry_safety`,
both assigned at discovery from fixed rules (navigation, back and extraction have no effect; an
allowed click or type is reversible; a control the policy calls risky is irreversible; anything
unclassifiable is unknown; extraction and plain navigation are safe to repeat, a checkpointed
reversible action or back is verified before a retry, a blind click, type or back and every
irreversible or unknown action are never retried); decision nodes branch on typed guards
(`input_equals`, `text_visible`, `url_matches`, `element_present`, `dialog_contains`, `always`
as the fallback); terminal nodes end as `success`, `business_outcome` or `failure`. Locators are
ladders, most stable first, and are parameterized like values; a parameterized ladder never
falls back onto a structural rung that would pick the discovery-time item. Validation on build
and load rejects malformed guards, undeclared placeholders, dead ends, unreachable nodes and
cycles. Artifacts are immutable files, one per version; approval writes a new version.

Provenance of the live graph names the campaign, the three discovery runs, the planner
(`openai:gpt-6-astra`), each scenario's node path, and after approval the reviewer, the draft's
SHA-256 and the digests of the three stability reports. It never holds a secret or a transcript.

## 3. Determinism & error handling

Replay resolves targets from the ladder, substitutes parameters, and never asks anything to
choose; the engine, the stability command and approval import no planner or provider SDK, which
a subprocess test proves. Waiting is bounded polling against perception, never fixed sleeps.
After an action, classification runs in a fixed order: checkpoint met wins; a screen that has
not changed is still loading; a declared business outcome on a changed screen ends the run
with its code; a node without a checkpoint hands the screen to its edges; a timeout escalates.
Absence outcomes ("the product is not listed") are judged only by the node that looks for the
item. Nine live replays of the three paths agreed with each other and with discovery to the cent.

During discovery a surface action that cannot complete is classified by the adapter as not
performed, performed without effect, or unknown: a not-performed click returns to the planner
with the reason in its history and is bounded per screen before the stuck handoff; an unknown
outcome is proven through the planner's stated expectation or handed to a person, and an
irreversible action is never repeated automatically. The classification comes from the phase
that failed (the explicit actionability trial and inspection, or the real action), never from
error wording. Native radios and checkboxes are activated through their browser-associated labels
under Playwright's normal actionability checks, then the input, then a guarded keyboard press
allowed only when the obstruction belongs to the same choice structure; never forced. Controls that are hidden, inert or clipped are not listed, and disabled or covered ones are
marked; a `select` action chooses from native selects, comboboxes and autocomplete fields by
option semantics and verifies acceptance; a planner repeating itself without progress is warned
once and then stopped through the stuck handoff. Perception keeps
visible leaf text whatever its tag and supplements it from the accessibility tree within a bound;
a value still absent from the list can be extracted through a bounded, grounded visual fallback in
which the model only proposes boxes and the page element under each box supplies the value, parsed
by the contract and replayed by structure or by an exact reading point. Visible
text that cannot take a click itself is clicked through the nearest enclosing control that is
actionable by its own semantics (native control, anchor, bound label, explicit interactive
role), recorded as that control; with no such control the failure is a structured
`unactionable_target`, eligible for the bounded vision fallback and otherwise a handoff. Transient
loads are retried with backoff and a reload, but each graph node's `retry_safety`
decides whether an action that may already have happened is repeated: `safe` repeats,
`verify_before_retry` re-checks the checkpoint first and asks a person when it cannot tell,
`never_retry` never repeats. Known dialogs are cleared by declared recoveries; unknown ones go to
a person. The result contract is `success` with typed outputs, `business_outcome` with a stable
code, or `failure` with the node, what was expected and what was observed, plus two traces:
`executed_path`, the action nodes completed once each in order (what reuse may import), and
`performed_attempts`, every physical action attempted including repeats (diagnostic only; a
restart clears the path but not the count). Recoveries and locator fallbacks are counted and
reported by stability runs rather than hidden.

A checkpoint is recorded only when it is evidence: expected text absent before the action and
present after it, a typed value shown back, or a page change proven by the URL; text that was
already on screen (a site-wide heading) is never recorded as proof. An expectation the screen
contradicted is not quietly cleared and the action recorded as done: discovery looks for
independent proof (a safe address change, a native control state that flipped, the acted control itself
becoming selected by browser state rather than styling, judged from native, accessibility, attribute,
associated-control or exact class-token evidence, and considered for every click whatever the planner
expected, or a commit inside the
acted control's own field group where a field gave up its value and exactly one new removable or selected
token carries it, counted as one logical token per removal control however deeply the page nests it),
records that instead when it exists, and otherwise classifies the action `action_effect_unverified`, records no
node for it, and refuses to dispatch it again before measurable progress or a person's decision.
Outputs are assigned once (a second `set` is refused before it runs) or
grown in order with `append`, which replay rebuilds across pages and never duplicates on a retry.
A direct discovery may declare its outputs up front (`--output-contract`, the same shape a campaign
spec uses, with optional list cardinality); the contract then overrides whatever the planner
proposes, an append beyond `max_items` is refused as "already complete", `done` is rejected with
the names of the missing or incomplete outputs until every required output is complete, and a
replay that reaches success re-validates its outputs against the same contract before reporting it.
Model-provider failures are retried at one shared boundary, bounded, classified by status or
exception class rather than message text, and logged per retry; a provider that stays down ends
discovery with a concise project-owned error, saved evidence and a nonzero exit, never a traceback.

## 4. Heterogeneity & multi-tenant

The artifact never mentions HTML. A desktop or legacy application needs only a new `Surface`
whose `observe` yields roles, names and containing items and whose `resolve` walks the same
ladder kinds; agent, merge, replay, policy and escalation are unchanged. Live testing showed the
value of keeping perception rules generic: an anchor with `role="button"` and no `href`, and an
icon-only control with only a numeric badge, are now perceived by their test id, and text
nested inside a control is not a separate thing to click. Perception must also be stable between
discovery and replay: a rule that reclassified `<a href role="button">` broke a recorded path, was
caught by stability runs, and was replaced by "the native tag role wins".

A second target, the MemberOps Sandbox (`examples/member_ops/`, a fictional back-office app served
from the standard library), exists to show the same generic mechanisms on realistic enterprise
markup: table-based forms named by the cell to their left, an icon-only control named by its test
id, repeated labels across account rows, two lookup methods and two account types as selectors,
path-specific inputs, and server modes that reproduce a transient failure, a known interstitial,
an unknown modal that stays until a person acknowledges it, a one-time expired session, and an
irreversible final action kept in a separate, supervised-only capability
(`member_account_open`) so the safe `member_account_prepare` artifact never contains it. Campaign
specs may carry reviewer-declared outcomes, validated by the artifact's own rules, so states
discovery never visits still classify deterministically. No live model run against it has been
made yet; the suite drives it through the real browser adapter with a scripted planner.

Accessibility support is Chromium-specific and confined to the adapter; the seam, the artifact
and the engine are unchanged. For controls neither source can expose (a canvas, an unnamed
icon) discovery has an opt-in, bounded screenshot fallback: after the planner is stuck, one
masked viewport screenshot goes to the provider, which may propose exactly one click or type
with a bounding box and an expected text; the proposal is validated, policy-checked, performed
once and kept only if structured perception verifies it, with a small per-run budget and one
attempt per unchanged screen. The recorded node keeps a coordinate rung bound to its viewport;
replay never uses a model, uses the coordinates last, and refuses them in another viewport.
Visual extraction is unsupported by design.

Approved artifacts also form a verified library that every discovery composes from
automatically: `library -> state-matched candidate retrieval -> planner chooses reuse or a novel
action -> deterministic segment replay -> checkpoint verification -> nodes inlined into the new
graph`. The catalog admits only approved, valid, same-surface artifacts whose file names match
their contents; a candidate is a segment cut at verified boundaries (the entry, the state after a
checkpointed action, a resolvable decision) whose entry state is proven on the current screen,
that contains no irreversible or unknown action, has its inputs available and ends at a
checkpoint, so an artifact is reusable from the middle. Before every planner decision the
matching candidates are described by name only, and the planner's one tool call selects a
`candidate_id` or an ordinary action; matching a screen is necessary but not sufficient, the
segment must advance the goal, and the model never regenerates a segment's actions. A selected
segment is replayed by the same engine with the irreversible policy forced to deny and its
completed nodes are inlined exactly, so the new artifact is self-contained and several artifacts
can be composed in one discovery, bounded by a budget, one attempt per segment per screen and a
no-progress check. A failure before acting continues on the same session; after checkpointed
actions the proven nodes are kept; an unproven physical action is never repeated and the session
is restarted cleanly with the artifact excluded (or handed to an operator). A named forced prefix
remains available as an override. This is composition of verified work, not a cache of clicks.

Multi-tenant reuse is a layering the schema is shaped for but does not implement: a base
artifact per vendor product, per-tenant `entry_url` and `allowed_hosts`, overlays for individual
ladders, and the `drift_signal` from each resolution as a per-tenant stability record. The
lifecycle already gates which version a tenant may run unattended.

## 5. Escalation & handoff

Discovery escalates when the planner is stuck, when it keeps choosing denied actions, and for any
risky action. Replay escalates on an unknown dialog, an unresolvable target, an unmet checkpoint,
a decision with no matching edge, an action whose result is uncertain and must not be repeated,
and any irreversible or unknown effect that needs confirmation. The request carries the capability,
its natural-language goal, the node, the reason, the observed text and a screenshot.
`SessionControl` has a single owner; the escalator transfers it to the human, hands the *same*
surface to the operator, and takes it back in a `finally`. The console operator lists the
numbered controls and drives them; every human action is recorded, redacted by field name.
Dispositions: `resume` re-checks the node before redoing anything, `restart` starts over once,
`abort` returns the original failure, `approve`/`deny` answer a confirmation. Live evidence shows
requests that nobody answered (a `Finish` confirmation and a stuck gate); an interactive session
was not recorded and the index says how to rehearse one.

## 6. Safety

Host and control allowlists are checked before every action in both paths and an artifact cannot
override them. Risk is layered: the policy's name-based classification of irreversible controls
(`Finish`, `Pay`, `Delete`, ...; `Close` by its wording, and a bare `Close` by browser-derived
evidence about its container: proven dismissal chrome is reversible, a container naming a resource
closure is irreversible, no evidence is unknown and confirmed), the node's declared `effect`, and the run's
`--irreversible-policy` (`deny`, `confirm`, `allow`); `unknown` always needs a person, a
conflict between the policy and the graph resolves to the stricter view, and stability runs always
use `deny`, so an irreversible node fails safely and stays ineligible. Sensitive inputs reach
neither the model nor disk. Unattended replay requires an approved artifact, a draft is refused
before a browser exists, and every bundle carries the exact artifact, a redacted result and a
manifest with its digest. Approval recomputes a report's eligibility from its run records and
derives required path coverage from the graph's own gate, so neither a flipped flag nor stripped
provenance lowers the bar. Limits: risk classification is name-based and English; redaction is
pattern-based; the allowlist is host-level.

## 7. Cuts

Deliberate: no cryptographic report signing (approval is a local review record; a consistently
rewritten report would pass); no authenticated web operator console (a terminal); no per-tenant
runtime infrastructure; no unrestricted LLM fallback during replay; no repeated testing of real
irreversible commits (stability targets prepare-only flows, which is why every workflow here
stops before `Finish`); one implemented browser surface; the replay engine stays an internal
component behind the lifecycle boundary. Also not done: locator normalization in the
merge key (a coordinate rung that differs between runs would prevent sharing), nested artifacts,
and guard minimization (guards carry the complete selector assignment on purpose).
