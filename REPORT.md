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
role, an accessible name, visible text and the item they belong to. What gets recorded is the
reusable flow, never the transcript: each performed action becomes a step with a locator ladder
and a checkpoint that was actually seen, inputs become `{{placeholders}}`, and outcomes the
model saw or declared are kept with their source.

A campaign runs one such discovery per declared scenario on a fresh session and merges the
verified traces by a prefix tree into a graph: shared identical prefixes, a decision node where
paths differ, guards on the complete selector assignment, no merging after divergence, and an
entry gate that admits only the declared assignments. Replay walks the graph deterministically.
The lifecycle around it is `draft -> stability -> approve -> approved -> unattended`.

Key trade-offs: text checkpoints are simple and surface-agnostic but coarse, so classification
order carries the weight (Section 3); the prefix tree never guesses that two screens reached by
different routes are the same, at the cost of duplicated suffixes (a live graph has 28 action
nodes for 37 recorded steps); and the model is asked only for one action at a time with no
transcript, which keeps artifacts provider-independent but makes discovery slower.

## 2. Artifact schema

Schema 1.0 is a linear list of steps. Schema 2.0, produced by campaigns, replaces it with
`entry_node`, `nodes` and `edges` while keeping the same contract fields: `inputs` (typed,
`sensitive`, and for graphs `selector` and `required_when` conditions), `outputs` (type,
requiredness and the regex that reads each value), declared `outcomes`, a `success` checkpoint
and `provenance`. An action node carries what a step does plus an `effect` (`none`, `reversible`,
`irreversible`, `unknown`) and a `retry_safety`; decision nodes branch on typed guards
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
choose; the engines, the stability command and approval import no planner or provider SDK, which
a subprocess test proves. Waiting is bounded polling against perception, never fixed sleeps.
After an action, classification runs in a fixed order: checkpoint met wins; a screen that has
not changed is still loading; a declared business outcome on a changed screen ends the run
with its code; a step without a checkpoint hands the screen to its edges; a timeout escalates.
Absence outcomes ("the product is not listed") are judged only by the step that looks for the
item. Nine live replays of the three paths agreed with each other and with discovery to the cent.

Transient loads are retried with backoff and a reload, but each graph node's `retry_safety`
decides whether an action that may already have happened is repeated: `safe` repeats,
`verify_before_retry` re-checks the checkpoint first and asks a person when it cannot tell,
`never_retry` never repeats. Known dialogs are cleared by declared recoveries; unknown ones go to
a person. The result contract is `success` with typed outputs, `business_outcome` with a stable
code, or `failure` with the node, what was expected and what was observed. Recoveries and
locator fallbacks are counted and reported by stability runs rather than hidden.

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
(`Finish`, `Pay`, `Delete`, ...), the node's declared `effect`, and the run's
`--irreversible-policy` (`deny`, `confirm`, `allow`); `unknown` always needs a person, a
conflict between the policy and the graph resolves to the stricter view, and stability runs always
use `deny`, so an irreversible step fails safely and stays ineligible. Sensitive inputs reach
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
stops before `Finish`); one implemented browser surface; the low-level replay engines stay
internal components behind the lifecycle boundary. Also not done: locator normalization in the
merge key (a coordinate rung that differs between runs would prevent sharing), nested artifacts,
and guard minimization (guards carry the complete selector assignment on purpose).
