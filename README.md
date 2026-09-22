# Pathloom

**LLM-discovered UI workflows compiled into deterministic automation.**

Pathloom lets a language model work out *once* how to accomplish a goal on a real web
application, records the result as a typed, versioned **capability artifact**, and then replays
that artifact **deterministically, with no model in the loop**: typed inputs, typed outputs,
explicit business-outcome classification, safety guardrails, and a real human handoff on the
same live session. A person declares the scenarios worth recording; the model never explores on
its own.

```text
scenarios/<name>.json  ->  discover-campaign  (one LLM discovery per declared scenario)
                       ->  prefix-tree merge  ->  artifacts/<name>.vN.json  (draft capability graph)
                       ->  stability          (N fresh unattended replays per path, no LLM)
                       ->  approve            (human review record -> next version, status approved)
                       ->  replay --operator none  (unattended, approved artifacts only)
```

The demonstration target is the public **SauceDemo** test shop. The recorded capability logs in,
adds a product by one of two routes (from the product list or from its details page), opens the
cart by one of two routes (the cart control or the cart URL), fills the checkout form and reads the
total on the Checkout Overview page.

> **Every workflow stops before `Finish`.** Placing the order is an irreversible action: it is
> never clicked unattended, stability runs refuse it, and even a supervised replay asks a person
> first. No run in this repository has ever placed an order.

Design write-up: [`REPORT.md`](REPORT.md) (short) and [`docs/design-notes.md`](docs/design-notes.md)
(detail). Assignment: [`docs/assignment.pdf`](docs/assignment.pdf). Evidence:
[`evidence/INDEX.md`](evidence/INDEX.md). Example artifact: [`artifacts/checkout_paths.v3.json`](artifacts/checkout_paths.v3.json)
(approved) and its draft [`artifacts/checkout_paths.v2.json`](artifacts/checkout_paths.v2.json).

## Architecture

One Python process, four seams, no services. The Python package is still called `cua`.

| Module | Role |
|---|---|
| `src/cua/surface.py` | `Surface` seam and the Playwright/Chromium adapter: perception as an operator sees it (role, name, text, enclosing item, computed states), merged from the page projection and Chromium's accessibility tree; locator ladders; secret masking |
| `src/cua/planner.py` | `Planner` seam; `ClaudePlanner` and `OpenAIPlanner`, one strict tool call per decision, stateless |
| `src/cua/agent.py` | Discovery loop: observe, decide, policy, act, verify, record; parameterizes values and locators; outputs and outcomes |
| `src/cua/library.py`, `reuse.py` | The verified artifact library: catalog of approved artifacts, state-matched segment candidates, deterministic segment replay and inlining; the discovery session owner (forced prefix, clean restarts) |
| `src/cua/campaign.py` | Human-declared scenarios: one discovery run per scenario on a fresh session, campaign evidence and summary |
| `src/cua/merge.py` | Conservative prefix-tree merge of verified traces into one graph with an entry selector gate |
| `src/cua/artifact.py`, `models.py` | The one capability artifact schema (a guarded acyclic graph, `schema_version: "2.0"`): the linear builder discovery uses, validation, serialization, immutable save and load |
| `src/cua/replay.py` | The one deterministic replay engine (an internal component): ladders, checkpoints, guards, effect policy, retry safety, escalation, the execution trace |
| `src/cua/policy.py`, `escalation.py`, `evidence.py` | Allowlists and risk classification, same-session human handoff, redacted logs and evidence bundles |
| `src/cua/lifecycle.py`, `stability.py`, `approval.py` | Draft/approved gate, multi-run stability reports, local approval record |
| `src/cua/__main__.py` | CLI: `discover`, `discover-campaign`, `stability`, `approve`, `replay` |
| `examples/member_ops/` | MemberOps Sandbox, a fictional back-office app with runtime modes for each result class (see below) |

**Discovery uses an LLM; replay never does.** The replay engine, the stability command and the
approval command do not import the planner or either provider SDK, and the test suite proves it
in a clean interpreter.

**One artifact schema.** Every artifact, whether `discover` recorded it or a campaign merged it,
is the same capability graph: action nodes (what to do, with an `effect` and a `retry_safety`),
decision nodes (typed guards choose the next edge) and terminal nodes. A single discovery yields
a linear graph, `s1 -> s2 -> ... -> success`, joined by `always` edges; a campaign merges several
such graphs into a branching one; one engine replays both. `schema_version` is the file format
and is always `"2.0"`; the `N` in `artifacts/<name>.vN.json` is the capability revision (a
re-recording or an approval writes the next one), not a schema version. A file with any other
schema version is refused with a clear error.

## Setup

Python 3.11+ (developed on 3.14).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

Environment variables (only discovery needs a key; nothing is read from `.env`):

| Variable | Needed by | Default |
|---|---|---|
| `OPENAI_API_KEY` | `discover`, `discover-campaign` with `--provider openai` | none |
| `ANTHROPIC_API_KEY` | the same commands with `--provider anthropic` | none |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` | optional model overrides | `gpt-6-astra` / `claude-opus-5` |
| `OPENAI_VISION_MODEL` / `ANTHROPIC_VISION_MODEL` | optional model for the `--vision-fallback` screenshot call | the configured model |

Never paste a key into a command line or a file in this repository.

## The lifecycle, with the exact commands

**1. Discover a campaign (LLM).** The spec declares the goal, the selector inputs that pick a
path (`product_route`, `cart_route`), the three scenarios to record, and the output contract
every run must satisfy. Each scenario is discovered on its own browser session; the three
verified traces are merged into one draft graph.

```bash
python -m src.cua discover-campaign --spec scenarios/checkout_paths.json --provider openai --operator none
```

**2. Verify stability (no LLM).** Three fresh unattended replays per declared selector
assignment, always with the irreversible policy set to `deny`; each writes
`evidence/stability-<id>/report.json`.

```bash
for combo in "inventory cart_link" "details cart_link" "inventory cart_url"; do
  set -- ${=combo}   # zsh; in bash: set -- $combo
  python -m src.cua stability --artifact artifacts/checkout_paths.v2.json --runs 3 \
    --param username=standard_user --param password=secret_sauce --sensitive password \
    --param "product_name=Sauce Labs Backpack" --param first_name=Test --param last_name=User --param postal_code=10001 \
    --param product_route=$1 --param cart_route=$2
done
```

**3. Approve (human, no browser, no LLM).** One eligible report per path admitted by the graph's
entry gate. The draft is never modified; the next version is written with `status: approved`.

```bash
python -m src.cua approve --artifact artifacts/checkout_paths.v2.json \
  --report evidence/stability-20260915T005943Z-2a01/report.json \
  --report evidence/stability-20260915T010110Z-81a7/report.json \
  --report evidence/stability-20260915T010241Z-7b2c/report.json \
  --reviewer soroush
```

**4. Replay unattended (no LLM).** Only an approved artifact may run with `--operator none`; a
draft is refused with `artifact_not_approved` before a browser exists.

```bash
python -m src.cua replay --artifact artifacts/checkout_paths.v3.json --operator none \
  --param username=standard_user --param password=secret_sauce --sensitive password \
  --param "product_name=Sauce Labs Backpack" --param first_name=Test --param last_name=User --param postal_code=10001 \
  --param product_route=details --param cart_route=cart_link
```

Change `--param username=locked_out_user` to see a declared business outcome
(`business_outcome` / `locked_out`), or `--param product_route=details --param cart_route=cart_url`
to see an undeclared combination refused at the entry gate before any action. A draft can be
replayed supervised with `--operator console` (a warning is printed and logged); add `--headed`
to watch the browser, and take over the same session from the terminal whenever the run asks.

**5. Offline tests.**

```bash
python -m pytest
```

313 tests, all offline. The browser and the model are replaced at their protocol boundaries by
`tests/fake_surface.py` and `tests/scripted_planner.py`. The 38 tests marked `browser` (perception
rules on a local page, and the MemberOps campaigns driven through the real Chromium adapter by a
scripted planner, then replayed in every runtime mode) start a local headless Chromium with no
network and skip when it is absent; they take about nine minutes, so
`python -m pytest -m "not browser"` (275 tests, under a minute) is the quick loop.

## MemberOps Sandbox: a controlled back-office target

SauceDemo is public and simple. `examples/member_ops/` is a fictional banking back-office
application, served locally from the Python standard library, built so every part of Pathloom's
result taxonomy can be shown deterministically: two lookup methods and two account types give a
real multi-path campaign, and server modes reproduce the awkward states a production UI throws
at automation. Everything is in memory and fictional: no real members, accounts, credentials or
network beyond the local socket. Pathloom drives it through the browser only.

```bash
python -m examples.member_ops --port 8765 --mode normal        # then open http://127.0.0.1:8765/
curl -X POST http://127.0.0.1:8765/__reset                      # clear sessions and created accounts
```

Training credentials: username `operator`, password `training_only` (pass it with `--sensitive password`;
it is a training credential for a fictional app, but Pathloom treats it like any secret). Members:
`1001` / phone `5550101` Alex Morgan (normal), `1003` / `5550103` Sam Patel (normal), `1002` /
`5550102` Jordan Lee (restricted: access denied). Any other lookup value is "not found". The
opening deposit must be a number of at least 25.00. The review page's last button,
**Confirm and Open Account**, is the irreversible action; the campaign goal stops before it.
The existing policy already classifies that label as irreversible (it contains "confirm"), and a
label such as "Create Account" is on the self-registration block list and would be denied
outright, which is why the sandbox uses this wording.

| Mode (`--mode`) | What happens | Pathloom result it demonstrates |
|---|---|---|
| `normal` | plain flow | `success` with typed outputs; `business_outcome` for a not-found member, a restricted member (`403` page), invalid credentials or an invalid deposit; `no_matching_edge` at the entry gate for the undeclared `phone + checking` combination, before any action |
| `slow_once` | the first profile load per browser session answers `503` after 1.5 s | a transient error, retried with backoff and reload, reported as a recovery |
| `known_interstitial` | a maintenance notice dialog after sign-in until dismissed | a recoverable outcome: the campaign is discovered in `normal` mode, and the reviewer-declared `dismiss_maintenance_notice` outcome lets a later replay in this mode dismiss the dialog deterministically and record a recovery |
| `unexpected_dialog` | a compliance dialog on the profile that no artifact knows, shown until it is acknowledged; reloading does not clear it | `unknown_dialog` handoff: the console operator clicks `Acknowledge` on the same session and types `resume`; automation re-checks the checkpoint and continues |
| `session_expired_once` | the first review submission per session bounces to the sign-in page | an unmet checkpoint handoff; the operator chooses `restart` immediately, the artifact starts from its entry node and signs in again itself, the expiry does not repeat |
| `slow_after_open_once` | the account is opened but the first confirmation page per session answers `503` | with the `member_account_open` capability: an uncertain result after an irreversible action is never repeated automatically; a person is asked |
| any, `--irreversible-policy` | the final button (`member_account_open` only) | `deny` refuses it, `confirm` asks a person, `allow` runs it once in the sandbox; the node is `never_retry` |

Do not sign in by hand before choosing `restart`: the restarted flow expects the sign-in page at
its entry checkpoint. When recording handoff evidence, act through the console operator's own
commands (`observe`, `click <n>`, `type <n> <text>`): those are captured as structured human
actions in the intervention record, whereas a click made directly in the headed browser window is
visible on screen but not recorded as a human action.

Two capabilities are declared, on purpose:

```text
member_account_prepare   safe: stops at the review page, extracts the outputs, eligible for
                         stability runs and approval; its artifact contains no irreversible action
member_account_open      performs the side effect: continues through "Confirm and Open Account",
                         discovered supervised (the final click needs approval), used only for
                         supervised effect-policy demonstrations, never for unattended approval
```

Both specs carry **reviewer-declared outcomes** (`outcomes` in the campaign JSON, a generic
Pathloom feature): a reviewer writes the business and recoverable outcomes discovery never
visits, using the artifact's own outcome structure and the application's exact visible text.
Every scenario's discovery receives them, and for any code the reviewer declared the reviewer's
definition is authoritative: a planner outcome with the same code is replaced (silently when
identical, with a `reviewer_outcome_overrode_planner` warning in the scenario's run log when it
differs), planner outcomes with other codes are kept, and the merged graph classifies them
deterministically at replay. For MemberOps: `invalid_credentials`,
`member_not_found`, `member_access_denied`, `invalid_opening_deposit`, and the recoverable
`dismiss_maintenance_notice`, all shared by both capabilities (so either campaign is discovered in
`normal` mode and still recovers the notice when replayed in `known_interstitial` mode).

The campaign specification is `scenarios/member_account_prepare.json`: selectors `lookup_method`
and `account_type`, three declared scenarios (`member_id + savings`, `phone + savings`,
`member_id + checking`), each carrying only the inputs its path needs, so the merged graph gets
`required_when` conditions for `member_id` and `phone`, one output contract (`member_name`,
`account_type`, `opening_deposit`) and the reviewer outcomes above. The planned commands are

```bash
python -m examples.member_ops --port 8765 --mode normal                # in one terminal
python -m src.cua discover-campaign --spec scenarios/member_account_prepare.json --provider openai --operator none
python -m src.cua discover-campaign --spec scenarios/member_account_open.json --provider openai --operator console
```

**No live MemberOps discovery, replay, stability report, handoff recording or approved artifact
exists yet.** What does exist is offline and local-browser proof: HTTP tests for every screen and
mode; tests of both specs, the outcome contract and the merged graph's conditional inputs and
gate; and a Chromium integration module in which a scripted planner discovers all three prepare
paths and the open capability through the real adapter, then the graphs replay each path, refuse
the undeclared one without an action, classify the four business outcomes, recover the
maintenance notice, ride out the transient load, resolve the persistent dialog through the
console-operator seam with the click recorded, recover from the expired session by `restart`
alone, obey `deny`, `confirm` and `allow` on the final click, and never repeat it after an
uncertain result.

## What the evidence shows

`evidence/INDEX.md` maps every retained folder to what it proves. In short, on SauceDemo: three
stability reports over nine live replays of the draft graph with total `32.39` every time and no
recovery, intervention or locator fallback, the approval record, an approved unattended replay,
an undeclared combination stopped at the gate with zero actions, and a locked-out user
classified as a business outcome. On MemberOps: supervised replays of the three declared paths,
three stability reports, an approved revision and its replays including a same-session operator
handoff, and the open capability refused under `deny` and run once under `confirm` with a
recorded approval. The proof that an irreversible final click is never performed unattended
lives in the offline suite and in the MemberOps `irreversible_denied` replay.

## The verified artifact library: automatic composition

Approved artifacts are a growing library of verified UI knowledge, and every discovery draws on
it automatically:

```text
verified artifact library  ->  state-matched candidate retrieval  ->  planner chooses reuse or a novel action
                           ->  deterministic segment replay  ->  checkpoint verification
                           ->  reusable nodes inlined into the new graph
```

At the start of a discovery, `artifacts/` is scanned once into a catalog: only files that load as
the capability graph schema, whose name and revision match their contents, that are approved and
valid, that belong to the same entry URL and stay within the discovery's host allowlist, and whose
sensitive inputs are marked sensitive here too. Anything else is skipped with a value-free reason
in the log. Then, before **every** planner decision, the current screen is matched against the
catalog: a segment may start at an artifact's entry, after any action whose checkpoint was
verified, or at a decision whose guards can be evaluated now, and it is a candidate only when
that boundary state is proven against the live observation and the parameters, it contains no
`irreversible` or `unknown` action, every input it needs is available, and it ends at a
checkpoint. An artifact is therefore reusable from the middle: on the authenticated member-search
screen, `sign in -> find member -> open profile` offers only `find member -> open profile`, never
the sign-in. Candidates are described to the planner by name, outline, inputs and outputs (never
values), and the same single decision either selects one `candidate_id` or proposes an ordinary
action; the planner is told to pick a segment only when it directly advances what remains of the
goal, never merely because it matches the screen, never to repeat work, and to prefer a verified
segment over inventing the same clicks. An id that was not offered in that turn is rejected.

A selected segment is rechecked (digest, entry state) and replayed by the ordinary engine on the
same live session with no model and the irreversible policy forced to `deny`; completed nodes
from `executed_path` are inlined exactly (action, ladder, checkpoint, `effect`, `retry_safety`),
extractions only under the output-contract rule, recoverable outcomes alongside. A discovery may
compose several artifacts (`login`, then `member lookup`, then the genuinely new part), bounded by
`--max-auto-reuses` (default 5), by one attempt per segment per screen, and by a no-progress
check. Failure is handled by what the segment touched: before any physical action, discovery
simply continues on the same session; after only checkpointed actions, those nodes are kept and
discovery continues from the proven state; after a physical action whose result is unproven, the
action is never repeated and the session is never handed on silently: a console operator may
`resume` (vouching for the screen) or `restart`, and with no operator the session is closed and
discovery restarts cleanly with that artifact excluded, at most twice. A segment's business
outcome is never a success: it is recorded, and discovery continues only when the goal declares
that outcome; otherwise a clean restart follows. The resulting artifact is completely
self-contained: deleting every source artifact afterwards does not affect its replay.

This is automatic composition of verified segments, not action-result caching: matching a screen
is necessary but not sufficient, only approved and safe segments are eligible, the model selects
among already verified candidates and never regenerates their actions, and replay stays
model-free. Flags: `--no-auto-reuse` turns the library off, `--max-auto-reuses N` bounds it,
`--library-dir DIR` reads another directory. The older flags remain as optional overrides:
`--reuse-capability NAME` / `--reuse-artifact PATH` (or `reuse_capability` / `reuse_artifact` in a
campaign spec) force one named approved artifact to run whole as the initial prefix before the
planner sees anything; automatic reuse continues afterwards unless `--no-auto-reuse` is given.
Campaign scenarios get independent catalogs, sessions, reuse histories and failure handling.
No live reuse run has been recorded; the suite proves the library on the fake shop and, in
Chromium, on MemberOps (login and member lookup composed automatically before the new part).

## Browser perception

The Chromium adapter merges two structured sources into one observation. The **page projection**
(a script run in the page) supplies visible text, geometry, the structural path used as the
action handle, and the item a control belongs to. **Chromium's computed accessibility tree**,
read over the DevTools protocol, supplies computed roles, accessible names and states
(`checked`, `expanded`, `selected`, `pressed`, and `disabled`, `required`, `readonly` when true).
Nodes are mapped back to their backing elements by recomputing the same structural path from
one `DOM.getDocument` call, so the merge costs two protocol calls plus one page call for
enrichment, never a call per node. Rules: a projected element keeps its native role, name and
geometry and gains states (an `<a href role="button">` stays a link, so recorded artifacts keep
resolving); a generic text element that the tree says is a control is upgraded to that role and
name; a control the projection never listed (a native checkbox, a radio, a tab) is added with
its geometry and text. Elements are ordered by document position and deduplicated by backing
element. Accessibility-only controls get the usual locator ladder (role and name, structural
path, coordinates) and are clicked or typed through the same handle, so they discover, save and
replay like any other control, with no model involved. If the tree cannot be read, the
observation is the projection alone, object for object, and the adapter records why. Password
values are never read from either source, registered secrets are masked in every name, text and
dialog, and screenshots mask them in page text and in the current values of text-like inputs and
textareas for the capture only (restored exactly afterwards, with no application event fired).
Limits:
this is Chromium-specific and lives entirely inside the adapter; the planner's text rendering
shows states but not geometry.

**Choice controls.** A native radio or checkbox is activated the way a person does: the adapter
confirms the element behind the reference really is such an input, refuses a disabled one, leaves
an already-selected radio alone, asks the browser for the control's own associated labels (`for=`
or wrapping) and clicks the first visible one that passes Playwright's actionability checks, else
the input itself. When both pointer paths are obstructed by something that belongs to the same
choice structure (the control's own label; for a radio also a sibling of its native group, same
name and form, or that sibling's label; never an independent same-named checkbox), it focuses the
control normally, confirms it holds focus, and presses the native activation key; an unrelated overlay, an open modal elsewhere, an inert or hidden subtree or a
control that will not take focus refuse the keyboard path before any key is sent. The resulting
state is always verified. Nothing is ever forced, scripted or clicked through an unrelated
element. Every action has two phases, and the phase that fails decides what the adapter reports:
inspection and the explicit actionability trial (never dispatched), or the real click, fill or key
press (unknown); a readable state that did not change is reported as dispatched without effect.
The wording of a browser error is kept only as a diagnostic and never decides anything.

**Ordered multi-element extraction.** When one output is an ordered list assembled from several
visible items (the steps of a route), the planner selects exactly those items, in order, with
`extract_many`; discovery records one node carrying one locator ladder per item and a `list`
output whose `items` rule (type and regex) parses every element, and replay resolves each ladder
on its own, in the declared order, with no model and no screenshot. A list is all or nothing:
every declared target is a position, so one item that is missing or does not parse means the list
is not emitted (a required list fails or hands off as usual; an optional list is absent as a whole
and logged with the failing position), never shortened. A malformed extraction of
either kind (no target, no output name, an empty, duplicated or oversized selection) is refused
before it runs and never recorded.

**Collecting one list across pages, and `back`.** An output is recorded once: a second `extract`
into a name that already holds a value is refused before it runs (nothing is silently
overwritten) and the planner is told to use `output_mode: "append"` or a new name. `append` grows
a `list` output in order: an `extract` adds one item, an `extract_many` adds all of its items, all
or nothing, and the artifact node carries `"mode": "append"` (a node without the key keeps the
one-shot `set` semantics). Replay rebuilds the list across pages in node order, and a node that is
attempted again after a transient error adds its items only once. To leave a detail page the
planner has a generic `back` action: the browser's own history, never a re-requested or guessed
URL; it is recorded like any action (effect `none`, verified before any retry when it has a
checkpoint, otherwise never repeated blindly) and replayed with no model. A `back` with no history
behind it is a not-performed action failure.

**URL checkpoints prove the change, not the page.** A recorded checkpoint must be false against the screen
before the action and true against the screen after it, so a predicate that already held proves nothing. A
path is asserted only when the path itself changed; a click that moved only the query or the fragment is
asserted through the smallest changed part of it, with declared input values stored as their placeholders.
Sensitive keys, values any redaction rule rewrites, credential-shaped values and unstable keys such as
session or campaign parameters are never written into an artifact. When no part of a URL change can be
asserted safely and falsely-before, the performed action is recorded with no checkpoint and its conservative
retry safety, rather than weak evidence. The same rule decides whether a URL change can prove an action the
surface lost track of.

**Checkpoints are evidence.** The planner's expected text becomes a step's checkpoint only when
the screen did not show it before the action and shows it afterwards; a site-wide heading that
was already there proves nothing, is never recorded as the checkpoint, and the planner is told
to expect text that only appears afterwards. A typed value is proven by the control showing it
back; a `navigate`, `back` or `click` whose expected text was already on screen is proven by the
page change instead, recorded as a `url_contains` checkpoint on the new path, which replay
verifies the same way.

**An action nothing can prove is never recorded and never repeated blindly.** An action with no
requested expectation and one whose explicit expectation the screen contradicted are different
things. When an expectation fails, discovery looks for independent deterministic proof that the
action did something: a page address that changed in a way that can be asserted safely, or a
native control state that flipped (checked, selected, expanded, pressed), the acted control itself
becoming selected, or a commit inside the acted control's own field group. A click the browser merely reports as dispatched is not proof.

Closing a modal, panel or sidebar often changes nothing a planner could predict. When the safety rules
already prove a control is a dismissal of interface chrome, its own disappearance proves the dismissal
happened: the control was on screen before the click and is gone after, matched by structure rather than
by its label. That never applies to an ordinary button, link or submission, nor to anything destructive or
ambiguous, where vanishing may just mean the page re-rendered. Replay clicks the step normally and then
checks that the recorded control is really gone, failing safely if it survives, becomes ambiguous, or is
replaced by another control of the same name.

Many controls neither navigate nor change any text: clicking one simply selects it. That counts as proof
when the control went from not selected to selected, judged only by state the browser itself reports: its
native selectedness, the accessibility state it publishes about itself, a boolean state attribute a custom
control sets, a control structurally associated with it such as a label's input or a hidden control it
names, or, as a last resort, an exact state word in its class list drawn from a small closed vocabulary.
That last rung matches whole tokens only, so words that merely contain "selected" never count, and
ambiguous words such as "active" are excluded. No class name is ever written into the artifact. A change of colour or class alone is never proof, a control that was already selected proves
nothing, and one whose selectedness the browser cannot report yields no proof rather than a wrong answer.
This is considered for every click, whatever the planner expected. Genuinely new text still wins, because
it describes the screen the next step acts on, but text that was already there proves nothing and the
selection is recorded in its place rather than nothing at all. The recorded checkpoint names no markup
detail: replay resolves the step's own control and asks the same question, failing safely if it is absent,
ambiguous, or no longer selected.

A commit is the common case where a control neither navigates nor changes its own state: it moves a value
out of a field and into the group as a selected thing. It counts as proof only when an editable field in
that group gave up its value **and** exactly one new token appeared carrying its own removal control or a
selected state. The token must be new, and it must sit in the acted control's nearest meaningful container
(a form, search landmark, fieldset, group or labelled region), so a chip elsewhere on the page, a toast,
unrelated new copy, two ambiguous new chips, and the value merely staying in the field are all refused,
each with its reason in the log.

Because chips are rarely marked up semantically, one visible chip is usually several nested containers
that all look like tokens. They are collapsed to one logical token per removal control, choosing the
innermost container that still carries the label. Tokens are never merged by their text, so two chips
reading the same words stay two selections, and two chips with separate removal controls stay ambiguous. The recorded checkpoint asks replay to find that selected value again,
which page copy mentioning it does not satisfy. A sensitive or opaque value is compared but never written
down: such a commit keeps the unverified handoff instead. With proof, the wrong guess is discarded and the proof
is recorded in its place. Without proof, the action is classified `action_effect_unverified`, **no
node enters the graph**, and the planner is told the action was unverified rather than told it
succeeded. The same action on the same control at the same address is then refused before it can
be dispatched a second time, and a person is brought in; a recorded output clears that refusal,
because it is measurable progress. Nothing is retried automatically by inference from an action's
effect, so buttons, toggles, submissions, irreversible and unknown actions are all treated alike.
The bounded vision fallback follows the same rule: an unverified visual action ends the visual
budget, so a possibly dispatched action is never proposed again. In a run log, a step with no
checkpoint is written as `no_checkpoint_recorded`; `checkpoint_passed` only ever means a
checkpoint was actually verified.

**A declared output contract for direct discovery (`--output-contract PATH`).** The file is a
JSON object keyed by output name in the artifact's own output shape, the same object a campaign
spec carries under `outputs`: `{"type", "required", "pattern"}` for a scalar, or `{"type": "list",
"required", "items": {"type", "pattern"}, "min_items", "max_items"}` for a list. When it is
supplied it is authoritative: the recorded type, requiredness, item rule, pattern and cardinality
are the contract's whatever the planner proposes, a name the contract does not declare is refused
before it is read and the planner is told the valid names, `set` never replaces a recorded output, an
extraction whose shape does not fit the declared type (a single read into a list, an append into
a scalar) is refused before it runs, and an `append` that would push a list past `max_items` is
refused whole with a history entry saying the output is already complete. `done` is accepted only
when every required output is recorded, fits its type and satisfies `min_items`/`max_items`;
otherwise the history names exactly the missing or incomplete outputs and the planner continues
within the step budget. The same check runs when a replay reaches the success terminal: an
incomplete or malformed accumulated output is a structured `outputs_incomplete` failure, never a
success. Without a contract discovery infers output definitions as before, and omitted
cardinality keys keep the existing list behaviour. Contradictory bounds are refused when the file
loads.

**Provider failures.** Both planners share one bounded retry boundary: a rate limit, a 5xx or a
connection or timeout error (classified by status or exception class, never by message text) is
retried after a short backoff, at most three attempts in all, each retry logged as a structured
`planner_retry` event (provider, attempt, category, error type, status); an authentication,
quota or bad-request failure is never retried. When the provider still cannot answer, discovery
ends with a project-owned `planner unavailable` error: the run's evidence and failure screenshot
are saved, the CLI prints one concise line and exits with status 2, a campaign records the
scenario as failed and cleans up as for any other failure, and no traceback or provider text
reaches the log.

**Controls built from nonsemantic elements.** Some applications build a control from an element with no
interactive tag, no ARIA control role and no `tabindex`. Perception offers such an element as a button only
when the platform's own view of it says so: it is visible with a rendered box, not hidden, inert, disabled,
clipped or covered, not `pointer-events: none`, it has a non-empty accessible name from `aria-label` or
`title`, its computed cursor is a pointer, and it is the nearest such element, never an ancestor that
contains a real control or another candidate. At most forty per observation. No framework internals, listener
registries, class names or site-specific selectors are consulted. The control is then selected, clicked
through the ordinary actionability trial, recorded with the ordinary locator ladder and replayed with no
model; native and ARIA controls keep their existing roles and precedence, and nothing new is written into
an artifact.

**Controls that vanish before the action.** When a control the current screen proved existed has gone from
the page by the time the action runs, the attempt fails at once with the structured cause `stale_target` and
`performed: no`: nothing was dispatched, so discovery simply re-observes and the planner chooses again, and
replay returns a structured failure having performed nothing. The check is a single immediate query for the
reference perception recorded, so waits for controls that were never observed, or that may legitimately
appear later, are unchanged.

**Controls that cannot be operated.** A control that is hidden, `aria-hidden`, inert or clipped
away is not on the operator's screen and is not listed; a disabled or covered one is listed with a
`disabled` or `covered` state so the planner does not choose it blindly. Cover is decided by a hit
test at the control's centre that tolerates the control's own descendants and ancestors, its
labels and anything inside the same control, so a label or an icon over a control is never cover;
a control outside the viewport is not called covered. The same evidence applies to controls that
only the accessibility tree reports. The runtime actionable-ancestor recovery is unchanged.

**Choosing a value (`select`).** Typing into a dropdown, combobox or autocomplete field is not a
selection. The `select` action takes the control and a value (a `{{placeholder}}` in the
artifact): a native `<select>` is set by visible option text; a combobox or text input is cleared and the value
typed through real keystrokes, because many autocompletes filter their list from key events alone and
never react to a value set programmatically; the adapter then waits a bounded time for visible suggestions and looks only in the
popup the control names through `aria-controls` or `aria-owns` when that popup exists and is
visible; an unlinked control considers every container that is a plausible popup for it:
one holding candidate rows, visible, not containing the control, sitting directly under or over
the control's own span. Matching then runs inside each container separately, containers with no
match are ignored, and the selection proceeds only when exactly one container holds exactly one
match. Matches in several containers, or several matches inside the one container, are a
structured ambiguity. Arbitrary page text is never searched. Inside it the candidates are semantic options and menu
items, else the actionable rows, else the plain-text children, so a custom widget built from
divs offers its suggestions like any other. Matching is exact text first, then exact once a trailing
parenthesised gloss is stripped from the displayed option (a name the caller knows still matches
an option shown as "... (EPA)"), then prefix, then containing, with case and spacing normalized,
and at each level exactly one match is chosen: several matches, or several
unlinked popups, are a structured `ambiguous_selection` failure in which nothing is clicked and
document order decides nothing, for native selects with duplicate labels too. The chosen option is
clicked through its own semantics and the control must accept it. A field that offers no suggestions at all within that wait gets one Enter, and only one: it is accepted
solely when deterministic new state proves it, such as the field resolving into something other than the
raw query, a stored value, an active descendant or a new chip. A visible list that is ambiguous or simply
does not match keeps its existing refusal and Enter is never tried there; an Enter that only navigates, or
changes nothing provable, is handed off and records nothing. Nothing chosen is
`performed: no`; a typed value no suggestion accepted is `performed: yes` and never a selection; a
failure after the choice is `unknown` and handled by the existing rules. Acceptance is proven only by deterministic widget state:
the control's value now reading as the chosen option (never merely the text that was typed), the
active descendant naming it, a stored value the widget writes its choice into, or a newly
appearing chip or tag with its own removal control. A failed selection is
never rescued by what the screen shows: the value a `select` types into the control looks the same
whether or not an option was accepted, so only the adapter's own acceptance check can confirm one,
and a select node is recorded only after that check passes. During discovery an
ambiguity is told to the planner with the same per-screen bound as other pre-action failures; at
replay it is an `ambiguous_selection` failure, never a retry or a guess. The node is reversible, verified before a retry
only with a checkpoint, and replays with no model.

**What a person does during a discovery handoff.** When automation hands a discovery over and the
operator navigates, clicks or types before choosing `resume`, those performed actions become
ordinary action nodes in the artifact, in execution order, before the automation continues. They
go through the same locator ladders, placeholder substitution, policy check, effect and retry
classification, checkpoint rule and redaction as automated actions: a typed value equal to a
declared parameter is stored as its placeholder, a `Finish`-like click stays irreversible and is
gated at replay like any other. `restart` abandons the attempt and its human actions; `abort`,
`approve`, `deny`, `observe` and the control transfers are never nodes. An action the surface lost
track of, one the policy would deny at replay, or a sensitive value typed into a field that is not a
declared parameter cannot be represented safely: discovery fails with an
`unrecordable_human_action` error instead of building an artifact that depends on it. Handoffs
during replay stay intervention evidence and never change the artifact. The intervention evidence
and provenance are unchanged; the artifact's provenance additionally lists `human_steps`.

**Abandoned actions before a human recovery.** When the person's first recovery action is a
navigation to another page, discovery looks at the trailing automated actions performed on the
page the person left: a run with effects `none` or `reversible` only, no recorded output, and no
boundary (the entry navigation, an irreversible or unknown effect, an imported or human step).
If the screen the person left from is the screen that run started on (same URL, same controls
outside any dialog), the run changed nothing lasting and is pruned; the recovery actions then
follow the last kept node, and a redacted `speculative_suffix_pruned` event lists the pruned node
ids and the reason. If the screen differs, actions with abandonment evidence (a contradicted
expectation, a detected loop) cannot be proven harmless and discovery fails clearly; actions
without such evidence are state the person continued from and are kept. Every attempt stays in
the log and the intervention evidence.

**Links that open a new tab.** A browser can emit its page event several ticks after the click call
returns, so an empty list is never read as "this was a same-page click" until a short bounded window has
fully elapsed; once a page has appeared, collection continues briefly so two windows opened on separate
ticks are both seen. That first window is a floor on the cost of every click, which is why it is kept
small. When a click opens exactly one new browser page, the adapter waits
for it with the ordinary bounded load handling, checks its address against the host allowlist, and
makes it the active page: every later observation, screenshot, accessibility read, extraction and
action uses it, in discovery and in replay alike. The opener is kept, so `back` closes the child
page and returns to it. The click is never repeated merely because the original page looked
unchanged. Several new pages are an `ambiguous_selection` failure with none chosen, and a
destination outside the allowlist is closed with the run left on the original page.

**Toggles and searches.** A control the platform proves is a disclosure toggle (an explicit
`aria-expanded` state, or a native `<summary>`) only expands or collapses content, so it is
reversible whatever its wording says. A submission is reversible when browser evidence proves it
runs a query: the control sits in a search landmark or a form whose name says it searches, or its
own accessible purpose is to search, find or apply filters. A bare `Submit` stays unknown, a
control that moves money or destroys data stays irreversible wherever it sits, and the planner's
reason and the goal never lower a verdict.

**Unique locator rungs.** A role or text rung resolves only when exactly one perceived control
matches all of its fields, the enclosing item included; several matches make the rung ambiguous
and replay continues down that ladder to its structural rung, which names the control that was
recorded. For a parameterized single target, that structural rung is accepted only when it points
to one of the controls that already matched the parameterized semantic rung; it cannot rescue a
different or missing input value. Document order never picks a control. When every rung is missing or ambiguous the node
fails with `ambiguous_target` before anything is clicked, typed, selected or read, for every
single-target action as much as for lists. Duplicate output values are never deduplicated; the
targets are resolved correctly instead. The exact coordinate rung keeps its viewport and scroll
rules.

**Replay judges the control it resolved.** A targeted replay action resolves its locator first, read-only,
and the policy then judges that live element, which carries the runtime-only evidence a stored locator
cannot: the control's native kind, its enclosing landmark and the platform's dismissal metadata. The host
and action allowlist is still checked before anything is resolved or touched, missing and ambiguous targets
still fail before acting, and navigation and `back` are unchanged. The stricter verdict always wins, so live
evidence that makes a stored effect look less safe is honoured; none of this metadata is written into an
artifact.

**Ordered results across detail pages.** When a contract declares several list outputs of the same
length, the run must fix which results it is reporting before it leaves the list page. An append
into one of those lists is refused until an ordered snapshot exists, and the planner is told to
read one whole identity column first with `extract_many`, in displayed order, after applying and
verifying the requested filtering and sorting. That column is accepted only if the same identities, compared as
normalized visible text rather than by any structural reference, still appear exactly once each and
in the same order after a bounded re-observation. A page that merely re-renders its rows is stable;
a reordered, replaced, missing or duplicated result is not, and is refused within that bound with
nothing partial recorded. Later detail pages are
opened by the recorded identity rather than by row position, so a list that reorders while a detail
page is open cannot change the output order, and every aligned list stays on the same positions.
A task with one list, or one whose lists are read entirely from the list page, is untouched, and
the snapshot is runtime state only: the artifact holds the concrete successful path and replays
without it.

**Distinct list targets.** Several items of one list can share the same wording, and a parameter
can make their semantic rungs identical, so each item keeps its own structural rung through
parameterization: the ladders stay distinct, the duplicate-target rule still applies, and a list
may legitimately hold the same value three times when three distinct elements produced it. An
`extract_many` reads its targets as an ordered one-to-one mapping
onto distinct page elements, judged by structural reference, never text or position: when a
target's first rung only reaches an element an earlier target already took, replay continues down
that target's ladder to its structural rung. When no complete distinct assignment exists the node
fails with `ambiguous_target` and no partial or shortened list is emitted.

**Loops.** Discovery keeps a window of action signatures with the fingerprint of the screen each
one started from. A signature is executable behaviour only: the kind, the target, the
parameterized value, the output name, the list targets and the chosen reusable action. The
planner's prose and its expected text are never part of it, so rewording an action does not make
it look new. Cycles of one, two or three states are recognised, so `Search → Refine → Search →
Refine` is caught as readily as the same click twice. The second repetition puts one clear warning
in the planner's history; the third refuses the action before it runs and hands off through the
structured stuck path (cause `planner_loop`), well short of the step budget and without saving an
artifact. An action the policy would have to confirm is stopped at the first repeat. Only real
progress clears the window: a new or grown output, a human recovery or restart. A passed
checkpoint and a changed URL are not progress by themselves, since a two-screen cycle produces
both on every turn. Opening different results, or returning after recording something new, is
ordinary work and is never a loop.

**Visible leaf text, whatever its tag.** Perception lists an element's own visible text
regardless of the tag it sits in (an `<aside>`, `<small>`, `<time>` or custom element as much as a
`<span>`), plus text a stylesheet draws into it when that text carries a letter or digit, unless a
control, heading or cell already emitted covers it, or the element is hidden, `aria-hidden`,
inert, clipped away by an overflow-hiding ancestor, or laid out outside the document. Rendered
static text the projection still cannot see (a text node under a zero-box wrapper) is added from
the accessibility tree through its backing element, bounded per screen so a long page never
floods the planner. Ordering, masking, de-duplication and the planner's element cap are unchanged.

**Grounded visual extraction (opt-in, discovery only).** When a value the goal asks for is drawn
on screen but absent from the structured list, the planner reports `stuck` with cause
`missing_data` and the output's name. Only when `--vision-fallback` is on, that output is declared
in the output contract and not yet recorded, and the usual per-run and per-screen budgets allow
it, the vision model is shown the masked viewport and may answer with `visual_extract`: one box
per value, in output order. The model only says where to look. For every box the adapter finds
the page element with visible text of its own drawn under it, at the exact viewport and scroll
position of the capture, reads that element's text, and parses it under the declared type,
pattern and cardinality; whatever the model read is dropped and never becomes a value. One box
with no page-backed text (a canvas, a picture) rejects the whole proposal and a person is asked,
so a list stays all or nothing. The recorded node is an ordinary `extract` or `extract_many`
whose ladder is the element's label or structural rung with the exact capture point last, marked
to read page text there; replay needs no model and refuses the point in another viewport or
scroll position. Provenance records the step, output, box count and ladder kinds, never image
bytes, model text or secrets.

**Text nested inside a clickable container.** A planner often picks the visible text of a row or
card rather than the control around it. The adapter clicks the exact element first, under the
browser's ordinary actionability rules (which already accept text inside a button, a link or a
`role=button`). When the trial proves the element itself cannot take the click, nothing has
happened yet, and the adapter looks up the nearest enclosing control that is actionable by its
own semantics: a native control, an anchor with an `href`, a label bound to a control, or an
explicit interactive ARIA role. That control gets the same trial and, if it passes, one click,
and the recorded node carries that control's ordinary locator ladder. A container that merely
holds the text, or reacts to clicks only through a script handler, is never chosen; an overlay
covering the control is never clicked through; the child and the parent are never both clicked
physically. When no enclosing control is provably actionable the failure is a not-performed
`unactionable_target`, which is eligible for the same bounded vision fallback as a missing
control (only when `--vision-fallback` is on, the action was definitely not performed, and the
per-run and per-screen budgets allow it), with every existing protection: masked screenshot,
policy check, expected text absent before and newly present after, exact viewport and scroll
recorded, nothing recorded when verification fails. When vision cannot identify and verify the
target, a person is asked. A click that may already have been dispatched keeps the existing
rule: proven through its expected text or handed to a person, never sent to vision, never
repeated.

**The `Close` rule.** The policy never treats the word alone as a business operation, nor as
harmless. Its wording is judged first: `Close account`, `Close position` or `Close order` name a
known resource and are irreversible; `Close menu`, `Close the notice` or `Close glossary` name a
piece of the interface and are reversible; a noun the policy does not know is reversible only when
the control's container is named after it or proves a dismissal, otherwise unknown. A bare `Close` (with or without an icon glyph) is judged by browser-derived
evidence about the control's container, never by the planner's reason or the goal: a container
that names a resource closure makes it irreversible; a container that names a risky operation
leaves it unknown; a proven dismissal context makes it reversible, that is the platform's own
dismissal metadata (a dialog form, a popover hide target, a framework dismiss attribute), an
enclosing dialog, menu, panel or region, or a panel-like heading such as `Settings`, and never a
form's submit control; with no such evidence the verdict is unknown and a person confirms. The
control's accessible name and visible text are normalized into one label, so a button whose name
and text both read `Close` is reported once. A proven dismissal records the click as reversible;
the replay engine's stricter handling of a policy-versus-graph conflict is unchanged.

**Action failures during discovery.** A surface action that cannot complete never ends the run by
itself. When the adapter proves nothing physical happened, the failure and its reason are written
into the planner's history and the planner chooses again on the re-observed screen; the same
action failing twice on the same screen goes to the ordinary stuck handoff. When the action may
have happened, the expectation the planner named is checked first and a proven one is recorded
exactly once; otherwise nothing is repeated automatically, above all an irreversible or
unknown-effect action, and a person decides through the same handoff. `max_steps` still bounds
the whole run, a failed action is never written into the artifact, and no screenshot is taken
for a plain timeout.

**Vision fallback (opt-in, discovery only).** When structured perception cannot expose a control
the planner needs (something drawn on a `<canvas>`, an icon with no name, a purely visual widget),
`--vision-fallback` lets discovery take one bounded extra step: only after the planner reports
`stuck` with the explicit cause `missing_control` (a provider refusal, malformed output or plain
uncertainty goes straight to a person), a masked **viewport** screenshot in CSS pixels (page text
and form values masked, restored afterwards, no events fired; the image size is checked against
the viewport) goes to the same provider with the structured observation, the redacted parameters
and the viewport size, and the model may propose exactly one `visual_click` or `visual_type` with
a bounding box, an expected post-action text and a confidence, or `no_target`. The proposal is
validated (finite coordinates inside the viewport, confidence at least 0.6, an expected text
that is not already on screen, only declared placeholders), then goes through the normal policy
check (allowlist, blocked controls, risk confirmation), is performed once by coordinates, and is
kept only when structured perception proves the expected text newly appeared; otherwise a person
is asked and nothing is repeated. Budgets:
`--max-vision-attempts` per run (default 2), never twice for an unchanged screen, no retry after
`no_target`. The recorded node's ladder is an exact coordinate rung (`exact: true`) bound to the viewport and scroll
position it was captured at; a role-and-name rung is added only when a structured element with
that identity really sits under the box, so a guessed name can never send replay to a different
control. **Replay never calls a model or takes a screenshot for one**: an exact rung resolves to
the recorded mouse point itself, is refused in another viewport or scroll position, and the
recorded checkpoint is verified afterwards. Visual extraction is deliberately
unsupported because replay could not reproduce it. Coordinate nodes are weaker than semantic
ones, so such artifacts stay drafts until stability runs and approval say otherwise. Optional
`OPENAI_VISION_MODEL` / `ANTHROPIC_VISION_MODEL` pick a different model for the visual call;
a model without image support makes the fallback unavailable and discovery escalates as before.
The browser suite proves the whole loop on a local kiosk page whose only button is drawn on a
canvas. Limits: coordinates depend on layout and viewport; only click and type are possible;
one attempt per screen means a wrong first guess goes to a human.

## Safety in one paragraph

A host allowlist and a blocked-control list are checked before every action in discovery and in
replay, and cannot be overridden by an artifact. Clicks on controls that look irreversible
(`Finish`, `Pay`, `Delete`, ...) require a human; `Close` is judged by its wording, so dismissing a
dialog, notice, menu or panel is harmless while closing an account, a case or any other persistent
thing stays protected. Graph nodes carry an explicit `effect`
that the `--irreversible-policy` decides on; `unknown` effects always need a person. Sensitive
inputs reach neither the model nor disk: the model sees `{{password}}`, observations, logs and
screenshots (page text and form values) are masked, and an artifact containing a secret refuses
to save. Every replay leaves a bundle with the
exact artifact that ran, a redacted result and a manifest with its SHA-256.
