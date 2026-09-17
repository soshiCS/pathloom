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

## Verified prefix reuse

A discovery may start from an approved capability that already performs its beginning, for
example `sign in -> find member -> open profile` before `prepare a savings sub-account`:

```bash
python -m src.cua discover --reuse-capability member_profile_lookup ...   # newest approved version by name
python -m src.cua discover --reuse-artifact artifacts/member_profile_lookup.v3.json ...   # or exactly this file
```

(`reuse_capability` / `reuse_artifact` do the same at the top level of a campaign spec; the two
are mutually exclusive, and one prefix per discovery.) Before any browser exists, the prefix is
checked: approved, unchanged digest, its required inputs present with matching sensitivity, the
same entry URL, no hosts beyond the discovery's allowlist, and no irreversible, unknown or risky
action anywhere in it. It is then replayed on the live session with no operator and no model,
under the `discovery_reuse` purpose (approved only, irreversible policy forced to deny). If it
succeeds, the planner's first look is the screen it ended on and the executed path (the action
nodes the engine completed, in order, on the branch actually taken) becomes the opening nodes of
the new recording; if it fails, ends in a business outcome, crashes, or its path cannot be
imported, that session is closed and ordinary discovery starts on a fresh one, so the model never
continues on a possibly dirty session. The new artifact is self-contained: nodes are renumbered
and otherwise copied exactly (action, ladder, checkpoint, `effect`, `retry_safety`), decision and
terminal nodes are never imported, extractions only when the new discovery declares the same
output contract, and provenance names the source, its version and digest, the executed path and
the run id, never a parameter value.
A name with no approved artifact logs `reuse_not_found` and discovery proceeds normally; an
explicit path that is missing, unreadable or not approved is a configuration error. This is
reuse of a verified, checkpointed prefix, not a cache of isolated clicks, which would lack the
preconditions that make a step safe to repeat. No live reuse run has been recorded; the suite
proves it on the fake shop and, in Chromium, on MemberOps.

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
(`Finish`, `Pay`, `Delete`, ...) require a human, and graph nodes carry an explicit `effect`
that the `--irreversible-policy` decides on; `unknown` effects always need a person. Sensitive
inputs reach neither the model nor disk: the model sees `{{password}}`, observations, logs and
screenshots (page text and form values) are masked, and an artifact containing a secret refuses
to save. Every replay leaves a bundle with the
exact artifact that ran, a redacted result and a manifest with its SHA-256.
