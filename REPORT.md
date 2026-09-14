# Design Report

## 1. Architecture

One Python process, five seams, no queues or services. The two paths share the same
`Surface`, `Policy`, and `Escalator`, and differ only in who decides the next action.

```text
                 discovery (once, model in the loop)          replay (many, no model)
                 ---------------------------------------      ----------------------------------
 goal+params --> agent.discover                                replay.replay <-- artifact + params
                   observe  : Surface.observe()                  settle     : clear known dialogs
                   decide   : Planner.decide()  (LLM)            policy     : Policy.check(step)
                   policy   : Policy.check(action)               target     : locator ladder
                   act      : Surface.click/type/navigate        act        : Surface.*
                   verify   : expected text seen?                classify   : checkpoint | business | escalate
                   record   : steps, outputs, outcomes         --> ReplayResult
                 --> artifact.build/save                       stuck? --> Escalator --> human on the same Surface
```

**Key decisions.**

- *Perception is an operator's view, not the DOM.* `Surface.observe()` returns a flat list of
  controls with role, accessible name, visible text, the item they belong to, and a screen box.
  The browser adapter builds it by walking rendered elements and naming them the way a person
  would: an explicit label when the author gave one, else the text; the placeholder or the cell
  to the left of a field; row plus column header for a table cell; and, for every control, the
  title of the nearest enclosing item ("the Add to cart button for the Backpack"). Nothing
  outside `surface.py` knows HTML exists. Password fields are perceived with an empty value.
- *The planner picks exactly one action per call and is stateless.* Anthropic uses the Messages
  API and OpenAI uses the Responses API. Each decision is one call with one forced strict tool
  (`choose_action`), the goal, the visible parameters, a
  compact history of prior actions and their results, and the numbered control list (with the
  enclosing item shown only where a name is ambiguous). No transcript is carried, so the
  artifact cannot depend on it. Parameters are written as `{{name}}` by the model, in typed text
  and URLs alike; sensitive ones are shown to it only as the placeholder.
- *Discovery records what was done; verification decides what is asserted.* Every performed
  action becomes a step. The planner's expected text becomes the step's checkpoint only if it
  actually appeared; otherwise the step is kept without one and the planner is told. Clicks made
  while a modal was open are recorded as *recoverable outcomes*, not as steps.
- *Replay is a fixed rule set over the artifact.* It imports no planner (a subprocess test loads
  the replay module in a clean interpreter and asserts neither the planner module nor either LLM SDK
  is present). Every branch it takes is either declared in the artifact or a rule in `replay.py`.
- *Trade-offs.* Single process and synchronous calls keep the vertical slice reviewable; the
  seams (`Surface`, `Planner`, `Operator`) are where a queue or a remote operator console would
  plug in. Text-based checkpoints are simple and surface-agnostic but coarser than structured
  assertions; the ordering rules in Section 3 compensate for their main failure mode.

## 2. Artifact schema

`artifacts/<name>.v<N>.json`, one immutable file per version, plain JSON, validated on build and
on load (`artifact.validate`). Top level:

| Field | Purpose |
|---|---|
| `schema_version`, `name`, `version`, `status` | Versioning and review state (`draft` after discovery; `approved` is the intended gate for unattended replay) |
| `description` | The natural-language goal, for humans and for an agent choosing a capability |
| `surface` | `{kind, app, entry_url, allowed_hosts}`: where the capability runs and where it is allowed to go |
| `inputs` | `name -> {type, required, sensitive, description}`: the typed contract a caller supplies |
| `outputs` | `name -> {type, required, pattern, description, example}`: what the caller gets back and how it is read |
| `steps` | Ordered `{id, action, target, value, checkpoint, risk}` |
| `outcomes` | Declared non-success states: `{code, kind: business \| recoverable, source, detect, recover?}` |
| `success` | Final checkpoint asserted after the last step |
| `provenance` | Run id, timestamp, planner name, step count, intervention count. Never the transcript |

**Targets are locator ladders**, ordered most stable first: `role`+`name` (plus the enclosing
item's `context` when the name alone was ambiguous on screen), visible `text` (for controls whose
text is their identity, or a `Label:` prefix for label/value text such as `Total: $ 32.39`),
structural `css` path, and screen `coords`. Replay walks the ladder one rung at a time and logs
which rung matched; any rung above zero is logged as a `drift_signal`.

**Locators are parameterized, not just values.** The recorded artifact targets `button "Add to
cart"` in `{{product_name}}` and `link "View details for {{product_name}}"`; replay substitutes
the input before resolving. When a locator depends on an input, its structural and coordinate
rungs are dropped at recording time and ignored at replay: they point at whatever was there
during discovery, and the first live run proved that falling back to them silently adds the
discovery-time product when a different one is requested. Parameterization is whole-token
(`User` never matches inside `standard_user`), and validation rejects placeholders that are not
declared inputs, in values and in locators alike.

**Outputs are typed and may be optional.** `type` is inferred from the discovery sample
(`number` for `$ 29.99`), `pattern` says which capture group to read, `required: false` turns an
absent value into `null`. **Secrets never enter the artifact**: the model sees `{{password}}`,
the step stores `{{password}}`, and `save()` refuses to write an artifact in which a sensitive
value survived parameterization. **Outcomes carry a `source`** (`observed`, `planner`,
`reviewer`): in the recorded run the model declared invalid credentials and a missing product,
and the reviewer declared the locked-out user and the rejected checkout form with `--outcome`,
because discovery never visits those states.

## 3. Determinism & error handling

**Determinism.** Replay resolves targets from the ladder, substitutes parameters, and never asks
anything to choose. Waiting is explicit: the adapter waits for the page to stop rendering, and
after each action replay polls perception until the step's checkpoint holds (bounded), rather
than sleeping fixed amounts or trusting that a click worked. The recorded checkpoints cover the
important states: `Products` after login, `Remove` after adding the product, `Your Cart` on the
cart page, `Checkout: Your Information` after Checkout, and `Checkout: Overview` after Continue
(also the success condition).

**Classification after an action**, in this order, because order is what makes text matching
safe: (1) checkpoint met: step passed; (2) screen unchanged since before the action: still
loading, keep waiting; (3) a declared *business* outcome's text is on the changed screen: return
`business_outcome` with its code (`Epic sadface: ... locked out`, `Error: ... is required`);
(4) a step without a checkpoint passes after that one check; (5) timeout: escalate. Outcome text
is matched case-insensitively because the model declares it from expectation.

**Absence outcomes.** "The requested product is not listed" cannot be detected by text that
appears; it is declared as `text_missing: "{{product_name}}"`. Such outcomes are judged only
when a step whose target names that text cannot resolve it. The first live run showed why the
scope matters: judged globally, a locked-out login was reported as "product not found" because
the product is, trivially, absent from the login page.

**Taxonomy in the result contract** (`ReplayResult`):

- *Expected business outcomes* (`status: business_outcome`, `outcome_code`, `step_id`):
  `invalid_credentials`, `user_locked_out`, `product_not_listed`, `checkout_info_missing`.
- *Recoverable conditions* never change the status; they are appended to `recoveries`: known
  interstitials (dialog text matches a `recoverable` outcome, its `recover` click is applied,
  bounded per step), transient loads (`TransientError` on a timeout or a 5xx document; backoff,
  reload, re-check the checkpoint before re-acting, three attempts; the entry page included),
  and absent optional outputs.
- *Hard failures* (`status: failure`) carry `outcome_code`, `step_id`, `expected`, `observed`:
  `transient_retries_exhausted`, `checkpoint_not_met`, `target_not_found`, `unknown_dialog`,
  `output_not_found`, `policy_denied`, `risky_step_not_confirmed`, `missing_inputs`,
  `success_condition_not_met`. An unknown dialog, an unresolvable required target, an unmet
  checkpoint, or a risky step first goes to a human (Section 5); it becomes a failure only if
  nobody resolves it.

**UI drift** is handled by the ladder (a renamed label falls through to the structural path; a
restyled page keeps its role and name) and reported through `drift_signal`; the one exception
is the deliberate refusal to fall back for parameterized targets, where a wrong element is worse
than a stop.

## 4. Heterogeneity & multi-tenant

**Surface abstraction.** The seam is `Surface` plus the shape of `Observation`/`Element`. The
artifact never mentions HTML: steps target controls by role, name, and enclosing item, fall back
to a structural path (an opaque string the surface issued and only the surface interprets) and
finally to screen coordinates. A legacy web app with framesets and no IDs is served by the same
adapter: name inference from layout and the coordinate rung already assume no clean DOM. A
desktop app needs a new adapter only: `observe()` from the OS accessibility tree (roles, names
and the parent container map directly onto role, name and context), `ref` becomes the
accessibility node path, `coords` becomes a screenshot-grounded click, and `TransientError`
wraps the app being busy. `agent.py`, `replay.py`, `policy.py`, and `escalation.py` are unchanged.

**Multi-tenant reuse.** The artifact is already the reusable object: the flow, the ladders, the
inputs and outputs, the outcomes. The intended layering is base artifact per vendor product plus
a per-tenant overlay: `surface.entry_url` and `allowed_hosts` are tenant values, `inputs` and
`outputs` are the shared contract, and an overlay may override individual ladders or add
outcomes without re-recording. Drift is detected, not guessed: the `drift_signal` on each
resolution and the checkpoint results feed a per-tenant stability record, so a tenant whose
replay keeps resolving on rung 2 is flagged for a targeted re-discovery of that one step.
`status: draft -> approved` gates which version each tenant runs unattended. None of this
plumbing is built; the schema was shaped so it does not need to be rebuilt to add it.

## 5. Escalation & handoff

**Detecting stuck.** Discovery escalates when the planner says `stuck`, when the same action is
denied repeatedly, and for any risky action (confirmation). Replay escalates on an unknown dialog,
an unresolvable required target, an unmet checkpoint after the wait, and a risky step. The
`InterventionRequest` carries capability, step id, kind (`stuck` or `confirm`), the reason, the
observed screen text, and a screenshot taken at that moment (secrets masked).

**Control-transfer model.** `SessionControl` is the single owner record: `automation` or `human`.
`Escalator.request()` snapshots evidence, transfers ownership to `human`, hands the *same*
`Surface` object to the `Operator`, and transfers ownership back in a `finally`, logging every
transition. Automation checks `require(automation)` before it acts, so a bug that tried to act
during a handoff fails loudly. `ConsoleOperator` is the minimal but real operator surface: a
terminal loop that lists the numbered controls and executes `click`, `type`, `navigate`,
`observe` on the live session (same browser, same cookies, same cart), recording each action
with typed secrets redacted by field name. With `--headed` a person can also just use the
browser window and then type `resume`.

**Handing back.** The operator ends with a disposition: `resume` (replay re-checks the current
step's checkpoint first and only re-runs it if unmet, so manual completion is not repeated),
`restart` (from step 1 on the repaired session, bounded to one restart), `abort` (a failure
carrying the original reason), or `approve` / `deny` for a confirmation. Human actions are
recorded in the run log and in `ReplayResult.interventions`. The live evidence shows the
confirmation path on `Finish`; resume and restart are proven on the fake surface in the tests.

## 6. Safety

- *Allowlist.* `Policy(allowed_hosts, allowed_actions)` is checked before every action in both
  paths. Navigation is judged by its destination, everything else by the page it acts on. Replay
  re-checks, so a hand-edited artifact cannot leave the allowlist (tested).
- *Risky vs safe.* Typing and reading are safe; clicks whose control name matches an
  irreversible or money-moving pattern (`Finish`, `Place order`, `Pay`, `Buy`, `Purchase`,
  `Submit`, `Transfer`, `Delete`, ...) are `confirm`. Confirmation rather than block because a
  goal may legitimately require them; rather than flag-only because this is money. A step
  recorded as risky is re-confirmed on every replay; unattended replays therefore fail on risky
  steps by construction. The evidence includes a replay of an artifact copy with a `Finish` step
  appended: it stopped with `risky_step_not_confirmed` and the order was never placed.
- *Blocked controls.* Account creation and deletion are denied outright with no confirmation
  offered. Logging in is not blocked: back-office goals require it, and the credentials are
  supplied by the caller, never by the model.
- *Data handling.* Sensitive inputs never reach the model or disk. Every log event and the
  artifact pass through `redact()`: sensitive key names, SSN and card shapes, API-key shapes, and
  any registered secret value. Password fields are perceived with an empty value, and screenshots
  mask any text node that contains a registered secret before capture (SauceDemo prints its test
  password on the login page). Model reasoning is logged, never stored in the artifact.
- *Limits.* Risk classification is name-based and English-only; a mislabeled button would be
  classified safe. Redaction is pattern-based and would not catch free-text PII such as a full
  name in an observation. The allowlist is host-level, not route-level. The operator console has
  no authentication or audit identity of its own.

## 7. Cuts

Left out deliberately: the operator console is a terminal, not a co-browsing UI; `status:
approved` exists in the schema but no `approve` command enforces it; no per-tenant overlay
resolver or stability scoring; no assisted LLM fallback during replay; recoverable outcomes only
know how to click; the model only declares outcomes it can infer, so states discovery never hits
are added by a reviewer with `--outcome`; one surface (Chromium) and one target (SauceDemo).

Known brittle spots in the recorded artifact: the model read the product name on the overview
through a structural path (the single cart item), which holds for a one-item cart only; the
entry checkpoint is the login page's first heading; two checkpoints after typing assert text that
is present regardless. All are visible in the artifact for a reviewer to tighten.

Next, in order: (1) an `approve` gate plus a replay-N stability score, because it turns
`drift_signal` into an operational signal; (2) per-tenant overlays over a base artifact with the
resolver in `artifact.load`; (3) a bounded, policy-checked single-step LLM recovery on
`checkpoint_not_met`, recorded as evidence; (4) an accessibility-tree desktop `Surface` to prove
the seam; (5) route-level allowlists and a real operator identity on interventions.

## Evidence index

See `evidence/INDEX.md`. Each folder holds `run.jsonl` (structured, redacted log), screenshots,
and for replays `result.json`.
