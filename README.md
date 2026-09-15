# Computer-Use Automation System (interface.ai take-home)

An LLM discovers how to accomplish a goal on a real web application **once**; the run is
recorded as a typed, versioned **capability artifact**; the artifact is then **replayed
deterministically** (no model in the loop) with input parameters, structured outputs,
explicit error classification, safety guardrails, and a real human-handoff path.

```text
goal + params  ->  LLM discovery (observe -> decide -> policy -> act -> verify -> record)
               ->  artifacts/<name>.vN.json (the capability contract)
               ->  deterministic replay  ->  {success + outputs | business_outcome | failure}
                                          ->  human takes over the same live session when stuck
```

The demonstration target is the public **SauceDemo** test shop (`https://www.saucedemo.com/`),
a multi-step flow with a login, a product list, a cart, a checkout form, and an overview page.
The capability logs in with the site's published test credentials, adds the requested
product, fills synthetic checkout data, reads the totals on the Checkout Overview page, and
**stops before Finish**. `Finish` is classified as an irreversible action: it is never clicked
without explicit human confirmation, even if an artifact asks for it.

Design write-up: [`REPORT.md`](REPORT.md). Assignment: [`docs/assignment.pdf`](docs/assignment.pdf).

## What is in the box

| Path | Role |
|---|---|
| `src/cua/models.py` | Shared data shapes (Observation, Action, Artifact, Step, ReplayResult, InterventionRequest, ...) |
| `src/cua/surface.py` | `Surface` seam + `PlaywrightSurface` (accessibility-style perception, item context, locator ladder, secret masking) |
| `src/cua/planner.py` | `Planner` seam plus interchangeable `ClaudePlanner` and `OpenAIPlanner` adapters (one strict tool call per decision) |
| `src/cua/agent.py` | Discovery loop; records parameterized steps, typed outputs, recoverable and business outcomes |
| `src/cua/artifact.py` | Parameterize (values and locators), build, validate, version, save, load |
| `src/cua/graph.py` | Capability graph schema 2.0: typed guards, validation (reachability, acyclicity), JSON round trip, and the version-1 adapter |
| `src/cua/replay.py` | Deterministic replay of schema 1.0 artifacts: ladder targeting, checkpoints, retries, outcome classification, escalation |
| `src/cua/graph_replay.py` | Deterministic replay of schema 2.0 graphs on the same helpers: guard-based edge selection, effect policy (`--irreversible-policy`), per-node retry safety |
| `src/cua/campaign.py` | Human-declared multi-scenario discovery: JSON spec, one discovery run per scenario, campaign evidence and summary |
| `src/cua/merge.py` | Conservative prefix-tree merge of verified linear traces into one graph, with conditional inputs and campaign provenance |
| `src/cua/lifecycle.py` | Draft/approved gate for unattended replay, version-aware loading and engine dispatch, replay evidence bundles (result, artifact copy, manifest with SHA-256) |
| `src/cua/stability.py` | N fresh unattended replays with `irreversible-policy: deny`, metrics and eligibility in `report.json` |
| `src/cua/approval.py` | Local review record: eligible reports for the exact draft digest (one per selector assignment for a campaign graph) become the next `approved` version |
| `src/cua/policy.py` | Host/action allowlists, blocked controls, risky-action classification, redaction |
| `src/cua/escalation.py` | `SessionControl` ownership, `Escalator`, `ConsoleOperator` (drives the same live session) |
| `src/cua/evidence.py` | Redacted JSONL run log, screenshots, evidence bundles |
| `src/cua/__main__.py` | CLI: `discover`, `discover-campaign`, `replay` (engine picked by `schema_version`), `stability`, `approve` |
| `scenarios/` | Campaign specifications (`checkout_paths.json` declares three routes through the shop) |
| `tests/` | Deterministic offline tests on an in-memory fake shop surface and a scripted planner |
| `artifacts/` | Saved capability artifacts |
| `evidence/` | Logs, screenshots, artifact copy and result JSON from the recorded runs |

## Setup

Python 3.11+ (developed on 3.14). No other services are needed.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

Keys and config (only for `discover`, the real LLM run; replay never uses the model):

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # needed when --provider anthropic
export OPENAI_API_KEY=sk-...             # needed when --provider openai
export ANTHROPIC_MODEL=claude-opus-5     # optional; default shown
export OPENAI_MODEL=gpt-6-astra          # optional; default shown
```

Tests run fully offline. The `discover` and `replay` commands reach the live public site and
are the explicit integration/demo path, not part of the test suite. Keep them low-volume: one
discovery is about 15 model calls and a handful of page loads; one replay is six page loads.

## Demo path (real SauceDemo)

The password is a public test credential, but it is treated as sensitive: with `--sensitive
password` the model only ever sees `{{password}}`, the artifact stores the placeholder, the
logs mask the value, and screenshots blank it out of the rendered page (SauceDemo prints it
in its own help text).

**1. Discover with Anthropic** (the default provider; needs its API key; about two minutes):

```bash
python -m src.cua discover --provider anthropic --operator none \
  --goal "Log in to SauceDemo using the supplied test credentials, find product {{product_name}}, add it to the cart, complete the checkout information using synthetic test data, reach the Checkout Overview page, and return the product name, subtotal, tax, and total. Stop before clicking Finish." \
  --url https://www.saucedemo.com/ --name checkout_review \
  --param username=standard_user --param password=secret_sauce --sensitive password \
  --param "product_name=Sauce Labs Backpack" --param first_name=Test --param last_name=User --param postal_code=10001 \
  --outcome "user_locked_out=locked out" --outcome "checkout_info_missing=is required"
```

Or run the same discovery with OpenAI by changing only the provider:

```bash
python -m src.cua discover --provider openai --operator none \
  --goal "Log in to SauceDemo using the supplied test credentials, find product {{product_name}}, add it to the cart, complete the checkout information using synthetic test data, reach the Checkout Overview page, and return the product name, subtotal, tax, and total. Stop before clicking Finish." \
  --url https://www.saucedemo.com/ --name checkout_review \
  --param username=standard_user --param password=secret_sauce --sensitive password \
  --param "product_name=Sauce Labs Backpack" --param first_name=Test --param last_name=User --param postal_code=10001 \
  --outcome "user_locked_out=locked out" --outcome "checkout_info_missing=is required"
```

Use `--model MODEL_ID` to override the selected provider's model for one run.

Writes `artifacts/checkout_review.vN.json` and `evidence/discovery-<run>/`. Every input is
recorded as a `{{placeholder}}`, including inside locators ("the Add to cart button in the
`{{product_name}}` card"). The two `--outcome` flags declare states discovery never visits
(a locked-out user, a rejected checkout form); the model declares the rest itself.

**2. Replay deterministically** (no LLM involved):

```bash
python -m src.cua replay --artifact artifacts/checkout_review.v1.json \
  --param username=standard_user --param password=secret_sauce --sensitive password \
  --param "product_name=Sauce Labs Backpack" --param first_name=Test --param last_name=User --param postal_code=10001
```

Prints a `ReplayResult` such as:

```json
{"status": "success", "outputs": {"product_name": "Sauce Labs Backpack", "subtotal": 29.99, "tax": 2.4, "total": 32.39}, ...}
```

**3. Expected business outcomes** (same command, different inputs):

| Input change | Result |
|---|---|
| `--param username=locked_out_user` | `business_outcome` / `user_locked_out` at the login step |
| `--param password=wrong` | `business_outcome` / `invalid_credentials` |
| `--param "product_name=Sauce Labs Unicorn"` | `business_outcome` / `product_not_listed` at the add-to-cart step |
| `--param postal_code=` | `business_outcome` / `checkout_info_missing` at the Continue step |

Transient loads (timeouts, 5xx documents) are retried with backoff; persistent ones become a
hard failure with the step, what was expected, and what was observed.

**4. Human handoff on the same live session.** Whenever replay cannot proceed (an unknown
dialog, an unmet checkpoint, a control it cannot find, or a risky step such as `Finish`), it
hands the live browser to you in the terminal with the context and a screenshot. You can
`observe`, `click <n>`, `type <n> <text>`, `navigate <url>`, then `resume`, `restart`,
`abort`, or answer `approve` / `deny` to a confirmation. Add `--headed` to also see the window:

```bash
python -m src.cua replay --artifact artifacts/checkout_review.v1.json --headed \
  --param username=standard_user --param password=secret_sauce --sensitive password \
  --param "product_name=Sauce Labs Backpack" --param first_name=Test --param last_name=User --param postal_code=10001
```

Useful flags: `--operator none` runs unattended (interventions come back unresolved, so a risky
step fails instead of running), `--allow-host` sets the allowlist (defaults to the entry host),
`--outcome CODE=TEXT` / `--missing-outcome CODE=TEXT` let a reviewer declare outcomes,
`--quiet` stops echoing log events.

**5. Capability graphs (schema 2.0).** `replay` also runs graph artifacts: action nodes carry
what a step does, decision nodes branch on typed guards (`input_equals`, `text_visible`,
`url_matches`, `element_present`, `dialog_contains`, with `always` as the last-resort fallback),
and terminal nodes end the run as `success`, `business_outcome`, or `failure`. Each action node
declares an `effect` and a `retry_safety`; `--irreversible-policy deny|confirm|allow` (default
`confirm`) decides what happens at an `irreversible` action, an `unknown` effect is refused under
`deny` and otherwise always needs a human, and the global allowlist can never be overridden. Discovery still records linear
artifacts; `graph.from_linear` converts one in memory and `graph.load_graph` loads either version.

**6. Multi-scenario discovery.** A JSON campaign spec declares one capability, the *selector*
inputs whose values pick a path, and the scenarios a person wants recorded (see
`scenarios/checkout_paths.json`). Each scenario is discovered on its own fresh session with its
own log and evidence, in order; only when all succeed are the traces merged into one draft
graph by a prefix tree (an entry decision that admits only the declared selector assignments,
shared identical prefixes, a decision node with `input_equals` guards on the complete selector
assignment at the first difference, no suffix merging) and saved as the next artifact version. Path-specific inputs get `required_when` conditions. A failed scenario
ends the campaign with all evidence kept and nothing saved; `evidence/campaign-<id>/summary.json`
maps every scenario to its run, trace and graph path.

```bash
python -m src.cua discover-campaign --spec scenarios/checkout_paths.json --provider anthropic --operator none
python -m src.cua replay --artifact artifacts/checkout_paths.v1.json \
  --param username=standard_user --param password=secret_sauce --sensitive password \
  --param "product_name=Sauce Labs Backpack" --param first_name=Test --param last_name=User \
  --param cart_route=cart_url --param cart_url=https://www.saucedemo.com/cart.html \
  --param zip_source=postal_code --param postal_code=10001
```

**7. Lifecycle: draft, stability, approval, unattended replay.** Discovery and campaigns produce
drafts. A draft can be replayed only supervised (`--operator console`, with a warning) or by the
`stability` command, which replays one invocation N times, each on a fresh browser and session,
unattended, with `irreversible-policy: deny`, and writes `evidence/stability-<id>/report.json`:
artifact identity and SHA-256 digest, parameter names (never values), counts by status and
outcome, success and clean-run rates, recoveries, interventions and locator drift signals, a
record per run linking its evidence, and `eligible_for_approval` (at least three completed runs,
all `success`, no intervention, one digest throughout; recoveries and locator fallbacks are
reported but do not disqualify). `approve` checks every report against the draft's exact digest,
re-derives every summary field and the eligibility verdict from the report's own run records
(an edited flag or total is refused), and requires one report per selector assignment admitted
by the graph's entry gate, cross-checked against campaign provenance when present, then saves
the next immutable version with `status: approved` and approval provenance (reviewer, timestamp,
source version and digest, report paths and digests, tested assignments). The draft file is
never modified. `replay --operator none` refuses a draft with `artifact_not_approved` before any
browser exists, and still writes an evidence bundle; the same rule is enforced inside the
library (`lifecycle.replay_any` takes an explicit purpose: supervised, stability or unattended),
so a caller that already opened a surface gets the refusal before any action. This is a local
review workflow, not cryptographic signing or authentication: anyone who can write to
`artifacts/` can approve, and a report rewritten consistently end to end would pass the checks.

```bash
python -m src.cua stability --artifact artifacts/checkout_review.v1.json --runs 3 \
  --param username=standard_user --param password=secret_sauce --sensitive password \
  --param "product_name=Sauce Labs Backpack" --param first_name=Test --param last_name=User --param postal_code=10001
python -m src.cua approve --artifact artifacts/checkout_review.v1.json \
  --report evidence/stability-<id>/report.json --reviewer soroush
python -m src.cua replay --artifact artifacts/checkout_review.v2.json --operator none ...
```

Every replay bundle under `evidence/replay-<id>/` holds `run.jsonl`, screenshots, `result.json`
(redacted with the run's secret values, like the log), a byte-exact copy of the artifact that
ran, and `manifest.json` (artifact path, schema and capability versions, status, SHA-256,
parameter names only).

Deliberate limitation: stability approval targets prepare-only or read-only flows that finish
before any irreversible effect, which is why it runs with `deny`. Repeatedly testing a flow that
really commits would need a sandbox, an idempotency key, or a rollback mechanism.

## Tests

```bash
python -m pytest
```

226 tests, all deterministic and offline: policy (allowlists, blocked controls, `Finish` and
other irreversible actions requiring confirmation, redaction), artifact (word-boundary
parameterization of values and locators, validation, versioning, save/load, refusal to save a
leaked secret), replay (determinism, a subprocess proof that the replay module never imports
the planner or either LLM SDK, typed outputs, the four business outcomes, recoverable notice,
transient retry, hard failure, unknown dialog, tampered artifact, `Finish` blocked unattended
and only clicked after explicit approval, no structural fallback onto the wrong product,
locator ladder fallback), capability graph (schema 2.0 round trip, a branched graph, every
validation failure, version-1 conversion with metadata and step order preserved, version-1 loading
and replay unchanged), graph replay (a converted graph replays action-for-action like the linear
artifact across every outcome scenario, all six guard kinds, priority and fallback selection, no
matching edge, the three terminals, all three retry-safety modes, `deny`/`confirm`/`allow`
irreversible policies, unknown effects, global denial overriding `allow`, resume/restart/abort/
approve/deny, engine dispatch by schema version, no planner or SDK import), campaigns and merging
(spec validation, one session per scenario, prefix sharing and branching, multiple divergence points,
identical and prefix traces, no suffix merging, deterministic ids, conflicts rejected, conditional
inputs, sensitive path-specific values off disk, a failed or crashing scenario saving nothing with
its session closed and secrets redacted, merged graph replayed per scenario and refusing undeclared
selector combinations before any action), lifecycle (stability runs on both schemas with fresh
closed sessions, metrics and eligibility, recoveries and drift reported without disqualifying,
irreversible and unknown actions denied, crashes recorded, no planner import; approval creating an
immutable approved version with the draft byte-identical, digest, eligibility, coverage and
malformed-report rejections, no browser; the unattended gate blocking drafts with an evidence
bundle and admitting approved artifacts), escalation (ownership transfer, same-session operation, redacted
typed password, resume/restart, the goal on every handoff), discovery (fully parameterized recording, checkpoints for the
important states, typed outputs, outcomes, password never reaching the planner, `Finish`
requiring approval during discovery, denials, stuck, max steps, and Anthropic/OpenAI tool-call binding). The browser
and the LLM are replaced at their protocol boundaries by `tests/fake_surface.py` and
`tests/scripted_planner.py`.

## Evidence

`evidence/` contains one folder per run (`run.jsonl` structured log, screenshots, and for
replays a `result.json`), plus a copy of the artifact next to the discovery run that produced
it. `evidence/INDEX.md` says which folder demonstrates what.
