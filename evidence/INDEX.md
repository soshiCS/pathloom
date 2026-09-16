# Evidence index

Every folder below was produced against the live public SauceDemo site. Each holds `run.jsonl`
(the structured, redacted log of what the system did and why), screenshots, and for replays a
redacted `result.json`, a byte-exact copy of the artifact that ran and `manifest.json` with its
SHA-256. The password was always a sensitive input: the retained run bundles and artifacts contain
no sensitive literal, only `{{password}}` or `[REDACTED]`, and the login page's printed password is
masked in screenshots. The rehearsal command at the end of this file shows SauceDemo's public,
published test credential in plain text, as the README's commands do; no real credential or API key
is present anywhere in this repository. **No run ever clicked `Finish`.** "LLM" says whether a model
made decisions in that run.

## Campaign `checkout_paths` (schema 2.0 graph)

| Folder | What it proves | Artifact | Result | LLM |
|---|---|---|---|---|
| `campaign-20260915T005111Z-ddde` | The campaign log and `summary.json` mapping the three declared scenarios to their discovery runs, traces and graph node paths; traces merged by prefix tree into one draft | `checkout_paths.v2.json` (draft) | succeeded, 3/3 scenarios, 32 nodes, 36 edges | orchestration only |
| `discovery-20260915T005111Z-0634` | Scenario `inventory_cart_link`: add from the product list, open the cart control; every planner decision and its reason logged; trace `inventory_cart_link.trace.json` | draft trace | 12 steps, goal reached | `openai:gpt-6-astra` |
| `discovery-20260915T005155Z-ca52` | Scenario `details_cart_link`: open the product details page first, then the cart control | draft trace | 13 steps, goal reached | `openai:gpt-6-astra` |
| `discovery-20260915T005245Z-1ce1` | Scenario `inventory_cart_url`: add from the list, navigate to the cart URL | draft trace | 12 steps, goal reached | `openai:gpt-6-astra` |
| `stability-20260915T005943Z-2a01` | Three fresh unattended replays of `inventory + cart_link` with `irreversible-policy: deny`; `report.json` recomputed at approval | `checkout_paths.v2.json` | 3/3 `success`, total 32.39, 0 recoveries, 0 interventions, 0 drift, eligible | no |
| `replay-20260915T005943Z-a600`, `replay-20260915T010012Z-b615`, `replay-20260915T010041Z-717f` | The three child replays of the report above | `checkout_paths.v2.json` | `success` / 32.39 | no |
| `stability-20260915T010110Z-81a7` | Same for `details + cart_link` | `checkout_paths.v2.json` | 3/3 `success`, eligible | no |
| `replay-20260915T010110Z-e848`, `replay-20260915T010140Z-634f`, `replay-20260915T010211Z-bfcb` | Its child replays | `checkout_paths.v2.json` | `success` / 32.39 | no |
| `stability-20260915T010241Z-7b2c` | Same for `inventory + cart_url` | `checkout_paths.v2.json` | 3/3 `success`, eligible | no |
| `replay-20260915T010241Z-28d2`, `replay-20260915T010310Z-17a5`, `replay-20260915T010339Z-d428` | Its child replays | `checkout_paths.v2.json` | `success` / 32.39 | no |
| (artifact) | `approve` turned the draft into `artifacts/checkout_paths.v3.json` (`status: approved`, provenance names reviewer, source digest, the three report digests and the three tested assignments); the draft file is byte-identical | `checkout_paths.v3.json` | approved | no |
| `replay-20260915T010449Z-a837` | Approved unattended replay (`--operator none`) of `details + cart_link` through the lifecycle gate; manifest says `approved`, artifact copy is the exact file | `checkout_paths.v3.json` | `success` / total 32.39, 12 actions, stopped on Checkout Overview | no |
| `replay-20260915T010520Z-5a69` | An undeclared combination (`details + cart_url`) refused by the entry selector gate: zero surface actions, a stuck request nobody answered, structured failure | `checkout_paths.v3.json` | `failure` / `no_matching_edge` at `d1`, 0 actions | no |
| `replay-20260915T010527Z-7562` | `username=locked_out_user`: the site's "locked out" message is an outcome the planner declared during discovery | `checkout_paths.v3.json` | `business_outcome` / `locked_out` at `s4` | no |

Removed as superseded: a first campaign attempt (recorded before selector values were made literal
and before the perception fixes), a first stability batch run before the perception regression
fix, and the earlier Books to Scrape and website-chatbot experiments. Version 1 of
`checkout_paths` was that first attempt, which is why the retained draft is version 2.

## Earlier single artifact `checkout_review` (schema 1.0, linear)

| Folder | What it proves | Artifact | Result | LLM |
|---|---|---|---|---|
| `discovery-20260911T103032Z-8890` | The first real discovery: 15 planner decisions, login, context-qualified add-to-cart, cart, checkout form, overview, four extracts, `done` before Finish | `checkout_review.v1.json` | 15 steps | `claude-opus-5` |
| `replay-20260911T103209Z-7241` | Deterministic replay of that artifact | `checkout_review.v1.json` | `success`: product, subtotal 29.99, tax 2.4, total 32.39 | no |
| `replay-20260911T103248Z-6d17` | `username=locked_out_user` as a declared business outcome | `checkout_review.v1.json` | `business_outcome` / `user_locked_out` at `s4` | no |
| `replay-20260911T103255Z-9d01` | `product_name=Sauce Labs Unicorn`: absence outcome scoped to the product step, no structural fallback allowed | `checkout_review.v1.json` | `business_outcome` / `product_not_listed` at `s5` | no |
| `replay-20260911T103303Z-7ec7` | Empty `postal_code`: the form's "is required" error as a declared outcome | `checkout_review.v1.json` | `business_outcome` / `checkout_info_missing` at `s11` | no |
| `replay-20260911T103321Z-7bf4` | A copy of the artifact with a `Finish` step appended, replayed unattended: the risky step asks for confirmation, no operator answers, the order is never placed | copy of `checkout_review.v1.json` + Finish | `failure` / `risky_step_not_confirmed`, one recorded intervention | no |

These older bundles predate the manifest and artifact copy that replay bundles carry now.

## Human handoff

The same-session handoff (control transfer, human actions recorded, resume/restart/abort/approve/
deny) is exercised by the offline test suite and appears live in `replay-20260911T103321Z-7bf4`
(a confirmation request answered by nobody) and `replay-20260915T010520Z-5a69` (a stuck request
answered by nobody). No interactive operator session was recorded in this environment, and none
is claimed. To rehearse one yourself, make a draft whose Login locator cannot resolve, replay it
headed and supervised, and at the operator prompt type `observe`, then `click <n>` on the real
Login control, then `resume`; automation re-checks the `Products` checkpoint and continues without
repeating the action. Use the operator's commands rather than clicking in the browser window: only
operator commands are recorded as structured human actions in the intervention record.
This is an intentional UI-drift rehearsal, not a naturally occurring failure:

```bash
python - <<'PY'
import json
g = json.load(open("artifacts/checkout_paths.v2.json"))
node = next(n for n in g["nodes"] if n["id"] == "s4")
node["action"]["target"]["strategies"] = [{"kind": "role", "role": "button", "name": "Log in (renamed)"}]
g["provenance"]["note"] = "handoff rehearsal: Login locator made unresolvable on purpose"
json.dump(g, open("artifacts/handoff_rehearsal.v1.json", "w"), indent=2)
PY
python -m src.cua replay --artifact artifacts/handoff_rehearsal.v1.json --operator console --headed \
  --param username=standard_user --param password=secret_sauce --sensitive password \
  --param "product_name=Sauce Labs Backpack" --param first_name=Test --param last_name=User --param postal_code=10001 \
  --param product_route=inventory --param cart_route=cart_link
```
