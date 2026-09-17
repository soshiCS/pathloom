# Evidence index

Every folder below was produced against the live public SauceDemo site or the local MemberOps
Sandbox, replaying the graph artifacts in `artifacts/` with no model. Each holds `run.jsonl`
(the structured, redacted log of what the system did and why), screenshots, a redacted
`result.json`, a byte-exact copy of the artifact that ran and `manifest.json` with its SHA-256;
stability folders hold `report.json` and a log linking their child replays. The password was
always a sensitive input: the retained run bundles and artifacts contain no sensitive literal,
only `{{password}}` or `[REDACTED]`, and the login page's printed password is masked in
screenshots. The rehearsal command at the end of this file shows SauceDemo's public,
published test credential in plain text, as the README's commands do; no real credential or API key
is present anywhere in this repository. **No run ever clicked `Finish`.** No run below involved a
model.

## Capability `checkout_paths` (SauceDemo)

| Folder | What it proves | Artifact | Result |
|---|---|---|---|
| `stability-20260915T005943Z-2a01` | Three fresh unattended replays of the draft graph's `inventory + cart_link` path with `irreversible-policy: deny`; `report.json` recomputed at approval | `checkout_paths.v2.json` | 3/3 `success`, total 32.39, 0 recoveries, 0 interventions, 0 drift, eligible |
| `replay-20260915T005943Z-a600`, `replay-20260915T010012Z-b615`, `replay-20260915T010041Z-717f` | The three child replays of the report above | `checkout_paths.v2.json` | `success` / 32.39 |
| `stability-20260915T010110Z-81a7` | Same for `details + cart_link` | `checkout_paths.v2.json` | 3/3 `success`, eligible |
| `replay-20260915T010110Z-e848`, `replay-20260915T010140Z-634f`, `replay-20260915T010211Z-bfcb` | Its child replays | `checkout_paths.v2.json` | `success` / 32.39 |
| `stability-20260915T010241Z-7b2c` | Same for `inventory + cart_url` | `checkout_paths.v2.json` | 3/3 `success`, eligible |
| `replay-20260915T010241Z-28d2`, `replay-20260915T010310Z-17a5`, `replay-20260915T010339Z-d428` | Its child replays | `checkout_paths.v2.json` | `success` / 32.39 |
| (artifact) | `approve` turned the draft into `artifacts/checkout_paths.v3.json` (`status: approved`, provenance names reviewer, source digest, the three report digests and the three tested assignments); the draft file is byte-identical | `checkout_paths.v3.json` | approved |
| `replay-20260915T010449Z-a837` | Approved unattended replay (`--operator none`) of `details + cart_link` through the lifecycle gate; manifest says `approved`, artifact copy is the exact file | `checkout_paths.v3.json` | `success` / total 32.39, 12 actions, stopped on Checkout Overview |
| `replay-20260915T010520Z-5a69` | An undeclared combination (`details + cart_url`) refused by the entry selector gate: zero surface actions, a stuck request nobody answered, structured failure | `checkout_paths.v3.json` | `failure` / `no_matching_edge` at `d1`, 0 actions |
| `replay-20260915T010527Z-7562` | `username=locked_out_user`: the site's "locked out" message is a declared business outcome of the graph | `checkout_paths.v3.json` | `business_outcome` / `locked_out` at `s4` |

The draft graph is revision 2 of `checkout_paths` (revision 1 was an earlier attempt that was
removed), which is why the approved revision is 3.

## Capabilities `member_account_prepare` and `member_account_open` (MemberOps Sandbox, local)

These folders were produced against the local MemberOps Sandbox (`python -m examples.member_ops`,
loopback only) with the same commands as above. The training credential `training_only` was a
sensitive input throughout; the bundles hold no sensitive literal.

| Folder | What it proves | Artifact | Result |
|---|---|---|---|
| `replay-20260915T094237Z-fb65`, `replay-20260915T094400Z-ad10`, `replay-20260915T094558Z-aa69` | Supervised replays (`--operator console`, draft warning logged) of the three declared paths: `member_id + savings`, `phone + savings`, `member_id + checking` | `member_account_prepare.v1.json` | `success` each, typed outputs |
| `stability-20260915T094738Z-35ac`, `stability-20260915T094953Z-ce54`, `stability-20260915T095200Z-d68d` | Three fresh unattended replays per declared selector assignment, irreversible policy `deny` | `member_account_prepare.v1.json` | 3/3 `success` each, eligible |
| `replay-20260915T094738Z-450a`, `replay-20260915T094808Z-95a8`, `replay-20260915T094839Z-5865`, `replay-20260915T094953Z-da67`, `replay-20260915T095025Z-ba14`, `replay-20260915T095057Z-7e5f`, `replay-20260915T095200Z-5373`, `replay-20260915T095231Z-3c30`, `replay-20260915T095301Z-bb21` | The nine child replays of the three reports | `member_account_prepare.v1.json` | `success` |
| (artifact) | `approve` turned the draft into `artifacts/member_account_prepare.v2.json` (`status: approved`) | `member_account_prepare.v2.json` | approved |
| `replay-20260915T221634Z-0e59`, `replay-20260915T223220Z-aa74` | Replays of the approved version (`phone + savings`, then `member_id + savings`) through the lifecycle boundary; the manifests say `approved` | `member_account_prepare.v2.json` | `success` |
| `replay-20260915T221903Z-f8d1` | Approved v2 replay of the `phone + savings` path in which a person took over on the same live session: a `stuck` request (`unknown_dialog`) answered with `resume` after 1 recorded operator action(s) | `member_account_prepare.v2.json` (approved) | `success` |
| `replay-20260915T222750Z-25c8` | The open capability under `--irreversible-policy deny`: the flow stops at the final click with a structured failure and no account is opened | `member_account_open.v1.json` | `failure` / `irreversible_denied` |
| `replay-20260915T223045Z-c926` | The same under `confirm` with a console operator who approved: the irreversible node ran once after a recorded `approve` | `member_account_open.v1.json` | `success` |

The graphs these runs exercise are the artifacts in `artifacts/`: `member_account_prepare` (safe,
stops at the review page; revision 1 draft, revision 2 approved) and `member_account_open`
(carries the irreversible final click; draft, supervised use only).

## Human handoff

The same-session handoff (control transfer, human actions recorded, resume/restart/abort/approve/
deny) is exercised by the offline test suite and appears live in `replay-20260915T010520Z-5a69`
(a stuck request answered by nobody), `replay-20260915T222750Z-25c8` (an irreversible node
refused outright under `deny`), `replay-20260915T223045Z-c926` (a confirmation answered with
`approve` by the console operator) and `replay-20260915T221903Z-f8d1` (an intervention resolved by
the console operator on the same session). To rehearse one yourself, make a draft whose Login
locator cannot resolve, replay it
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
