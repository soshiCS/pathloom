# Evidence index

All runs were made against the live public SauceDemo test site with the artifact
`checkout_review.v1.json` (a copy sits inside the discovery folder). The password was passed as
a sensitive input: it appears nowhere below except as `{{password}}` or `[REDACTED]`, and the
login-page screenshot has the site's own printed password masked. Each folder holds `run.jsonl`
(structured, redacted log of what the system did and why), screenshots, and for replays
`result.json` (the `ReplayResult` returned to the caller). No run ever clicked `Finish`.

| Folder | What it demonstrates | Result |
|---|---|---|
| `discovery-20260911T103032Z-8890` | Real LLM-driven discovery (`claude-opus-5`, 15 planner decisions with reasons): login, add `{{product_name}}` via a context-qualified locator, cart, checkout form, overview, four extracts, `done` before Finish | artifact v1, 15 steps |
| `replay-20260911T103209Z-7241` | Deterministic replay for the example inputs (no LLM), stops on Checkout Overview | `success`: product_name "Sauce Labs Backpack", subtotal 29.99, tax 2.4, total 32.39 |
| `replay-20260911T103248Z-6d17` | `username=locked_out_user`: the site's "locked out" message is a declared business outcome | `business_outcome` / `user_locked_out` at `s4` |
| `replay-20260911T103255Z-9d01` | `product_name=Sauce Labs Unicorn`: the product locator cannot resolve and no structural fallback is allowed; absence outcome scoped to that step | `business_outcome` / `product_not_listed` at `s5` |
| `replay-20260911T103303Z-7ec7` | Empty `postal_code`: the checkout form's "is required" error is a declared business outcome | `business_outcome` / `checkout_info_missing` at `s11` |
| `replay-20260911T103321Z-7bf4` | A copy of the artifact with a `Finish` step appended, replayed unattended: the risky step asks for confirmation, no operator answers, the order is never placed | `failure` / `risky_step_not_confirmed` at `s_finish`, one recorded intervention |
