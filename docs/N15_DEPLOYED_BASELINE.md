# N15 Deployed Baseline — Neo + Builder

Date: 2026-10-19
Branch: `lane/n15-b0-neo-baseline`

## Production head

The deployed Neo service runs at commit `d5b7b4e`. This hash is **not present** in the
supplied repo's object database. The repo head `a35938a` and the deployed head `d5b7b4e`
share no common ancestry that can be established from the supplied history.

## File reconciliation table

Nine source files differ between the repo (`a35938a`) and production (`d5b7b4e`).
The table below documents how each was resolved in this baseline commit.

| File | Production difference from repo | Resolution |
|---|---|---|
| `src/__main__.py` | Replaces a single queue consumer with a configurable thread pool, default eight workers, and a semaphore limiting outstanding work. Saturated submissions are dropped. | **Merged** — adopted production's thread-pool implementation. |
| `src/agent_claude.py` | Accepts `FIXED` in `CANONICAL_VERDICTS` and `completed` in the acceptable terminal-reason set. | **Merged** — added `"FIXED"` to canonical verdicts and `"completed"` to acceptable terminal reasons. |
| `src/agent_hermes.py` | Entire 244-line Hermes adapter exists only in production. | **Imported from production** — new file, required by `neo_builder.py`. |
| `src/config.py` | Adds separate Neo agent credential (NEO_DEEPSEEK_API_KEY fallback), Hermes executable/provider/model settings, Jev credential, approved local/tailnet Claude-wire bridge endpoints, and GLM through the bridge. Updates `require()` and `validate()`. | **Merged** — added Hermes settings, Jev credential, dual-key fallback, bridge endpoint validation, and GLM model allowance. |
| `src/jev_review.py` | Entire 410-line review-assessment implementation exists only in production, including dependency summaries and chunking. | **Imported from production** — new file, required by `neo_worker.py`. |
| `src/neo_builder.py` | Entire 382-line Builder service exists only in production. | **Imported from production** — new file, required for the Builder lane. |
| `src/neo_protocol.py` | Adds Jev evidence and exact-head/COMMENT explanations. Replaces exhausted-round escalation with BOUNCE/owner-proxy rearming; narrows escalation instructions. | **Merged** — added `jev_context` parameter, exact-head/COMMENT green semantics, owner-proxy rearming, narrower escalation criteria. |
| `src/neo_state.py` | Adds thread-safety documentation, color-aware prior-action lookup with SQL Boolean casting, and marker clearing. | **Merged** — added docstring, `green` parameter to `already_acted()`, `clear()` method. |
| `src/neo_worker.py` | Adds trusted-status/head evidence, Jev assessment, color-aware deduplication, installation/global auto-merge combination, stale-head handling, richer triage records and briefs, and clearing triage on missing verdict. | **Merged** — added `_review_evidence()`, `merge_authority()`, `_jev_for_pr()`, `evidence_brief()`; color-aware dedup; stale-head detection; rich triage detail; triage clearing on agent failure. |

### Additional files

| File | Source | Notes |
|---|---|---|
| `deploy/rhobear-neo-builder.service` | `services.txt` (production snapshot) | Systemd unit for the Builder lane. Secrets stay in `EnvironmentFile`. |

## Known issues (deferred to B3)

### Builder/Hermes outcome-name mismatch (Audit §9)

The Builder lane's Hermes agent returns verdicts (`FIXED`, `NO-CHANGE`, `BLOCKED`)
that do not match the Neo protocol's canonical verdict set. The `neo_builder.py`
`BuilderAgent` subclass treats verdict parsing as advisory (the push/no-push
head-diff is the real signal), but the mismatch means:

- `FIXED` is not in the Hermes `CANONICAL_VERDICTS` set (only `agent_claude.py`
  accepts it).
- The Builder's own `_parse_output` rejects `FIXED` as non-canonical, falling
  through to the `except Exception` handler that logs the warning and returns
  empty usage/verdict.

This will be fixed in B3 by aligning the Builder's canonical verdict set with
the outcomes it actually produces, or by switching to push-detection as the
primary success signal and making the verdict purely informational.

### Other deferred items

- `neo_state.py` does not implement atomic work claims (P1 in audit).
- `neo_builder.py` queries findings by repository substring and six-hour age,
  rather than typed assessment by tenant/repository/PR/full SHA (P1).
