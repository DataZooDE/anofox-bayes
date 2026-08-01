# anofox_bayes Telemetry

`anofox_bayes` collects **anonymous, privacy-preserving usage telemetry** so we
can see which model families and diagnostics are used, on which platforms, and
where they fail — and prioritise accordingly. It is **on by default** and
**trivial to turn off**.

Telemetry is emitted through the shared
[`DataZooDE/posthog-telemetry`](https://github.com/DataZooDE/posthog-telemetry)
library and follows the cross-product **`telemetry_schema: 2`** envelope
(`posthog-telemetry/TELEMETRY-SCHEMA.md`). Ingestion is the EU PostHog cloud.

## How to turn it off

Any one of these fully short-circuits telemetry — when disabled, **nothing leaves
the machine** (the opt-out is enforced at the transport, not just at the call
sites):

```sql
SET anofox_telemetry_enabled = false;   -- DuckDB setting (per session)
```

```bash
export DATAZOO_DISABLE_TELEMETRY=1       # environment (1|true|yes)
```

Telemetry is also auto-disabled when a CI environment is detected (`CI`,
`GITHUB_ACTIONS`, `GITLAB_CI`, and similar).

If you build from source, telemetry can be compiled out entirely: the
`ANOFOX_TELEMETRY_ENABLED` compile definition guards every call site, and when it
is off the shared library's header stubs (`POSTHOG_TELEMETRY_DISABLED`) turn each
call into an inline no-op. This is already the case on MinGW builds, where the
OpenSSL dependency is unavailable — those binaries contain no telemetry code at
all.

You can also redirect telemetry to a PostHog project you control:

```sql
SET anofox_telemetry_key = 'phc_your_own_project_key';
```

## The guarantee: bounded, enumerated, non-PII

Every property we send is **either** a constant drawn from a small,
code-controlled enumeration **or** a pure number (durations, counts). The library
additionally clamps every outgoing string to 512 bytes as a backstop.

This matters more here than in a general-purpose extension, because a Bayesian fit
is called with a config that names business columns and a relation that contains
customer data. Neither is ever read by the instrumentation.

We **never** send: table names, column names, group keys, the contents of the
`config` struct (family slots, priors, seeds, draw counts), the data being fitted,
posterior draws, `model_id` values, data fingerprints, `WHERE`/`FILTER` clauses,
SQL text, row or result data, or error messages. Only the fixed strings and
numbers described below leave the machine.

The instrumentation is centralised in the extension entry point
(`src/anofox_bayes_extension.cpp`) plus two `RecordFunctionCall` sites, and the
shared telemetry library header (`posthog-telemetry/include/telemetry.hpp`).

## What is collected

### Envelope (attached to every event)

`product` (`anofox_bayes`), `product_version`, `product_edition` (`commercial` —
`anofox_bayes` is BSL-licensed, not an OSS community extension),
`telemetry_schema` (`2`), `duckdb_version`, `os`, `arch`, `platform`, `is_ci`,
`is_container`, a per-process `$session_id`, and — once associated — the
`deployment` group. `distinct_id` is the SHA-256 of a machine id: a **stable,
pseudonymous** identifier, not tied to any personal data.

### Events

| Event | When | Properties (beyond the envelope) |
|---|---|---|
| `extension_loaded` | the `anofox_bayes` extension loads | — |
| `function_executed` | an instrumented function runs — **aggregated** per function per session (not per row, not per draw) | `function_name`, `call_count`, `duration_ms_p50` |

`extension_loaded` fires once, at extension load, from `LoadInternal`.

### Instrumented function names

The `function_name` property is drawn from this fixed set and nothing else:

| `function_name` | Recorded at |
|---|---|
| `anofox_bayes_fit` | bind time, once per `anofox_bayes_fit(...)` call in a query |
| `diagnostics_aggregates` | aggregate registration, i.e. once at extension load |

Note what is *not* in that list. The family name (`conjugate_anomaly`,
`pooled_gaussian`), the engine, the number of draws and the fit status are **not**
sent, even though they are enumerated constants and would be useful to us — they
are recorded nowhere in the envelope, because the smallest surface that answers
"is this extension being used at all" is the right one to start from.

## Function-call aggregation

`RecordFunctionCall(function_name)` aggregates in-process into a single
`function_executed` event per function per session (carrying `call_count` and
`duration_ms_p50`), flushed at session end. Instrumentation is placed at bind /
register time, never on a per-row `GetChunk` path, so a fit that emits four
million draw rows produces O(1) telemetry rows, not a firehose.

## Enterprise / account analytics

`anofox_bayes` associates only the `deployment` group. It has no license key, so
no `account` group is associated.
