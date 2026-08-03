"""Aggregator: depends on every demo so one `uv sync` installs all seven scripts.

Deliberately empty of code. The shared shell lives in `anofox_bayes_demo`
(singular) under `lib/`, and each demo is its own workspace member under
`agents/`; this package exists only so that `uv sync` at the workspace root pulls
all of them into one environment.
"""
