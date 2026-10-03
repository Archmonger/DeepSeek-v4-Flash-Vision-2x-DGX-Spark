# Per-request cache reporting

Both supported launcher paths pass `--enable-prompt-tokens-details` —
`scripts/launch/ds4-vision-tp2.sh` and `scripts/launch/ds4-vision-tp4.sh` — so
OpenAI-compatible clients can read `usage.prompt_tokens_details.cached_tokens`
in Chat Completions responses. For streaming requests, send
`"stream_options": {"include_usage": true}` and read the final usage chunk.
A cold request may report zero; enabling reporting does not guarantee a hit.

This is response metadata only: it does not enable or retune prefix caching,
change cache retention, or change the KV pool. Prometheus cache metrics at
`/metrics` are separate and do not depend on this flag.

Existing servers only pick up this change when the operator next recreates or
restarts them. The accompanying CPU-only regression
(`scripts/check/test-prompt-token-details.py`) checks the launch arguments on both
launcher paths; it does not start a server or prove live cache reuse.

**Known gap:** the [sparkrun](../sparkrun/README.md) recipe does **not** currently pass this
flag, so `cached_tokens` is not reportable on that path. It is a serve-path drift of exactly
the kind upstream PR #56 introduced; adding the flag there is a recipe change, so it has been
left for an explicit decision.

Related: [CURRENT.md](../CURRENT.md) lists what each serve path passes, and
[`PATCHES.md`](PATCHES.md) covers the KV-cache work behind the cached-token counts.
