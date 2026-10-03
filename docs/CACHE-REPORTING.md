# Per-request cache reporting

Every supported serving path passes `--enable-prompt-tokens-details` — both sparkrun recipes
(`sparkrun/ds4-vision-exp-tp2.yaml`, `sparkrun/ds4-vision-exp-tp4.yaml`) and both legacy launchers
(`scripts/launch/ds4-vision-tp2.sh`, `scripts/launch/ds4-vision-tp4.sh`) — so
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

This was a drift for a while: the flag landed on the launchers first and the recipes lagged. Both
sides are covered now, and `scripts/check/test-prompt-token-details.py` fails CI if either path
loses the flag — the same class of drift upstream PR #56 introduced cannot silently return.

Related: [CURRENT.md](../CURRENT.md) lists what each serve path passes, and
[`PATCHES.md`](PATCHES.md) covers the KV-cache work behind the cached-token counts.
