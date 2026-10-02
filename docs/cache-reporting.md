# Per-request cache reporting

Every serve path in this repo passes `--enable-prompt-tokens-details` —
`docker-compose.dspark.yml` and both `launchers/ds4-vision-tp{2,4}.sh` — so
OpenAI-compatible clients can read `usage.prompt_tokens_details.cached_tokens`
in Chat Completions responses. For streaming requests, send
`"stream_options": {"include_usage": true}` and read the final usage chunk.
A cold request may report zero; enabling reporting does not guarantee a hit.

This is response metadata only: it does not enable or retune prefix caching,
change cache retention, or change the KV pool. Prometheus cache metrics at
`/metrics` are separate and do not depend on this flag.

Existing servers only pick up this change when the operator next recreates or
restarts them. The accompanying CPU-only regression
(`scripts/test-prompt-token-details.py`) checks the launch arguments on all three
serve paths; it does not start a server or prove live cache reuse.

Related: [CURRENT.md](../CURRENT.md) lists what each serve path passes, and
[`PATCHES.md`](PATCHES.md) covers the KV-cache work behind the cached-token counts.
