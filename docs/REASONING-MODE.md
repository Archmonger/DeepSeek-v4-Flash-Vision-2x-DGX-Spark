# Reasoning / thinking mode

How thinking mode behaves on this serving stack: the response field names, the prompt
mechanics, how to enable it, and what `reasoning_effort` actually does here. All of
it is measured on this runtime.

## Default state

Reasoning is **off by default** in this recipe:

```
--default-chat-template-kwargs '{"thinking":false}'
```

## The response field is `reasoning`, not `reasoning_content`

| Shape | Field |
| --- | --- |
| Non-streaming | `choices[0].message.reasoning` |
| Streaming | `choices[0].delta.reasoning` |

There is **no** `reasoning_content` key in a response on this runtime — it is
deprecated and only accepted on *input*. Clients reading `reasoning_content` see an
empty value and conclude reasoning extraction is broken. In a client that renders a
reasoning panel, this shows up as sitting on "Thinking…" until the whole response
lands. A small translating proxy in front of the server is enough for such clients.

Credit @vinicius-symetrix (PR #13) for independently reporting the streaming half
of this.

## `<|im_start|>` is written into the prompt, not generated

The prompt tail carries the tag; the completion never starts with it:

```
thinking off →  ...<｜Assistant｜><|im_end|>
thinking on  →  ...<｜Assistant｜><|im_start|>
```

So in thinking mode the model's output *starts* with reasoning text and ends with
`<|im_end|>`; there is no opening tag in the completion. **A missing `<|im_start|>` in
the output is correct behaviour.**

If you see `<|im_end|>` inside `content`, the server is missing
`--reasoning-parser deepseek_v4` and `--reasoning-config`.

## Enabling it

- Per request: `chat_template_kwargs: {"thinking": true}`
- Per request: a top-level `reasoning_effort` of `low`/`high`/`max`
- Server-wide: `--default-chat-template-kwargs '{"thinking":true}'`

## Never combine `reasoning_effort: "none"` with `thinking: true`

`"none"` forces chat-mode formatting while the reasoning parser stays armed for
thinking; with no `<|im_end|>` in the output, the parser puts the *entire* response
into `reasoning` and returns `content: null`. The failure is reproducible 4/4:
real `completion_tokens`, `finish_reason: stop`, empty `content`.

`reasoning_effort` on its own is fine.

## `reasoning_effort` on this stack

`reasoning_effort: "low"`, `"medium"` and `"high"` all produce **no** effort
prefix. They normalise to an internal `"high"` that has no injection branch, so
nothing is added to the prompt.

Only `"max"`/`"xhigh"` inject — and what they inject is the model's **high** text
(79 tokens), not its `max` text (96). **The model's `max` effort is unreachable in
this tokenizer mode**, and selecting `"high"` here means running at the vendor's
*low*.

Verify in three requests; `prompt_tokens` is the whole witness:

```bash
for E in low high max; do
  printf '%-5s ' "$E"
  curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"deepseek-v4-flash-dspark\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],
         \"max_tokens\":1,\"reasoning_effort\":\"$E\"}" \
  | python3 -c 'import json,sys; print("prompt_tokens =", json.load(sys.stdin)["usage"]["prompt_tokens"])'
done
# this stack:  low 5 | high 5 (identical — no prefix) | max 84 (+79)
# api.deepseek.com, same messages: low 681 | high 760 (+79) | max 773
```

Reported by @Capicua25x (issue #25), whose measurements establish that the
runtime's internal `"high"` is not the vendor's `"high"`.

PR #24 vendors the checkpoint's own three-level effort table and restores the
distinction. **Note the behaviour change that brings:** with that table in place,
`reasoning_effort: "max"` injects the model's real `max` text instead of its high
text, so existing callers of `"max"` get a materially stronger instruction.
`"low"` and omitted stay byte-identical.

## `chat_template.jinja` is ignored under `--tokenizer-mode deepseek_v4`

With `--tokenizer-mode deepseek_v4`, a `chat_template.jinja` in the model directory
is ignored — prompt formatting comes from the checkpoint's built-in encoder, not a
Jinja template. This is why the HuggingFace discussion #26 workaround has no effect
here.

It is stronger than that: **explicitly passing
`--chat-template /path/to/chat_template.jinja` is also ignored** — the flag shows up
in the engine's `non-default args` and changes nothing (issue #25).

There is no way to reach the template's three-way effort split without leaving
`tokenizer_mode=deepseek_v4`, which this recipe requires for DSpark.

## Thinking mode needs output budget

Reasoning consumes `max_tokens` before any content is produced, so a small cap
yields `finish_reason: length` with empty `content`. Benchmarking thinking mode at a
low cap measures truncation, not the model.

## What thinking buys (measured)

Execution-graded harness: 20 frozen LiveCodeBench-style problems, 3 public + up to 40
private tests each (a problem counts only when every private test passes), plus a
procedural seed-generated suite of 48 cases.

| | procedural suite | LCB, one-shot | LCB, failures retried at a 32k budget |
|---|---|---|---|
| thinking off (recipe default) | 0.875 | 12/20 | 13/20 |
| **thinking on, effort high** | **0.979** | 11/20 | **20/20** |

The one-shot number goes *down* with reasoning enabled: at an 8k cap the model spends
the budget thinking and gets truncated. Every one of the nine failures was
`finish_reason: length`, and all nine passed at 32k. Report the thinking setting and
retry length-capped failures, or the result measures the cap rather than the model.
