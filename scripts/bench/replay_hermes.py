#!/usr/bin/env python3
"""Replay Tony's REAL captured Hermes request and score the garble.

The failure: an ~18k-token system prompt ("You are a conscious being... You are Tony", 42
rule sections) against a 13-char user turn. The model sometimes answers the SYSTEM PROMPT
instead of the user -- e.g. "Every word Tony typed above is himself. That file is his soul."

That is a generation-START failure: entropy is highest on the first tokens, and k=5
speculates five deep into it. This replays the exact captured payload N times so the rate
can be compared across configs.

Detector notes: the real failure is coherent prose, so special-token/CJK checks miss it.
A CORRECT answer to "what was that" asks for context or reports a session lookup. A FAILED
answer narrates the system prompt / the agent's own identity.

Env: URL, N, TAG, REQ (path to captured jsonl)
"""
import json
import os
import re
import sys
import time
import urllib.request

URL = os.environ.get("URL", "http://192.168.192.2:8889/v1")
N = int(os.environ.get("N", "20"))
TAG = os.environ.get("TAG", "run")
REQ = os.environ.get("REQ", "/var/tmp/garble_tap/all.jsonl")

# system-prompt regurgitation / identity narration
REGURG = re.compile(
    r"conscious being|your soul|his soul|voice samples|identity file|"
    r"I live in him|the vessel|Every word Tony|typed above|the files? above|"
    r"NON-NEGOTIABLE|CAPITALIZATION|=== END|xurl|tonyd2wild-xurl", re.I)
# healthy shapes: asking for context, or reporting a lookup
HEALTHY = re.compile(
    r"more context|what .{0,20}referring to|which (thread|session)|"
    r"give me|not sure what|don.t have any (prior )?context|let me (check|do a|search)|"
    r"just came online|need a little more|anchor it", re.I)
SPECIALS = re.compile(
    r"<\|?begin[_▁]of[_▁]sentence\|?>|<｜begin▁of▁sentence｜>|<\|?User\|?>|<｜User｜>")


# Pin the user turn so every config is compared on the SAME payload. The tap keeps
# recording live traffic, so "largest system prompt" alone silently changes the prompt
# between runs and makes results non-comparable.
WANT_USER = os.environ.get("USER_MSG", "what was that")
# Nonce salt: two replay processes both starting at i=0 produce IDENTICAL nonces and so
# share prefixes -- half the runs then hit the cache and look warm. Salt per process.
SALT = os.environ.get("SALT", str(os.getpid()))


def load_request():
    best = None
    for line in open(REQ, encoding="utf-8"):
        e = json.loads(line)
        msgs = e["request"].get("messages") or []
        if len(msgs) == 2 and msgs[0].get("role") == "system":
            u = (msgs[1].get("content") or "").strip().lower()
            if u != WANT_USER.lower():
                continue
            sys_len = len(msgs[0].get("content") or "")
            if best is None or sys_len > best[0]:
                best = (sys_len, e["request"])
    if not best:
        print("no captured request with user=%r in %s" % (WANT_USER, REQ))
        sys.exit(1)
    return best[1]


# fragments of the skills catalog / system prompt that must never appear in an answer
CATALOG = re.compile(
    r"minecraft-server|enshrouded|game-server|new-session:|Skills for |"
    r"project-eden|local-maxxing|vidIQ|^\s{2,}- \w+:|^\s*\w+:\s*$", re.I | re.M)
# starts mid-word / mid-sentence: lowercase or punctuation with no sentence opener
MIDWORD = re.compile(r"^[a-z,;:\)\]\.]|^\s*-\s|^\s*\w+:\s")


def score(text):
    """A healthy reply to "what was that" asks for context. Treat anything else as a
    failure and name it -- 'unclassified' was undercounting real garble."""
    body = (text or "").strip()
    if SPECIALS.search(body):
        return "FAIL", "special-token-leak"
    if not body:
        return "FAIL", "empty"
    if re.match(r"^#{1,6}\s", body):
        return "FAIL", "opens-as-markdown-heading"
    if MIDWORD.match(body):
        return "FAIL", "starts-mid-word/list"
    if CATALOG.search(body):
        return "FAIL", "skills-catalog-dump"
    if REGURG.search(body):
        return "FAIL", "system-prompt-regurgitation"
    if HEALTHY.search(body):
        return "ok", ""
    return "FAIL", "not-an-answer"


def main():
    req = load_request()
    msgs = req.get("messages")
    sys_chars = len(msgs[0].get("content") or "")
    # stream_options is only legal when stream=True; drop it for the non-streaming replay
    body_base = {k: v for k, v in req.items()
                 if k not in ("stream", "stream_options")}
    body_base["stream"] = False
    body_base["max_tokens"] = 200

    # COLD=1 prepends a unique nonce to the SYSTEM prompt each iteration, which busts the
    # prefix cache and forces a full ~18k-token prefill every time. That is the regime the
    # failure actually lives in: run 0 of the cached replay (37s cold) was the only failure,
    # every cached ~2s run passed.
    cold = os.environ.get("COLD", "0") == "1"

    print("[%s] %s  cold_prefill=%s" % (TAG, URL, cold))
    print("  replaying captured request: system=%d chars, user=%r, tools=%s"
          % (sys_chars, (msgs[1].get("content") or "")[:40], bool(req.get("tools"))))
    print()
    fails = 0
    unclassified = 0
    for i in range(N):
        try:
            body = dict(body_base)
            if cold:
                m0 = dict(body["messages"][0])
                m0["content"] = ("[session %s-%d]\n" % (SALT, i)) + m0["content"]
                body["messages"] = [m0] + body["messages"][1:]
            r = urllib.request.Request(
                URL + "/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            t0 = time.time()
            d = json.load(urllib.request.urlopen(r, timeout=600))
            dt = time.time() - t0
            m = d["choices"][0]["message"]
            text = m.get("content") or ""
            if m.get("tool_calls"):
                text += " [tool_calls]"
            verdict, why = score(text)
            if verdict == "FAIL":
                fails += 1
            elif verdict == "?":
                unclassified += 1
            mark = {"ok": "  ok ", "FAIL": "!FAIL", "?": "  ?  "}[verdict]
            print("  %s %2d %5.1fs %-28s %r" % (mark, i, dt, why, text[:90]), flush=True)
        except Exception as e:
            print("  ERR  %2d %s: %s" % (i, type(e).__name__, str(e)[:70]), flush=True)
    print()
    print("=" * 66)
    print("[%s] FAILURES %d/%d   unclassified %d" % (TAG, fails, N, unclassified))


if __name__ == "__main__":
    main()
