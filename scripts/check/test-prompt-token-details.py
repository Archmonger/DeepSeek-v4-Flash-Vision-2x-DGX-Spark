#!/usr/bin/env python3
"""CPU-only regression for the serve argv of the **supported** launchers.

Supported serve paths are `scripts/launch/ds4-vision-tp2.sh` and
`scripts/launch/ds4-vision-tp4.sh`. Nothing here parses a Compose file: the old
two-node Compose lane was removed because it could not load the current Vision-Exp
checkpoint (no vision-port bind mounts, no `--hf-overrides` multimodal alias, no
Patch 6 mount), so it is no longer a serve path this repo stands behind.

Two things are enforced here:

1. `--enable-prompt-tokens-details` on every launcher, so `cached_tokens` is
   reportable on all supported paths (upstream PR #56 only touched the Compose
   lane, which is why the shell launchers originally drifted — see
   docs/CACHE-REPORTING.md).
2. The `@Wpnx330` capture-size rule: under speculative decoding the CUDA-graph
   capture buckets are multiples of `1 + k`, so the capture size must be a
   multiple of `1 + k` **and** at least `--max-num-seqs`. A value that floors
   below the concurrency silently drops the excess requests off the captured path
   into eager/piecewise. See docs/LAUNCH-FLAGS.md and docs/PATCHES.md (Patch A).
"""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# launcher -> --max-num-seqs it is expected to run
LAUNCHERS = {
    "scripts/launch/ds4-vision-tp2.sh": 12,
    "scripts/launch/ds4-vision-tp4.sh": 64,
}


def flag(source: str, name: str) -> str | None:
    """First value passed to `--name` on a serve line."""
    match = re.search(rf"--{re.escape(name)}[= ](\S+)", source)
    return match.group(1).rstrip("\\") if match else None


class PromptTokenDetailsTests(unittest.TestCase):
    def test_every_launcher_reports_cached_tokens(self):
        for rel in LAUNCHERS:
            with self.subTest(launcher=rel):
                source = (ROOT / rel).read_text()
                self.assertEqual(source.count("--enable-prefix-caching"), 1)
                self.assertEqual(source.count("--enable-prompt-tokens-details"), 1)
                # guard against the two ways this flag is commonly mistyped
                self.assertEqual(source.count("--enable-prompt-token-details"), 0)
                self.assertEqual(source.count("--no-enable-prompt-tokens-details"), 0)

    def test_capture_size_rule(self):
        for rel, max_num_seqs in LAUNCHERS.items():
            with self.subTest(launcher=rel):
                source = (ROOT / rel).read_text()
                spec = json.loads(re.search(r"SPEC='(.+?)'", source).group(1))
                k = spec["num_speculative_tokens"]
                bucket = 1 + k

                seqs = int(flag(source, "max-num-seqs"))
                cap = int(flag(source, "max-cudagraph-capture-size"))

                self.assertEqual(seqs, max_num_seqs, f"{rel}: --max-num-seqs drifted")
                self.assertEqual(
                    cap % bucket,
                    0,
                    f"{rel}: --max-cudagraph-capture-size {cap} is not a multiple of "
                    f"1+k ({bucket}); capture buckets round to multiples of {bucket}",
                )
                self.assertGreaterEqual(
                    cap,
                    seqs,
                    f"{rel}: capture size {cap} floors below max-num-seqs {seqs}; "
                    f"requests above it fall off the captured CUDA-graph path",
                )

    def test_spec_is_dspark_probabilistic(self):
        for rel in LAUNCHERS:
            with self.subTest(launcher=rel):
                source = (ROOT / rel).read_text()
                spec = json.loads(re.search(r"SPEC='(.+?)'", source).group(1))
                self.assertEqual(spec["method"], "dspark")
                self.assertEqual(spec["draft_sample_method"], "probabilistic")


if __name__ == "__main__":
    unittest.main()
