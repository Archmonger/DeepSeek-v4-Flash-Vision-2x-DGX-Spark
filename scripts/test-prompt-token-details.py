#!/usr/bin/env python3
"""CPU-only regression for every serve argv in this repo (Compose + shell launchers)."""
import shlex
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PromptTokenDetailsTests(unittest.TestCase):
    def test_compose_serve_args(self):
        source = (ROOT / "docker-compose.dspark.yml").read_text()
        marker = "exec /opt/env/bin/vllm serve "
        self.assertEqual(source.count(marker), 1)
        argv = shlex.split(source.split(marker, 1)[1], comments=True)
        self.assertEqual(argv.count("--enable-prompt-tokens-details"), 1)
        self.assertIn("--enable-prefix-caching", argv)
        self.assertNotIn("--enable-prompt-token-details", argv)
        self.assertNotIn("--no-enable-prompt-tokens-details", argv)
        self.assertIn("${NODE_RANK}", argv)
        self.assertIn("${HEADLESS:+--headless}", argv)

    # Upstream PR #56 only touched docker-compose.dspark.yml, which left the shell
    # launchers reporting no cached_tokens. These cover the other serve paths so the
    # two cannot drift apart again.
    LAUNCHERS = (
        "launchers/ds4-vision-tp2.sh",
        "launchers/ds4-vision-tp4.sh",
    )

    def test_launcher_serve_args(self):
        for rel in self.LAUNCHERS:
            with self.subTest(launcher=rel):
                source = (ROOT / rel).read_text()
                self.assertEqual(source.count("--enable-prefix-caching"), 1)
                self.assertEqual(source.count("--enable-prompt-tokens-details"), 1)
                self.assertEqual(source.count("--enable-prompt-token-details"), 0)
                self.assertEqual(source.count("--no-enable-prompt-tokens-details"), 0)


if __name__ == "__main__":
    unittest.main()
