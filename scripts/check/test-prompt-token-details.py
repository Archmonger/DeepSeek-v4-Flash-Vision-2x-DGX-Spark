#!/usr/bin/env python3
"""CPU-only regression over every supported serve path.

Two jobs:

1. Guard the serve-argv rules on the launchers and the sparkrun recipes.
   * `--enable-prompt-tokens-details` present exactly once, never mistyped, so
     `usage.prompt_tokens_details.cached_tokens` is reportable on every path
     (docs/CACHE-REPORTING.md).
   * `--max-cudagraph-capture-size` is a multiple of `1+k` and >= `--max-num-seqs`.
     Under spec decode the CUDA-graph capture buckets are multiples of `1+k`, so a value
     that floors below the concurrency drops the excess requests off the captured path into
     eager/piecewise **silently** (@Wpnx330, docs/LAUNCH-FLAGS.md, Patch A in PATCHES.md).
   * The spec stays DSpark + probabilistic, and thinking stays on by default.

2. Keep `sparkrun/*.yaml` 1:1 with its launcher. The launcher is the reference; the recipe
   must reach it with no missing flags, no extra flags, and equal values. The only launcher-only
   flags allowed are the cluster flags sparkrun's vllm-distributed runtime appends itself.
   Full list of deliberate deviations: docs/SPARKRUN-PARITY.md.

Nothing here starts a server. It parses shell and YAML, so it runs in CI in milliseconds.
"""
import json
import re
import shlex
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# (label, launcher, recipe, expected --max-num-seqs)
PAIRS = (
    ("TP2", "scripts/launch/ds4-vision-tp2.sh", "sparkrun/ds4-vision-exp-tp2.yaml", 12),
    ("TP4", "scripts/launch/ds4-vision-tp4.sh", "sparkrun/ds4-vision-exp-tp4.yaml", 64),
)

# Flags sparkrun's vllm-distributed runtime appends; a recipe must not hardcode them.
SPARKRUN_MANAGED_FLAGS = {
    "--nnodes",
    "--node-rank",
    "--master-addr",
    "--master-port",
    "--headless",
}

# Host/transport knobs sparkrun detects per host, or offline-serving knobs that only make sense
# for the launcher's pre-staged weights. docs/SPARKRUN-PARITY.md sections 3 and 5.
SPARKRUN_MANAGED_ENV = {
    "NCCL_IB_HCA",
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "NCCL_IB_GID_INDEX",
    "VLLM_HOST_IP",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
}

# Same role, deliberately different root: launcher /vllm-cache vs recipe /cache/runtime.
# Checked by leaf name rather than full value.
PATH_LEVEL_ENV = {
    "VLLM_CACHE_ROOT",
    "DG_JIT_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "TILELANG_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
}

FLAG_RE = re.compile(r"--([a-z][a-z0-9-]*)")


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


def launcher_serve_argv(rel: str) -> list:
    """The argv the container actually runs, from the launcher's `-lc "..."` block."""
    source = read(rel)
    block = re.search(r'-lc\s*"((?:[^"\\]|\\.)*)"', source, re.S)
    assert block, f"{rel}: no `-lc \"...\"` serve block found"
    inner = block.group(1).replace('\\"', '"')

    # Resolve the shell variables the serve line interpolates.
    for name in ("SPEC", "REASON"):
        m = re.search(rf"^{name}='(.*?)'$", source, re.M)
        if m:
            inner = inner.replace(f"${name}", m.group(1))
    for name in ("PORT", "MODEL_IN_CONTAINER"):
        m = re.search(rf'^{name}="([^"]*)"', source, re.M)
        if m:
            inner = inner.replace(f"${name}", m.group(1))
    inner = inner.replace("$HEADLESS", "--headless")

    serve = inner.split("vllm serve", 1)[1]
    return shlex.split(serve, comments=True)


def launcher_env(rel: str) -> dict:
    """`-e KEY=VALUE` pairs from the docker run, with `${VAR:-default}` resolved."""
    env = {}
    for key, raw in re.findall(r'-e\s+([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|\'[^\']*\'|\S+)', read(rel)):
        value = raw.strip("\"'")
        value = re.sub(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-([^}]*)\}", r"\1", value)
        env[key] = value
    return env


def recipe(rel: str) -> dict:
    return yaml.safe_load(read(rel))


def recipe_serve_argv(rel: str) -> list:
    """The argv sparkrun renders: every {placeholder} resolved from `defaults`."""
    doc = recipe(rel)
    defaults = dict(doc.get("defaults", {}))
    defaults.setdefault("model", doc.get("model", "{model}"))

    def render(text: str) -> str:
        for _ in range(4):  # values may themselves reference {keys}
            rendered = re.sub(
                r"\{(\w+)\}",
                lambda m: str(defaults.get(m.group(1), m.group(0))),
                text,
            )
            if rendered == text:
                break
            text = rendered
        assert "{" not in re.sub(r"'[^']*'", "", rendered), (
            f"{rel}: unresolved placeholder in command: {rendered!r}"
        )
        return rendered

    serve = render(doc["command"]).split("vllm serve", 1)[1]
    return shlex.split(serve, comments=True)


def strip_noise(argv: list) -> list:
    """Drop whitespace-only tokens leaked by shell/YAML line continuations.

    A bare `\\n` token does not start with `-`, so `to_flags` would happily read it as
    the value of the boolean flag before it and report `--enable-flashinfer-autotune
    ['\n']`. Filter here so every consumer sees a clean argv.
    """
    return [tok for tok in argv if tok.strip()]


def to_flags(argv: list) -> dict:
    """argv -> {flag: [values...]}; value None marks a presence-only boolean flag."""
    argv = strip_noise(argv)
    flags: dict = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            i += 1
            continue
        if "=" in token:
            name, value = token.split("=", 1)
            flags.setdefault("--" + name, []).append(value)
            i += 1
            continue
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if nxt is None or nxt.startswith("-"):
            flags.setdefault(token, []).append(None)
            i += 1
        else:
            flags.setdefault(token, []).append(nxt)
            i += 2
    return flags


def spec_of(text: str, pattern: str) -> dict:
    return json.loads(re.search(pattern, text, re.S).group(1))


def launcher_spec(rel: str) -> dict:
    return spec_of(read(rel), r"SPEC='(\{.*?\})'")


def recipe_spec(rel: str) -> dict:
    return spec_of(read(rel), r"speculative_config: '(\{.*?\})'")


def nums(flags: dict, name: str):
    values = flags.get(name, [])
    assert len(values) == 1, f"{name}: expected exactly one, got {values}"
    return int(values[0])


class ServePathTests(unittest.TestCase):
    """Rules that must hold on every supported serve path."""

    def test_every_path_reports_cached_tokens(self):
        for label, launcher, recipe_rel, _ in PAIRS:
            for kind, rel in (("launcher", launcher), ("recipe", recipe_rel)):
                argv = (launcher_serve_argv(launcher) if kind == "launcher"
                        else recipe_serve_argv(recipe_rel))
                flags = to_flags(argv)
                for flag in ("--enable-prefix-caching", "--enable-prompt-tokens-details"):
                    with self.subTest(path=f"{label}/{kind}", flag=flag):
                        self.assertEqual(len(flags.get(flag, [])), 1,
                                        f"{label}/{kind}: {flag} must appear exactly once "
                                        f"in the serve argv, got {flags.get(flag)}")
                # The two ways this flag gets mistyped. Raw text, because a typo never shows up
                # under the correct name.
                with self.subTest(path=f"{label}/{kind}", guard="typo"):
                    self.assertEqual(read(rel).count("--enable-prompt-token-details"), 0)
                    self.assertEqual(read(rel).count("--no-enable-prompt-tokens-details"), 0)

    def test_capture_size_rule(self):
        for label, launcher, recipe_rel, expected_seqs in PAIRS:
            for kind, rel, spec in (
                ("launcher", launcher, launcher_spec(launcher)),
                ("recipe", recipe_rel, recipe_spec(recipe_rel)),
            ):
                with self.subTest(path=f"{label}/{kind}"):
                    argv = (launcher_serve_argv(launcher) if kind == "launcher"
                           else recipe_serve_argv(recipe_rel))
                    flags = to_flags(argv)
                    k = spec["num_speculative_tokens"]
                    bucket = 1 + k
                    seqs = nums(flags, "--max-num-seqs")
                    cap = nums(flags, "--max-cudagraph-capture-size")

                    self.assertEqual(seqs, expected_seqs,
                                     f"{label}/{kind}: --max-num-seqs drifted")
                    self.assertEqual(cap % bucket, 0,
                                     f"{label}/{kind}: capture size {cap} is not a multiple "
                                     f"of 1+k ({bucket}); capture buckets round to {bucket}")
                    self.assertGreaterEqual(cap, seqs,
                                           f"{label}/{kind}: capture size {cap} floors below "
                                           f"--max-num-seqs {seqs}; requests above it fall "
                                           f"off the captured CUDA-graph path")

    def test_spec_is_dspark_probabilistic(self):
        for label, launcher, recipe_rel, _ in PAIRS:
            for kind, spec in (("launcher", launcher_spec(launcher)),
                              ("recipe", recipe_spec(recipe_rel))):
                with self.subTest(path=f"{label}/{kind}"):
                    self.assertEqual(spec["method"], "dspark")
                    self.assertEqual(spec["draft_sample_method"], "probabilistic")

    def test_thinking_enabled_by_default(self):
        for label, launcher, recipe_rel, _ in PAIRS:
            for kind, rel in (("launcher", launcher), ("recipe", recipe_rel)):
                with self.subTest(path=f"{label}/{kind}"):
                    flags = to_flags(launcher_serve_argv(launcher) if kind == "launcher"
                                    else recipe_serve_argv(recipe_rel))
                    value = flags["--default-chat-template-kwargs"][0]
                    self.assertEqual(json.loads(value), {"thinking": True})

    def test_no_repetition_penalty_override(self):
        """--override-generation-config on the DSpark path is an illegal-memory-access crash.

        Checked against the resolved argv, not the file text, so a comment warning about the
        flag does not trip the guard.
        """
        for label, launcher, recipe_rel, _ in PAIRS:
            for kind, argv in (("launcher", launcher_serve_argv(launcher)),
                              ("recipe", recipe_serve_argv(recipe_rel))):
                with self.subTest(path=f"{label}/{kind}"):
                    flags = to_flags(argv)
                    self.assertNotIn("--override-generation-config", flags)
                    joined = " ".join(v for vals in flags.values() for v in vals if v)
                    self.assertNotIn("repetition_penalty", joined)


class LauncherRecipeParityTests(unittest.TestCase):
    """sparkrun recipes must equal their launcher."""

    def test_no_launcher_flag_missing_from_recipe(self):
        for label, launcher, recipe_rel, _ in PAIRS:
            with self.subTest(label):
                lflags = to_flags(launcher_serve_argv(launcher))
                rflags = to_flags(recipe_serve_argv(recipe_rel))
                missing = set(lflags) - set(rflags) - SPARKRUN_MANAGED_FLAGS
                self.assertFalse(missing,
                               f"{label}: launcher passes {sorted(missing)} but the recipe "
                               f"does not (docs/SPARKRUN-PARITY.md §2)")

    def test_no_recipe_only_flags(self):
        for label, launcher, recipe_rel, _ in PAIRS:
            with self.subTest(label):
                lflags = to_flags(launcher_serve_argv(launcher))
                rflags = to_flags(recipe_serve_argv(recipe_rel))
                extra = set(rflags) - set(lflags)
                self.assertFalse(extra,
                               f"{label}: recipe adds {sorted(extra)} that the launcher does "
                               f"not pass — fix the launcher or document the deviation")

    def test_shared_flag_values_match(self):
        for label, launcher, recipe_rel, _ in PAIRS:
            lflags = to_flags(launcher_serve_argv(launcher))
            rflags = to_flags(recipe_serve_argv(recipe_rel))
            for flag in sorted(set(lflags) & set(rflags)):
                with self.subTest(label=label, flag=flag):
                    self.assertEqual(lflags[flag], rflags[flag],
                                     f"{label}: {flag} differs — launcher {lflags[flag]} "
                                     f"vs recipe {rflags[flag]}")

    def test_environment_parity(self):
        for label, launcher, recipe_rel, _ in PAIRS:
            lenv = launcher_env(launcher)
            renv = recipe(recipe_rel).get("env", {})
            with self.subTest(label=label):
                self.assertTrue(lenv, f"{label}: no `-e` pairs parsed from the launcher")

            for key, lvalue in sorted(lenv.items()):
                if key in SPARKRUN_MANAGED_ENV:
                    continue
                with self.subTest(label=label, var=key):
                    self.assertIn(key, renv,
                                  f"{label}: launcher sets {key} but the recipe does not")
                    if key in PATH_LEVEL_ENV:
                        rvalue = str(renv[key])
                        self.assertTrue(rvalue.startswith("/"),
                                      f"{label}: {key} must be an absolute container path")
                        self.assertEqual(rvalue.rstrip("/").rsplit("/", 1)[-1],
                                        lvalue.rstrip("/").rsplit("/", 1)[-1],
                                        f"{label}: {key} leaf differs from the launcher "
                                        f"(path-level parity, different root)")
                    else:
                        self.assertEqual(str(renv[key]), lvalue,
                                        f"{label}: {key} differs — launcher {lvalue!r} vs "
                                        f"recipe {str(renv[key])!r}")

            extra = set(renv) - set(lenv)
            self.assertFalse(extra,
                            f"{label}: recipe env adds {sorted(extra)} the launcher does not set")

    def test_recipes_pin_the_same_payload(self):
        pins = {rel: recipe(rel)["defaults"]["ds4_src_commit"] for _, _, rel, _ in PAIRS}
        self.assertEqual(len(set(pins.values())), 1,
                        f"recipes pin different payload commits: {pins}")


if __name__ == "__main__":
    unittest.main()
