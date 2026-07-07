#!/usr/bin/env python3
"""adapters.py — workload adapters: profile ANY GPU codebase, not just cuVSLAM.

The harness (profiling/harness/profile.py, profiling/regime.py) needs three
things from a workload: (1) an argv to launch it, (2) the env it needs, and
(3) an optional quality-of-result extractor so the regime can prove the
profilers don't perturb the computation. An Adapter provides exactly that.

Built-in adapters:

  cuvslam   the stack's own TOML runner (`run.py <config>`) — the default
            whenever the config has no [workload] table. QoR = the [eval]
            report (ATE/RPE vs ground truth).

  command   ANY command line, described in the config itself:

                # configs/workloads/my_dnn.toml
                [workload]
                cmd = "python3 train.py --data /mnt/data/imagenet --iters 100"
                cwd = "/home/me/my_dnn_repo"        # optional
                [workload.env]                       # optional
                CUDA_VISIBLE_DEVICES = "0"
                [qor]                                # optional
                stdout_regex = "final_loss=([0-9.eE+-]+)"   # captured as the QoR
                tolerance = 0.02                            # relative
                [profiling]
                nvbit = true                         # deep-trace marker

            That covers torch/DNN training or inference scripts, cuBLAS/CUDA
            benchmarks, image-processing pipelines — anything launchable from
            a shell. Annotate stages with NVTX ranges in the codebase and the
            nsys DAG/stage analyses pick them up exactly like cuVSLAM's.

Selection is automatic: a `[workload]` table in the config selects `command`;
otherwise `cuvslam`. Force one with --adapter.
"""
from __future__ import annotations

import os
import re
import shlex


class Adapter:
    """One workload family the harness can launch and (optionally) score."""

    name = "abstract"

    def __init__(self, config_path: str, config_text: str, repo_root: str,
                 venv_python: str):
        self.config_path = config_path
        self.text = config_text
        self.repo_root = repo_root
        self.venv_python = venv_python

    def argv(self) -> list:
        raise NotImplementedError

    def env(self) -> dict:
        return {}

    def cwd(self) -> str:
        return self.repo_root

    def qor(self) -> dict | None:
        """Post-run quality-of-result, e.g. {'ape_m': 0.0196}. None = no QoR."""
        return None

    def describe(self) -> dict:
        return {"adapter": self.name,
                "workload_cmd": " ".join(self.argv()),
                "cwd": self.cwd()}


class CuvslamAdapter(Adapter):
    """The stack's TOML runner — historical default, byte-compatible."""

    name = "cuvslam"

    def argv(self):
        return [self.venv_python, os.path.join(self.repo_root, "run.py"),
                self.config_path]

    def qor(self):
        m = re.search(r'report\s*=\s*"([^"]+)"', self.text)
        if not m or not os.path.isfile(m.group(1)):
            return None
        body = open(m.group(1)).read()
        ape = re.search(r"RMSE APE[^(]*\(([0-9.]+) m\)", body)
        matched = re.search(r"matched poses[^0-9]*([0-9]+)", body)
        out = {}
        if ape:
            out["ape_m"] = float(ape.group(1))
        if matched:
            out["matched_poses"] = int(matched.group(1))
        return out or None


class CommandAdapter(Adapter):
    """Any command line, straight from the config's [workload] table."""

    name = "command"

    def _field(self, key, table="workload"):
        m = re.search(rf'(?ms)^\[{table}\].*?^{key}\s*=\s*"([^"]+)"', self.text)
        return m.group(1) if m else None

    def argv(self):
        cmd = self._field("cmd")
        if not cmd:
            raise ValueError(f"{self.config_path}: [workload] table needs cmd = \"...\"")
        return shlex.split(cmd)

    def cwd(self):
        return self._field("cwd") or self.repo_root

    def env(self):
        m = re.search(r"(?ms)^\[workload\.env\]\s*$(.*?)(?=^\[|\Z)", self.text)
        if not m:
            return {}
        env = {}
        for line in m.group(1).splitlines():
            kv = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"', line)
            if kv:
                env[kv.group(1)] = kv.group(2)
        return env

    def qor(self):
        # the harness stores workload stdout next to the run; regex it if asked
        rex = self._field("stdout_regex", table="qor")
        if not rex:
            return None
        stdout_path = getattr(self, "stdout_path", None)
        if not stdout_path or not os.path.isfile(stdout_path):
            return None
        m = re.search(rex, open(stdout_path, errors="replace").read())
        return {"qor_value": float(m.group(1))} if m else None


ADAPTERS = {"cuvslam": CuvslamAdapter, "command": CommandAdapter}

# ── drop-in adapters: profiling/adapters.d/<name>.py ─────────────────────────
# A user integrating their own codebase writes ONE file defining a subclass of
# Adapter with a unique `name`; it is discovered here and selectable via
# `--adapter <name>` or `[workload] adapter = "<name>"` in the config.
# Files starting with "_" are skipped (keep examples/docs there).
_DROPIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapters.d")
if os.path.isdir(_DROPIN_DIR):
    import importlib.util as _ilu
    for _f in sorted(os.listdir(_DROPIN_DIR)):
        if not _f.endswith(".py") or _f.startswith("_"):
            continue
        try:
            _spec = _ilu.spec_from_file_location(f"adapters_d.{_f[:-3]}",
                                                 os.path.join(_DROPIN_DIR, _f))
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            for _v in vars(_mod).values():
                if (isinstance(_v, type) and issubclass(_v, Adapter)
                        and _v.name not in ("abstract",) and _v.name not in ADAPTERS):
                    ADAPTERS[_v.name] = _v
        except Exception as _e:  # noqa: BLE001 — a broken drop-in must not kill the harness
            print(f"[adapters] skipping adapters.d/{_f}: {_e}")


def pick_adapter(config_path: str, repo_root: str, venv_python: str,
                 forced: str | None = None) -> Adapter:
    """Select the adapter for a config.

    Priority: --adapter flag > `[workload] adapter = "x"` in the config >
    auto ([workload] table present -> command, else cuvslam)."""
    text = open(config_path).read()
    if forced and forced != "auto":
        cls = ADAPTERS[forced]
    else:
        named = re.search(r'(?ms)^\[workload\].*?^adapter\s*=\s*"([^"]+)"', text)
        if named and named.group(1) in ADAPTERS:
            cls = ADAPTERS[named.group(1)]
        else:
            cls = CommandAdapter if re.search(r"(?m)^\[workload\]", text) else CuvslamAdapter
    return cls(config_path, text, repo_root, venv_python)
