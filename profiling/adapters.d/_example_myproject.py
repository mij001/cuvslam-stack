"""Example drop-in adapter — copy to <yourname>.py (no leading underscore) to
activate. See docs/ADAPTERS.md for the full contract.

The harness discovers every Adapter subclass in this directory; select yours
with `--adapter myproject` or `[workload] adapter = "myproject"` in a config.
"""
from adapters import Adapter


class MyProjectAdapter(Adapter):
    """Launch + QoR for an out-of-tree GPU codebase."""

    name = "myproject"

    def argv(self):
        # any deterministic command; self.config_path is the harness config,
        # so workload parameters can live in it (read them however you like)
        return ["./build/bench", "--iters", "100"]

    def cwd(self):
        return "/home/me/myproject"        # your codebase checkout

    def env(self):
        return {"CUDA_VISIBLE_DEVICES": "0"}

    def qor(self):
        # anything comparable run-to-run: a loss, a checksum, an error metric.
        # The regime compares this across plain/nsys/ncu/nvbit runs to prove
        # the profilers did not perturb your computation.
        try:
            import json
            return json.load(open("/home/me/myproject/results/metrics.json"))
        except OSError:
            return None
