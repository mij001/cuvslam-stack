#!/usr/bin/env python3
"""Minimal example GPU workload for the command adapter (profiling/adapters.py).

Any GPU codebase profiles the same way: describe how to launch it in a config's
[workload] table and the whole harness (nsys/ncu/nvbit capture, DAG, roofline,
classification, substrate verdicts) applies unchanged. NVTX ranges — like the
ones below — become pipeline stages in the nsys analyses, exactly as cuVSLAM's
internal annotations do.

Run it under the harness:
    python3 profiling/harness/profile.py \
        --config configs/workload_torch_matmul.toml --profiler nsys \
        --hw profiling/hw/rtx2000ada_sm89.toml
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=2048, help="matrix dimension")
parser.add_argument("--iters", type=int, default=100)
args = parser.parse_args()

try:
    import torch
except ImportError:
    raise SystemExit("this example needs pytorch (pip install torch)")

if not torch.cuda.is_available():
    raise SystemExit("no CUDA device")

dev = torch.device("cuda")
a = torch.randn(args.n, args.n, device=dev)
b = torch.randn(args.n, args.n, device=dev)

torch.cuda.nvtx.range_push("warmup")
for _ in range(5):
    a @ b
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("matmul_steady")
c = a
for _ in range(args.iters):
    c = (c @ b).relu()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

# a scalar the harness can extract as QoR (proves profiling didn't perturb it)
print(f"checksum={c.float().abs().mean().item():.6e}")
