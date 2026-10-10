"""CUDA preflight checks, FP32 settings and runtime metadata for the sensitivity study."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import scipy
import torch


def configure_cuda_fp32():
    """Disable TF32 and cuDNN autotuning and request deterministic cuDNN kernels.

    Must be called in each spawned worker as well as in the parent process.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def cuda_preflight():
    """Run a small CUDA LSTM forward/backward pass and a sparse matmul on toy inputs.

    Raises RuntimeError if CUDA is unavailable or a check fails; there is no
    CPU fallback. Returns a dict describing the device and numeric settings.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run inside a GPU "
            "allocation; check nvidia-smi. This run will NOT fall back to CPU."
        )
    configure_cuda_fp32()
    device = torch.cuda.current_device()
    cuda_device = torch.device("cuda", device)
    properties = torch.cuda.get_device_properties(device)
    # Fork the RNG so the checks do not change the simulation's random streams.
    with torch.random.fork_rng(devices=[device]):
        model = torch.nn.LSTM(3, 8, batch_first=True).cuda(device)
        x = torch.ones(4, 5, 3, device=cuda_device, requires_grad=True)
        output, _ = model(x)
        output.square().mean().backward()
        if not torch.isfinite(output).all() or not torch.isfinite(x.grad).all():
            raise RuntimeError("CUDA LSTM forward/backward produced non-finite values")
        indices = torch.tensor([[0, 1], [1, 0]], device=cuda_device)
        adjacency = torch.sparse_coo_tensor(indices, torch.ones(2, device=cuda_device), (2, 2)).coalesce()
        dense = torch.tensor([[1., 2.], [3., 4.]], device=cuda_device)
        torch.testing.assert_close(torch.sparse.mm(adjacency, dense), dense.flip(0))
        torch.cuda.synchronize(device)
    del model, x, output, adjacency, dense, indices, _
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "logical_device": device, "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_gib": total / 2**30, "free_memory_gib": free / 2**30,
        "lstm_forward_backward": "passed", "sparse_mm": "passed",
        "amp": False, "tf32": False, "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }


def runtime_metadata(device_name):
    """Record library versions and the resolved device; on CUDA, run the preflight and query nvidia-smi."""
    resolved = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    if resolved == "auto":
        resolved = "cpu"
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": resolved, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if resolved == "cuda":
        result["gpu"] = cuda_preflight()
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=True,
            )
            result["nvidia_smi_inventory"] = proc.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            result["nvidia_smi_inventory"] = None
    return result


def main():
    """Print CUDA runtime metadata as JSON."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(runtime_metadata("cuda"), indent=2))


if __name__ == "__main__":
    main()
