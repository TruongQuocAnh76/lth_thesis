"""Model efficiency metrics shared across experiment runners."""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.model import get_model

MaskDict = Dict[str, Any]
MaskApplier = Callable[[nn.Module, MaskDict], None]


def infer_input_shape(dataset_name: str) -> tuple[int, int, int]:
    """Infer a single-sample input shape for the configured dataset."""
    dataset = dataset_name.lower()
    if dataset in {"mnist", "fashion_mnist"}:
        return (3, 32, 32)
    return (3, 32, 32)


def _mask_density(mask: Any) -> float:
    """Return the non-zero ratio of a pruning mask."""
    if mask is None:
        return 1.0

    if isinstance(mask, torch.Tensor):
        total = mask.numel()
        if total == 0:
            return 1.0
        return float(torch.count_nonzero(mask).item()) / float(total)

    mask_array = np.asarray(mask)
    if mask_array.size == 0:
        return 1.0
    return float(np.count_nonzero(mask_array)) / float(mask_array.size)


def estimate_model_flops(
    model: nn.Module,
    sample_input: torch.Tensor,
    masks: Optional[MaskDict] = None,
) -> float:
    """Estimate Conv2d/Linear FLOPs with optional mask-aware scaling.

    This counts arithmetic work for Conv2d and Linear layers only.
    When masks are provided, per-layer FLOPs are scaled by each layer's
    non-zero mask density instead of profiling the zero-masked dense graph.
    """
    module_names = {module: name for name, module in model.named_modules()}
    hooks = []
    total_flops = 0.0
    was_training = model.training

    def conv_hook(module: nn.Conv2d, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal total_flops
        density = 1.0
        name = module_names.get(module)
        if masks is not None and name in masks:
            density = _mask_density(masks[name])
        if density <= 0.0:
            return

        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        bias_ops = 1 if module.bias is not None else 0
        flops_per_output = (2 * kernel_ops) + bias_ops
        total_flops += float(output.numel()) * float(flops_per_output) * density

    def linear_hook(module: nn.Linear, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal total_flops
        density = 1.0
        name = module_names.get(module)
        if masks is not None and name in masks:
            density = _mask_density(masks[name])
        if density <= 0.0:
            return

        bias_ops = 1 if module.bias is not None else 0
        flops_per_output = (2 * module.in_features) + bias_ops
        total_flops += float(output.numel()) * float(flops_per_output) * density

    try:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(linear_hook))

        model.eval()
        with torch.no_grad():
            model(sample_input)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    return float(total_flops)


def measure_latency(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    num_runs: int = 30,
    warmup: int = 10,
) -> tuple[float, float]:
    """Measure average latency and throughput."""
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_runs):
            model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    model.train(was_training)

    average_seconds = (end - start) / num_runs
    latency_ms = average_seconds * 1000.0
    throughput = 1.0 / average_seconds if average_seconds > 0 else float("inf")
    return float(latency_ms), float(throughput)


def compute_efficiency_metrics(
    *,
    model_name: str,
    num_classes: int,
    dataset_name: str,
    device: torch.device | str,
    final_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    masks: Optional[MaskDict] = None,
    mask_applier: Optional[MaskApplier] = None,
) -> Dict[str, Any]:
    """Compute FLOPs and latency metrics for dense and pruned models."""
    torch_device = device if isinstance(device, torch.device) else torch.device(device)
    input_shape = infer_input_shape(dataset_name)
    sample_input = torch.randn((1, *input_shape), device=torch_device)

    dense_model = get_model(model_name, num_classes=num_classes).to(torch_device)
    dense_flops = estimate_model_flops(dense_model, sample_input)
    dense_latency_ms, dense_throughput = measure_latency(dense_model, sample_input, torch_device)

    pruned_model = get_model(model_name, num_classes=num_classes).to(torch_device)
    if final_state_dict is not None:
        pruned_model.load_state_dict(copy.deepcopy(final_state_dict))
    if masks is not None and mask_applier is not None:
        mask_applier(pruned_model, masks)

    pruned_flops = estimate_model_flops(pruned_model, sample_input, masks=masks)
    pruned_latency_ms, pruned_throughput = measure_latency(pruned_model, sample_input, torch_device)

    flops_reduction = None
    if pruned_flops > 0:
        flops_reduction = float(dense_flops) / float(pruned_flops)

    return {
        "flops_reduction": flops_reduction,
        "dense_flops": float(dense_flops),
        "pruned_flops": float(pruned_flops),
        "dense_latency_ms": dense_latency_ms,
        "pruned_latency_ms": pruned_latency_ms,
        "dense_throughput": dense_throughput,
        "pruned_throughput": pruned_throughput,
        "flops_backend": "internal_mask_aware_estimator",
        "flops_note": (
            "Counts Conv2d/Linear FLOPs. Pruned FLOPs are estimated from mask density "
            "because zero-masked dense models do not change operator shapes."
        ),
    }
