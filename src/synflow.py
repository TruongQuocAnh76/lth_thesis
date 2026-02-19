"""
SynFlow (Synaptic Flow) pruning at initialisation.

SynFlow is a data-free, iterative pruning method that avoids layer collapse
by computing scores from a surrogate loss based on the product of absolute
weights across all layers:

    R = 1^T ( prod_{l=1}^{L} |theta^{[l]}| ) 1

The per-parameter score (synaptic saliency) is:

    S = (dR / d theta) * theta          (element-wise)

Key properties:
  - **Data-free**: uses all-ones input instead of real data.
  - **Iterative**: re-evaluates scores after each pruning step.
  - **Global**: ranks scores across the entire network.
  - **No layer-collapse**: positive, conserved scores (Theorems 1-2).
  - **Exponential schedule**: prunes to target compression over n steps.

Reference:
    Tanaka et al., "Pruning neural networks without any data by iteratively
    conserving synaptic flow", NeurIPS 2020.

See also: Synaptic-Flow/Pruners/pruners.py (SynFlow class) and
          Synaptic-Flow/prune.py (prune_loop with exponential schedule).
"""

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper: linearise / restore model
# ---------------------------------------------------------------------------

@torch.no_grad()
def _linearise(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Replace all parameters with their absolute values.

    This effectively removes non-linearities (ReLU, etc.) from the
    computational graph for the surrogate objective, because
    abs(weight) is always non-negative.

    Returns:
        Dictionary of original parameter signs so we can restore later.
    """
    signs: Dict[str, torch.Tensor] = {}
    for name, param in model.state_dict().items():
        signs[name] = torch.sign(param)
        param.abs_()
    return signs


@torch.no_grad()
def _restore(model: nn.Module, signs: Dict[str, torch.Tensor]):
    """Restore original parameter signs after scoring."""
    for name, param in model.state_dict().items():
        if name in signs:
            param.mul_(signs[name])


# ---------------------------------------------------------------------------
# Core: compute SynFlow scores for one iteration
# ---------------------------------------------------------------------------

def compute_synflow_scores(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, ...] = (3, 32, 32),
) -> Dict[str, torch.Tensor]:
    """Compute SynFlow saliency scores for all prunable parameters.

    Algorithm (one pass):
      1. Linearise: replace params with |params|.
      2. Forward an all-ones input through the network.
      3. Backward sum(output) to get gradients.
      4. Score = |grad * param| for each prunable weight.
      5. Restore original signs.

    Args:
        model: Neural network (can be masked, eval mode recommended).
        device: Torch device (cpu / cuda).
        input_shape: Spatial input shape **without** batch dim, e.g. (3, 32, 32).

    Returns:
        Dictionary ``{layer_name: score_tensor}`` for every Conv2d / Linear
        layer in the model.  Scores are non-negative.
    """
    model.eval()

    # Step 1 – linearise
    signs = _linearise(model)

    # Step 2 – forward all-ones input
    ones_input = torch.ones([1] + list(input_shape), device=device)
    output = model(ones_input)

    # Step 3 – backward
    torch.sum(output).backward()

    # Step 4 – collect scores  S = |grad * param|
    scores: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if module.weight.grad is not None:
                score = (module.weight.grad * module.weight.data).detach().abs()
                scores[name] = score.cpu()
            module.weight.grad = None          # reset for next iteration

    # Step 5 – restore signs
    _restore(model, signs)

    # Zero all remaining gradients
    model.zero_grad()

    return scores


# ---------------------------------------------------------------------------
# Iterative SynFlow pruning (Algorithm 1 from the paper)
# ---------------------------------------------------------------------------

def synflow_pruning(
    model: nn.Module,
    device: torch.device,
    target_sparsity: float,
    num_iters: int = 100,
    input_shape: Tuple[int, ...] = (3, 32, 32),
) -> Dict[str, np.ndarray]:
    """Run iterative SynFlow pruning (Algorithm 1).

    Pseudocode (from paper):
        Input: network f(x; θ₀), compression ratio ρ, iteration steps n
        μ = 1  (all-ones mask)
        for k in [1, …, n]:
            θ_μ ← μ ⊙ θ₀
            R ← 1ᵀ (∏ |θ_μ^{[l]}|) 1
            S ← (∂R / ∂θ_μ) ⊙ θ_μ
            τ ← percentile of S at (1 − ρ^{−k/n})
            μ ← (S > τ)
        return μ ⊙ θ₀

    The exponential schedule prunes gently at the start and more
    aggressively later, which is critical for avoiding layer collapse.

    Args:
        model: Model at initialisation (weights will **not** be modified;
               a deep copy is used internally).
        device: Torch device.
        target_sparsity: Fraction of weights to prune (e.g. 0.9 → keep 10%).
        num_iters: Number of iterative pruning rounds (default 100).
        input_shape: Spatial input shape without batch dim.

    Returns:
        Dictionary ``{layer_name: mask_ndarray}`` compatible with
        ``src.util.apply_masks_to_model``.
    """
    # Work on a deep copy so the caller's model stays pristine
    model_copy = copy.deepcopy(model).to(device)

    # Initialise masks (all ones)
    masks: Dict[str, torch.Tensor] = {}
    for name, module in model_copy.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            masks[name] = torch.ones_like(module.weight.data)

    # Compression ratio ρ  (fraction of weights to *keep*)
    # target_sparsity = 0.9 → keep_ratio = 0.1
    keep_ratio = 1.0 - target_sparsity
    if keep_ratio <= 0:
        # Edge case: prune everything
        return {k: np.zeros_like(v.numpy()) for k, v in masks.items()}

    for k in range(1, num_iters + 1):
        # Apply current mask to weights
        with torch.no_grad():
            for name, module in model_copy.named_modules():
                if name in masks:
                    module.weight.data.mul_(masks[name].to(device))

        # Compute scores
        scores = compute_synflow_scores(model_copy, device, input_shape)

        # Exponential schedule: sparsity at step k
        # sparse_k = keep_ratio ^ (k / n)
        # This means: at step 1 we keep many, at step n we keep keep_ratio
        sparse_k = keep_ratio ** (k / num_iters)

        # Global threshold: keep top sparse_k fraction of scores
        all_scores = torch.cat([s.view(-1) for s in scores.values()])
        num_params = all_scores.numel()
        num_to_keep = int(sparse_k * num_params)
        num_to_keep = max(num_to_keep, 1)

        # Find threshold (keep scores > threshold)
        if num_to_keep < num_params:
            threshold, _ = torch.kthvalue(
                all_scores.view(-1),
                num_params - num_to_keep + 1
            )
        else:
            threshold = torch.tensor(0.0)

        # Update masks
        for name in masks:
            if name in scores:
                masks[name] = (scores[name] > threshold).float()
            # else: keep existing mask (shouldn't happen)

    # Convert to numpy (project-wide format)
    named_masks: Dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        named_masks[name] = mask.numpy()

    return named_masks


# ---------------------------------------------------------------------------
# Convenience: apply SynFlow masks and return sparsity report
# ---------------------------------------------------------------------------

def apply_synflow_masks(
    model: nn.Module,
    masks: Dict[str, np.ndarray],
):
    """Apply SynFlow masks to model weights in-place.

    Args:
        model: Model whose weights will be zeroed out according to masks.
        masks: Named masks from :func:`synflow_pruning`.
    """
    for name, module in model.named_modules():
        if name in masks:
            mask_t = torch.from_numpy(masks[name]).to(module.weight.device)
            module.weight.data.mul_(mask_t.float())


def get_synflow_sparsity(masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Report per-layer and overall sparsity of SynFlow masks.

    Args:
        masks: Named masks from :func:`synflow_pruning`.

    Returns:
        Dictionary with per-layer sparsity fractions and an ``'overall'`` key.
    """
    sparsity: Dict[str, float] = {}
    total_params = 0
    total_pruned = 0
    for name, mask in masks.items():
        n = mask.size
        p = int(np.sum(mask == 0))
        sparsity[name] = p / n if n > 0 else 0.0
        total_params += n
        total_pruned += p
    sparsity["overall"] = total_pruned / total_params if total_params else 0.0
    return sparsity
