"""
GraSP (Gradient Signal Preservation) pruning at initialization.

GraSP finds a sparse mask **before training** that preserves the ability
of gradients to propagate through the network. It keeps weights whose
removal would most damage |∇_θ L|², i.e. the total gradient flow.

Algorithm overview (one-shot, no training required):
  1. Sample a small batch from the training data.
  2. Compute first-order gradients g = ∇_θ L  (with computational graph).
  3. Form the gradient-flow scalar z = Σ (g_i)².
  4. Backprop z to obtain the Hessian–gradient product Hg = ∂z/∂θ.
  5. Score each weight:  S(θ) = −θ · Hg   (element-wise).
  6. Keep the top-k weights by score; prune the rest.

Reference implementation:
    GraSP/pruner/GraSP.py (Wang et al., 2020)
"""

import copy
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helper: balanced data sampling
# ---------------------------------------------------------------------------

def fetch_balanced_data(
    dataloader: DataLoader,
    num_classes: int,
    samples_per_class: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a balanced mini-batch with equal representation per class.

    This mirrors GraSP_fetch_data from the reference implementation.

    Args:
        dataloader: Training dataloader to draw from.
        num_classes: Number of classes in the dataset.
        samples_per_class: How many samples to collect for each class.

    Returns:
        Tuple of (inputs, targets) tensors with
        ``num_classes * samples_per_class`` samples total.
    """
    buckets_x: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]
    buckets_y: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]
    filled = set()

    dataloader_iter = iter(dataloader)
    while len(filled) < num_classes:
        inputs, targets = next(dataloader_iter)
        for i in range(inputs.size(0)):
            cls = targets[i].item()
            if cls >= num_classes:
                continue
            if len(buckets_x[cls]) >= samples_per_class:
                filled.add(cls)
                continue
            buckets_x[cls].append(inputs[i : i + 1])
            buckets_y[cls].append(targets[i : i + 1])
            if len(buckets_x[cls]) == samples_per_class:
                filled.add(cls)

    X = torch.cat([torch.cat(bx, dim=0) for bx in buckets_x], dim=0)
    y = torch.cat([torch.cat(by, dim=0) for by in buckets_y], dim=0).view(-1)
    return X, y


# ---------------------------------------------------------------------------
# Core: compute GraSP scores
# ---------------------------------------------------------------------------

def _get_prunable_weights(net: nn.Module) -> List[torch.nn.Parameter]:
    """Return weight tensors eligible for pruning (Conv2d / Linear only).

    Bias and BatchNorm parameters are intentionally excluded—pruning
    them hurts training stability and they contribute very few params.
    """
    weights = []
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weights.append(module.weight)
    return weights


def compute_grasp_scores(
    net: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    samples_per_class: int = 25,
    num_iters: int = 1,
    T: float = 200.0,
) -> Dict[nn.Module, torch.Tensor]:
    """Compute per-weight GraSP importance scores.

    The function implements stages 1–4 of the tutorial:
      • Accumulate first-order gradients over ``num_iters`` balanced
        mini-batches (split in two halves following the reference).
      • Build the gradient-flow objective z = g^T · g_f and backprop
        to get the Hessian–gradient product via ``z.backward()``.
      • Score: S(θ) = −θ · Hg.

    Args:
        net: Model at **random initialisation** (not trained).
        dataloader: Training dataloader.
        device: Device for computation (cpu / cuda).
        num_classes: Number of classes in the dataset.
        samples_per_class: Samples per class in each balanced batch.
        num_iters: Number of balanced batches to accumulate gradients over.
        T: Temperature scaling divisor applied to logits before the loss.
           Stabilises second-order signal (default 200 matches reference).

    Returns:
        Dictionary mapping ``nn.Module`` (Conv2d/Linear) → element-wise
        GraSP score tensor (same shape as the layer's weight).
    """
    # Work on a deep copy so we don't mutate the caller's model
    net_copy = copy.deepcopy(net).to(device)
    net_copy.zero_grad()

    weights = _get_prunable_weights(net_copy)
    for w in weights:
        w.requires_grad_(True)

    # ------------------------------------------------------------------
    # Stage 1 – accumulate first-order gradients  g = Σ ∇L_i
    # ------------------------------------------------------------------
    # The reference splits each balanced batch in two halves: the first
    # half is used *without* create_graph (cheaper) and the second half
    # *with* create_graph to allow the second-order backward later.
    # We reproduce that behaviour exactly.

    grad_w: Optional[List[torch.Tensor]] = None  # accumulated grads
    inputs_halves: List[torch.Tensor] = []
    targets_halves: List[torch.Tensor] = []

    for _ in range(num_iters):
        inputs, targets = fetch_balanced_data(
            dataloader, num_classes, samples_per_class
        )
        N = inputs.size(0)
        inputs = inputs.to(device)
        targets = targets.to(device)

        # --- first half (no graph needed) ---
        out1 = net_copy(inputs[: N // 2]) / T
        loss1 = F.cross_entropy(out1, targets[: N // 2])
        grad_p = autograd.grad(loss1, weights)
        if grad_w is None:
            grad_w = list(grad_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] = grad_w[idx] + grad_p[idx]

        # --- second half (no graph needed either — graph comes in stage 2) ---
        out2 = net_copy(inputs[N // 2 :]) / T
        loss2 = F.cross_entropy(out2, targets[N // 2 :])
        grad_p = autograd.grad(loss2, weights, create_graph=False)
        if grad_w is None:
            grad_w = list(grad_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] = grad_w[idx] + grad_p[idx]

        # Store halves for the second pass
        inputs_halves.append(inputs[: N // 2])
        targets_halves.append(targets[: N // 2])
        inputs_halves.append(inputs[N // 2 :])
        targets_halves.append(targets[N // 2 :])

    # ------------------------------------------------------------------
    # Stages 2–3 – gradient-flow objective z = g^T · g_f  and Hg
    # ------------------------------------------------------------------
    # For each stored half-batch, compute a *new* forward/backward with
    # create_graph=True so that z.backward() gives us Hg via the weight
    # .grad attributes.

    for inputs_h, targets_h in zip(inputs_halves, targets_halves):
        out = net_copy(inputs_h) / T
        loss = F.cross_entropy(out, targets_h)

        grad_f = autograd.grad(loss, weights, create_graph=True)

        # z = Σ_i  (g_i · g_f_i)  — effectively g^T g_f
        z = 0
        count = 0
        for module in net_copy.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    # ------------------------------------------------------------------
    # Stage 4 – score:  S(θ) = −θ · Hg
    # ------------------------------------------------------------------
    # Map scores back to the *original* model's modules so callers can
    # look up by module reference.
    original_modules = [
        m for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    copy_modules = [
        m for m in net_copy.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
    ]

    scores: Dict[nn.Module, torch.Tensor] = {}
    for orig_m, copy_m in zip(original_modules, copy_modules):
        scores[orig_m] = (
            -copy_m.weight.data * copy_m.weight.grad
        ).cpu()

    return scores


# ---------------------------------------------------------------------------
# Masking: convert scores → binary masks
# ---------------------------------------------------------------------------

def grasp_masks_from_scores(
    scores: Dict[nn.Module, torch.Tensor],
    sparsity: float,
) -> Dict[nn.Module, torch.Tensor]:
    """Threshold GraSP scores into binary keep/prune masks.

    **Higher** score ⇒ more important ⇒ **keep**.
    We keep the top ``(1 − sparsity)`` fraction of weights.

    Args:
        scores: Per-module GraSP scores from :func:`compute_grasp_scores`.
        sparsity: Fraction of weights to prune (0 = keep all, 1 = prune all).

    Returns:
        Dictionary mapping the same ``nn.Module`` keys to float tensors
        of 0s and 1s (same shape as each layer's weight).
    """
    eps = 1e-10

    # Flatten, normalise, find threshold
    all_scores = torch.cat([s.view(-1) for s in scores.values()])
    norm_factor = torch.abs(all_scores.sum()) + eps
    all_scores = all_scores / norm_factor

    num_to_keep = int((1 - sparsity) * all_scores.numel())
    num_to_keep = max(num_to_keep, 1)  # keep at least one weight
    threshold, _ = torch.topk(all_scores, num_to_keep, sorted=True)
    accept = threshold[-1]

    masks: Dict[nn.Module, torch.Tensor] = {}
    for module, s in scores.items():
        normalised = s.view(-1) / norm_factor
        masks[module] = (normalised >= accept).float().view(s.shape)

    return masks


# ---------------------------------------------------------------------------
# Public API: one-call GraSP pruning
# ---------------------------------------------------------------------------

def grasp(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    sparsity: float,
    num_classes: int = 10,
    samples_per_class: int = 25,
    num_iters: int = 1,
    T: float = 200.0,
) -> Dict[str, np.ndarray]:
    """Run GraSP pruning and return masks in the project-wide format.

    This is the main entry point for GraSP.  It:
      1. Computes GraSP scores on the **uninitialised** model.
      2. Converts scores to binary masks at the requested sparsity.
      3. Applies the masks to the model weights **in-place**.
      4. Returns masks as ``{layer_name: np.ndarray}`` dicts, matching
         the format used by :func:`src.util.apply_masks_to_model` and
         the rest of the project (IMP, Early-Bird, etc.).

    Args:
        model: Model at random initialisation (**not** trained).
        dataloader: Training dataloader (only a small sample is used).
        device: Torch device.
        sparsity: Fraction of prunable weights to remove (e.g. 0.9 = 90%).
        num_classes: Number of dataset classes.
        samples_per_class: Balanced samples per class for GraSP batches.
        num_iters: Number of balanced batches to average gradients over.
        T: Temperature scaling for logits (default 200).

    Returns:
        Dictionary ``{layer_name: mask_ndarray}`` compatible with the
        project-wide mask utilities in ``src.util`` and ``src.pruning``.
    """
    # Stage 0–4: compute scores
    scores = compute_grasp_scores(
        model, dataloader, device,
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        num_iters=num_iters,
        T=T,
    )

    # Stage 5: threshold into masks
    module_masks = grasp_masks_from_scores(scores, sparsity)

    # Convert module-keyed masks → name-keyed numpy masks (project format)
    name_to_module = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    }
    module_to_name = {v: k for k, v in name_to_module.items()}

    named_masks: Dict[str, np.ndarray] = {}
    for module, mask in module_masks.items():
        name = module_to_name[module]
        named_masks[name] = mask.numpy()

    # Stage 5 (cont.): apply masks to model weights in-place
    for name, module in name_to_module.items():
        if name in named_masks:
            mask_tensor = torch.from_numpy(named_masks[name]).to(module.weight.device)
            module.weight.data.mul_(mask_tensor)

    return named_masks


def get_grasp_sparsity(masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Report per-layer and overall sparsity of GraSP masks.

    Args:
        masks: Named masks from :func:`grasp`.

    Returns:
        Dictionary with per-layer sparsity and an ``'overall'`` entry.
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
