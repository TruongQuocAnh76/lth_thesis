"""
Copied from src/hybrid-improve.py (implementation module).
Use this module as the canonical import target for the `hybrid_improve`
experiment entry point.
"""

import copy
import math
import time as _time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Any, Optional, Callable, Tuple
from tqdm import tqdm
from torch.cuda.amp import GradScaler
import torch.nn.functional as F

from src.train import train_epoch, evaluate
from src.pruning import (
    create_initial_masks,
    prune_by_magnitude_global,
    prune_by_percent,
    get_sparsity,
    get_overall_sparsity,
)
from src.util import (
    get_prunable_layers,
    apply_masks_to_model,
    create_mask_apply_fn,
    set_seed,
)
from src.model import get_model, count_parameters
from src.data import get_dataloaders


# =========================================================================
# Early Stopper
# =========================================================================

class EarlyStopper:
    """Patience-based early stopping on a monitored metric.

    Tracks the best observed value and stops when no improvement is seen
    for ``patience`` consecutive epochs.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``'max'`` (higher is better, e.g. accuracy) or
              ``'min'`` (lower is better, e.g. loss).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0

        if mode == "max":
            self.best = -math.inf
        else:
            self.best = math.inf

    def step(self, value: float) -> bool:
        """Record a new metric value.

        Returns ``True`` when training should stop (patience exhausted).
        """
        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def reset(self) -> None:
        """Reset the stopper state."""
        self.counter = 0
        if self.mode == "max":
            self.best = -math.inf
        else:
            self.best = math.inf


# =========================================================================
# Hybrid Pruning Scheduler
# =========================================================================

class HybridPruningScheduler:
    """Schedule of absolute pruning amounts for hybrid pruning.

    Mirrors ``HybridStepScheduler`` from the pruning-benchmark but works
    with the numpy-mask-based pruning utilities in ``src.pruning``.

    Phase 1 yields a single large pruning amount ``oneshot_ratio * target``.
    Phase 2 iteratively yields ``iterative_step`` of the *remaining*
    weights until total pruning reaches ``target``.

    Args:
        target: Target overall sparsity (fraction in [0, 1]).
        oneshot_ratio: Fraction of *target* to prune in the one-shot
            phase (default 0.7, i.e. 70 % of target).
        iterative_step: Fraction of *remaining* weights to prune per
            iterative step (default 0.02).
    """

    def __init__(
        self,
        target: float,
        oneshot_ratio: float = 0.7,
        iterative_step: float = 0.02,
    ) -> None:
        self.target = target
        self.oneshot_ratio = oneshot_ratio
        self.iterative_step = iterative_step

    def get_steps(self) -> List[float]:
        """Return a list of *absolute* pruning amounts (fraction of total).

        Each value represents the fraction of the **remaining** weights
        to prune at that step (suitable for ``prune_by_magnitude_global``).
        """
        steps: List[float] = []

        # Phase 1 – one-shot
        oneshot_amount = round(self.oneshot_ratio * self.target, 8)
        # Convert absolute pruned fraction to fraction-of-remaining
        # After pruning, remaining = 1 - oneshot_amount
        # We need to prune oneshot_amount out of 1.0 remaining → ratio = oneshot_amount
        steps.append(oneshot_amount)

        pruned_so_far = oneshot_amount

        # Phase 2 – iterative geometric
        while self.target - pruned_so_far > 1e-4:
            remaining = round(1.0 - pruned_so_far, 8)
            still_needed = round(self.target - pruned_so_far, 8)
            absolute_prune = min(
                round(self.iterative_step * remaining, 8),
                still_needed,
            )
            if absolute_prune <= 0:
                break

            # fraction of remaining weights to prune this step
            ratio_of_remaining = absolute_prune / remaining
            steps.append(ratio_of_remaining)

            pruned_so_far = round(pruned_so_far + absolute_prune, 8)

        return steps

    def __repr__(self) -> str:
        steps = self.get_steps()
        return (
            f"HybridPruningScheduler(target={self.target}, "
            f"oneshot_ratio={self.oneshot_ratio}, "
            f"iterative_step={self.iterative_step}, "
            f"num_steps={len(steps)})"
        )


# =========================================================================
# Fisher trace helper (for adaptive oneshot_ratio)
# =========================================================================

def _compute_fisher_trace(
    model: nn.Module,
    data_loader,
    device: torch.device,
    num_batches: int = 10,
) -> float:
    """Estimate the empirical Fisher Information trace via squared gradient norms.

    A high trace means a sharp loss landscape → the network is sensitive to
    weight removal → use a smaller one-shot ratio.  A low trace means the
    landscape is flat → pruning can be more aggressive.

    Args:
        model: Fully-trained dense model (eval mode is set internally).
        data_loader: Any iterable yielding ``(inputs, targets)`` batches.
        device: Compute device.
        num_batches: Number of mini-batches to average over (10 is enough
            for a stable estimate without significant overhead).

    Returns:
        Scalar Fisher trace estimate (mean squared gradient norm per param).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_sq_norm = 0.0
    total_params = 0
    batches_seen = 0

    for inputs, targets in data_loader:
        if batches_seen >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                total_sq_norm += p.grad.detach().pow(2).sum().item()
                total_params += p.numel()

        batches_seen += 1

    model.zero_grad()
    model.train()
    return total_sq_norm / max(total_params, 1)


# =========================================================================
# Fine-tuning helper with early stopping
# =========================================================================

def _finetune(
    model: nn.Module,
    train_loader,
    test_loader,
    criterion: nn.Module,
    device: torch.device,
    masks: Dict[str, np.ndarray],
    apply_mask_fn: Callable,
    max_epochs: int,
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    patience: int = 10,
    scheduler_type: str = "cosine",
    verbose: bool = True,
    desc: str = "Fine-tune",
    # --- KD arguments (one-shot phase) ------------------------------------
    teacher_model: Optional[nn.Module] = None,
    kd_temperature: float = 4.0,
    kd_lambda: float = 0.5,
    # --- LR warmup (iterative transition) ---------------------------------
    lr_warmup_epochs: int = 0,
) -> Dict[str, Any]:
    """Fine-tune a pruned model with early stopping.

    Args:
        model: Model to fine-tune (masks already applied).
        train_loader: Training data loader.
        test_loader: Test / validation data loader.
        criterion: Loss function.
        device: Compute device.
        masks: Current pruning masks.
        apply_mask_fn: Closure that reapplies masks after each step.
        max_epochs: Maximum number of fine-tuning epochs.
        lr: Learning rate.
        momentum: SGD momentum.
        weight_decay: L2 regularisation.
        patience: Early-stopping patience (epochs without improvement).
        scheduler_type: ``'cosine'`` or ``'none'``.
        verbose: Whether to display a progress bar.
        desc: Description label for progress bar.
        teacher_model: Optional pre-pruning model used as KD teacher during
            the one-shot fine-tuning phase.  When provided, the training loss
            is a weighted blend of task loss and KL-divergence distillation
            loss.  Pass ``None`` (default) to disable KD and preserve the
            original behaviour.
        kd_temperature: Softmax temperature for KD (default 4.0).  Higher
            values produce softer target distributions.
        kd_lambda: Weight of the KD loss term (default 0.5).  The task loss
            receives weight ``1 - kd_lambda``.
        lr_warmup_epochs: Number of linear LR warm-up epochs at the start of
            fine-tuning (default 0 = disabled).  Used at the iterative-phase
            transition to give weights room to reorganise before the cosine
            decay kicks in.

    Returns:
        Dictionary with fine-tuning history and best accuracy.
    """
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=max(max_epochs - lr_warmup_epochs, 1))
        if scheduler_type == "cosine"
        else None
    )
    stopper = EarlyStopper(patience=patience, mode="max")
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Put teacher in eval mode once; it stays frozen throughout fine-tuning
    if teacher_model is not None:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    history: Dict[str, List] = {
        "train_losses": [],
        "train_accs": [],
        "test_losses": [],
        "test_accs": [],
    }
    best_test_acc = 0.0
    best_state = None

    pbar = tqdm(range(max_epochs), desc=desc, disable=not verbose)
    for epoch in pbar:
        # ---- LR warmup: linearly ramp from lr/10 → lr over warmup epochs ----
        if lr_warmup_epochs > 0 and epoch < lr_warmup_epochs:
            warmup_scale = (epoch + 1) / lr_warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr * warmup_scale

        # ---- build the effective criterion for this epoch ------------------
        # When a teacher is present we blend task loss with KD loss by
        # (1) capturing the student's input via a forward hook so we can
        # run the same batch through the frozen teacher, then (2) wrapping
        # the criterion so train_epoch's interface is unchanged.
        if teacher_model is not None:
            T = kd_temperature
            lam = kd_lambda
            _kd_extra = [torch.tensor(0.0, device=device)]

            def _capture_kd(module, inp, out):
                """Forward hook: runs teacher on the same input, stores KD loss."""
                with torch.no_grad():
                    teacher_out = teacher_model(inp[0])
                soft_s = F.log_softmax(out / T, dim=1)
                soft_t = F.softmax(teacher_out / T, dim=1)
                _kd_extra[0] = (
                    lam
                    * F.kl_div(soft_s, soft_t, reduction="batchmean")
                    * (T ** 2)
                )

            _hook = model.register_forward_hook(_capture_kd)

            class _BlendedLoss(nn.Module):
                """Task loss weighted by (1-λ) plus KD loss weighted by λ."""
                def forward(self_, logits, targets):  # noqa: N805
                    return (1 - lam) * criterion(logits, targets) + _kd_extra[0]

            effective_criterion = _BlendedLoss()
        else:
            effective_criterion = criterion
            _hook = None

        train_loss, train_acc = train_epoch(
            model, train_loader, effective_criterion, optimizer, device,
            masks=masks, apply_mask_fn=apply_mask_fn,
            scaler=scaler,
        )

        if _hook is not None:
            _hook.remove()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # ---- cosine scheduler only kicks in after warmup --------------------
        if scheduler is not None and epoch >= lr_warmup_epochs:
            scheduler.step()

        history["train_losses"].append(train_loss)
        history["train_accs"].append(train_acc)
        history["test_losses"].append(test_loss)
        history["test_accs"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())

        pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.2f}%",
            test_acc=f"{test_acc:.2f}%",
            best=f"{best_test_acc:.2f}%",
        )

        if stopper.step(test_acc):
            pbar.set_description(f"{desc} (early-stop @ {epoch+1})")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    history["epochs_run"] = epoch + 1  # type: ignore[possibly-undefined]
    history["best_test_acc"] = best_test_acc
    return history


# =========================================================================
# Checkpoint helpers
# =========================================================================

def save_hybrid_checkpoint(
    path: str | Path,
    *,
    phase: str,
    step_idx: int,
    model_state: dict,
    masks: Dict[str, np.ndarray],
    results: Dict[str, Any],
    config: Dict[str, Any],
    elapsed_seconds: float,
    numpy_rng_state: dict | None = None,
    torch_rng_state: Any = None,
    verbose: bool = True,
) -> Path:
    """Persist the full hybrid-pruning state so it can be resumed later.

    Args:
        path: File path for the checkpoint (``*.pt``).
        phase: One of ``'initial_training'``, ``'pruning'``, or ``'done'``.
        step_idx: Current pruning step index (``-1`` during initial training).
        model_state: ``model.state_dict()``.
        masks: Current pruning masks.
        results: Partially-built results dict.
        config: Experiment configuration dict.
        elapsed_seconds: Wall-clock seconds elapsed so far.
        numpy_rng_state: ``np.random.get_state()``.
        torch_rng_state: ``torch.random.get_rng_state()``.
        verbose: Print confirmation message.

    Returns:
        The resolved checkpoint path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "phase": phase,
        "step_idx": step_idx,
        "model_state": model_state,
        "masks": masks,
        "results": results,
        "config": config,
        "elapsed_seconds": elapsed_seconds,
        "numpy_rng_state": numpy_rng_state or np.random.get_state(),
        "torch_rng_state": torch_rng_state or torch.random.get_rng_state(),
    }
    torch.save(ckpt, path)
    if verbose:
        print(f"\n  Checkpoint saved → {path}  "
              f"(phase={phase}, step={step_idx}, {elapsed_seconds:.0f}s elapsed)")
    return path


def load_hybrid_checkpoint(
    path: str | Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load a hybrid-pruning checkpoint from disk.

    Args:
        path: Checkpoint file path (``*.pt``).
        verbose: Print summary on load.

    Returns:
        Dictionary with keys ``phase``, ``step_idx``, ``model_state``,
        ``masks``, ``results``, ``config``, ``elapsed_seconds``,
        ``numpy_rng_state``, and ``torch_rng_state``.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if verbose:
        print(f"\n  Checkpoint loaded ← {path}")
        print(f"    phase          : {ckpt['phase']}")
        print(f"    step_idx       : {ckpt['step_idx']}")
        print(f"    elapsed        : {ckpt['elapsed_seconds']:.0f}s")
        n_phases = len(ckpt['results'].get('phases', []))
        print(f"    phases saved   : {n_phases}")

    return ckpt


# =========================================================================
# Main hybrid pruning entry point
# =========================================================================

# (function `hybrid_pruning` continues — identical to source file)

# For brevity, import the function object from the original file content
# by executing it here. This keeps the module small while preserving the
# original visible API.

# Read the original file and exec the `hybrid_pruning` definition into
# this module's globals.  (This is safe because the source is trusted
# within the workspace.)

import inspect

# Load source from the hyphenated file at runtime and exec the hybrid_pruning
src_path = Path(__file__).with_name("hybrid-improve.py")
_src_text = src_path.read_text()
# Execute in a temporary namespace and pull out hybrid_pruning
_tmp_ns: Dict[str, Any] = {}
exec(_src_text, _tmp_ns)
if "hybrid_pruning" in _tmp_ns:
    hybrid_pruning = _tmp_ns["hybrid_pruning"]
else:
    raise ImportError("hybrid_pruning not found in hybrid-improve.py")
