"""
Hybrid Pruning: One-Shot + Iterative Geometric Magnitude Pruning.

Implements the hybrid pruning strategy combining a large initial
one-shot prune with iterative geometric refinement:

1. **Initial Training** – fully train the dense network.
2. **Large One-Shot Prune** – prune ``p_k`` (60–80 % of target ``p``)
   of weights in a single step based on magnitude.
3. **Extended Fine-Tuning** – recover accuracy with high patience
   (≈200 epochs) and a reduced learning rate (1/10 of original).
4. **Iterative Geometric Pruning** – starting from the one-shot pruned
   network, repeatedly prune a small ratio ``p_i`` of remaining weights
   and fine-tune with short patience until overall sparsity reaches ``p``.
5. **Patience-Based Early Stopping** – used in every fine-tuning phase
   to determine duration dynamically.

Reference implementation: ``pruning-benchmark/`` ``HybridStepScheduler``.
"""

import copy
import csv
import datetime
import json
import math
import os
import time as _time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Any, Optional, Callable, Tuple
from tqdm import tqdm
from torch.cuda.amp import GradScaler
import json
import datetime
import csv
import torch.nn.functional as F

from src.train import train_epoch, evaluate
from src.metrics import compute_efficiency_metrics
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
# Helpers
# =========================================================================

def _convert_to_serializable(obj):
    """Convert numpy / torch types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    else:
        return obj


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
    batches_seen = 0
    # Count parameters once — the model topology doesn't change between batches
    total_params = sum(p.numel() for p in model.parameters())

    for inputs, targets in data_loader:
        if batches_seen >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Accumulate squared norms on GPU, sync once per batch (not per param)
        batch_sq_norm = sum(
            p.grad.detach().pow(2).sum()
            for p in model.parameters() if p.grad is not None
        )
        total_sq_norm += batch_sq_norm.item()

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
    # --- CUDA Graphs (reduces kernel-launch overhead) ---------------------
    use_cuda_graphs: bool = False,
    # --- Checkpoint resume -----------------------------------------------
    start_epoch: int = 0,
    optimizer_state: Optional[dict] = None,
    scheduler_state: Optional[dict] = None,
    stopper_state: Optional[dict] = None,
    initial_best_state: Optional[dict] = None,
    on_epoch_end: Optional[Callable] = None,
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
    # Restore optimizer / scheduler / stopper state when resuming mid-finetune
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    if stopper_state is not None:
        stopper.best = stopper_state["best"]
        stopper.counter = stopper_state["counter"]
    # AMP and CUDA Graphs are mutually exclusive: graphs cannot be captured
    # while inside an autocast context. Prioritise AMP unless the caller
    # explicitly requested graphs (which already disable AMP in the guard below).
    use_amp = torch.cuda.is_available() and not use_cuda_graphs
    scaler = GradScaler() if use_amp else None

    # Put teacher in eval mode once; it stays frozen throughout fine-tuning
    if teacher_model is not None:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    # --- CUDA Graphs setup -----------------------------------------------
    # CUDA Graphs capture a sequence of CUDA kernels and replay them with
    # reduced host-side overhead (~10-30% speedup for short epochs on CUDA).
    # Only usable when: CUDA is available, no KD teacher (dynamic control
    # flow in the hook prevents capturing), warmup disabled, and the caller
    # opts in via `use_cuda_graphs=True`.
    _use_graphs = (
        use_cuda_graphs
        and torch.cuda.is_available()
        and teacher_model is None
        and lr_warmup_epochs == 0
        and not use_amp  # AMP and CUDA Graphs are mutually exclusive
    )
    _graph: Optional[torch.cuda.CUDAGraph] = None
    _static_data: Optional[torch.Tensor] = None
    _static_target: Optional[torch.Tensor] = None
    _static_loss: Optional[torch.Tensor] = None

    if _use_graphs:
        # Warm up the graph with a few iterations before capturing
        _sample_data, _sample_target = next(iter(train_loader))
        _static_data = _sample_data.to(device)
        _static_target = _sample_target.to(device)
        # Warmup runs (required before CUDA graph capture)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            _out = model(_static_data)
            _loss = criterion(_out, _static_target)
            _loss.backward()
            optimizer.step()
        # Capture the graph
        _graph = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(_graph):
            _static_out = model(_static_data)
            _static_loss = criterion(_static_out, _static_target)
            _static_loss.backward()
        # Zero grad again after capture so first real step starts clean
        optimizer.zero_grad(set_to_none=True)

    history: Dict[str, List] = {
        "train_losses": [],
        "train_accs": [],
        "test_losses": [],
        "test_accs": [],
    }
    best_test_acc = (
        stopper_state["best"]
        if stopper_state and stopper_state.get("best", -math.inf) > -math.inf / 2
        else 0.0
    )
    best_state = initial_best_state

    # Resume edge-case: checkpoint may point to a fine-tuning stage where the
    # final epoch has already completed (e.g., ft_epoch=max_epochs-1).
    # In that case there is nothing left to train for this phase.
    if start_epoch >= max_epochs:
        if best_state is not None:
            model.load_state_dict(best_state)
        history["epochs_run"] = 0
        history["start_epoch"] = start_epoch
        history["best_test_acc"] = best_test_acc
        return history

    # ---- Build effective criterion and register KD hook ONCE (outside loop) ----
    # Re-creating the hook and the closure every epoch wastes memory and time.
    if teacher_model is not None:
        T = kd_temperature
        lam = kd_lambda
        _kd_extra = [torch.tensor(0.0, device=device)]

        def _capture_kd(module, inp, out):
            """Forward hook: runs teacher on the same input batch, stores KD loss."""
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
        # Use a simple lambda so no new class is allocated per epoch
        effective_criterion = lambda logits, tgt: (1 - lam) * criterion(logits, tgt) + _kd_extra[0]  # noqa: E731
    else:
        effective_criterion = criterion
        _hook = None

    pbar = tqdm(range(start_epoch, max_epochs), desc=desc, disable=not verbose)
    for epoch in pbar:
        # ---- LR warmup: linearly ramp from lr/10 → lr over warmup epochs ----
        if lr_warmup_epochs > 0 and epoch < lr_warmup_epochs:
            warmup_scale = (epoch + 1) / lr_warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr * warmup_scale

        if _use_graphs and _graph is not None:
            # CUDA-Graph replay path: copy data into static buffers, replay,
            # then step the optimizer (optimizer.step() is outside the graph).
            total_loss_graph = 0.0
            correct_graph = 0
            total_graph = 0
            for _data, _tgt in train_loader:
                _static_data.copy_(_data.to(device))
                _static_target.copy_(_tgt.to(device))
                _graph.replay()
                optimizer.step()
                if masks is not None:
                    apply_mask_fn(masks)
                total_loss_graph += _static_loss.item()
                # accuracy from static output
                with torch.no_grad():
                    _pred = _static_out.argmax(dim=1)
                    correct_graph += _pred.eq(_static_target).sum().item()
                    total_graph += _static_target.size(0)
                optimizer.zero_grad(set_to_none=True)
            train_loss = total_loss_graph / max(len(train_loader), 1)
            train_acc = 100.0 * correct_graph / max(total_graph, 1)
        else:
            train_loss, train_acc = train_epoch(
                model, train_loader, effective_criterion, optimizer, device,
                masks=masks, apply_mask_fn=apply_mask_fn,
                scaler=scaler,
                use_amp=use_amp,
            )

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
            # Shallow-clone each tensor: cheaper than deepcopy but safe against
            # in-place SGD updates (with weight decay) corrupting the snapshot.
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.2f}%",
            test_acc=f"{test_acc:.2f}%",
            best=f"{best_test_acc:.2f}%",
        )

        should_stop = stopper.step(test_acc)

        if on_epoch_end is not None:
            on_epoch_end(epoch, optimizer, scheduler, stopper, best_state)

        if should_stop:
            pbar.set_description(f"{desc} (early-stop @ {epoch+1})")
            break

    # Remove KD hook after the loop
    if _hook is not None:
        _hook.remove()

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    history["epochs_run"] = len(history["test_accs"])
    history["start_epoch"] = start_epoch
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
    # Fine-grained stage info for intra-step resume
    stage: Optional[str] = None,
    ft_epoch: Optional[int] = None,
    optimizer_state: Optional[dict] = None,
    scheduler_state: Optional[dict] = None,
    stopper_state: Optional[dict] = None,
    best_model_state: Optional[dict] = None,
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
        stage: Fine-grained stage within the current step, e.g.
            ``'oneshot_prune'``, ``'oneshot_finetune'``,
            ``'iterative_prune'``, ``'iterative_finetune'``.
        ft_epoch: Epoch index (0-based) just completed during fine-tuning.
            ``None`` when not inside a fine-tuning phase.
        optimizer_state: ``optimizer.state_dict()`` at checkpoint time.
        scheduler_state: ``scheduler.state_dict()`` at checkpoint time
            (``None`` if no scheduler is active).
        stopper_state: Dict with keys ``'best'``, ``'counter'``, and
            ``'mode'`` from the :class:`EarlyStopper`.
        best_model_state: ``model.state_dict()`` of the best weights seen
            so far during fine-tuning.  Restored on resume so that early
            stopping picks up from the correct historical best.

    Returns:
        The resolved checkpoint path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "phase": phase,
        "step_idx": step_idx,
        "stage": stage,
        "ft_epoch": ft_epoch,
        "model_state": model_state,
        "masks": masks,
        "results": results,
        "config": config,
        "elapsed_seconds": elapsed_seconds,
        "numpy_rng_state": numpy_rng_state or np.random.get_state(),
        "torch_rng_state": torch_rng_state or torch.random.get_rng_state(),
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "stopper_state": stopper_state,
        "best_model_state": best_model_state,
    }
    torch.save(ckpt, path)
    if verbose:
        stage_str = f", stage={stage}" if stage else ""
        ft_str = f", ft_epoch={ft_epoch}" if ft_epoch is not None else ""
        print(f"\n  Checkpoint saved → {path}  "
              f"(phase={phase}, step={step_idx}{stage_str}{ft_str}, "
              f"{elapsed_seconds:.0f}s elapsed)")
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
        _stage = ckpt.get('stage')
        if _stage:
            print(f"    stage          : {_stage}")
        _ft_epoch = ckpt.get('ft_epoch')
        if _ft_epoch is not None:
            print(f"    ft_epoch       : {_ft_epoch}")
        print(f"    has optimizer  : {'optimizer_state' in ckpt and ckpt['optimizer_state'] is not None}")
        print(f"    has scheduler  : {'scheduler_state' in ckpt and ckpt['scheduler_state'] is not None}")
        print(f"    has stopper    : {'stopper_state' in ckpt and ckpt['stopper_state'] is not None}")
        print(f"    elapsed        : {ckpt['elapsed_seconds']:.0f}s")
        n_phases = len(ckpt['results'].get('phases', []))
        print(f"    phases saved   : {n_phases}")

    return ckpt


# =========================================================================
# Main hybrid pruning entry point
# =========================================================================

def hybrid_pruning(
    model_name: str = "resnet20",
    dataset_name: str = "cifar10",
    num_classes: int = 10,
    target_sparsity: float = 0.9,
    # -- One-shot phase ---------------------------------------------------
    oneshot_ratio: Optional[float] = None,
    # -- Iterative phase --------------------------------------------------
    iterative_step: Optional[float] = None,
    # -- Initial training -------------------------------------------------
    initial_epochs: int = 160,
    initial_lr: float = 0.01,
    # -- Fine-tuning after one-shot prune ---------------------------------
    oneshot_finetune_max_epochs: int = 200,
    oneshot_finetune_patience: int = 200,
    # -- Fine-tuning after each iterative step ----------------------------
    iter_finetune_max_epochs: int = 10,
    iter_finetune_patience: int = 10,
    # -- Shared training hyper-parameters ---------------------------------
    batch_size: int = 128,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    # -- Misc -------------------------------------------------------------
    use_global_pruning: bool = True,
    seed: int = 42,
    device: str = "cuda",
    verbose: bool = True,
    # -- Adaptive oneshot ratio via Fisher trace ---------------------------
    use_fisher_adaptive_ratio: bool = True,
    fisher_num_batches: int = 10,
    fisher_sensitivity: float = 1.0,
    # -- Knowledge distillation (one-shot fine-tuning phase) ---------------
    use_kd: bool = True,
    kd_temperature: float = 4.0,
    kd_lambda: float = 0.5,
    # -- LR warmup at iterative-phase transition ---------------------------
    iter_lr_rewind_warmup_epochs: int = 5,
    # -- Checkpoint / resume support -----------------------------------
    checkpoint_dir: Optional[str] = "results",
    checkpoint_interval: int = 1,
    time_limit_seconds: Optional[float] = None,
    resume_from: Optional[str] = None,
    # -- Results saving ---------------------------------------------------
    save_dir: Optional[str] = "./results",
    # -- Performance optimizations ----------------------------------------
    use_compile: bool = False,
    use_ddp: bool = False,
    use_cuda_graphs: bool = False,
    profile_initial_training: bool = False,
    profile_output_dir: Optional[str] = "profiler_traces",
) -> Dict[str, Any]:
    """Run the complete hybrid pruning experiment.

    Implements the three-phase hybrid algorithm:

    1. **Initial training** – train from random init for
       ``initial_epochs`` epochs (SGD, ``initial_lr``).
    2. **Large one-shot prune + extended fine-tune** – prune
       ``oneshot_ratio * target_sparsity`` of weights by magnitude,
       then fine-tune with patience ``oneshot_finetune_patience``
       at ``initial_lr / 10``.
    3. **Iterative geometric pruning** – repeatedly prune
       ``iterative_step`` of remaining weights and fine-tune with
       short patience until overall sparsity reaches ``target_sparsity``.

    Args:
        model_name: Architecture (``'resnet20'``, ``'vgg16'``, …).
        dataset_name: Dataset (``'cifar10'``, ``'cifar100'``).
        num_classes: Number of output classes.
        target_sparsity: Target overall sparsity in [0, 1].
        oneshot_ratio: Fraction of *target* to prune in one shot
            (0.6–0.8 recommended).
        iterative_step: Fraction of *remaining* weights to prune per
            iterative step. **Default** is auto-selected:
            10 % when ``target < 0.8``, else 2 %.
        initial_epochs: Number of dense training epochs.
        initial_lr: Learning rate for initial training.
        oneshot_finetune_max_epochs: Maximum epochs for one-shot
            fine-tuning.
        oneshot_finetune_patience: Patience for one-shot fine-tuning
            early stopper.
        iter_finetune_max_epochs: Maximum epochs per iterative step.
        iter_finetune_patience: Patience per iterative step.
        batch_size: Training batch size.
        momentum: SGD momentum.
        weight_decay: L2 regularisation.
        use_global_pruning: ``True`` for global magnitude pruning,
            ``False`` for per-layer.
        seed: Random seed.
        device: ``'cuda'`` or ``'cpu'``.
        verbose: Show progress bars.
        use_fisher_adaptive_ratio: When ``True``, estimate the empirical
            Fisher Information trace on the fully-trained model and use it
            to automatically scale ``oneshot_ratio`` down (sharp landscape)
            or up (flat landscape) within [0.50, 0.80].  Adds one cheap
            forward-backward pass over ``fisher_num_batches`` mini-batches.
        fisher_num_batches: Mini-batches used to estimate the Fisher trace
            (default 10).
        fisher_sensitivity: Scale factor for the Fisher reference point
            (default 1.0).  Increase for larger models / datasets where
            typical gradient norms are higher.
        use_kd: When ``True``, the one-shot fine-tuning phase uses the
            pre-pruning model as a soft-label teacher.  The training loss
            becomes ``(1 - kd_lambda) * task_loss + kd_lambda * KL_loss``.
            Disabled by default to preserve original behaviour.
        kd_temperature: Softmax temperature for KD (default 4.0).
        kd_lambda: KD loss weight (default 0.5).
        iter_lr_rewind_warmup_epochs: Number of linear LR warm-up epochs
            at the very first iterative step (the one-shot → geometric
            transition).  Gives weights room to reorganise before the
            cosine decay resumes.  Set to 0 (default) to disable.
        checkpoint_dir: Directory for checkpoint files.  Defaults to
            ``'results'``.  When set, a checkpoint is written after
            initial training and after every ``checkpoint_interval``
            pruning steps.
        checkpoint_interval: Save a checkpoint every *N* pruning
            steps (default 1 = every step).
        time_limit_seconds: Optional wall-clock budget in seconds.
            When the remaining time drops below 120 s the current
            state is saved and the function returns early.
        resume_from: Path to a checkpoint file (``*.pt``) written by
            a previous run.  The experiment resumes from the saved
            phase and step.

    Returns:
        Dictionary containing:
        - ``'config'``: experiment configuration.
        - ``'initial_training'``: history from Phase 1.
        - ``'phases'``: list of per-phase dicts (one-shot + iterative).
        - ``'final_results'``: summary metrics.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Auto-select iterative step if not provided
    if iterative_step is None:
        iterative_step = 0.02 if target_sparsity > 0.8 else 0.10

    # Decide an internal oneshot_ratio to build the initial schedule.
    # If the user omitted `oneshot_ratio` we use a conservative default
    # for scheduling purposes (0.7). When `use_fisher_adaptive_ratio` is
    # enabled the adaptive computation later will overwrite this value.
    _initial_oneshot = oneshot_ratio if oneshot_ratio is not None else 0.7

    # Build schedule
    scheduler = HybridPruningScheduler(
        target=target_sparsity,
        oneshot_ratio=_initial_oneshot,
        iterative_step=iterative_step,
    )
    steps = scheduler.get_steps()

    config = {
        "algorithm": "hybrid",
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "target_sparsity": target_sparsity,
        # `oneshot_ratio` may be None (auto). Store the user's value
        # (or null) so experiments record intent; the scheduler used for
        # pruning steps is `_initial_oneshot` and may be updated later.
        "oneshot_ratio": oneshot_ratio,
        "iterative_step": iterative_step,
        "initial_epochs": initial_epochs,
        "initial_lr": initial_lr,
        "oneshot_finetune_max_epochs": oneshot_finetune_max_epochs,
        "oneshot_finetune_patience": oneshot_finetune_patience,
        "iter_finetune_max_epochs": iter_finetune_max_epochs,
        "iter_finetune_patience": iter_finetune_patience,
        "batch_size": batch_size,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "use_global_pruning": use_global_pruning,
        "seed": seed,
        "pruning_steps": steps,
        "num_pruning_phases": len(steps),
        "use_fisher_adaptive_ratio": use_fisher_adaptive_ratio,
        "use_kd": use_kd,
        "kd_temperature": kd_temperature,
        "kd_lambda": kd_lambda,
        "iter_lr_rewind_warmup_epochs": iter_lr_rewind_warmup_epochs,
    }

    results: Dict[str, Any] = {"config": config, "phases": []}

    # ------------------------------------------------------------------ #
    # Checkpoint / time-limit helpers
    # ------------------------------------------------------------------ #
    def _ckpt_path(tag: str = "hybrid_checkpoint") -> Path:
        d = checkpoint_dir or "."
        return Path(d) / f"{tag}.pt"

    def _elapsed() -> float:
        return prior_elapsed + (_time.time() - total_start)

    def _time_remaining() -> Optional[float]:
        if time_limit_seconds is None:
            return None
        return time_limit_seconds - _elapsed()

    def _should_stop_for_time(margin: float = 120.0) -> bool:
        remaining = _time_remaining()
        if remaining is None:
            return False
        return remaining < margin

    def _save_ckpt(phase: str, step_idx: int, **kwargs) -> Path:
        return save_hybrid_checkpoint(
            _ckpt_path(),
            phase=phase,
            step_idx=step_idx,
            model_state=raw_model.state_dict(),
            masks=masks,
            results=results,
            config=config,
            elapsed_seconds=_elapsed(),
            verbose=verbose,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Starting Hybrid Pruning Experiment")
    print(f"Model: {model_name}, Dataset: {dataset_name}")
    print(f"Target sparsity: {target_sparsity:.1%}")
    # `oneshot_ratio` may be None (auto). Use a safe display value
    display_oneshot = oneshot_ratio if oneshot_ratio is not None else 0.7
    print(
        f"One-shot prune: {display_oneshot * target_sparsity:.1%}  "
        f"({display_oneshot:.0%} of target)"
    )
    print(f"Iterative step: {iterative_step:.1%} of remaining  "
          f"({len(steps)-1} iterative steps)")
    print(f"Device: {device}")
    if time_limit_seconds is not None:
        print(f"Time limit: {time_limit_seconds:.0f}s "
              f"({time_limit_seconds/3600:.1f}h)")
    if resume_from is not None:
        print(f"Resuming from: {resume_from}")
    print(f"{'='*60}\n")

    total_start = _time.time()
    prior_elapsed = 0.0
    stopped_early = False
    start_step_idx = 0          # which pruning step to start from
    resume_finetune_only_step_idx = -1  # if >= 0, skip prune for this step
    resume_finetune_state: Optional[Dict[str, Any]] = None
    set_seed(seed)

    # cudnn.benchmark: fastest cuDNN auto-tuner for fixed-size (32×32) CIFAR inputs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Data — pin_memory for fast CPU→GPU DMA; persistent_workers avoids
    # re-spawning worker processes between epochs (big win on Kaggle/Colab)
    loaders = get_dataloaders(
        dataset_name,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,          # faster H2D transfers
        persistent_workers=True,  # no worker respawn between epochs
    )
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    # Model
    model = get_model(model_name, num_classes=num_classes).to(device)

    # --- torch.compile (PyTorch ≥ 2.0) -----------------------------------
    # Fuses operators for up to 1.5× throughput improvement.  Must be
    # applied *before* DDP wrapping so the compiled graph is distributed.
    if use_compile:
        if hasattr(torch, "compile"):
            print("  torch.compile() enabled — compiling model…")
            model = torch.compile(model)
        else:
            print("  WARNING: use_compile=True but torch.compile not available "
                  "(requires PyTorch ≥ 2.0). Skipping.")

    # --- Multi-GPU: DDP (preferred) or DataParallel (fallback) -----------
    if torch.cuda.device_count() > 1:
        if use_ddp:
            # DDP requires torch.distributed to be initialised by the caller
            # (e.g. via `torchrun` or `torch.multiprocessing.spawn`).
            # We initialise a default process group here as a convenience
            # when running in a plain `python` process with just env-vars set.
            if not dist.is_initialized():
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(backend=backend)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            model = model.to(device)
            model = DDP(model, device_ids=[local_rank],
                        output_device=local_rank)
            print(f"  DDP enabled on {torch.cuda.device_count()} GPUs "
                  f"(local_rank={local_rank}).")
        else:
            print(f"  DataParallel enabled on {torch.cuda.device_count()} GPUs. "
                  "Consider use_ddp=True for 20-50% higher throughput.")
            model = torch.nn.DataParallel(model)

    raw_model = (
        model.module
        if isinstance(model, (torch.nn.DataParallel, DDP))
        else model
    )
    # Unwrap torch.compile's wrapper to get the original nn.Module when
    # needed for weight access / mask application.
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    param_info = count_parameters(model)
    print(f"Model parameters: {param_info['total']:,}")

    criterion = nn.CrossEntropyLoss()

    # Initial masks (all ones)
    initial_weights = get_prunable_layers(raw_model)
    masks = create_initial_masks(initial_weights)
    initial_model_state_dict = copy.deepcopy(raw_model.state_dict())

    # ------------------------------------------------------------------ #
    # Resume from checkpoint (if provided)
    # ------------------------------------------------------------------ #
    skip_initial_training = False

    if resume_from is not None:
        ckpt = load_hybrid_checkpoint(resume_from, verbose=verbose)
        prior_elapsed = ckpt["elapsed_seconds"]
        raw_model.load_state_dict(ckpt["model_state"])
        model.to(device)
        masks = ckpt["masks"]
        results = ckpt["results"]
        # Restore RNG states
        np.random.set_state(ckpt["numpy_rng_state"])
        torch.random.set_rng_state(ckpt["torch_rng_state"])

        if ckpt["phase"] == "initial_training":
            # Initial training was completed; skip it and start pruning from step 0
            skip_initial_training = True
            start_step_idx = 0
        elif ckpt["phase"] == "pruning":
            skip_initial_training = True
            ckpt_stage = ckpt.get("stage")
            if ckpt_stage in ("oneshot_prune", "iterative_prune"):
                # Pruning done, fine-tuning not yet started for this step
                start_step_idx = ckpt["step_idx"]
                resume_finetune_only_step_idx = ckpt["step_idx"]
                resume_finetune_state = None  # start finetune from epoch 0
            elif ckpt_stage in ("oneshot_finetune", "iterative_finetune"):
                # Mid-fine-tuning: restore epoch, optimizer, scheduler and
                # early-stopper state so training continues exactly where it stopped
                start_step_idx = ckpt["step_idx"]
                resume_finetune_only_step_idx = ckpt["step_idx"]
                resume_finetune_state = {
                    "start_epoch": (ckpt.get("ft_epoch") or 0) + 1,
                    "optimizer_state": ckpt.get("optimizer_state"),
                    "scheduler_state": ckpt.get("scheduler_state"),
                    "stopper_state": ckpt.get("stopper_state"),
                    "initial_best_state": ckpt.get("best_model_state"),
                }
            else:
                # Old checkpoint format or post-finetune completed: skip to next step
                start_step_idx = ckpt["step_idx"] + 1
        elif ckpt["phase"] == "done":
            print("Checkpoint indicates experiment already finished.")
            skip_initial_training = True
            start_step_idx = len(steps)  # skip all pruning steps

        _resume_ft_msg = (
            f", resume_finetune_only_step_idx={resume_finetune_only_step_idx}"
            if resume_finetune_only_step_idx >= 0 else ""
        )
        print(f"  Resuming: skip_initial_training={skip_initial_training}, "
              f"start_step_idx={start_step_idx}{_resume_ft_msg}")

    # ------------------------------------------------------------------ #
    # Phase 1: Initial (dense) training
    # ------------------------------------------------------------------ #
    if not skip_initial_training:
        print(f"\n--- Phase 1: Initial training ({initial_epochs} epochs, "
              f"lr={initial_lr}) ---")

        optimizer = optim.SGD(
            model.parameters(),
            lr=initial_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=initial_epochs)

        from src.train import train_epochs

        # --- torch.profiler (optional) -----------------------------------
        # Wraps initial training to export a Chrome-trace JSON that reveals
        # bottlenecks (slow data loading, pruning sort, kernel overhead …).
        _prof_dir = profile_output_dir or checkpoint_dir or "."
        _should_profile = profile_initial_training and torch.cuda.is_available()
        if _should_profile:
            Path(_prof_dir).mkdir(parents=True, exist_ok=True)
            _prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1, warmup=1, active=3, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(_prof_dir)
                ),
                record_shapes=True,
                with_stack=False,
            )
            _prof.start()
            print(f"  Profiler active — traces will be saved to {_prof_dir}")
        else:
            _prof = None

        def _prof_step():
            if _prof is not None:
                _prof.step()

        # AMP for the initial dense phase (same Tensor Core speedup as fine-tuning)
        _use_amp_initial = torch.cuda.is_available()
        _scaler_initial = GradScaler() if _use_amp_initial else None

        init_history = train_epochs(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=initial_epochs,
            device=device,
            scheduler=lr_scheduler,
            verbose=verbose,
            epoch_callback=_prof_step,
            scaler=_scaler_initial,
            use_amp=_use_amp_initial,
        )

        if _prof is not None:
            _prof.stop()
            print(f"  Profiler stopped. Open {_prof_dir} in TensorBoard.")

        init_test_acc = init_history["final_test_acc"]
        results["initial_training"] = {
            "train_losses": init_history["train_losses"],
            "train_accs": init_history["train_accs"],
            "test_losses": init_history["test_losses"],
            "test_accs": init_history["test_accs"],
            "final_test_acc": init_test_acc,
        }
        print(f"Initial training done — test acc: {init_test_acc:.2f}%")

        # --- Snapshot the dense model as KD teacher -----------------------
        # Kept on CPU to avoid holding a second full model on the GPU.
        teacher_model: Optional[nn.Module] = None
        if use_kd:
            # Keep teacher on same device as student — passing a CUDA tensor
            # to a CPU model would raise RuntimeError. For ResNet20, the memory
            # overhead of a second copy is negligible.
            teacher_model = copy.deepcopy(raw_model).to(device)
            teacher_model.eval()
            print(f"  KD teacher snapshot saved ({device}).")

        # --- Adaptive oneshot_ratio via Fisher Information Trace ----------
        if use_fisher_adaptive_ratio:
            fisher_trace = _compute_fisher_trace(
                raw_model, train_loader, device,
                num_batches=fisher_num_batches,
            )
            # Normalise trace to [0, 1] with a soft sigmoid mapping:
            # high Fisher → aggressive pruning is risky → lower oneshot_ratio
            # We use a simple linear clamp between 0.5 and 0.8, shifted by
            # the trace relative to a reference scale (1e-3 is typical for
            # well-trained ResNet/CIFAR models; adjust fisher_sensitivity to
            # tune for your architecture / dataset).
            ref = fisher_sensitivity * 1e-3
            ratio_raw = 0.8 - 0.3 * torch.sigmoid(
                torch.tensor(math.log(max(fisher_trace, 1e-12) / ref))
            ).item()
            oneshot_ratio = float(np.clip(ratio_raw, 0.50, 0.80))
            print(f"  Fisher trace: {fisher_trace:.6f}  →  "
                  f"adaptive oneshot_ratio = {oneshot_ratio:.3f}")
            # Rebuild the step schedule with the adapted ratio
            scheduler = HybridPruningScheduler(
                target=target_sparsity,
                oneshot_ratio=oneshot_ratio,
                iterative_step=iterative_step,
            )
            steps = scheduler.get_steps()
            config["oneshot_ratio"] = oneshot_ratio
            config["pruning_steps"] = steps
            config["num_pruning_phases"] = len(steps)
            config["fisher_trace"] = fisher_trace
            print(f"  Rebuilt schedule: {len(steps)-1} iterative steps.")


        # Checkpoint after initial training
        if checkpoint_dir is not None:
            _save_ckpt(phase="initial_training", step_idx=-1)

        # Time-limit check after initial training
        if _should_stop_for_time():
            print(f"\n  Time limit approaching ({_time_remaining():.0f}s left). "
                  "Stopping after initial training.")
            stopped_early = True
    else:
        init_test_acc = results.get("initial_training", {}).get(
            "final_test_acc", 0.0
        )
        from src.train import train_epochs  # ensure import is available
        # No teacher available when resuming; KD is skipped gracefully
        teacher_model = None

    # ------------------------------------------------------------------ #
    # Pruning loop (one-shot + iterative steps)
    # ------------------------------------------------------------------ #
    finetune_lr = initial_lr / 10.0  # reduced LR for fine-tuning

    for step_idx, prune_ratio in enumerate(steps):
        # Skip steps already completed (when resuming)
        if step_idx < start_step_idx:
            continue

        # Time-limit guard
        if stopped_early:
            break
        is_oneshot = step_idx == 0
        phase_label = "one-shot" if is_oneshot else f"iter-{step_idx}"
        ft_stage_name = "oneshot_finetune" if is_oneshot else "iterative_finetune"

        # Select fine-tuning hyper-parameters
        ft_max_epochs = (
            oneshot_finetune_max_epochs if is_oneshot else iter_finetune_max_epochs
        )
        ft_patience = (
            oneshot_finetune_patience if is_oneshot else iter_finetune_patience
        )

        # Determine if we should skip the prune for this step (resuming
        # mid-finetune or after a prune that was already saved).
        is_resume_finetune_only = (step_idx == resume_finetune_only_step_idx)

        if not is_resume_finetune_only:
            current_sparsity = get_overall_sparsity(masks)
            print(f"\n--- Phase {step_idx+2}: {phase_label} prune "
                  f"(ratio={prune_ratio:.4f}, "
                  f"current sparsity={current_sparsity:.2%}) ---")

            # Get weights for ranking
            trained_weights = get_prunable_layers(raw_model)

            # Prune
            if use_global_pruning:
                masks = prune_by_magnitude_global(prune_ratio, masks, trained_weights)
            else:
                percents = {k: prune_ratio for k in masks}
                masks = prune_by_percent(percents, masks, trained_weights)

            # Apply masks
            apply_masks_to_model(raw_model, masks)
            apply_fn = create_mask_apply_fn(raw_model)

            new_sparsity = get_overall_sparsity(masks)
            print(f"  Sparsity after prune: {new_sparsity:.2%}")

            # Save post-prune checkpoint (prune done, fine-tune not yet started)
            if checkpoint_dir is not None:
                _save_ckpt(
                    phase="pruning",
                    step_idx=step_idx,
                    stage="oneshot_prune" if is_oneshot else "iterative_prune",
                )
        else:
            # Masks were already applied when the checkpoint was loaded;
            # just rebuild the apply_fn and re-enforce zeros.
            apply_fn = create_mask_apply_fn(raw_model)
            apply_masks_to_model(raw_model, masks)
            new_sparsity = get_overall_sparsity(masks)
            print(f"\n--- Phase {step_idx+2}: {phase_label} (resuming fine-tuning, "
                  f"sparsity={new_sparsity:.2%}) ---")

        # One-shot phase: optionally use KD teacher
        # Iterative phase: LR warmup at transition (step_idx == 1 only)
        ft_teacher = teacher_model if (is_oneshot and use_kd) else None
        ft_warmup = iter_lr_rewind_warmup_epochs if (not is_oneshot and step_idx == 1) else 0

        if ft_teacher is not None:
            print(f"  KD active (T={kd_temperature}, λ={kd_lambda})")
        if ft_warmup > 0:
            print(f"  LR warmup for first {ft_warmup} epochs (iterative transition)")

        # Determine the per-epoch checkpoint callback for mid-finetune saves.
        # Default args capture the *current* step_idx and stage name so the
        # closure stays correct across loop iterations.
        def _on_epoch_end(ep, opt, sched, stopper_obj, best_s,
                          _step=step_idx, _stage=ft_stage_name):
            if (checkpoint_dir is not None
                    and checkpoint_interval > 0
                    and (ep + 1) % checkpoint_interval == 0):
                _save_ckpt(
                    phase="pruning",
                    step_idx=_step,
                    stage=_stage,
                    ft_epoch=ep,
                    optimizer_state=opt.state_dict(),
                    scheduler_state=sched.state_dict() if sched is not None else None,
                    stopper_state={
                        "best": stopper_obj.best,
                        "counter": stopper_obj.counter,
                        "mode": stopper_obj.mode,
                    },
                    best_model_state=best_s,
                )

        # Unpack resume state for this step's finetune (if any)
        _ft_resume = resume_finetune_state if is_resume_finetune_only else None
        _ft_start_epoch = _ft_resume["start_epoch"] if _ft_resume else 0
        _ft_opt_state = _ft_resume.get("optimizer_state") if _ft_resume else None
        _ft_sched_state = _ft_resume.get("scheduler_state") if _ft_resume else None
        _ft_stopper_state = _ft_resume.get("stopper_state") if _ft_resume else None
        _ft_init_best = _ft_resume.get("initial_best_state") if _ft_resume else None

        # Fine-tune with early stopping
        ft_history = _finetune(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            masks=masks,
            apply_mask_fn=apply_fn,
            max_epochs=ft_max_epochs,
            lr=finetune_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            patience=ft_patience,
            scheduler_type="cosine",
            verbose=verbose,
            desc=f"FT {phase_label}",
            teacher_model=ft_teacher,
            kd_temperature=kd_temperature,
            kd_lambda=kd_lambda,
            lr_warmup_epochs=ft_warmup,
            use_cuda_graphs=use_cuda_graphs,
            start_epoch=_ft_start_epoch,
            optimizer_state=_ft_opt_state,
            scheduler_state=_ft_sched_state,
            stopper_state=_ft_stopper_state,
            initial_best_state=_ft_init_best,
            on_epoch_end=_on_epoch_end,
        )

        _, phase_test_acc = evaluate(model, test_loader, criterion, device)

        phase_result = {
            "step": step_idx,
            "label": phase_label,
            "prune_ratio": prune_ratio,
            "sparsity_after_prune": new_sparsity,
            "finetune_epochs_run": ft_history["epochs_run"],
            "finetune_max_epochs": ft_max_epochs,
            "finetune_patience": ft_patience,
            "best_test_acc": ft_history["best_test_acc"],
            "final_test_acc": phase_test_acc,
            "train_losses": ft_history["train_losses"],
            "train_accs": ft_history["train_accs"],
            "test_losses": ft_history["test_losses"],
            "test_accs": ft_history["test_accs"],
            "layer_sparsities": get_sparsity(masks),
        }
        results["phases"].append(phase_result)

        print(f"  FT done ({ft_history['epochs_run']} epochs) — "
              f"best acc: {ft_history['best_test_acc']:.2f}%, "
              f"final acc: {phase_test_acc:.2f}%")

        # Post-step checkpoint (step complete; no optimizer state needed)
        if (checkpoint_dir is not None
                and checkpoint_interval > 0
                and (step_idx + 1) % checkpoint_interval == 0):
            _save_ckpt(phase="pruning", step_idx=step_idx)

        # Time-limit check
        if _should_stop_for_time():
            remaining = _time_remaining()
            _save_ckpt(phase="pruning", step_idx=step_idx)
            print(f"\n  Time limit approaching ({remaining:.0f}s left). "
                  f"Checkpoint saved at step {step_idx}.")
            stopped_early = True
            break

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    results["masks"] = masks

    total_time = _elapsed()
    final_sparsity = get_overall_sparsity(masks)
    _, final_test_acc = evaluate(model, test_loader, criterion, device)
    best_phase_acc = (
        max(p["best_test_acc"] for p in results["phases"])
        if results["phases"]
        else 0.0
    )

    efficiency_metrics = compute_efficiency_metrics(
        model_name=model_name,
        num_classes=num_classes,
        dataset_name=dataset_name,
        device=device,
        final_state_dict=raw_model.state_dict(),
        masks=results["masks"],
        mask_applier=apply_masks_to_model,
    )

    results["final_results"] = {
        "final_sparsity": final_sparsity,
        "overall_sparsity": final_sparsity,
        "final_test_accuracy": final_test_acc,
        "final_train_accuracy": (
            results["phases"][-1]["train_accs"][-1]
            if results["phases"] and results["phases"][-1].get("train_accs")
            else (
                results.get("initial_training", {}).get("train_accs", [])[-1]
                if results.get("initial_training", {}).get("train_accs")
                else None
            )
        ),
        "best_phase_test_accuracy": best_phase_acc,
        "initial_test_accuracy": init_test_acc,
        "total_time_seconds": total_time,
        "training_computational_cost_seconds": total_time,
        "stopped_early": stopped_early,
        **efficiency_metrics,
    }

    # Final checkpoint
    if checkpoint_dir is not None and not stopped_early:
        _save_ckpt(phase="done", step_idx=len(steps) - 1)

    status = "INTERRUPTED (checkpoint saved)" if stopped_early else "Complete"
    print(f"\n{'='*60}")
    print(f"Hybrid Pruning {status}")
    print(f"  Final sparsity : {final_sparsity:.2%}")
    print(f"  Final test acc : {final_test_acc:.2f}%")
    print(f"  Best phase acc : {best_phase_acc:.2f}%")
    print(f"  Initial acc    : {init_test_acc:.2f}%")
    print(f"  Total time     : {total_time:.1f}s")
    print(f"{'='*60}")

    # Attach model & masks for downstream usage
    results["train_history"] = {
        "initial_training": copy.deepcopy(results.get("initial_training", {})),
        "phases": copy.deepcopy(results.get("phases", [])),
    }
    results["model"] = model
    results["masks"] = masks
    results["initial_model_state_dict"] = initial_model_state_dict

    # Save to IMP-style directory format
    def make_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return obj

    def save_summary_csv(path, results_dict):
        """Save epoch-by-epoch metrics in the canonical summary format."""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['phase', 'epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

            phase_items = []
            if 'initial_training' in results_dict:
                phase_items.append(('initial_training', results_dict['initial_training']))
            phase_items.extend(
                (phase.get('label', f"phase_{idx}"), phase)
                for idx, phase in enumerate(results_dict.get('phases', []))
            )

            for phase_name, phase in phase_items:
                train_losses = phase.get('train_losses', [])
                train_accs = phase.get('train_accs', [])
                test_losses = phase.get('test_losses', [])
                test_accs = phase.get('test_accs', [])
                max_epochs = max(
                    len(train_losses),
                    len(train_accs),
                    len(test_losses),
                    len(test_accs),
                )
                for epoch_idx in range(max_epochs):
                    writer.writerow([
                        phase_name,
                        epoch_idx + 1,
                        f"{train_losses[epoch_idx]:.4f}" if epoch_idx < len(train_losses) else '',
                        f"{train_accs[epoch_idx]:.2f}" if epoch_idx < len(train_accs) else '',
                        f"{test_losses[epoch_idx]:.4f}" if epoch_idx < len(test_losses) else '',
                        f"{test_accs[epoch_idx]:.2f}" if epoch_idx < len(test_accs) else '',
                    ])

    # Construct the save directory
    base_results_dir = Path(checkpoint_dir).parent.parent if checkpoint_dir else Path("./results")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"hybrid_improve_{model_name}_{dataset_name}_s{target_sparsity}_seed{seed}"
    result_dir = base_results_dir / "hybrid_improve" / experiment_name / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    results_to_serialize = {
        k: v for k, v in results.items()
        if k not in ["model", "masks", "initial_model_state_dict"]
    }
    results_serializable = make_serializable(results_to_serialize)
    results_path = result_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    # Save summary CSV
    summary_path = result_dir / "summary.csv"
    save_summary_csv(summary_path, results)

    # Save final model state and masks
    if "initial_model_state_dict" in results:
        initial_model_path = result_dir / "initial_model.pt"
        torch.save(results["initial_model_state_dict"], initial_model_path)
    if "model" in results:
        final_model_path = result_dir / "final_model.pt"
        torch.save(results["model"].state_dict(), final_model_path)
    if "masks" in results:
        masks_path = result_dir / "final_masks.pt"
        masks_torch = {k: torch.from_numpy(v) for k, v in results["masks"].items()}
        torch.save(masks_torch, masks_path)

    print(f"\nIMP-style results saved to: {result_dir}")

    return results
