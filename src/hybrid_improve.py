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
    # --- CUDA Graphs (reduces kernel-launch overhead) ---------------------
    use_cuda_graphs: bool = False,
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
        and scaler is None  # AMP and graphs need extra care; skip for safety
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
    use_fisher_adaptive_ratio: bool = False,
    fisher_num_batches: int = 10,
    fisher_sensitivity: float = 1.0,
    # -- Knowledge distillation (one-shot fine-tuning phase) ---------------
    use_kd: bool = False,
    kd_temperature: float = 4.0,
    kd_lambda: float = 0.5,
    # -- LR warmup at iterative-phase transition ---------------------------
    iter_lr_rewind_warmup_epochs: int = 5,
    # -- Checkpoint / resume support -----------------------------------
    checkpoint_dir: Optional[str] = "results",
    checkpoint_interval: int = 1,
    time_limit_seconds: Optional[float] = None,
    resume_from: Optional[str] = None,
    # -- Performance optimizations ----------------------------------------
    use_compile: bool = False,
    use_ddp: bool = False,
    use_cuda_graphs: bool = False,
    profile_initial_training: bool = False,
    profile_output_dir: Optional[str] = None,
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

    def _save_ckpt(phase: str, step_idx: int) -> Path:
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
    start_step_idx = 0  # which pruning step to start from
    set_seed(seed)

    # Data
    loaders = get_dataloaders(dataset_name, batch_size=batch_size, num_workers=4)
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
            # Initial training was completed; skip it and start pruning
            skip_initial_training = True
            start_step_idx = 0
        elif ckpt["phase"] == "pruning":
            skip_initial_training = True
            start_step_idx = ckpt["step_idx"] + 1  # resume from next step
        elif ckpt["phase"] == "done":
            print("Checkpoint indicates experiment already finished.")
            skip_initial_training = True
            start_step_idx = len(steps)  # skip all pruning steps

        print(f"  Resuming: skip_initial_training={skip_initial_training}, "
              f"start_step_idx={start_step_idx}")

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
            teacher_model = copy.deepcopy(raw_model).cpu()
            teacher_model.eval()
            print("  KD teacher snapshot saved (CPU).")

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

        # Select fine-tuning hyper-parameters
        ft_max_epochs = (
            oneshot_finetune_max_epochs if is_oneshot else iter_finetune_max_epochs
        )
        ft_patience = (
            oneshot_finetune_patience if is_oneshot else iter_finetune_patience
        )

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
        if ft_teacher is not None:
            print(f"  KD active (T={kd_temperature}, λ={kd_lambda})")
        if ft_warmup > 0:
            print(f"  LR warmup for first {ft_warmup} epochs (iterative transition)")

        # Fine-tune with early stopping
        # One-shot phase: optionally use KD teacher
        # Iterative phase: LR warmup at transition (step_idx == 1 only)
        ft_teacher = teacher_model if (is_oneshot and use_kd) else None
        ft_warmup = iter_lr_rewind_warmup_epochs if (not is_oneshot and step_idx == 1) else 0

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

        # Periodic checkpoint
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
    total_time = _elapsed()
    final_sparsity = get_overall_sparsity(masks)
    _, final_test_acc = evaluate(model, test_loader, criterion, device)
    best_phase_acc = (
        max(p["best_test_acc"] for p in results["phases"])
        if results["phases"]
        else 0.0
    )

    results["final_results"] = {
        "final_sparsity": final_sparsity,
        "final_test_accuracy": final_test_acc,
        "best_phase_test_accuracy": best_phase_acc,
        "initial_test_accuracy": init_test_acc,
        "total_time_seconds": total_time,
        "stopped_early": stopped_early,
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
    results["model"] = model
    results["masks"] = masks

    return results