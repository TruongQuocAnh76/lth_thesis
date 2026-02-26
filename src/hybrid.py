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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Any, Optional, Callable
from tqdm import tqdm

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

    Returns:
        Dictionary with fine-tuning history and best accuracy.
    """
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=max_epochs)
        if scheduler_type == "cosine"
        else None
    )
    stopper = EarlyStopper(patience=patience, mode="max")

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
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            masks=masks, apply_mask_fn=apply_mask_fn,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        if scheduler is not None:
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
# Main hybrid pruning entry point
# =========================================================================

def hybrid_pruning(
    model_name: str = "resnet20",
    dataset_name: str = "cifar10",
    num_classes: int = 10,
    target_sparsity: float = 0.9,
    # -- One-shot phase ---------------------------------------------------
    oneshot_ratio: float = 0.7,
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

    Returns:
        Dictionary containing:
        - ``'config'``: experiment configuration.
        - ``'initial_training'``: history from Phase 1.
        - ``'phases'``: list of per-phase dicts (one-shot + iterative).
        - ``'final_results'``: summary metrics.
    """
    import time as _time

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Auto-select iterative step if not provided
    if iterative_step is None:
        iterative_step = 0.02 if target_sparsity > 0.8 else 0.10

    # Build schedule
    scheduler = HybridPruningScheduler(
        target=target_sparsity,
        oneshot_ratio=oneshot_ratio,
        iterative_step=iterative_step,
    )
    steps = scheduler.get_steps()

    config = {
        "algorithm": "hybrid",
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "target_sparsity": target_sparsity,
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
    }

    results: Dict[str, Any] = {"config": config, "phases": []}

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Starting Hybrid Pruning Experiment")
    print(f"Model: {model_name}, Dataset: {dataset_name}")
    print(f"Target sparsity: {target_sparsity:.1%}")
    print(f"One-shot prune: {oneshot_ratio*target_sparsity:.1%}  "
          f"({oneshot_ratio:.0%} of target)")
    print(f"Iterative step: {iterative_step:.1%} of remaining  "
          f"({len(steps)-1} iterative steps)")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    total_start = _time.time()
    set_seed(seed)

    # Data
    loaders = get_dataloaders(dataset_name, batch_size=batch_size, num_workers=4)
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    # Model
    model = get_model(model_name, num_classes=num_classes).to(device)
    param_info = count_parameters(model)
    print(f"Model parameters: {param_info['total']:,}")

    criterion = nn.CrossEntropyLoss()

    # Initial masks (all ones)
    initial_weights = get_prunable_layers(model)
    masks = create_initial_masks(initial_weights)

    # ------------------------------------------------------------------ #
    # Phase 1: Initial (dense) training
    # ------------------------------------------------------------------ #
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
    )
    init_test_acc = init_history["final_test_acc"]
    results["initial_training"] = {
        "train_losses": init_history["train_losses"],
        "train_accs": init_history["train_accs"],
        "test_losses": init_history["test_losses"],
        "test_accs": init_history["test_accs"],
        "final_test_acc": init_test_acc,
    }
    print(f"Initial training done — test acc: {init_test_acc:.2f}%")

    # ------------------------------------------------------------------ #
    # Pruning loop (one-shot + iterative steps)
    # ------------------------------------------------------------------ #
    finetune_lr = initial_lr / 10.0  # reduced LR for fine-tuning

    for step_idx, prune_ratio in enumerate(steps):
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
        trained_weights = get_prunable_layers(model)

        # Prune
        if use_global_pruning:
            masks = prune_by_magnitude_global(prune_ratio, masks, trained_weights)
        else:
            percents = {k: prune_ratio for k in masks}
            masks = prune_by_percent(percents, masks, trained_weights)

        # Apply masks
        apply_masks_to_model(model, masks)
        apply_fn = create_mask_apply_fn(model)

        new_sparsity = get_overall_sparsity(masks)
        print(f"  Sparsity after prune: {new_sparsity:.2%}")

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

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    total_time = _time.time() - total_start
    final_sparsity = get_overall_sparsity(masks)
    _, final_test_acc = evaluate(model, test_loader, criterion, device)
    best_phase_acc = max(p["best_test_acc"] for p in results["phases"])

    results["final_results"] = {
        "final_sparsity": final_sparsity,
        "final_test_accuracy": final_test_acc,
        "best_phase_test_accuracy": best_phase_acc,
        "initial_test_accuracy": init_test_acc,
        "total_time_seconds": total_time,
    }

    print(f"\n{'='*60}")
    print("Hybrid Pruning Complete")
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
