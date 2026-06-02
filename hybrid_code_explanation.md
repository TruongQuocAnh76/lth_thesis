# Hybrid Pruning — Code ⇄ Methodology Mapping

This note maps the Hybrid pseudocode in methodology to how it is actually implemented in [src/hybrid.py](src/hybrid.py#L461).

- Methodology section: [Khoá_luận_tốt_nghiệp/chapters/methodology.tex](Khoá_luận_tốt_nghiệp/chapters/methodology.tex) (Hybrid subsection, Algorithm `algo:hybrid-main`, `algo:hybrid-prune`, `algo:hybrid-finetune`).
- Implementation entry point: `hybrid_pruning(...)` in [src/hybrid.py](src/hybrid.py#L461).

## Quick citations (code locations)

- **Initial masks:** [src/hybrid.py](src/hybrid.py#L699) — `initial_weights = get_prunable_layers(raw_model)`; [src/hybrid.py](src/hybrid.py#L700) — `masks = create_initial_masks(initial_weights)`
- **One-shot amount p_k = alpha * p:** [src/hybrid.py](src/hybrid.py#L212) — `oneshot_amount = round(self.oneshot_ratio * self.target, 8)`
- **Iterative schedule:** [src/hybrid.py](src/hybrid.py#L221) — schedule loop computing `absolute_prune` / `ratio_of_remaining`; default policy: [src/hybrid.py](src/hybrid.py#L585) — `iterative_step = 0.02 if target_sparsity > 0.8 else 0.10`
- **Build schedule / steps:** [src/hybrid.py](src/hybrid.py#L593) — `steps = scheduler.get_steps()`
- **Pruning loop (one-shot → iterative):** [src/hybrid.py](src/hybrid.py#L809) — `for step_idx, prune_ratio in enumerate(steps):`; one-shot detection: [src/hybrid.py](src/hybrid.py#L817) — `is_oneshot = step_idx == 0`
- **Get weights & magnitude pruning:** [src/hybrid.py](src/hybrid.py#L847) — `trained_weights = get_prunable_layers(raw_model)`; global prune: [src/hybrid.py](src/hybrid.py#L851) — `prune_by_magnitude_global(...)`; per-layer: [src/hybrid.py](src/hybrid.py#L854) — `prune_by_percent(...)`
- **Apply masks to model:** [src/hybrid.py](src/hybrid.py#L857) — `apply_masks_to_model(raw_model, masks)`
- **Fine-tune wrapper:** [src/hybrid.py](src/hybrid.py#L253) — `_finetune(...)` delegates epochs to `train_epoch(..., masks=masks, apply_mask_fn=apply_mask_fn)` ([src/hybrid.py](src/hybrid.py#L331)–[src/hybrid.py](src/hybrid.py#L333))
- **Patience-based early stopping:** [src/hybrid.py](src/hybrid.py#L59) — `class EarlyStopper`; `_finetune` uses `stopper.step(test_acc)` ([src/hybrid.py](src/hybrid.py#L358))
- **Best-state tracking & restore:** [src/hybrid.py](src/hybrid.py#L349) — `best_state = copy.deepcopy(...)`; restore: [src/hybrid.py](src/hybrid.py#L364) — `model.load_state_dict(best_state)`
- **Fine-tune LR (eta = eta0/10):** [src/hybrid.py](src/hybrid.py#L807) — `finetune_lr = initial_lr / 10.0`
- **Checkpoint / resume helpers:** [src/hybrid.py](src/hybrid.py#L375) — `save_hybrid_checkpoint`; [src/hybrid.py](src/hybrid.py#L428) — `load_hybrid_checkpoint`
- **Warm-start from dense run (skip initial training):** [src/hybrid.py](src/hybrid.py#L708) — `if dense_run_dir is not None: ...`


## 1) Main loop pseudocode (`algo:hybrid-main`) vs code

### Pseudocode: initialize all-one mask
- **Methodology:** initialize `m <- 1^W`.
- **Code:** create initial masks from prunable layers:
  - `initial_weights = get_prunable_layers(raw_model)`
  - `masks = create_initial_masks(initial_weights)`
  - Reference: `src/hybrid.py#L699`

### Pseudocode: compute one-shot amount `p_k = alpha * p`
- **Methodology:** `p_k <- alpha * p`.
- **Code:** scheduler phase-1 amount is exactly `oneshot_ratio * target`:
  - `oneshot_amount = round(self.oneshot_ratio * self.target, 8)`
  - Reference: `src/hybrid.py#L212`

### Pseudocode: geometric iterative schedule
- **Methodology:** iterative pruning with small ratio on remaining weights until target sparsity is reached.
- **Code:** schedule generation loop:
  - `while self.target - pruned_so_far > 1e-4:`
  - `absolute_prune = min(iterative_step * remaining, still_needed)`
  - `ratio_of_remaining = absolute_prune / remaining`
  - References: `src/hybrid.py#L221`, `src/hybrid.py#L232`

### Pseudocode: recommendation for `p_i`
- **Methodology:** when `p < 0.8`, use about `0.10`; else around `0.02`.
- **Code:** auto policy matches this:
  - `iterative_step = 0.02 if target_sparsity > 0.8 else 0.10`
  - Reference: `src/hybrid.py#L585`

### Pseudocode: phase 1 then phase 2 loop
- **Methodology:** one-shot prune + long fine-tune, then iterative prune + short fine-tune until target reached.
- **Code:**
  - Build schedule: `steps = scheduler.get_steps()` (`steps[0]` is one-shot, rest iterative)
  - Loop: `for step_idx, prune_ratio in enumerate(steps):`
  - One-shot detection: `is_oneshot = step_idx == 0`
  - References: `src/hybrid.py#L593`, `src/hybrid.py#L809`, `src/hybrid.py#L817`

### Pseudocode: stop when `s(m) >= p`
- **Methodology:** `while s(m) < p` continue.
- **Code implementation detail:** stop condition is encoded in the precomputed schedule (`get_steps`) instead of runtime `while s(m) < p`; schedule terminates once `pruned_so_far` reaches `target`.
  - References: `src/hybrid.py#L221`, `src/hybrid.py#L809`

## 2) `Prune(theta, m, r)` pseudocode (`algo:hybrid-prune`) vs code

### Pseudocode: select alive weights, rank by `|theta|`, prune lowest `r`
- **Methodology:** sort alive weights by magnitude ascending, set lowest `k` to zero in mask.
- **Code:**
  - read current trained weights: `trained_weights = get_prunable_layers(raw_model)`
  - global magnitude pruning: `masks = prune_by_magnitude_global(prune_ratio, masks, trained_weights)`
  - or per-layer: `masks = prune_by_percent(...)`
  - apply updated mask: `apply_masks_to_model(raw_model, masks)`
  - References: `src/hybrid.py#L847`, `src/hybrid.py#L851`, `src/hybrid.py#L854`, `src/hybrid.py#L857`

### Magnitude criterion equation `score(theta_i)=|theta_i|`
- **Methodology:** Eq. magnitude score.
- **Code:** implemented inside pruning utility calls (`prune_by_magnitude_global` / `prune_by_percent`) imported from `src.pruning` and invoked in `hybrid_pruning`.
  - Invocation refs: `src/hybrid.py#L40`, `src/hybrid.py#L41`, `src/hybrid.py#L851`, `src/hybrid.py#L854`

## 3) `FineTune(...)` pseudocode (`algo:hybrid-finetune`) vs code

### Pseudocode: SGD with mask reapplied after updates
- **Methodology:** update by SGD with masked objective, then re-apply mask each step.
- **Code:** `_finetune(...)` delegates each epoch to `train_epoch(..., masks=masks, apply_mask_fn=apply_mask_fn)`.
  This passes the mask reapplication closure to training so masked weights stay zero after optimizer updates.
  - References: `src/hybrid.py#L253`, `src/hybrid.py#L331`, `src/hybrid.py#L333`

### Pseudocode: patience-based early stopping with threshold `delta`
- **Methodology:** stop when validation accuracy has no significant improvement for `rho` epochs.
- **Code:** `EarlyStopper` with `min_delta` and `patience`; `_finetune` uses `stopper.step(test_acc)` each epoch.
  - References: `src/hybrid.py#L59`, `src/hybrid.py#L88`, `src/hybrid.py#L321`, `src/hybrid.py#L358`

### Pseudocode: keep best `theta*`
- **Methodology:** track best validation accuracy and return best parameters.
- **Code:** `_finetune` caches `best_state` and restores it at end.
  - References: `src/hybrid.py#L349`, `src/hybrid.py#L364`

### Pseudocode: one-shot fine-tune uses larger patience, iterative uses smaller patience
- **Methodology:** `rho_k` large (around 200), `rho_i = rho_k/20` smaller.
- **Code:** separate arguments and branch by phase:
  - defaults: `oneshot_finetune_patience=50`, `iter_finetune_patience=10`
  - phase selection in loop by `is_oneshot`
  - References: `src/hybrid.py#L475`, `src/hybrid.py#L478`, `src/hybrid.py#L824`, `src/hybrid.py#L825`

### Pseudocode: reduced LR during fine-tuning (`eta = eta0/10`)
- **Methodology:** fine-tune with reduced LR.
- **Code:** `finetune_lr = initial_lr / 10.0`.
  - Reference: `src/hybrid.py#L807`

## 4) Practical implementation details beyond pseudocode

- Supports resume/checkpoint/time-limit (`save_hybrid_checkpoint`, `load_hybrid_checkpoint`), which are engineering additions not shown in the mathematical pseudocode.
  - References: `src/hybrid.py#L375`, `src/hybrid.py#L428`

- Supports warm-start from an existing dense run (`dense_run_dir`) to skip phase-1 dense training.
  - Reference: `src/hybrid.py#L708`

## 5) Fidelity summary

The implementation is faithful to the Hybrid methodology:
- uses magnitude-based pruning,
- performs a large one-shot prune followed by geometric iterative pruning,
- uses patience-based fine-tuning with best-checkpoint restore,
- applies reduced fine-tune learning rate,
- and reaches target sparsity via a schedule equivalent to the pseudocode stop condition.

The main difference is **engineering form**: instead of a runtime `while s(m) < p`, it precomputes a step schedule that guarantees convergence to the target sparsity.
