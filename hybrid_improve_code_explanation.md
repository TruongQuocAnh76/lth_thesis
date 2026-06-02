# Improved Hybrid Pruning — Code ⇄ Methodology Mapping

This file maps the Improved Hybrid pseudocode in [Khoá_luận_tốt_nghiệp/chapters/contribution.tex](Khoá_luận_tốt_nghiệp/chapters/contribution.tex#L2-L150) to the implementation in [src/hybrid_improve.py](src/hybrid_improve.py#L761).

## Quick citations (code locations)

- **Entry point:** [src/hybrid_improve.py](src/hybrid_improve.py#L761) — `hybrid_pruning(...)`
- **Scheduler (one-shot + iterative):** [src/hybrid_improve.py](src/hybrid_improve.py#L236) — `HybridPruningScheduler.get_steps()`; initial build: [src/hybrid_improve.py](src/hybrid_improve.py#L937) — `steps = scheduler.get_steps()`
- **Fisher trace (adaptive r_k):** [src/hybrid_improve.py](src/hybrid_improve.py#L286) — `_compute_fisher_trace(...)`; adaptive rebuild: [src/hybrid_improve.py](src/hybrid_improve.py#L1302) / [src/hybrid_improve.py](src/hybrid_improve.py#L1348)
- **KD teacher snapshot:** [src/hybrid_improve.py](src/hybrid_improve.py#L1273) — `teacher_model = copy.deepcopy(raw_model)`
- **KD hook / criterion:** [src/hybrid_improve.py](src/hybrid_improve.py#L538) — `_hook = model.register_forward_hook(_capture_kd)` (KD integrated into `_finetune`)
- **Pruning loop (one-shot → iterative):** [src/hybrid_improve.py](src/hybrid_improve.py#L1360) — `for step_idx, prune_ratio in enumerate(steps):`
- **Global magnitude prune call:** [src/hybrid_improve.py](src/hybrid_improve.py#L1408) — `prune_by_magnitude_global(...)` (or per-layer at [src/hybrid_improve.py](src/hybrid_improve.py#L1411))
- **Apply masks to model:** [src/hybrid_improve.py](src/hybrid_improve.py#L1414) — `apply_masks_to_model(raw_model, masks)`
- **Fine-tune wrapper (KD / warmup / early-stop):** [src/hybrid_improve.py](src/hybrid_improve.py#L343) — `_finetune(...)` signature (see `teacher_model`, `kd_temperature`, `kd_lambda`, `lr_warmup_epochs` args)
- **LR warmup implementation:** [src/hybrid_improve.py](src/hybrid_improve.py#L549-L552) — linear warmup scale inside `_finetune`
- **Compute sparsity / bookkeeping:** [src/hybrid_improve.py](src/hybrid_improve.py#L1111) — `create_initial_masks(...)`; [src/hybrid_improve.py](src/hybrid_improve.py#L1398) — `get_overall_sparsity(masks)`; final sparsity: [src/hybrid_improve.py](src/hybrid_improve.py#L1563)
- **Checkpoint helpers:** [src/hybrid_improve.py](src/hybrid_improve.py#L637) — `save_hybrid_checkpoint`; [src/hybrid_improve.py](src/hybrid_improve.py#L719) — `load_hybrid_checkpoint`

---

## Mapping: Contribution.tex → Implementation

**Fisher-based adaptive one-shot ratio (Eq. 1 / Eq. adaptive_rk)**
- Methodology: estimate empirical Fisher trace and map to $r_k$ via a sigmoid mapping.
- Code: `_compute_fisher_trace(model, train_loader, device, num_batches)` computes the squared-gradient trace ([src/hybrid_improve.py](src/hybrid_improve.py#L286)); the trace is converted to `oneshot_ratio` and clamped to [0.50, 0.80], then the scheduler is rebuilt (`scheduler = HybridPruningScheduler(...); steps = scheduler.get_steps()`) — see adaptive rebuild at [src/hybrid_improve.py](src/hybrid_improve.py#L1302) and [src/hybrid_improve.py](src/hybrid_improve.py#L1348).

**KD loss during one-shot fine-tune (Eq. kd_loss / kd_kl)**
- Methodology: snapshot teacher $	heta_0$ before one-shot; after pruning use combined task + KD loss only for the one-shot phase.
- Code: teacher snapshot saved after initial training when `use_kd` is true (`teacher_model = copy.deepcopy(raw_model)`) — [src/hybrid_improve.py](src/hybrid_improve.py#L1273). `_finetune` accepts `teacher_model`, `kd_temperature`, and `kd_lambda` and registers a forward hook to compute the KD term (`_hook = model.register_forward_hook(_capture_kd)`) — [src/hybrid_improve.py](src/hybrid_improve.py#L538). In the main pruning loop the teacher is passed only for the one-shot step: `ft_teacher = teacher_model if (is_oneshot and use_kd) else None` — [src/hybrid_improve.py](src/hybrid_improve.py#L1438).

**LR warm-up at iterative transition (Eq. lr_warmup)**
- Methodology: for the first iterative step, perform `E_w` linear warmup epochs then resume cosine decay.
- Code: `_finetune` exposes `lr_warmup_epochs` (signature) — [src/hybrid_improve.py](src/hybrid_improve.py#L343); linear warmup applied per-epoch via `warmup_scale = (epoch + 1) / lr_warmup_epochs` — [src/hybrid_improve.py](src/hybrid_improve.py#L549-L552). The loop sets `ft_warmup = iter_lr_rewind_warmup_epochs if (not is_oneshot and step_idx == 1) else 0` so warmup runs only at the one-shot→iter transition — [src/hybrid_improve.py](src/hybrid_improve.py#L1439).

**Prune + Fine-tune loop (Phase 2 & 3 in pseudocode)**
- Methodology: one-shot prune (`r_k * p`), long fine-tune with KD, then geometric iterative pruning with short fine-tunes.
- Code: schedule built by `HybridPruningScheduler.get_steps()` ([src/hybrid_improve.py](src/hybrid_improve.py#L236), built at [src/hybrid_improve.py](src/hybrid_improve.py#L937)); loop runs at [src/hybrid_improve.py](src/hybrid_improve.py#L1360). Pruning uses global magnitude pruning (`prune_by_magnitude_global(prune_ratio, masks, trained_weights)`) — [src/hybrid_improve.py](src/hybrid_improve.py#L1408). Masks are applied via `apply_masks_to_model(raw_model, masks)` — [src/hybrid_improve.py](src/hybrid_improve.py#L1414). Fine-tuning with KD/warmup/early-stop is delegated to `_finetune(...)` — [src/hybrid_improve.py](src/hybrid_improve.py#L1494).

**Engineering details**
- Checkpointing with intra-step resume (stage, ft_epoch, optimizer/scheduler/stopper state): `save_hybrid_checkpoint(...)` and `load_hybrid_checkpoint(...)` — [src/hybrid_improve.py](src/hybrid_improve.py#L637) / [src/hybrid_improve.py](src/hybrid_improve.py#L719).
- Warm-start from an existing dense run is supported (`dense_run_dir`) and the code recomputes `oneshot_ratio` and `teacher_model` when resuming — see the resume branches at [src/hybrid_improve.py](src/hybrid_improve.py#L1325-L1332).

---

If you want, I can:
- Add inline citations inside the LaTeX `contribution.tex` near the Improved-Hybrid subsection, or
- Normalize links in the other algorithm MD files the same way.

Which of those should I do next?
