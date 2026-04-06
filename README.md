# Đề tài

1. **Adaptive Hybrid Pruning with Dynamic Switching** (Cải Tiến Pruning Hybrid Thích Ứng Để Tìm Vé Số May Mắn Trong Mạng Nơ-ron Sâu): cải tiến phương pháp hybrid oneshot - iterative pruning của Janusz et al. (2025) bằng cách tìm phương pháp giúp threshold chuyển từ oneshot sang iterative adaptive thay vì sử dụng 1 constant.
2. **Comparative Study of Pruning Methods Related to the Lottery Ticket Hypothesis** (So sánh các Phương pháp Pruning liên quan đến Giả thuyết Vé Số): thử nghiệm và so sánh <= 8 thuật toán pruning. (nếu cải tiến không khả quan)

# Usage


This project implements multiple pruning algorithms for the Lottery Ticket Hypothesis. You can run experiments using the unified command-line interface.

## Experiment Output Format

All experiment outputs (model checkpoints, masks, metrics, logs) are saved in a standardized format for every algorithm. For full details and example output, see [RESULTS_FORMAT.md](RESULTS_FORMAT.md).

**Key output files for each experiment:**
- `results.json`: Full config, training history, and all final metrics (including FLOPS, latency, training cost, etc.)
- `summary.csv`: Epoch-by-epoch training/test metrics
- `final_model.pt`: Final trained model weights
- `final_masks.pt`: Final pruning masks
- `initial_model.pt`: Initial model weights before pruning

All algorithms (IMP, Early-Bird, GraSP, SynFlow, GA, Hybrid, Hybrid-Improve) produce this unified output format for easy downstream analysis.

## Installation

```bash
pip install torch torchvision tqdm numpy
```

## Running Experiments


### 1. Dense Baseline (No Pruning)

Run a dense (no pruning) baseline for comparison:

```bash
python -m src.experiments --algorithm dense \
    --model resnet20 \
    --dataset cifar10 \
    --seed 42 \
    --epochs 160 \
    --batch_size 128 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --lr_gamma 0.1 \
    --device cuda
```

**Available Dense arguments:**
- `--epochs`: Number of training epochs (default: 160)
- `--batch_size`: Training batch size (default: 128)
- `--learning_rate`: Initial learning rate (default: 0.1)
- `--momentum`: SGD momentum (default: 0.9)
- `--weight_decay`: Weight decay (default: 5e-4)
- `--lr_milestones`: Epochs at which to reduce LR (default: [80, 120])
- `--lr_gamma`: LR decay factor (default: 0.1)
- `--device`: cuda or cpu (default: cuda)

### 2. Iterative Magnitude Pruning (IMP)


```bash
python -m src.experiments --algorithm imp \
    --model resnet20 \
    --dataset cifar10 \
    --seed 42 \
    --target_sparsity 0.9 \
    --num_iterations 10 \
    --epochs_per_iteration 160 \
    --batch_size 128 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --use_global_pruning True \
    --warmup_epochs 5 \
    --device cuda
```

**Available IMP arguments:**
- `--target_sparsity`: Target final sparsity (default: 0.9)
- `--num_iterations`: Number of pruning iterations (default: 10)
- `--epochs_per_iteration`: Training epochs per iteration (default: 160)
- `--use_global_pruning`: Use global magnitude pruning (default: True)
- `--warmup_epochs`: Warmup epochs (default: 5)

IMP implementation notes:
- LR scheduler uses `MultiStepLR` with milestones at 50% and 75% of `epochs_per_iteration` (`gamma=0.1`), matching paper-style IMP behavior.
- For each iteration, pruning decisions are made from the best-validation checkpoint in that iteration.
- Reported `test_accuracy` in iteration history remains the final-epoch value for backward compatibility; `best_test_accuracy` stores the best-epoch value.

### 2. Early-Bird (Channel Pruning with BN γ)


#### VGG Models:
```bash
python -m src.experiments --algorithm earlybird \
    --model vgg16 \
    --dataset cifar10 \
    --seed 42 \
    --target_sparsity 0.5 \
    --search_epochs 160 \
    --finetune_epochs 160 \
    --batch_size 256 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --l1_coef 1e-4 \
    --distance_threshold 0.1 \
    --patience 5 \
    --min_search_epochs 10 \
    --pruning_method global \
    --device cuda
```


#### ResNet (Block-wise Pruning):
```bash
python -m src.experiments --algorithm earlybird_resnet \
    --model resnet20 \
    --dataset cifar10 \
    --seed 42 \
    --target_sparsity 0.5 \
    --total_epochs 160 \
    --l1_coef 1e-4 \
    --batch_size 128 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --distance_threshold 0.1 \
    --patience 5 \
    --min_search_epochs 10 \
    --pruning_method global \
    --device cuda
```

**Available Early-Bird arguments:**
- `--target_sparsity`: Target channel sparsity (default: 0.5)
- `--search_epochs` / `--total_epochs`: Max epochs for ticket search
- `--finetune_epochs`: Fine-tuning epochs (default: 160)
- `--l1_coef`: L1 regularization for BN γ (default: 1e-4)
- `--distance_threshold`: Convergence threshold (default: 0.1)
- `--patience`: Convergence window size (default: 5)
- `--min_search_epochs`: Minimum epochs before convergence is allowed (default: 10)
- `--pruning_method`: 'global' or 'layerwise' (default: 'global')

### 3. GraSP (Gradient Signal Preservation)

One-shot pruning at initialization that preserves gradient flow:


```bash
python -m src.experiments --algorithm grasp \
    --model resnet20 \
    --dataset cifar10 \
    --seed 42 \
    --target_sparsity 0.9 \
    --epochs 160 \
    --samples_per_class 10 \
    --grasp_T 200.0 \
    --grasp_iters 1 \
    --batch_size 128 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --lr_milestones 80 120 \
    --lr_gamma 0.1 \
    --device cuda
```

**Available GraSP arguments:**
- `--target_sparsity`: Fraction of weights to prune (default: 0.9)
- `--epochs`: Training epochs after pruning (default: 160)
- `--samples_per_class`: Balanced samples per class for GraSP scoring (default: 10)
- `--grasp_T`: Temperature scaling for logits (default: 200.0)
- `--grasp_iters`: Gradient accumulation batches (default: 1)
- `--lr_milestones`: Epochs at which to reduce LR (default: [80, 120])
- `--lr_gamma`: LR decay factor (default: 0.1)

### 4. Genetic Algorithm (GA)

Evolves binary masks over a fixed, randomly-initialised network, then trains
the discovered sub-network from the masked initialization:


```bash
python -m src.experiments --algorithm genetic \
    --model resnet20 \
    --dataset cifar10 \
    --seed 42 \
    --population_size 100 \
    --rec_rate 0.3 \
    --mut_rate 0.1 \
    --mig_rate 0.1 \
    --par_rate 0.3 \
    --min_generations 100 \
    --max_generations 200 \
    --stagnation_threshold 50 \
    --use_adaptive_ab False \
    --use_loss_fitness True \
    --max_eval_batches 4 \
    --post_prune True \
    --epochs 160 \
    --batch_size 128 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --lr_milestones 80 120 \
    --lr_gamma 0.1 \
    --device cuda
```

**Available GA arguments:**
- `--population_size`: Population size (default: 100)
- `--rec_rate`: Recombination rate (default: 0.3)
- `--mut_rate`: Mutation rate (default: 0.1)
- `--mig_rate`: Migration rate (default: 0.1)
- `--par_rate`: Top fraction eligible for mating (default: 0.3)
- `--min_generations`: Minimum generations before stagnation check (default: 100)
- `--max_generations`: Maximum generations (default: 200)
- `--stagnation_threshold`: Stop after N gens without improvement (default: 50)
- `--use_adaptive_ab`: Use adaptive accuracy-bound initialisation (default: False)
- `--use_loss_fitness`: Use negative loss as fitness (default: True)
- `--max_eval_batches`: Limit batches per fitness evaluation (default: None)
- `--no_post_prune`: Disable post-evolutionary pruning (default: False)

#### Checkpoint / Resume Support (for Kaggle 12-hour sessions)

GA experiments can be paused and resumed across multiple sessions with wall-clock time limits.

**Arguments:**
- `--time_limit`: Wall-clock time limit in seconds (e.g. `39600` for 11 hours, leaving 1-hour safety margin)
- `--resume_from`: Path to a checkpoint file to resume from
- `--checkpoint_interval`: Save checkpoint every N GA generations / training epochs (default: 10)

**Checkpoint location:**
```
./results/genetic/ga_{model}_{dataset}_pop{size}_seed{seed}/checkpoints/
  ├── ga_checkpoint.pt           (GA phase checkpoint)
  └── train_checkpoint.pt        (training phase checkpoint)
```

**Example: Kaggle 12-hour session with auto-save**
```bash
# Session 1: Run with 39600s (~11h) time limit. Saves checkpoint before deadline.
python -m src.experiments --algorithm genetic --model resnet20 --dataset cifar10 \
    --population_size 100 --min_generations 100 --max_generations 200 \
    --time_limit 39600 --checkpoint_interval 10

# Session 2: Resume from checkpoint when session 1 is interrupted
python -m src.experiments --algorithm genetic --model resnet20 --dataset cifar10 \
    --population_size 100 --min_generations 100 --max_generations 200 \
    --time_limit 39600 --checkpoint_interval 10 \
    --resume_from ./results/genetic/ga_resnet20_cifar10_pop100_seed42/checkpoints/ga_checkpoint.pt
```

The experiment will:
1. **Auto-save** a checkpoint every N generations during GA evolution
2. **Auto-save** a checkpoint every N epochs during post-GA training
3. **Check wall-clock time** and save a final checkpoint + gracefully exit before the deadline
4. **Resume** seamlessly from the last checkpoint, preserving:
   - GA population, best individual, fitness cache
   - Training state (model weights, optimizer, scheduler, epoch)
   - All metrics and history

### 5. Hybrid Pruning (One-Shot + Iterative Geometric)

Combines a large one-shot magnitude prune with iterative geometric refinement.
Phases: dense training → large prune + extended fine-tune → iterative small prunes + short fine-tunes.


```bash
python -m src.experiments --algorithm hybrid \
    --model resnet20 \
    --dataset cifar10 \
    --seed 42 \
    --target_sparsity 0.9 \
    --oneshot_ratio 0.7 \
    --iterative_step 0.02 \
    --initial_epochs 160 \
    --initial_lr 0.01 \
    --oneshot_finetune_max_epochs 200 \
    --oneshot_finetune_patience 50 \
    --iter_finetune_max_epochs 30 \
    --iter_finetune_patience 10 \
    --oneshot_scheduler_type cosine \
    --iter_scheduler_type none \
    --batch_size 128 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --device cuda
```

**Available Hybrid arguments:**
- `--target_sparsity`: Target overall sparsity (default: 0.9)
- `--oneshot_ratio`: Fraction of target to prune in one shot (default: 0.7)
- `--iterative_step`: Fraction of remaining weights to prune per iterative step.
- `--initial_epochs`: Dense training epochs (default: 160)
- `--initial_lr`: Learning rate for initial training (default: 0.01).
- `--oneshot_finetune_max_epochs`: Max epochs for fine-tuning after one-shot prune (default: 200)
- `--oneshot_finetune_patience`: Early-stopping patience for one-shot fine-tuning (default: 50)
- `--iter_finetune_max_epochs`: Max epochs per iterative fine-tuning step (default: 30)
- `--iter_finetune_patience`: Early-stopping patience per iterative step (default: 10)
- `--oneshot_scheduler_type`: Scheduler for one-shot fine-tuning (`cosine` or `none`, default: `cosine`)
- `--iter_scheduler_type`: Scheduler for iterative fine-tuning (`cosine` or `none`, default: `none` = constant LR)

### 6. SynFlow (Synaptic Flow)

Data-free iterative pruning at initialization that preserves gradient flow:


```bash
python -m src.experiments --algorithm synflow \
    --model resnet20 \
    --dataset cifar10 \
    --seed 42 \
    --rho 10 \
    --synflow_iters 100 \
    --epochs 160 \
    --batch_size 128 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --lr_milestones 80 120 \
    --lr_gamma 0.1 \
    --device cuda
```

**Available SynFlow arguments:**
- `--rho`: Compression ratio ρ ≥ 1 (default: 10). ρ=1 no pruning, ρ=10 keep 10%, ρ=100 keep 1%
- `--synflow_iters`: Number of iterative SynFlow pruning rounds (default: 100)
- `--epochs`: Training epochs after pruning (default: 160)

### 7. Hybrid-Improve (Adaptive Hybrid Pruning)

The improved hybrid variant features adaptive oneshot ratio estimation, Knowledge Distillation (KD) support, and optimized training loops.


```bash
python -m src.experiments --algorithm hybrid_improve \
    --model resnet20 \
    --dataset cifar10 \
    --seed 42 \
    --target_sparsity 0.8 \
    --oneshot_ratio 0.7 \
    --iterative_step 0.02 \
    --initial_epochs 100 \
    --initial_lr 0.1 \
    --oneshot_finetune_max_epochs 200 \
    --oneshot_finetune_patience 100 \
    --iter_finetune_max_epochs 10 \
    --iter_finetune_patience 5 \
    --oneshot_scheduler_type cosine \
    --iter_scheduler_type none \
    --batch_size 128 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --device cuda
```

**Available Hybrid-Improve arguments:**
- `--target_sparsity`: Target overall sparsity (default: 0.8)
- `--oneshot_ratio`: Fraction of target to prune in one shot (default: 0.7)
- `--iterative_step`: Fraction of remaining weights per iterative step (default: 0.02)
- `--initial_epochs`: Dense training epochs (default: 100)
- `--initial_lr`: Learning rate for initial training (default: 0.1)
- `--oneshot_finetune_max_epochs`: Max epochs for Phase 2 fine-tuning (default: 200)
- `--oneshot_finetune_patience`: Patience for Phase 2 (default: 100)
- `--iter_finetune_max_epochs`: Max epochs per Phase 3 step (default: 10)
- `--iter_finetune_patience`: Patience per Phase 3 step (default: 5)
- `--oneshot_scheduler_type`: Scheduler for one-shot fine-tuning (`cosine` or `none`, default: `cosine`)
- `--iter_scheduler_type`: Scheduler for iterative fine-tuning (`cosine` or `none`, default: `none` = constant LR)

**Advanced Features:**
- `--checkpoint_interval`: Save checkpoint every N pruning stages (default: 1).
- `--time_limit`: Wall-clock time limit in seconds to gracefully exit and save.
- `--resume_from`: Path to a checkpoint file to resume the experiment.

### 8. Recompute Missing Canonical Metrics (Post-hoc)

If a previous run folder contains `results.json`, `final_model.pt`, `final_masks.pt`, and optionally `initial_model.pt`, but misses canonical efficiency metrics (`flops_reduction`, `dense_flops`, `pruned_flops`, latency/throughput fields) or `dense_test_accuracy`, you can backfill them without rerunning training:

```bash
python -m src.experiments --algorithm recompute_metrics \
    --result_dir ./results/imp/imp_resnet20_cifar10_s0.9_seed42/20260331_120000 \
    --device cpu
```

**Available recompute arguments:**
- `--result_dir`: Existing run output folder (must contain `results.json`, `final_model.pt`, `final_masks.pt`; `initial_model.pt` enables dense baseline backfill when available)
- `--override_model_name`: Optional override if `results.json` config is missing `model_name`
- `--override_dataset_name`: Optional override if `results.json` config is missing `dataset_name`
- `--override_num_classes`: Optional override if class count cannot be inferred

The command updates `results.json` in place and creates a timestamped backup:
- `results_before_metrics_recompute_YYYYMMDD_HHMMSS.json`

### 9. Common Arguments

All algorithms support:
- `--model`: Architecture (resnet20, resnet50, vgg16, vgg19)
- `--dataset`: Dataset (cifar10, cifar100, mnist)
- `--seed`: Random seed (default: 42)
- `--device`: cuda or cpu (default: cuda)
- `--batch_size`: Training batch size
- `--learning_rate`: Initial learning rate
- `--momentum`: SGD momentum (default: 0.9)
- `--weight_decay`: Weight decay

### 10. Quick Tests

Run quick tests with reduced iterations (for validation):
```bash
python -m src.experiments --mode quick_imp
python -m src.experiments --mode quick_earlybird
python -m src.experiments --mode quick_grasp
python -m src.experiments --mode quick_synflow
```
