# Early-Bird code explanation

This note maps the Early-Bird description in [Khoá_luận_tốt_nghiệp/chapters/methodology.tex](Khoá_luận_tốt_nghiệp/chapters/methodology.tex#L49) to the implementation in the codebase.

## 1. What the methodology says

The thesis describes Early-Bird as a two-stage process:

1. Search for a stable sparse ticket early in training.
2. Retrain the pruned subnet from that discovered ticket.

The key technical ideas are:

- use BatchNorm gamma values as the pruning signal,
- prune channels rather than individual weights,
- measure mask stability with normalized Hamming distance,
- stop the search when the last K distances stay below a threshold.

## 2. Where the code implements each step

### BatchNorm gamma extraction

The pruning signal comes from BatchNorm scaling factors. That is implemented in [extract_bn_gammas](src/earlybird.py#L7), which scans the model with `named_modules()` and collects `module.weight` from every BatchNorm layer.

### Channel mask construction

The mask creation logic is in [compute_channel_mask_from_bn_gamma](src/earlybird.py#L30). It turns the absolute gamma values into binary channel masks by keeping the largest-magnitude channels and pruning the smallest ones.

The function supports two policies:

- `global`, which ranks all channels across all BatchNorm layers together,
- `layerwise`, which keeps a per-layer quota and then adjusts globally to match the target sparsity.

This matches the thesis’s structured channel pruning view, while adding a practical implementation option for layerwise control.

### Mask distance and convergence

The stability test is implemented in [compute_channel_mask_hamming_distance](src/earlybird.py#L141) and [EarlyBirdFinder](src/earlybird.py#L208).

The convergence rule is handled by [_try_update_convergence](src/earlybird.py#L262):

- it waits until `min_search_epochs` has passed,
- it requires at least `patience` recorded distances,
- it checks whether `max(last K distances) < distance_threshold`.

That is the code version of the thesis condition in [Equation (mask distance)](Khoá_luận_tốt_nghiệp/chapters/methodology.tex#L79) and [Equation (EB condition)](Khoá_luận_tốt_nghiệp/chapters/methodology.tex#L90).

### Epoch-level search loop

The per-epoch recording step is in [record_epoch](src/earlybird.py#L294). It extracts BN gammas, computes the channel mask, compares it with the previous epoch’s mask, stores the distance, and updates convergence state.

The same logic is available offline in [record_from_bn_gammas](src/earlybird.py#L341), which is useful if the BN snapshots were saved during training.

### L1 regularization on BatchNorm gamma

The thesis says the search phase uses low-cost training with L1 pressure on the pruning signal. The implementation of that pressure is in [add_l1_regularization_to_bn](src/earlybird.py#L687).

During training, [_train_epoch_with_l1](src/experiments.py#L1009) adds that regularizer to the classification loss before backpropagation.

## 3. How the experiment runner follows the thesis

The Early-Bird experiment itself is defined in [EarlyBirdExperiment](src/experiments.py#L877).

### Search phase

In [PHASE 1](src/experiments.py#L1194), the code:

- creates an [EarlyBirdFinder](src/experiments.py#L1198),
- trains the model with BN L1 regularization,
- checks mask convergence at the end of each epoch via [record_epoch](src/experiments.py#L1106),
- retrieves the discovered ticket with [get_early_bird_ticket](src/experiments.py#L1222).

This is the implementation of the thesis search loop in [Algorithm EB Search](Khoá_luận_tốt_nghiệp/chapters/methodology.tex#L99).

### Residual alignment and mask expansion

The thesis describes channel masks at the structural level, but the model still needs weight-level masks to apply them in PyTorch.

That conversion happens in two steps:

- [align_channel_mask_for_residual_blocks](src/earlybird.py#L474) keeps residual-path BatchNorm layers consistent,
- [expand_channel_mask_to_conv_weights](src/earlybird.py#L618) expands channel masks into Conv2d weight masks.

This is especially important for ResNet-style blocks, where the skip path and main path must agree on the active channels.

### Fine-tuning phase

After the ticket is found, the code applies the discovered masks with [apply_masks_to_model](src/experiments.py#L1275) and then retrains the masked model in [PHASE 2](src/experiments.py#L1284).

That corresponds to the thesis statement that the discovered ticket is trained further after it is identified.

## 4. Small implementation differences from the thesis text

The code is faithful to the methodology, but it includes two practical extensions:

- it supports both `global` and `layerwise` channel pruning,
- it adds `min_search_epochs` so convergence cannot be declared too early.

Those are implementation safeguards, not changes to the core Early-Bird idea.

## 5. Short summary

In code, Early-Bird means:

- extract BN gamma values,
- prune channels by smallest gamma magnitudes,
- track mask stability epoch by epoch,
- stop when the last K mask distances are small enough,
- apply the resulting channel mask and finetune the pruned network.

The main entry points are [src/earlybird.py](src/earlybird.py) and [src/experiments.py](src/experiments.py).