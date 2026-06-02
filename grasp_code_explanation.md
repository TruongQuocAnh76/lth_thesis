# GraSP code explanation

This note maps the GraSP description in [Khoá_luận_tốt_nghiệp/chapters/methodology.tex](Khoá_luận_tốt_nghiệp/chapters/methodology.tex#L176) to the implementation in the codebase.

## 1. What the methodology says

GraSP finds a pruning mask at initialization by keeping weights that preserve gradient flow. The theoretical steps are:

- compute first-order gradients g at init,
- compute Hessian–gradient product Hg efficiently via second-order autodiff,
- score weights by S(θ) = −θ · Hg,
- threshold scores to produce a binary mask, then train the masked network from initialization.

## 2. Where the code implements each step

### Balanced sampling

GraSP requires a small balanced batch. The helper [fetch_balanced_data](src/grasp.py#L12) builds a class-balanced mini-batch from a DataLoader.

### Prunable-weight selection

Weights eligible for pruning (Conv2d and Linear) are collected by [_get_prunable_weights](src/grasp.py#L46).

### Accumulate first-order gradients

The main routine [compute_grasp_scores](src/grasp.py#L56) reproduces the reference behaviour:

- it makes a deep copy of the network and enables gradients on prunable weights,
- it samples balanced batches and splits each into two halves,
- it accumulates first-order gradients `grad_w` across halves (stage 1).

See the accumulation code at [compute_grasp_scores — accumulation loop](src/grasp.py#L86).

### Hessian–gradient product (Hg)

For each stored half-batch the code recomputes a forward/backward with `create_graph=True` so that the scalar z (the inner product between accumulated grads and fresh grads) can be backpropagated to obtain Hg via `.grad` on copied weights.

This is implemented in the second pass inside [compute_grasp_scores](src/grasp.py#L124) where `z.backward()` yields the Hessian–gradient product into `copy_m.weight.grad`.

### Score: S(θ) = −θ · Hg

After `z.backward()` the code computes per-weight scores `-copy_m.weight.data * copy_m.weight.grad` and maps them back to the original model modules (see [compute_grasp_scores — scoring & mapping](src/grasp.py#L160)).

### Threshold into masks

The helper [grasp_masks_from_scores](src/grasp.py#L194) concatenates all scores, selects the globally smallest scores to *keep* (equivalently pruning the top-p scores), and builds per-module float masks.

### Apply masks and return

The public API [grasp](src/grasp.py#L237) runs the full pipeline, converts module-keyed masks to name-keyed NumPy arrays (project-wide mask format), applies masks in-place to `module.weight.data`, and returns the named masks.

## 3. How it matches the thesis

- Equation~\eqref{eq:grasp_score} (S = −θ ⊙ Hg) is implemented directly in [compute_grasp_scores](src/grasp.py#L160).
- The Hessian–gradient trick from Equation~\eqref{eq:hg_trick} is realized by the `create_graph=True` backward over the scalar `z` (see [compute_grasp_scores](src/grasp.py#L124)).
- Mask construction via percentile thresholding is implemented in [grasp_masks_from_scores](src/grasp.py#L194) and matches the thesis masking rule.

## 4. Practical notes and differences

- The implementation follows the reference closely, including the two-half batching trick and temperature scaling `T` to stabilise the second-order signal (default `T=200`).
- Bias and BatchNorm parameters are deliberately excluded from pruning in [_get_prunable_weights](src/grasp.py#L46).
- The code returns masks in the same `{layer_name: np.ndarray}` format used by the rest of the project, making it interoperable with IMP and other experiment runners.

## 5. Short summary

In code, GraSP does:

- build a balanced mini-batch, accumulate first-order gradients,
- recompute forward/backward with `create_graph=True` to get Hg,
- compute scores `S = -θ · Hg`,
- threshold scores globally to produce binary masks,
- apply masks to model weights and return project-formatted masks.

Main entry points: `src/grasp.py::compute_grasp_scores`, `src/grasp.py::grasp_masks_from_scores`, `src/grasp.py::grasp`.
