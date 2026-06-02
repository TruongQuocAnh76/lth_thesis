# SynFlow — Code ⇄ Methodology Mapping

This document maps the SynFlow explanation in the thesis methodology to the exact implementation in the repository so you can cite and inspect the code quickly.

- **Methodology reference:** [methodology: Iterative Synaptic Flow Pruning](Khoá_luận_tốt_nghiệp/chapters/methodology.tex#L362-L485)

## Key implementation points

- **Linearise (replace parameters with |θ|)** — implemented in `_linearise` which saves signs and takes absolute values: [src/synflow.py](src/synflow.py#L43)

- **Restore original signs** — implemented in `_restore`: [src/synflow.py](src/synflow.py#L61)

- **Compute SynFlow scores (S = |grad * param|)** — one-pass scoring that (1) linearises, (2) forwards an all-ones input, (3) backward-sums the output, and (4) computes per-parameter scores: `compute_synflow_scores` [src/synflow.py](src/synflow.py#L72)

- **Iterative SynFlow pruning loop (Algorithm 1, exponential schedule, global thresholding)** — implemented in `synflow_pruning`. This function:
  - makes a deep copy of the model (preserves the caller's model),
  - initialises per-layer masks and original weight snapshots,
  - runs `num_iters` rounds where it rebuilds masked weights, calls `compute_synflow_scores`, computes the exponential schedule `sparse_k = keep_ratio ** (k / num_iters)`, finds the global threshold, and updates masks accordingly: [src/synflow.py](src/synflow.py#L129)

- **Apply masks in-place** — helper `apply_synflow_masks` zeros weights according to the returned masks (useful for finetuning or evaluation): [src/synflow.py](src/synflow.py#L236)

- **Sparsity report helper** — `get_synflow_sparsity` computes per-layer and overall sparsity from the mask dict: [src/synflow.py](src/synflow.py#L252)

## How the code matches the thesis

- Equation (1) / Eq.~\eqref{eq:synflow_loss} (SynFlow surrogate loss as product of |θ| across layers): the implementation enforces this by linearising weights (absolute values) and using an all-ones input to propagate the product through the network before computing gradients — see `_linearise` and `compute_synflow_scores` ([src/synflow.py](src/synflow.py#L43), [src/synflow.py](src/synflow.py#L72)).

- Score definition Eq.~\eqref{eq:synflow_score} (S = (∂R/∂θ) ⋅ θ): the code computes `score = (module.weight.grad * module.weight.data).abs()` after backpropagating `torch.sum(output)` (implements the derivative of the surrogate objective); see `compute_synflow_scores` ([src/synflow.py](src/synflow.py#L72)).

- Iterative schedule and percentile thresholding (algorithm pseudocode in thesis): implemented in `synflow_pruning`, including the exponential sparsity schedule `sparse_k = keep_ratio ** (k / num_iters)` and global percentile threshold computed via `torch.kthvalue` over concatenated scores; see [src/synflow.py](src/synflow.py#L129).

- Numerical stability note from the thesis (layer-wise normalisation / float64): the implementation does not perform explicit layer normalisation; if needed, consider converting model copy to `float64` before scoring or normalising layer weights in `compute_synflow_scores` prior to forward. The place to change is `synflow_pruning` where `model_copy` is created and before `compute_synflow_scores` is called: [src/synflow.py](src/synflow.py#L129).

## Quick usage examples

- Run pruning and get masks (keeps 1/ρ of parameters):

```
from src.synflow import synflow_pruning, apply_synflow_masks

# model: your nn.Module at init, device: torch.device('cpu') or cuda
masks = synflow_pruning(model, device=device, rho=10.0, num_iters=100)
apply_synflow_masks(model, masks)
```

## Notes & next steps

- I validated the main function anchors above. If you want, I can also add a short example in `notebooks/` that runs `synflow_pruning` on a small model (ResNet20) and plots per-iteration overall sparsity / score histogram.
