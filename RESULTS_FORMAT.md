## Experiment Output Format

All experiment results are saved in the `results/` directory, organized by algorithm, model, dataset, sparsity, and seed. Each experiment run produces a dedicated subdirectory, e.g.:

```
results/
  imp/imp_resnet20_cifar10_s0.9_seed42/
  earlybird/eb_resnet20_cifar10_sparsity0.5_seed42/
  grasp/grasp_resnet20_cifar10_s0.9_seed42/
  synflow/synflow_resnet20_cifar10_rho10_seed42/
  ga/ga_resnet20_cifar10_seed42/
  hybrid/hybrid_resnet20_cifar10_s0.9_seed42/
  hybrid-improve/hybrid_improve_resnet20_cifar10_s0.8_seed42/
```

Each experiment output directory contains the following files:

- `results.json`: Full experiment configuration, training history, and final metrics. Includes:
    - Algorithm parameters and config
    - Training/validation accuracy and loss curves
    - Final results (see below for metrics)
    - All new metrics: FLOPS reduction, dense/pruned FLOPS, inference latency, throughput, training computational cost
- `summary.csv`: Epoch-by-epoch training and test metrics (accuracy, loss, etc.)
- `final_model.pt`: Final trained model weights (PyTorch state_dict)
- `final_masks.pt`: Final pruning masks (PyTorch dict of binary masks)
- `initial_model.pt`: Initial model weights before pruning (PyTorch state_dict)

### Example: `results.json` structure

```json
{
  "config": { ... },
  "train_history": { ... },
  "final_results": {
    "final_test_accuracy": 91.23,
    "final_train_accuracy": 99.99,
    "overall_sparsity": 0.9,
    "flops_reduction": 0.85,
    "dense_flops": 41000000,
    "pruned_flops": 6100000,
    "dense_latency_ms": 2.1,
    "pruned_latency_ms": 1.2,
    "dense_throughput": 476.2,
    "pruned_throughput": 833.3,
    "training_computational_cost_seconds": 1234.56,
    "dense_test_accuracy": 69.42067,
    ...
  },
  ...
}
```