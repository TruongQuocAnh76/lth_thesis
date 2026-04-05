import numpy as np
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imp_scheduler_uses_multistep_and_expected_milestones():
    from experiments import IMPExperiment

    exp = IMPExperiment(
        model_name="resnet20",
        dataset_name="moons",
        num_classes=2,
        target_sparsity=0.5,
        num_iterations=1,
        epochs_per_iteration=10,
        seed=42,
        device="cpu",
        save_dir="/tmp",
    )

    model = exp._create_model()
    optimizer = exp._create_optimizer(model)
    scheduler = exp._create_scheduler(optimizer)

    assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
    assert sorted(scheduler.milestones.keys()) == [5, 7]


def test_imp_scheduler_keeps_nonzero_lr_at_end_of_iteration():
    from experiments import IMPExperiment

    exp = IMPExperiment(
        model_name="resnet20",
        dataset_name="moons",
        num_classes=2,
        target_sparsity=0.5,
        num_iterations=1,
        epochs_per_iteration=10,
        learning_rate=0.1,
        seed=42,
        device="cpu",
        save_dir="/tmp",
    )

    model = exp._create_model()
    optimizer = exp._create_optimizer(model)
    scheduler = exp._create_scheduler(optimizer)

    for _ in range(10):
        optimizer.step()
        scheduler.step()

    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr == pytest.approx(0.001)
    assert final_lr > 0.0


def test_imp_prunes_using_best_epoch_weights(monkeypatch):
    import experiments as exp_mod

    model = nn.Sequential(nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(0.0)

    weight_by_epoch = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    test_acc_by_epoch = [60.0, 90.0, 70.0, 50.0, 55.0, 53.0]
    step_idx = {"i": 0}
    captured = {}

    monkeypatch.setattr(exp_mod, "set_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(exp_mod, "get_dataloaders", lambda *_args, **_kwargs: {"train": [None], "test": [None]})
    monkeypatch.setattr(exp_mod, "count_parameters", lambda _model: {"total": 1})
    monkeypatch.setattr(exp_mod, "compute_efficiency_metrics", lambda **_kwargs: {})
    monkeypatch.setattr(exp_mod, "apply_masks_to_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(exp_mod, "create_mask_apply_fn", lambda _model: (lambda *_args, **_kwargs: None))
    monkeypatch.setattr(exp_mod, "get_overall_sparsity", lambda _masks: 0.0)
    monkeypatch.setattr(exp_mod, "get_sparsity", lambda _masks: {"0.weight": 0.0})
    monkeypatch.setattr(
        exp_mod,
        "create_initial_masks",
        lambda weights: {k: np.ones_like(v, dtype=np.float32) for k, v in weights.items()},
    )

    def fake_get_prunable_layers(_model):
        return {"0.weight": model[0].weight.detach().cpu().numpy().copy()}

    monkeypatch.setattr(exp_mod, "get_prunable_layers", fake_get_prunable_layers)

    def fake_train_epoch(**kwargs):
        current_step = step_idx["i"]
        optimizer = kwargs["optimizer"]
        optimizer.zero_grad()
        optimizer.step()
        with torch.no_grad():
            model[0].weight.fill_(weight_by_epoch[current_step])
        return 1.0, 50.0

    def fake_evaluate(*_args, **_kwargs):
        current_step = step_idx["i"]
        step_idx["i"] += 1
        return 1.0, test_acc_by_epoch[current_step]

    def fake_prune_by_magnitude_global(_rate, masks, trained_weights):
        captured["weight"] = float(trained_weights["0.weight"].reshape(-1)[0])
        return masks

    monkeypatch.setattr(exp_mod, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(exp_mod, "evaluate", fake_evaluate)
    monkeypatch.setattr(exp_mod, "prune_by_magnitude_global", fake_prune_by_magnitude_global)
    monkeypatch.setattr(exp_mod, "prune_by_percent", lambda _p, masks, _w: masks)
    monkeypatch.setattr(exp_mod.IMPExperiment, "_create_model", lambda self: model)
    monkeypatch.setattr(exp_mod.IMPExperiment, "_save_results", lambda self: None)

    exp = exp_mod.IMPExperiment(
        model_name="resnet20",
        dataset_name="moons",
        num_classes=2,
        target_sparsity=0.5,
        num_iterations=1,
        epochs_per_iteration=3,
        seed=42,
        device="cpu",
        save_dir="/tmp",
        use_global_pruning=True,
    )

    results = exp.run()

    assert captured["weight"] == pytest.approx(2.0)
    assert results["iterations"][0]["test_accuracy"] == pytest.approx(70.0)
    assert results["iterations"][0]["best_test_accuracy"] == pytest.approx(90.0)
