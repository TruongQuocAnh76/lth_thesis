import torch
import torch.nn as nn
import pytest


def _fake_history(best=64.1, last=63.2):
    return {
        "epochs_run": 3,
        "best_test_acc": best,
        "train_losses": [1.0, 0.9, 0.8],
        "train_accs": [50.0, 52.0, 54.0],
        "test_losses": [1.0, 0.9, 0.95],
        "test_accs": [63.0, best, last],
    }


def test_hybrid_finetune_none_scheduler_skips_cosine(monkeypatch):
    import src.hybrid as hybrid

    cosine_inits = {"count": 0}

    class FakeCosine:
        def __init__(self, optimizer, T_max):
            cosine_inits["count"] += 1

        def step(self):
            return None

    monkeypatch.setattr(hybrid, "CosineAnnealingLR", FakeCosine)
    monkeypatch.setattr(hybrid, "train_epoch", lambda *args, **kwargs: (0.5, 50.0))

    eval_values = iter([(0.4, 60.0), (0.4, 59.5), (0.4, 59.0)])
    monkeypatch.setattr(hybrid, "evaluate", lambda *args, **kwargs: next(eval_values))

    model = nn.Linear(4, 2)
    history = hybrid._finetune(
        model=model,
        train_loader=[None],
        test_loader=[None],
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        masks={},
        apply_mask_fn=lambda _: None,
        max_epochs=3,
        lr=0.01,
        patience=2,
        scheduler_type="none",
        verbose=False,
    )

    assert cosine_inits["count"] == 0
    assert history["epochs_run"] == 3


def test_hybrid_finetune_adjusts_inert_patience(monkeypatch):
    import src.hybrid as hybrid

    monkeypatch.setattr(hybrid, "train_epoch", lambda *args, **kwargs: (0.5, 50.0))

    # First epoch improves, then a long plateau.
    eval_values = [(0.4, 60.0)] + [(0.4, 59.0)] * 20
    iterator = iter(eval_values)
    monkeypatch.setattr(hybrid, "evaluate", lambda *args, **kwargs: next(iterator))

    model = nn.Linear(4, 2)
    history = hybrid._finetune(
        model=model,
        train_loader=[None],
        test_loader=[None],
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        masks={},
        apply_mask_fn=lambda _: None,
        max_epochs=10,
        lr=0.01,
        patience=10,
        scheduler_type="none",
        verbose=False,
    )

    assert history["epochs_run"] < 10


def test_hybrid_pruning_uses_phase_scheduler_defaults_and_final_metric(monkeypatch):
    import src.hybrid as hybrid
    import src.train as train_mod

    scheduler_types = []

    def fake_finetune(*args, **kwargs):
        scheduler_types.append(kwargs["scheduler_type"])
        return _fake_history()

    monkeypatch.setattr(hybrid, "_finetune", fake_finetune)
    monkeypatch.setattr(hybrid, "get_dataloaders", lambda *args, **kwargs: {"train": [None], "test": [None]})
    monkeypatch.setattr(hybrid, "get_model", lambda *args, **kwargs: nn.Sequential(nn.Linear(4, 2)))
    monkeypatch.setattr(
        hybrid,
        "count_parameters",
        lambda model: {"total": sum(p.numel() for p in model.parameters())},
    )
    monkeypatch.setattr(hybrid, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(hybrid, "evaluate", lambda *args, **kwargs: (0.4, 63.2))
    monkeypatch.setattr(
        hybrid,
        "compute_efficiency_metrics",
        lambda **kwargs: {"dense_flops": 1.0, "pruned_flops": 1.0, "flops_reduction": 0.0},
    )

    def fake_train_epochs(**kwargs):
        return {
            "train_losses": [1.0],
            "train_accs": [50.0],
            "test_losses": [1.0],
            "test_accs": [60.0],
            "final_test_acc": 60.0,
        }

    monkeypatch.setattr(train_mod, "train_epochs", fake_train_epochs)

    results = hybrid.hybrid_pruning(
        model_name="resnet20",
        dataset_name="moons",
        num_classes=2,
        target_sparsity=0.5,
        oneshot_ratio=0.5,
        iterative_step=1.0,
        initial_epochs=1,
        oneshot_finetune_max_epochs=5,
        iter_finetune_max_epochs=5,
        device="cpu",
        verbose=False,
        checkpoint_dir=None,
    )

    assert scheduler_types[0] == "cosine"
    assert scheduler_types[1] == "none"
    for phase in results["phases"]:
        assert phase["best_test_acc"] == pytest.approx(64.1)
        assert phase["final_test_acc"] == pytest.approx(63.2)


def test_hybrid_improve_finetune_none_scheduler_skips_cosine(monkeypatch):
    import src.hybrid_improve as hybrid_improve

    cosine_inits = {"count": 0}

    class FakeCosine:
        def __init__(self, optimizer, T_max):
            cosine_inits["count"] += 1

        def step(self):
            return None

    monkeypatch.setattr(hybrid_improve, "CosineAnnealingLR", FakeCosine)
    monkeypatch.setattr(hybrid_improve, "train_epoch", lambda *args, **kwargs: (0.5, 50.0))

    eval_values = iter([(0.4, 60.0), (0.4, 59.5), (0.4, 59.0)])
    monkeypatch.setattr(hybrid_improve, "evaluate", lambda *args, **kwargs: next(eval_values))

    model = nn.Linear(4, 2)
    history = hybrid_improve._finetune(
        model=model,
        train_loader=[None],
        test_loader=[None],
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        masks={},
        apply_mask_fn=lambda _: None,
        max_epochs=3,
        lr=0.01,
        patience=2,
        scheduler_type="none",
        verbose=False,
        use_cuda_graphs=False,
    )

    assert cosine_inits["count"] == 0
    assert history["epochs_run"] == 3
