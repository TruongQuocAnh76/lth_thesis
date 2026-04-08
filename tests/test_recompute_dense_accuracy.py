import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiments import recompute_efficiency_metrics_for_existing_run


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture()
def dense_artifact_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    results = {
        "config": {
            "model_name": "tiny",
            "dataset_name": "cifar10",
            "num_classes": 10,
            "batch_size": 4,
        },
        "final_results": {
            "total_time_seconds": 1.0,
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    model = TinyClassifier(num_classes=10)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        model.fc.bias[0] = 10.0

    torch.save(model.state_dict(), run_dir / "initial_model.pt")
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    torch.save({"fc.weight": torch.ones_like(model.fc.weight)}, run_dir / "final_masks.pt")

    return run_dir


@pytest.fixture()
def stubbed_experiment_api(monkeypatch):
    def fake_get_model(model_name, num_classes=10):
        return TinyClassifier(num_classes=num_classes)

    def fake_get_dataloaders(dataset_name, batch_size=128, **kwargs):
        inputs = torch.zeros(8, 4)
        targets = torch.zeros(8, dtype=torch.long)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)
        return {"test": loader}

    def fake_compute_efficiency_metrics(**kwargs):
        return {
            "flops_reduction": 2.0,
            "dense_flops": 1.0,
            "pruned_flops": 0.5,
            "dense_latency_ms": 1.0,
            "pruned_latency_ms": 0.5,
            "dense_throughput": 10.0,
            "pruned_throughput": 20.0,
            "training_computational_cost_seconds": 3.0,
        }

    monkeypatch.setattr("src.experiments.get_model", fake_get_model)
    monkeypatch.setattr("src.experiments.get_dataloaders", fake_get_dataloaders)
    monkeypatch.setattr("src.experiments.compute_efficiency_metrics", fake_compute_efficiency_metrics)


def test_recompute_metrics_backfills_dense_test_accuracy(dense_artifact_run, stubbed_experiment_api):
    result = recompute_efficiency_metrics_for_existing_run(
        result_dir=str(dense_artifact_run),
        device="cpu",
    )

    final_results = result["final_results"]
    assert final_results["dense_test_accuracy"] == pytest.approx(100.0)
    assert "layer_sparsities" in final_results
    assert final_results["layer_sparsities"]["fc.weight"] == pytest.approx(0.0)

    saved = json.loads((dense_artifact_run / "results.json").read_text())
    assert saved["final_results"]["dense_test_accuracy"] == pytest.approx(100.0)
    assert "layer_sparsities" in saved["final_results"]
    assert saved["final_results"]["layer_sparsities"]["fc.weight"] == pytest.approx(0.0)
    backups = list(dense_artifact_run.glob("results_before_metrics_recompute_*.json"))
    assert backups, "Expected a timestamped backup of the original results.json"


def test_recompute_metrics_leaves_dense_test_accuracy_null_without_initial_model(tmp_path, stubbed_experiment_api):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    results = {
        "config": {
            "model_name": "tiny",
            "dataset_name": "cifar10",
            "num_classes": 10,
            "batch_size": 4,
        },
        "final_results": {
            "total_time_seconds": 1.0,
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    model = TinyClassifier(num_classes=10)
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    torch.save({"fc.weight": torch.ones_like(model.fc.weight)}, run_dir / "final_masks.pt")

    result = recompute_efficiency_metrics_for_existing_run(
        result_dir=str(run_dir),
        device="cpu",
    )

    assert result["final_results"]["dense_test_accuracy"] is None
    saved = json.loads((run_dir / "results.json").read_text())
    assert saved["final_results"]["dense_test_accuracy"] is None


def test_recompute_metrics_adds_missing_linear_layer_sparsity(tmp_path, stubbed_experiment_api):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    results = {
        "config": {
            "model_name": "tiny",
            "dataset_name": "cifar10",
            "num_classes": 10,
            "batch_size": 4,
        },
        "final_results": {
            "total_time_seconds": 1.0,
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    model = TinyClassifier(num_classes=10)
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    # Mask payload intentionally omits the linear layer.
    torch.save({"dummy": torch.ones(1)}, run_dir / "final_masks.pt")

    result = recompute_efficiency_metrics_for_existing_run(
        result_dir=str(run_dir),
        device="cpu",
    )

    layer_sparsities = result["final_results"].get("layer_sparsities", {})
    assert layer_sparsities.get("fc") == pytest.approx(0.0)

    saved = json.loads((run_dir / "results.json").read_text())
    assert saved["final_results"]["layer_sparsities"].get("fc") == pytest.approx(0.0)
