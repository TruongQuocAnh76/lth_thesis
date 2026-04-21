import torch
import torch.nn as nn

from src import synflow


def _build_toy_model() -> nn.Module:
    model = nn.Sequential(
        nn.Linear(4, 3, bias=False),
        nn.ReLU(),
        nn.Linear(3, 2, bias=False),
    )

    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor(
                [
                    [1.0, -2.0, 3.0, -4.0],
                    [-5.0, 6.0, -7.0, 8.0],
                    [9.0, -10.0, 11.0, -12.0],
                ]
            )
        )
        model[2].weight.copy_(
            torch.tensor(
                [
                    [1.0, -2.0, 3.0],
                    [-4.0, 5.0, -6.0],
                ]
            )
        )

    return model


def test_linearise_and_restore_use_live_parameters():
    model = _build_toy_model()
    original = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }

    signs = synflow._linearise(model)

    for _, param in model.named_parameters():
        assert torch.all(param.data >= 0)

    synflow._restore(model, signs)

    for name, param in model.named_parameters():
        assert torch.allclose(param, original[name])


def test_synflow_pruning_keeps_nonzero_weights_with_deterministic_scores(
    monkeypatch,
):
    model = _build_toy_model()
    original = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }

    call_count = {"value": 0}

    def fake_compute_synflow_scores(model_copy, device, input_shape=(3, 32, 32)):
        call_count["value"] += 1
        del model_copy, device, input_shape
        return {
            "0": torch.arange(1, 13, dtype=torch.float32).reshape(3, 4),
            "2": torch.arange(13, 19, dtype=torch.float32).reshape(2, 3),
        }

    monkeypatch.setattr(synflow, "compute_synflow_scores", fake_compute_synflow_scores)

    masks = synflow.synflow_pruning(
        model=model,
        device=torch.device("cpu"),
        rho=2.0,
        num_iters=2,
        input_shape=(4,),
    )

    assert call_count["value"] == 2
    assert set(masks) == {"0", "2"}

    total_params = sum(mask.size for mask in masks.values())
    total_zeros = sum(int((mask == 0).sum()) for mask in masks.values())

    assert total_params == 18
    assert total_zeros == 9
    assert total_zeros < total_params

    for name, param in model.named_parameters():
        assert torch.allclose(param, original[name])