
import pytest
from unittest import mock
import torch

# Patch thop.profile globally for all tests
@pytest.fixture(autouse=True)
def patch_thop_profile():
    with mock.patch('thop.profile', return_value=(123456, None)):
        yield

def _run_and_check_flops(run_func, **kwargs):
    results = run_func(**kwargs)
    final = results.get('final_results', {})
    assert final.get('dense_flops') is not None, f"dense_flops is None for {run_func.__name__}"
    assert final.get('pruned_flops') is not None, f"pruned_flops is None for {run_func.__name__}"
    assert final.get('flops_reduction') is not None, f"flops_reduction is None for {run_func.__name__}"

def test_imp_flops():
    from src.experiments import run_imp_experiment
    _run_and_check_flops(run_imp_experiment, model_name="resnet20", dataset_name="moons", target_sparsity=0.5, num_iterations=1, epochs=1, seed=42, device="cpu", save_dir="/tmp")

def test_earlybird_flops():
    from src.experiments import run_earlybird_experiment
    _run_and_check_flops(run_earlybird_experiment, model_name="resnet20", dataset_name="moons", target_sparsity=0.5, search_epochs=1, finetune_epochs=1, l1_coef=1e-4, seed=42, device="cpu", save_dir="/tmp")

def test_grasp_flops():
    from src.experiments import run_grasp_experiment
    _run_and_check_flops(run_grasp_experiment, model_name="resnet20", dataset_name="moons", target_sparsity=0.5, epochs=1, samples_per_class=1, grasp_T=1.0, grasp_iters=1, learning_rate=0.1, lr_milestones=[1], lr_gamma=0.1, seed=42, device="cpu", save_dir="/tmp")

def test_synflow_flops():
    from src.experiments import run_synflow_experiment
    _run_and_check_flops(run_synflow_experiment, model_name="resnet20", dataset_name="moons", rho=2.0, synflow_iters=1, epochs=1, learning_rate=0.1, lr_milestones=[1], lr_gamma=0.1, seed=42, device="cpu", save_dir="/tmp")

def test_hybrid_flops():
    from src.experiments import run_experiment
    results = run_experiment(
        algorithm="hybrid",
        model="resnet20",
        dataset="moons",
        target_sparsity=0.5,
        initial_epochs=1,
        oneshot_finetune_max_epochs=1,
        iter_finetune_max_epochs=1,
        seed=42,
        device="cpu",
        save_dir="/tmp"
    )
    final = results.get('final_results', {})
    assert final.get('dense_flops') is not None, "dense_flops is None for hybrid"
    assert final.get('pruned_flops') is not None, "pruned_flops is None for hybrid"
    assert final.get('flops_reduction') is not None, "flops_reduction is None for hybrid"

def test_hybrid_improve_flops():
    from src.experiments import run_experiment
    results = run_experiment(
        algorithm="hybrid_improve",
        model="resnet20",
        dataset="moons",
        target_sparsity=0.5,
        initial_epochs=1,
        oneshot_finetune_max_epochs=1,
        iter_finetune_max_epochs=1,
        seed=42,
        device="cpu",
        save_dir="/tmp"
    )
    final = results.get('final_results', {})
    assert final.get('dense_flops') is not None, "dense_flops is None for hybrid_improve"
    assert final.get('pruned_flops') is not None, "pruned_flops is None for hybrid_improve"
    assert final.get('flops_reduction') is not None, "flops_reduction is None for hybrid_improve"
