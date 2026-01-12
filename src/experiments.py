"""
Experiment runner for Lottery Ticket Hypothesis pruning methods.

This module implements experiments for various pruning methods,
starting with Iterative Magnitude Pruning (IMP).
"""

import sys
import json
import copy
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Ensure project root on sys.path for script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model import get_model, count_parameters
from src.data import get_dataloaders
from src.train import train_epochs, evaluate
from src.pruning import (
    create_initial_masks,
    prune_by_percent,
    prune_by_magnitude_global,
    get_sparsity,
    get_overall_sparsity
)
from src.util import (
    load_config,
    set_seed,
    get_prunable_layers,
    apply_masks_to_model,
    create_mask_apply_fn,
)


class IMPExperiment:
    """Iterative Magnitude Pruning (IMP) experiment runner.
    
    Implements the Lottery Ticket Hypothesis experiment:
    1. Train network to completion
    2. Prune smallest magnitude weights
    3. Reset remaining weights to original initialization
    4. Repeat for multiple pruning iterations
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        num_classes: int,
        target_sparsity: float,
        num_iterations: int = 10,
        epochs_per_iteration: int = 160,
        batch_size: int = 128,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        seed: int = 42,
        device: str = "cuda",
        save_dir: str = "./results",
        use_global_pruning: bool = True,
        warmup_epochs: int = 5
    ):
        """Initialize IMP experiment.
        
        Args:
            model_name: Name of model architecture ('resnet20', 'resnet50', 'vgg16')
            dataset_name: Name of dataset ('cifar10', 'cifar100')
            num_classes: Number of output classes
            target_sparsity: Target final sparsity (e.g., 0.9 for 90% pruned)
            num_iterations: Number of pruning iterations
            epochs_per_iteration: Training epochs per pruning round
            batch_size: Training batch size
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: L2 regularization weight
            seed: Random seed for reproducibility
            device: Device to use ('cuda' or 'cpu')
            save_dir: Directory to save results
            use_global_pruning: If True, use global magnitude pruning; else per-layer
            warmup_epochs: Number of warmup epochs
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.target_sparsity = target_sparsity
        self.num_iterations = num_iterations
        self.epochs_per_iteration = epochs_per_iteration
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.use_global_pruning = use_global_pruning
        self.warmup_epochs = warmup_epochs
        
        # Calculate per-iteration pruning rate to achieve target sparsity
        # After n iterations: (1 - rate)^n = (1 - target_sparsity)
        # rate = 1 - (1 - target_sparsity)^(1/n)
        self.prune_rate_per_iteration = 1 - (1 - target_sparsity) ** (1 / num_iterations)
        
        # Results storage
        self.results = {
            'config': self._get_config_dict(),
            'iterations': [],
            'final_results': {}
        }
        
    def _get_config_dict(self) -> Dict:
        """Return experiment configuration as dictionary."""
        return {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'num_classes': self.num_classes,
            'target_sparsity': self.target_sparsity,
            'num_iterations': self.num_iterations,
            'epochs_per_iteration': self.epochs_per_iteration,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'seed': self.seed,
            'use_global_pruning': self.use_global_pruning,
            'prune_rate_per_iteration': self.prune_rate_per_iteration
        }
    
    def _create_model(self) -> nn.Module:
        """Create and return model instance."""
        model = get_model(self.model_name, num_classes=self.num_classes)
        return model.to(self.device)
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer for model."""
        return optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> Any:
        """Create learning rate scheduler."""
        return CosineAnnealingLR(optimizer, T_max=self.epochs_per_iteration)
    
    def run(self) -> Dict[str, Any]:
        """Run the complete IMP experiment.
        
        Returns:
            Dictionary containing all experiment results
        """
        print(f"\n{'='*60}")
        print(f"Starting IMP Experiment")
        print(f"Model: {self.model_name}, Dataset: {self.dataset_name}")
        print(f"Target Sparsity: {self.target_sparsity:.1%}")
        print(f"Pruning Iterations: {self.num_iterations}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Set random seed
        set_seed(self.seed)
        
        # Load data
        print("Loading dataset...")
        loaders = get_dataloaders(
            self.dataset_name,
            batch_size=self.batch_size,
            num_workers=4
        )
        train_loader = loaders['train']
        test_loader = loaders['test']
        
        # Create initial model and save initial weights
        print("Initializing model...")
        model = self._create_model()
        param_count = count_parameters(model)
        print(f"Model parameters: {param_count['total']:,}")
        
        # Store initial weights (the "lottery ticket" initialization)
        initial_weights = get_prunable_layers(model)
        initial_state_dict = copy.deepcopy(model.state_dict())
        
        # Create initial masks (all ones = no pruning)
        masks = create_initial_masks(initial_weights)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Run iterative magnitude pruning
        for iteration in range(self.num_iterations + 1):
            print(f"\n{'='*40}")
            print(f"Pruning Iteration {iteration}/{self.num_iterations}")
            current_sparsity = get_overall_sparsity(masks)
            print(f"Current Sparsity: {current_sparsity:.2%}")
            print(f"{'='*40}")
            
            # Reset model to initial weights and apply current mask
            model.load_state_dict(copy.deepcopy(initial_state_dict))
            apply_masks_to_model(model, masks)
            
            # Create fresh optimizer and scheduler
            optimizer = self._create_optimizer(model)
            scheduler = self._create_scheduler(optimizer)
            
            # Create mask application function
            apply_mask_fn = create_mask_apply_fn(model)
            
            # Train
            print(f"Training for {self.epochs_per_iteration} epochs...")
            history = train_epochs(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=self.epochs_per_iteration,
                device=self.device,
                scheduler=scheduler,
                masks=masks,
                apply_mask_fn=apply_mask_fn,
                verbose=True
            )
            
            # Evaluate final accuracy
            test_loss, test_acc = evaluate(model, test_loader, criterion, self.device)
            
            # Store iteration results
            iteration_result = {
                'iteration': iteration,
                'sparsity': current_sparsity,
                'remaining_weights_fraction': 1 - current_sparsity,
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'train_accuracy_final': history['train_accs'][-1] if history['train_accs'] else 0,
                'train_loss_final': history['train_losses'][-1] if history['train_losses'] else 0,
                'best_test_accuracy': max(history['test_accs']) if history['test_accs'] else test_acc,
                'layer_sparsities': get_sparsity(masks)
            }
            self.results['iterations'].append(iteration_result)
            
            print(f"\nIteration {iteration} Results:")
            print(f"  Test Accuracy: {test_acc:.2f}%")
            print(f"  Sparsity: {current_sparsity:.2%}")
            
            # Prune for next iteration (skip if last iteration)
            if iteration < self.num_iterations:
                # Get trained weights for pruning decision
                trained_weights = get_prunable_layers(model)
                
                if self.use_global_pruning:
                    # Global magnitude pruning
                    masks = prune_by_magnitude_global(
                        self.prune_rate_per_iteration,
                        masks,
                        trained_weights
                    )
                else:
                    # Per-layer magnitude pruning
                    prune_percents = {k: self.prune_rate_per_iteration for k in masks.keys()}
                    masks = prune_by_percent(prune_percents, masks, trained_weights)
        
        # Final results
        self.results['final_results'] = {
            'final_sparsity': get_overall_sparsity(masks),
            'final_test_accuracy': self.results['iterations'][-1]['test_accuracy'],
            'initial_test_accuracy': self.results['iterations'][0]['test_accuracy'],
            'accuracy_at_target_sparsity': self.results['iterations'][-1]['test_accuracy'],
        }
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save experiment results to disk."""
        # Create results directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"imp_{self.model_name}_{self.dataset_name}_s{self.target_sparsity}_seed{self.seed}"
        result_dir = self.save_dir / "imp" / experiment_name / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays in results to lists for JSON serialization
        results_serializable = self._make_serializable(self.results)
        
        # Save JSON results
        results_path = result_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # Save summary CSV
        summary_path = result_dir / "summary.csv"
        self._save_summary_csv(summary_path)
        
        print(f"\nResults saved to: {result_dir}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    def _save_summary_csv(self, path: Path):
        """Save iteration results as CSV."""
        import csv
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'iteration', 'sparsity', 'remaining_weights', 
                'test_accuracy', 'test_loss', 'train_accuracy', 'train_loss'
            ])
            # Data rows
            for result in self.results['iterations']:
                writer.writerow([
                    result['iteration'],
                    f"{result['sparsity']:.4f}",
                    f"{result['remaining_weights_fraction']:.4f}",
                    f"{result['test_accuracy']:.2f}",
                    f"{result['test_loss']:.4f}",
                    f"{result['train_accuracy_final']:.2f}",
                    f"{result['train_loss_final']:.4f}"
                ])


def run_imp_experiment(
    model_name: str = "resnet20",
    dataset_name: str = "cifar10",
    target_sparsity: float = 0.9,
    num_iterations: int = 10,
    epochs: int = 160,
    seed: int = 42,
    device: str = "cuda",
    save_dir: str = "./results"
) -> Dict[str, Any]:
    """Convenience function to run a single IMP experiment.
    
    Args:
        model_name: Model architecture name
        dataset_name: Dataset name
        target_sparsity: Target sparsity level
        num_iterations: Number of pruning iterations
        epochs: Training epochs per iteration
        seed: Random seed
        device: Compute device
        save_dir: Results directory
        
    Returns:
        Experiment results dictionary
    """
    num_classes = 10 if dataset_name == "cifar10" else 100
    
    experiment = IMPExperiment(
        model_name=model_name,
        dataset_name=dataset_name,
        num_classes=num_classes,
        target_sparsity=target_sparsity,
        num_iterations=num_iterations,
        epochs_per_iteration=epochs,
        seed=seed,
        device=device,
        save_dir=save_dir
    )
    
    return experiment.run()


def run_full_imp_study(config_path: str = "configs/experiment.yaml"):
    """Run the full IMP study across all configurations in the config file.
    
    This runs IMP experiments for:
    - Multiple datasets (CIFAR-10, CIFAR-100)
    - Multiple architectures (ResNet-20, ResNet-50, VGG-16)
    - Multiple sparsity levels
    - Multiple seeds
    
    Args:
        config_path: Path to experiment configuration file
    """
    config = load_config(config_path)
    
    datasets = config['datasets']
    models = config['models']
    sparsity_levels = config['pruning']['sparsity_levels']
    seeds = config['experiment_settings']['seeds']
    epochs = config['training']['epochs']
    num_iterations = config['pruning']['iterations']
    save_dir = config['logging']['save_dir']
    device = config['hardware']['device']
    
    all_results = []
    
    for dataset in datasets:
        dataset_name = dataset['name']
        num_classes = dataset['num_classes']
        
        for model_name in models:
            for sparsity in sparsity_levels:
                for seed in seeds:
                    print(f"\n{'#'*60}")
                    print(f"Running: {model_name} on {dataset_name}")
                    print(f"Sparsity: {sparsity}, Seed: {seed}")
                    print(f"{'#'*60}")
                    
                    try:
                        experiment = IMPExperiment(
                            model_name=model_name,
                            dataset_name=dataset_name,
                            num_classes=num_classes,
                            target_sparsity=sparsity,
                            num_iterations=num_iterations,
                            epochs_per_iteration=epochs,
                            seed=seed,
                            device=device,
                            save_dir=save_dir
                        )
                        
                        results = experiment.run()
                        all_results.append({
                            'model': model_name,
                            'dataset': dataset_name,
                            'sparsity': sparsity,
                            'seed': seed,
                            'results': results
                        })
                    except Exception as e:
                        print(f"Error running experiment: {e}")
                        continue
    
    # Save aggregated results
    agg_results_path = Path(save_dir) / "imp" / "aggregated_results.json"
    agg_results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(agg_results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nAll experiments complete!")
    print(f"Aggregated results saved to: {agg_results_path}")
    
    return all_results


# Quick experiment for testing/development
def run_quick_imp_test():
    """Run a quick IMP test with reduced parameters for debugging."""
    print("Running quick IMP test...")
    
    results = run_imp_experiment(
        model_name="resnet20",
        dataset_name="cifar10",
        target_sparsity=0.5,
        num_iterations=3,  # Reduced iterations
        epochs=5,  # Reduced epochs
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="./results"
    )
    
    print("\nQuick Test Results:")
    print(f"Initial Accuracy: {results['final_results']['initial_test_accuracy']:.2f}%")
    print(f"Final Accuracy: {results['final_results']['final_test_accuracy']:.2f}%")
    print(f"Final Sparsity: {results['final_results']['final_sparsity']:.2%}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run IMP experiments for Lottery Ticket Hypothesis")
    parser.add_argument("--mode", type=str, default="single", 
                       choices=["single", "full", "quick"],
                       help="Experiment mode: single, full study, or quick test")
    parser.add_argument("--model", type=str, default="resnet20",
                       help="Model architecture")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       help="Dataset name")
    parser.add_argument("--sparsity", type=float, default=0.9,
                       help="Target sparsity")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of pruning iterations")
    parser.add_argument("--epochs", type=int, default=160,
                       help="Epochs per iteration")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml",
                       help="Config file path")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        run_quick_imp_test()
    elif args.mode == "full":
        run_full_imp_study(args.config)
    else:
        run_imp_experiment(
            model_name=args.model,
            dataset_name=args.dataset,
            target_sparsity=args.sparsity,
            num_iterations=args.iterations,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device
        )
