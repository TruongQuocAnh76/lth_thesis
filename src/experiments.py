"""
Experiment runner for Lottery Ticket Hypothesis pruning methods.

This module implements experiments for various pruning methods,
starting with Iterative Magnitude Pruning (IMP).
"""

import sys
import json
import copy
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

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
    get_overall_sparsity,
)
# Early-Bird is in separate module - uses BN gamma, NOT weight magnitudes
from src.earlybird import (
    EarlyBirdFinder,
    extract_bn_gammas,
    compute_channel_mask_from_bn_gamma,
    expand_channel_mask_to_conv_weights,
    add_l1_regularization_to_bn,
    get_bn_layer_count,
    get_overall_channel_sparsity,
)
from src.util import (
    load_config,
    set_seed,
    get_prunable_layers,
    apply_masks_to_model,
    create_mask_apply_fn,
    compute_mask_overlap,
)
from src.grasp import grasp, get_grasp_sparsity
from src.synflow import synflow_pruning, apply_synflow_masks, get_synflow_sparsity
from src.ga import GAConfig, GeneticAlgorithmPruner


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
        self.initial_model_state_dict = initial_state_dict  # Save for later
        
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
        
        # Store final model state and masks for saving
        self.final_model_state_dict = copy.deepcopy(model.state_dict())
        self.final_masks = masks
        
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
        
        # Save model checkpoints if stored during experiment
        if hasattr(self, 'initial_model_state_dict'):
            initial_model_path = result_dir / "initial_model.pt"
            torch.save(self.initial_model_state_dict, initial_model_path)
        
        if hasattr(self, 'final_model_state_dict'):
            final_model_path = result_dir / "final_model.pt"
            torch.save(self.final_model_state_dict, final_model_path)
        
        if hasattr(self, 'final_masks'):
            masks_path = result_dir / "final_masks.pt"
            # Convert numpy arrays to torch tensors for consistency
            masks_torch = {k: torch.from_numpy(v) for k, v in self.final_masks.items()}
            torch.save(masks_torch, masks_path)
        
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


class EarlyBirdExperiment:
    """Early-Bird Lottery Ticket Discovery experiment runner.
    
    Implements the CORRECT Early-Bird algorithm (You et al., ICLR 2020):
    1. Train network with L1 regularization on BatchNorm γ values
    2. At end of each EPOCH, compute channel pruning mask from |γ|
    3. Monitor mask stability: max(last K distances) < ε
    4. When masks stabilize → early-bird ticket found
    5. Train pruned network from initialization using discovered ticket
    
    CRITICAL: Early-Bird uses BatchNorm γ (scaling factors) for channel pruning,
    NOT weight magnitudes. This is fundamentally different from IMP.
    
    Key properties:
    - Signal: BatchNorm γ values (NOT weight magnitudes)
    - Unit: Channels/filters (NOT individual weights) - structured pruning
    - Convergence: max(last K distances) < ε (NOT consecutive counter)
    - Timing: Checked every EPOCH (NOT arbitrary step intervals)
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        num_classes: int,
        target_sparsity: float,
        search_epochs: int = 40,
        finetune_epochs: int = 160,
        batch_size: int = 128,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        l1_coef: float = 1e-4,  # L1 regularization on BN gamma
        seed: int = 42,
        device: str = "cuda",
        save_dir: str = "./results",
        pruning_method: str = 'global',
        patience: int = 5,
        distance_threshold: float = 0.1,
    ):
        """Initialize Early-Bird experiment.
        
        Args:
            model_name: Name of model architecture (must have BatchNorm layers)
            dataset_name: Name of dataset
            num_classes: Number of output classes
            target_sparsity: Target channel sparsity for pruning
            search_epochs: Epochs to search for early-bird ticket
            finetune_epochs: Epochs to train with discovered ticket
            batch_size: Training batch size
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: L2 regularization weight
            l1_coef: L1 regularization coefficient on BN γ (crucial for Early-Bird!)
            seed: Random seed
            device: Device to use
            save_dir: Directory to save results
            pruning_method: 'global' or 'layerwise' channel pruning
            patience: Window size K for max(last K distances) < ε check
            distance_threshold: ε threshold for convergence
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.target_sparsity = target_sparsity
        self.search_epochs = search_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.l1_coef = l1_coef
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.pruning_method = pruning_method
        self.patience = patience
        self.distance_threshold = distance_threshold
        
        # Results storage
        self.results = {
            'config': self._get_config_dict(),
            'search_phase': {},
            'finetune_phase': {},
            'final_results': {}
        }
        
    def _get_config_dict(self) -> Dict:
        """Return experiment configuration as dictionary."""
        return {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'num_classes': self.num_classes,
            'target_sparsity': self.target_sparsity,
            'search_epochs': self.search_epochs,
            'finetune_epochs': self.finetune_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'l1_coef': self.l1_coef,
            'seed': self.seed,
            'pruning_method': self.pruning_method,
            'patience': self.patience,
            'distance_threshold': self.distance_threshold,
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
    
    def _create_scheduler(self, optimizer: optim.Optimizer, epochs: int) -> Any:
        """Create learning rate scheduler."""
        return CosineAnnealingLR(optimizer, T_max=epochs)
    
    def _train_epoch_with_l1(
        self,
        model: nn.Module,
        train_loader: Any,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> Tuple[float, float]:
        """Train one epoch with L1 regularization on BatchNorm γ.
        
        This is the key difference from regular training:
        We add L1 regularization to BN γ values to encourage sparsity,
        which makes Early-Bird convergence detection work.
        
        Returns:
            Tuple of (train_loss, train_accuracy)
        """
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Classification loss
            ce_loss = criterion(output, target)
            
            # L1 regularization on BatchNorm γ (CRUCIAL for Early-Bird!)
            l1_loss = add_l1_regularization_to_bn(model, self.l1_coef)
            
            # Total loss
            loss = ce_loss + l1_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += ce_loss.item()  # Report CE loss only for comparison
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        train_loss = epoch_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc
    
    def _search_for_ticket(
        self,
        model: nn.Module,
        train_loader: Any,
        test_loader: Any,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        finder: EarlyBirdFinder,
        num_epochs: int
    ) -> Dict[str, Any]:
        """Train while searching for Early-Bird ticket via BN γ convergence.
        
        Key difference from old implementation:
        - Checks mask at END of each EPOCH (not every step)
        - Uses BN γ values (not weight magnitudes)
        - Uses max(last K distances) < ε (not consecutive counter)
        
        Returns:
            Dictionary with training history and convergence info
        """
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        
        converged = False
        convergence_epoch = -1
        
        print(f"Searching for Early-Bird ticket via BN γ convergence...")
        print(f"L1 regularization coefficient: {self.l1_coef}")
        pbar = tqdm(range(num_epochs), desc="Search Phase")
        
        for epoch in pbar:
            # Train one epoch with L1 on BN γ
            train_loss, train_acc = self._train_epoch_with_l1(
                model, train_loader, criterion, optimizer
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Evaluate
            test_loss, test_acc = evaluate(model, test_loader, criterion, self.device)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            # Step scheduler
            if scheduler is not None:
                scheduler.step()
            
            # *** EPOCH-LEVEL MASK CHECK (correct Early-Bird behavior) ***
            if not converged:
                conv, distance = finder.record_epoch(model, epoch)
                
                if conv:
                    converged = True
                    convergence_epoch = epoch
                    stats = finder.get_statistics()
                    print(f"\n🎯 Early-Bird ticket found at epoch {epoch}!")
                    print(f"   Max of last {self.patience} distances: {stats.get('max_last_k', 'N/A'):.6f}")
                    print(f"   Threshold: {self.distance_threshold}")
            
            pbar.set_postfix({
                'train_acc': f'{train_acc:.2f}%',
                'test_acc': f'{test_acc:.2f}%',
                'converged': converged,
                'dist': f'{distance:.4f}' if distance is not None else 'N/A'
            })
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
            'converged': converged,
            'convergence_epoch': convergence_epoch,
            'total_epochs': num_epochs
        }
    
    def run(self) -> Dict[str, Any]:
        """Run the complete Early-Bird experiment.
        
        Returns:
            Dictionary containing all experiment results
        """
        print(f"\n{'='*60}")
        print(f"Starting Early-Bird Experiment (BN γ Channel Pruning)")
        print(f"Model: {self.model_name}, Dataset: {self.dataset_name}")
        print(f"Target Channel Sparsity: {self.target_sparsity:.1%}")
        print(f"Search Epochs: {self.search_epochs}")
        print(f"L1 Coefficient: {self.l1_coef}")
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
        
        # Create model and save initial weights
        print("Initializing model...")
        model = self._create_model()
        param_count = count_parameters(model)
        print(f"Model parameters: {param_count['total']:,}")
        
        # Check that model has BatchNorm layers (required for Early-Bird!)
        num_bn_layers = get_bn_layer_count(model)
        if num_bn_layers == 0:
            raise ValueError(
                f"Model {self.model_name} has no BatchNorm layers! "
                "Early-Bird requires BatchNorm γ values for channel pruning."
            )
        print(f"BatchNorm layers: {num_bn_layers}")
        
        # Store initial state (the lottery ticket initialization)
        initial_state_dict = copy.deepcopy(model.state_dict())
        self.initial_model_state_dict = initial_state_dict
        
        criterion = nn.CrossEntropyLoss()
        
        # ===================================================================
        # PHASE 1: Search for Early-Bird Ticket via BN γ Convergence
        # ===================================================================
        print(f"\n{'='*60}")
        print("PHASE 1: Searching for Early-Bird Ticket (BN γ Channel Pruning)")
        print(f"{'='*60}")
        
        # Initialize Early-Bird finder (uses BN γ, not weights!)
        finder = EarlyBirdFinder(
            target_sparsity=self.target_sparsity,
            patience=self.patience,
            distance_threshold=self.distance_threshold,
            pruning_method=self.pruning_method,
        )
        
        # Train with L1 regularization and search for stable mask
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer, self.search_epochs)
        
        search_results = self._search_for_ticket(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            finder=finder,
            num_epochs=self.search_epochs
        )
        
        # Get Early-Bird channel mask
        channel_mask = finder.get_early_bird_ticket()
        ticket_stats = finder.get_statistics()
        
        if channel_mask is None:
            print("\n⚠️  Warning: Early-Bird ticket not found within search epochs!")
            print("    Using final BN γ mask from search phase.")
            bn_gammas = extract_bn_gammas(model)
            channel_mask = compute_channel_mask_from_bn_gamma(
                bn_gammas, self.target_sparsity, self.pruning_method
            )
        
        # Convert channel mask to Conv weight mask
        weight_mask = expand_channel_mask_to_conv_weights(channel_mask, model)
        channel_sparsity = get_overall_channel_sparsity(channel_mask)
        
        print(f"\n{'='*60}")
        print("Search Phase Complete")
        print(f"Converged: {search_results['converged']}")
        if search_results['converged']:
            print(f"Convergence Epoch: {search_results['convergence_epoch']}")
        print(f"Channel Sparsity: {channel_sparsity:.2%}")
        print(f"Search Accuracy: {search_results['test_accs'][-1]:.2f}%")
        print(f"{'='*60}\n")
        
        # Store search results
        self.results['search_phase'] = {
            'train_losses': search_results['train_losses'],
            'train_accs': search_results['train_accs'],
            'test_losses': search_results['test_losses'],
            'test_accs': search_results['test_accs'],
            'converged': search_results['converged'],
            'convergence_epoch': search_results['convergence_epoch'],
            'total_epochs': search_results['total_epochs'],
            'channel_sparsity': channel_sparsity,
            'distance_history': ticket_stats['distance_history'],
            'final_test_acc': search_results['test_accs'][-1]
        }
        
        # ===================================================================
        # PHASE 2: Train with Early-Bird Ticket (Channel-Pruned Network)
        # ===================================================================
        print(f"\n{'='*60}")
        print("PHASE 2: Training with Early-Bird Channel Mask")
        print(f"{'='*60}")
        
        # Reset to initial weights and apply discovered channel mask
        model.load_state_dict(copy.deepcopy(initial_state_dict))
        apply_masks_to_model(model, weight_mask)
        
        # Create fresh optimizer and scheduler for finetuning
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer, self.finetune_epochs)
        apply_mask_fn = create_mask_apply_fn(model)
        
        # Train with the ticket (NO L1 regularization in finetune phase)
        print(f"Training for {self.finetune_epochs} epochs with discovered ticket...")
        finetune_results = train_epochs(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=self.finetune_epochs,
            device=self.device,
            scheduler=scheduler,
            masks=weight_mask,  # Use expanded weight mask from channel mask
            apply_mask_fn=apply_mask_fn,
            verbose=True
        )
        
        # Final evaluation
        final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, self.device)
        
        print(f"\n{'='*60}")
        print("Training Complete")
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")
        print(f"Channel Sparsity: {channel_sparsity:.2%}")
        print(f"{'='*60}\n")
        
        # Store finetune results
        self.results['finetune_phase'] = {
            'train_losses': finetune_results['train_losses'],
            'train_accs': finetune_results['train_accs'],
            'test_losses': finetune_results['test_losses'],
            'test_accs': finetune_results['test_accs'],
            'final_test_acc': final_test_acc,
            'final_test_loss': final_test_loss,
            'channel_sparsities': {k: float(1 - m.mean()) for k, m in channel_mask.items()}
        }
        
        # Final results
        self.results['final_results'] = {
            'channel_sparsity': channel_sparsity,
            'final_test_accuracy': final_test_acc,
            'search_test_accuracy': search_results['test_accs'][-1],
            'converged': search_results['converged'],
            'convergence_epoch': search_results['convergence_epoch'],
            'total_search_epochs': search_results['total_epochs'],
            'training_efficiency': (
                search_results['convergence_epoch'] / self.search_epochs
                if search_results['converged'] else 1.0
            )
        }
        
        # Store final model and masks
        self.final_model_state_dict = copy.deepcopy(model.state_dict())
        self.channel_mask = channel_mask
        self.weight_mask = weight_mask
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save experiment results to disk."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"earlybird_{self.model_name}_{self.dataset_name}_s{self.target_sparsity}_seed{self.seed}"
        result_dir = self.save_dir / "earlybird" / experiment_name / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON
        results_serializable = self._make_serializable(self.results)
        
        # Save JSON results
        results_path = result_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # Save summary CSV
        summary_path = result_dir / "summary.csv"
        self._save_summary_csv(summary_path)
        
        # Save model checkpoints
        if hasattr(self, 'initial_model_state_dict'):
            torch.save(self.initial_model_state_dict, result_dir / "initial_model.pt")
        
        if hasattr(self, 'final_model_state_dict'):
            torch.save(self.final_model_state_dict, result_dir / "final_model.pt")
        
        # Save channel mask (Early-Bird ticket)
        if hasattr(self, 'channel_mask'):
            masks_torch = {k: torch.from_numpy(v) for k, v in self.channel_mask.items()}
            torch.save(masks_torch, result_dir / "channel_mask.pt")
        
        # Save weight mask (expanded from channel mask)
        if hasattr(self, 'weight_mask'):
            masks_torch = {k: torch.from_numpy(v) for k, v in self.weight_mask.items()}
            torch.save(masks_torch, result_dir / "weight_mask.pt")
        
        print(f"\nResults saved to: {result_dir}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python native types."""
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
        """Save summary results as CSV."""
        import csv
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Phase', 'Metric', 'Value'])
            
            # Search phase
            writer.writerow(['Search', 'Converged', self.results['search_phase']['converged']])
            writer.writerow(['Search', 'Convergence Epoch', self.results['search_phase']['convergence_epoch']])
            writer.writerow(['Search', 'Final Test Acc', f"{self.results['search_phase']['final_test_acc']:.2f}"])
            writer.writerow(['Search', 'Channel Sparsity', f"{self.results['search_phase']['channel_sparsity']:.4f}"])
            
            # Finetune phase
            writer.writerow(['Finetune', 'Final Test Acc', f"{self.results['finetune_phase']['final_test_acc']:.2f}"])
            writer.writerow(['Finetune', 'Final Test Loss', f"{self.results['finetune_phase']['final_test_loss']:.4f}"])


def run_earlybird_experiment(
    model_name: str = "resnet20",
    dataset_name: str = "cifar10",
    target_sparsity: float = 0.5,
    search_epochs: int = 40,
    finetune_epochs: int = 160,
    l1_coef: float = 1e-4,
    seed: int = 42,
    device: str = "cuda",
    save_dir: str = "./results"
) -> Dict[str, Any]:
    """Convenience function to run a single Early-Bird experiment.
    
    This implements the CORRECT Early-Bird algorithm:
    - Uses BatchNorm γ values for channel importance (NOT weight magnitudes)
    - Performs channel/filter pruning (NOT individual weight pruning)
    - Applies L1 regularization on BN γ to encourage sparsity
    - Checks mask convergence at epoch level: max(last K distances) < ε
    
    Args:
        model_name: Model architecture name (must have BatchNorm layers!)
        dataset_name: Dataset name
        target_sparsity: Target CHANNEL sparsity (fraction of channels to prune)
        search_epochs: Epochs to search for ticket
        finetune_epochs: Epochs to train with ticket
        l1_coef: L1 regularization coefficient on BN γ (crucial for Early-Bird!)
        seed: Random seed
        device: Compute device
        save_dir: Results directory
        
    Returns:
        Experiment results dictionary
    """
    num_classes = 10 if dataset_name == "cifar10" else 100
    
    experiment = EarlyBirdExperiment(
        model_name=model_name,
        dataset_name=dataset_name,
        num_classes=num_classes,
        target_sparsity=target_sparsity,
        search_epochs=search_epochs,
        finetune_epochs=finetune_epochs,
        l1_coef=l1_coef,
        seed=seed,
        device=device,
        save_dir=save_dir
    )
    
    return experiment.run()


def run_quick_earlybird_test():
    """Run a quick Early-Bird test with reduced parameters for debugging.
    
    This tests the CORRECT Early-Bird implementation:
    - Uses BN γ for channel pruning
    - L1 regularization on BN γ
    - Epoch-level convergence check
    """
    print("Running quick Early-Bird test (BN γ channel pruning)...")
    
    results = run_earlybird_experiment(
        model_name="resnet20",
        dataset_name="cifar10",
        target_sparsity=0.3,  # 30% channels pruned
        search_epochs=5,  # Reduced search
        finetune_epochs=5,  # Reduced training
        l1_coef=1e-4,  # L1 on BN γ
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="./results"
    )
    
    print("\nQuick Test Results:")
    print(f"Converged: {results['final_results']['converged']}")
    if results['final_results']['converged']:
        print(f"Convergence Epoch: {results['final_results']['convergence_epoch']}")
    print(f"Search Accuracy: {results['final_results']['search_test_accuracy']:.2f}%")
    print(f"Final Accuracy: {results['final_results']['final_test_accuracy']:.2f}%")
    print(f"Channel Sparsity: {results['final_results']['channel_sparsity']:.2%}")
    
    return results


class GraSPExperiment:
    """GraSP (Gradient Signal Preservation) experiment runner.

    GraSP is a one-shot pruning method applied at initialisation:
    1. Initialise model (random init, NO training)
    2. Compute GraSP importance scores from gradient flow
    3. Prune to target sparsity in a single step
    4. Train the sparse network from the (masked) initialisation
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        num_classes: int,
        target_sparsity: float,
        epochs: int = 160,
        batch_size: int = 128,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        lr_milestones: List[int] = [80, 120],
        lr_gamma: float = 0.1,
        samples_per_class: int = 10,
        grasp_T: float = 200.0,
        grasp_iters: int = 1,
        seed: int = 42,
        device: str = "cuda",
        save_dir: str = "./results",
    ):
        """Initialise GraSP experiment.

        Args:
            model_name: Architecture name ('resnet20', 'vgg19', …)
            dataset_name: Dataset name ('cifar10', 'cifar100')
            num_classes: Number of output classes
            target_sparsity: Fraction of weights to prune (e.g. 0.9)
            epochs: Training epochs after pruning
            batch_size: Training batch size
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: L2 weight decay
            lr_milestones: Epochs at which to multiply LR by lr_gamma
            lr_gamma: LR decay factor
            samples_per_class: Balanced samples per class for GraSP
            grasp_T: Temperature scaling for GraSP logits
            grasp_iters: Number of balanced batches for gradient accumulation
            seed: Random seed
            device: 'cuda' or 'cpu'
            save_dir: Root directory for saved results
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.target_sparsity = target_sparsity
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_milestones = list(lr_milestones)
        self.lr_gamma = lr_gamma
        self.samples_per_class = samples_per_class
        self.grasp_T = grasp_T
        self.grasp_iters = grasp_iters
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)

        self.results: Dict[str, Any] = {
            'config': self._get_config_dict(),
            'pruning': {},
            'training': {},
            'final_results': {},
        }

    def _get_config_dict(self) -> Dict:
        return {
            'algorithm': 'grasp',
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'num_classes': self.num_classes,
            'target_sparsity': self.target_sparsity,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'lr_milestones': self.lr_milestones,
            'lr_gamma': self.lr_gamma,
            'samples_per_class': self.samples_per_class,
            'grasp_T': self.grasp_T,
            'grasp_iters': self.grasp_iters,
            'seed': self.seed,
        }

    def run(self) -> Dict[str, Any]:
        """Run the complete GraSP experiment.

        Returns:
            Dictionary containing pruning info, training history, and
            final accuracy / sparsity metrics.
        """
        from torch.optim.lr_scheduler import MultiStepLR as _MultiStepLR

        print(f"\n{'='*60}")
        print(f"Starting GraSP Experiment")
        print(f"Model: {self.model_name}, Dataset: {self.dataset_name}")
        print(f"Target Sparsity: {self.target_sparsity:.1%}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        # Seed
        set_seed(self.seed)

        # Data
        loaders = get_dataloaders(
            self.dataset_name, batch_size=self.batch_size, num_workers=4
        )
        train_loader = loaders['train']
        test_loader = loaders['test']

        # Model (Kaiming Normal init)
        model = get_model(self.model_name, num_classes=self.num_classes).to(self.device)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        param_info = count_parameters(model)
        print(f"Model parameters: {param_info['total']:,}")

        # ---- GraSP pruning (one-shot at init) ----
        print(f"Running GraSP pruning (T={self.grasp_T}, "
              f"samples={self.samples_per_class * self.num_classes}) …")
        masks = grasp(
            model, train_loader, self.device,
            sparsity=self.target_sparsity,
            num_classes=self.num_classes,
            samples_per_class=self.samples_per_class,
            num_iters=self.grasp_iters,
            T=self.grasp_T,
        )
        layer_sp = get_grasp_sparsity(masks)
        print(f"Achieved overall sparsity: {layer_sp['overall']*100:.2f}%")
        for name, sp in layer_sp.items():
            if name != 'overall':
                print(f"  {name:>30s}: {sp*100:.2f}%")

        self.results['pruning'] = {
            'overall_sparsity': layer_sp['overall'],
            'layer_sparsities': {k: v for k, v in layer_sp.items() if k != 'overall'},
        }

        # ---- Train ----
        apply_fn = create_mask_apply_fn(model)
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = _MultiStepLR(optimizer, milestones=self.lr_milestones,
                                 gamma=self.lr_gamma)
        criterion = nn.CrossEntropyLoss()

        print(f"Training for {self.epochs} epochs …")
        history = train_epochs(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=self.epochs,
            device=self.device,
            scheduler=scheduler,
            masks=masks,
            apply_mask_fn=apply_fn,
            verbose=True,
        )

        best_test = max(history['test_accs']) if history['test_accs'] else 0.0
        final_test = history['final_test_acc']

        self.results['training'] = {
            'train_losses': history['train_losses'],
            'train_accs': history['train_accs'],
            'test_losses': history['test_losses'],
            'test_accs': history['test_accs'],
        }
        self.results['final_results'] = {
            'best_test_accuracy': best_test,
            'final_test_accuracy': final_test,
            'overall_sparsity': layer_sp['overall'],
        }

        print(f"\n✓ Done — best test acc: {best_test:.2f}%, "
              f"final test acc: {final_test:.2f}%")

        # ---- Save ----
        self._save_results(model, masks)

        return self.results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _save_results(self, model: nn.Module, masks: Dict):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = (f"grasp_{self.model_name}_{self.dataset_name}"
                f"_s{self.target_sparsity}_seed{self.seed}")
        result_dir = self.save_dir / "grasp" / name / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        results_ser = _convert_to_serializable(self.results)
        with open(result_dir / "results.json", 'w') as f:
            json.dump(results_ser, f, indent=2)

        # Summary CSV
        self._save_summary_csv(result_dir / "summary.csv")

        # Checkpoints
        torch.save(model.state_dict(), result_dir / "final_model.pt")
        masks_torch = {k: torch.from_numpy(v) for k, v in masks.items()}
        torch.save(masks_torch, result_dir / "masks.pt")

        print(f"Results saved to: {result_dir}")

    def _save_summary_csv(self, path: Path):
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc',
                             'test_loss', 'test_acc'])
            t = self.results['training']
            for i in range(len(t.get('train_losses', []))):
                writer.writerow([
                    i + 1,
                    f"{t['train_losses'][i]:.4f}",
                    f"{t['train_accs'][i]:.2f}",
                    f"{t['test_losses'][i]:.4f}" if i < len(t.get('test_losses', [])) else '',
                    f"{t['test_accs'][i]:.2f}" if i < len(t.get('test_accs', [])) else '',
                ])


def run_grasp_experiment(
    model_name: str = "resnet20",
    dataset_name: str = "cifar10",
    target_sparsity: float = 0.9,
    epochs: int = 160,
    samples_per_class: int = 10,
    grasp_T: float = 200.0,
    grasp_iters: int = 1,
    learning_rate: float = 0.1,
    lr_milestones: List[int] = [80, 120],
    lr_gamma: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    save_dir: str = "./results",
) -> Dict[str, Any]:
    """Convenience function to run a single GraSP experiment.

    Args:
        model_name: Model architecture ('resnet20', 'vgg19', …)
        dataset_name: Dataset ('cifar10', 'cifar100')
        target_sparsity: Fraction of weights to prune
        epochs: Training epochs after pruning
        samples_per_class: Balanced samples per class for GraSP
        grasp_T: Temperature scaling for GraSP logits
        grasp_iters: Gradient-accumulation batches for GraSP
        learning_rate: Initial learning rate
        lr_milestones: Epochs to decay LR
        lr_gamma: LR decay factor
        seed: Random seed
        device: Compute device
        save_dir: Results directory

    Returns:
        Experiment results dictionary.
    """
    num_classes = 10 if dataset_name.lower() == "cifar10" else 100

    experiment = GraSPExperiment(
        model_name=model_name,
        dataset_name=dataset_name,
        num_classes=num_classes,
        target_sparsity=target_sparsity,
        epochs=epochs,
        samples_per_class=samples_per_class,
        grasp_T=grasp_T,
        grasp_iters=grasp_iters,
        learning_rate=learning_rate,
        lr_milestones=lr_milestones,
        lr_gamma=lr_gamma,
        seed=seed,
        device=device,
        save_dir=save_dir,
    )
    return experiment.run()


def run_quick_grasp_test():
    """Run a quick GraSP test with reduced parameters for debugging."""
    print("Running quick GraSP test …")

    results = run_grasp_experiment(
        model_name="resnet20",
        dataset_name="cifar10",
        target_sparsity=0.5,
        epochs=5,
        samples_per_class=10,
        grasp_T=200.0,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="./results",
    )

    print("\nQuick GraSP Test Results:")
    print(f"Overall Sparsity: {results['final_results']['overall_sparsity']:.2%}")
    print(f"Best Test Accuracy: {results['final_results']['best_test_accuracy']:.2f}%")
    print(f"Final Test Accuracy: {results['final_results']['final_test_accuracy']:.2f}%")
    return results


# ====================================================================
# Genetic Algorithm Experiment
# ====================================================================

class GAExperiment:
    """Genetic Algorithm experiment for finding Lottery Tickets.

    Evolves binary masks over a fixed, randomly-initialised network.
    After the GA discovers the best mask, the sub-network is trained
    from scratch (or from the original initialisation) using standard
    SGD, following the same post-pruning training protocol as IMP /
    SynFlow.

    Pipeline:
    1.  Initialise model (random init, NO training).
    2.  Run GA on the mask space to maximise sub-network quality.
    3.  (Optional) Post-evolutionary pruning for extra sparsity.
    4.  Train the discovered sub-network from the masked init.
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        num_classes: int,
        # GA hyper-parameters
        population_size: int = 100,
        rec_rate: float = 0.3,
        mut_rate: float = 0.1,
        mig_rate: float = 0.1,
        par_rate: float = 0.3,
        min_generations: int = 100,
        max_generations: int = 200,
        stagnation_threshold: int = 50,
        use_adaptive_ab: bool = False,
        initial_ab_threshold: float = 0.7,
        ab_decay_rate: float = 0.95,
        use_loss_fitness: bool = True,
        max_eval_batches: Optional[int] = None,
        post_prune: bool = True,
        # Training hyper-parameters (applied after GA)
        epochs: int = 160,
        batch_size: int = 128,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        lr_milestones: List[int] = [80, 120],
        lr_gamma: float = 0.1,
        seed: int = 42,
        device: str = "cuda",
        save_dir: str = "./results",
        # Checkpoint / resume
        resume_from: Optional[str] = None,
        time_limit_seconds: Optional[float] = None,
        checkpoint_interval: int = 10,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_classes = num_classes

        # Resume / time-limit
        self.resume_from = resume_from
        self.time_limit_seconds = time_limit_seconds
        self.checkpoint_interval = checkpoint_interval

        # Build checkpoint directory
        name_tag = (f"ga_{model_name}_{dataset_name}"
                    f"_pop{population_size}_seed{seed}")
        self._ckpt_dir = Path(save_dir) / "genetic" / name_tag / "checkpoints"

        # GA config
        self.ga_cfg = GAConfig(
            population_size=population_size,
            rec_rate=rec_rate,
            mut_rate=mut_rate,
            mig_rate=mig_rate,
            par_rate=par_rate,
            min_generations=min_generations,
            max_generations=max_generations,
            stagnation_threshold=stagnation_threshold,
            use_adaptive_ab=use_adaptive_ab,
            initial_ab_threshold=initial_ab_threshold,
            ab_decay_rate=ab_decay_rate,
            use_loss_fitness=use_loss_fitness,
            max_eval_batches=max_eval_batches,
            post_prune=post_prune,
            time_limit_seconds=time_limit_seconds,
            checkpoint_dir=str(self._ckpt_dir),
            checkpoint_interval=checkpoint_interval,
        )

        # Training config
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_milestones = list(lr_milestones)
        self.lr_gamma = lr_gamma
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)

        self.results: Dict[str, Any] = {
            'config': self._get_config_dict(),
            'ga': {},
            'training': {},
            'final_results': {},
        }

    def _get_config_dict(self) -> Dict:
        return {
            'algorithm': 'genetic',
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'num_classes': self.num_classes,
            'population_size': self.ga_cfg.population_size,
            'rec_rate': self.ga_cfg.rec_rate,
            'mut_rate': self.ga_cfg.mut_rate,
            'mig_rate': self.ga_cfg.mig_rate,
            'par_rate': self.ga_cfg.par_rate,
            'min_generations': self.ga_cfg.min_generations,
            'max_generations': self.ga_cfg.max_generations,
            'stagnation_threshold': self.ga_cfg.stagnation_threshold,
            'use_adaptive_ab': self.ga_cfg.use_adaptive_ab,
            'use_loss_fitness': self.ga_cfg.use_loss_fitness,
            'post_prune': self.ga_cfg.post_prune,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'lr_milestones': self.lr_milestones,
            'lr_gamma': self.lr_gamma,
            'seed': self.seed,
        }

    # ------------------------------------------------------------------ #
    # Training-phase checkpoint helpers
    # ------------------------------------------------------------------ #

    def _save_train_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        epoch: int,
        masks: Dict[str, np.ndarray],
        initial_state: Dict,
        ga_stats: Dict[str, Any],
        ga_time: float,
        history: Dict[str, list],
        elapsed_train: float,
    ) -> Path:
        """Save a training-phase checkpoint so training can resume later."""
        ckpt_path = self._ckpt_dir / "train_checkpoint.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        masks_torch = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                       for k, v in masks.items()}
        ckpt = {
            'phase': 'train',
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'initial_state': initial_state,
            'masks': masks_torch,
            'ga_stats': ga_stats,
            'ga_time': ga_time,
            'history': history,
            'elapsed_train_seconds': elapsed_train,
        }
        torch.save(ckpt, ckpt_path)
        print(f"\n  Training checkpoint saved → {ckpt_path}  "
              f"(epoch {epoch}/{self.epochs})")
        return ckpt_path

    @staticmethod
    def _load_train_checkpoint(
        path: str | Path,
    ) -> Dict[str, Any]:
        """Load a training-phase checkpoint."""
        return torch.load(Path(path), map_location='cpu', weights_only=False)

    def _time_remaining(self, start: float) -> Optional[float]:
        """Seconds remaining before the configured wall-clock limit."""
        if self.time_limit_seconds is None:
            return None
        return self.time_limit_seconds - (time.time() - start)

    def _should_stop_training(self, start: float, margin: float = 180.0) -> bool:
        """True when we are within *margin* seconds of the time limit."""
        remaining = self._time_remaining(start)
        if remaining is None:
            return False
        return remaining < margin

    # ------------------------------------------------------------------ #
    # Main run
    # ------------------------------------------------------------------ #

    def run(self) -> Dict[str, Any]:
        """Run the full GA→train pipeline with checkpoint / resume support.

        If ``self.resume_from`` points to a checkpoint file the experiment
        picks up where it left off:

        * **GA checkpoint** (``phase='evolve'``): resumes the GA search,
          then trains the discovered sub-network.
        * **Training checkpoint** (``phase='train'``): skips the GA phase
          entirely and resumes SGD training from the saved epoch.

        When a ``time_limit_seconds`` is configured the experiment will
        auto-save a checkpoint and exit gracefully before the deadline.

        Returns:
            Dictionary with GA stats, training history, and final metrics.
        """
        import time as _time
        from torch.optim.lr_scheduler import MultiStepLR as _MultiStepLR
        from src.train import train_epoch, evaluate

        wall_start = _time.time()

        print(f"\n{'='*60}")
        print(f"Starting GA Experiment")
        print(f"Model: {self.model_name}, Dataset: {self.dataset_name}")
        print(f"Population: {self.ga_cfg.population_size}, "
              f"Generations: {self.ga_cfg.min_generations}–{self.ga_cfg.max_generations}")
        print(f"Device: {self.device}")
        if self.resume_from:
            print(f"Resuming from: {self.resume_from}")
        if self.time_limit_seconds:
            print(f"Time limit: {self.time_limit_seconds:.0f}s "
                  f"({self.time_limit_seconds / 3600:.1f}h)")
        print(f"{'='*60}\n")

        # Seed
        set_seed(self.seed)

        # Data
        loaders = get_dataloaders(
            self.dataset_name, batch_size=self.batch_size, num_workers=4
        )
        train_loader = loaders['train']
        test_loader = loaders['test']

        # Model (Kaiming init — same as other experiments)
        model = get_model(self.model_name, num_classes=self.num_classes).to(self.device)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        param_info = count_parameters(model)
        print(f"Model parameters: {param_info['total']:,}")

        # Save initial weights for resetting after GA
        initial_state = copy.deepcopy(model.state_dict())

        # ============================================================== #
        #  Determine resume phase                                         #
        # ============================================================== #
        resume_phase = None          # None | 'evolve' | 'train'
        resume_ckpt = None
        if self.resume_from and Path(self.resume_from).is_file():
            resume_ckpt = torch.load(
                self.resume_from, map_location='cpu', weights_only=False
            )
            resume_phase = resume_ckpt.get('phase', 'evolve')
            print(f"  Checkpoint phase: {resume_phase}")

        # ============================================================== #
        #  Phase 1 — GA mask search (skip if resuming from training)      #
        # ============================================================== #
        if resume_phase == 'train':
            # Training checkpoint — reload masks and GA results from ckpt
            masks_torch = resume_ckpt['masks']
            masks: Dict[str, np.ndarray] = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in masks_torch.items()
            }
            ga_stats = resume_ckpt['ga_stats']
            ga_time = resume_ckpt['ga_time']
            initial_state = resume_ckpt['initial_state']
            print(f"Skipping GA phase (loaded from training checkpoint)")
        else:
            # Fresh GA or resume from GA checkpoint
            print(f"\nRunning Genetic Algorithm mask search …")
            ga_start = _time.time()

            resume_info = None
            if resume_phase == 'evolve':
                pruner, resume_info = GeneticAlgorithmPruner.load_checkpoint(
                    path=self.resume_from,
                    model=model,
                    train_loader=train_loader,
                    device=self.device,
                    config=self.ga_cfg,
                    verbose=True,
                )
            else:
                pruner = GeneticAlgorithmPruner(
                    model=model,
                    train_loader=train_loader,
                    device=self.device,
                    config=self.ga_cfg,
                    verbose=True,
                )

            masks, ga_stats = pruner.run(resume_info=resume_info)
            ga_time = _time.time() - ga_start

            # If GA stopped early due to time limit, save & return early
            if ga_stats.get('stopped_early', False):
                print("\nGA phase stopped early (time limit). "
                      "Re-run with --resume_from to continue.")
                self.results['ga'] = {
                    'stopped_early': True,
                    'total_generations': ga_stats['total_generations'],
                }
                return self.results

        # Compute sparsity from the discovered masks
        total_params = sum(m.size for m in masks.values())
        total_zeros = sum((m == 0).sum() for m in masks.values())
        overall_sparsity = total_zeros / total_params if total_params > 0 else 0.0
        layer_sparsities = {}
        for name, m in masks.items():
            layer_sparsities[name] = float(1.0 - m.mean())

        print(f"GA completed — overall sparsity: {overall_sparsity:.2%}")
        for name, sp in layer_sparsities.items():
            print(f"  {name:>30s}: {sp*100:.2f}%")

        self.results['ga'] = {
            'total_generations': ga_stats['total_generations'],
            'evolve_time_seconds': ga_stats.get('evolve_time_seconds', 0),
            'total_ga_time_seconds': ga_time,
            'best_performance': ga_stats.get('best_performance', 0),
            'best_sparsity': ga_stats.get('best_sparsity', 0),
            'cache_entries': ga_stats.get('cache_entries', 0),
            'overall_sparsity': overall_sparsity,
            'layer_sparsities': layer_sparsities,
            'generation_history': ga_stats.get('generations', []),
            'ga_config': ga_stats.get('config', {}),
        }

        # ============================================================== #
        #  Phase 2 — Train the discovered sub-network                     #
        # ============================================================== #
        model.load_state_dict(initial_state)
        apply_masks_to_model(model, masks)

        apply_fn = create_mask_apply_fn(model)
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = _MultiStepLR(optimizer, milestones=self.lr_milestones,
                                 gamma=self.lr_gamma)
        criterion = nn.CrossEntropyLoss()

        # Resume training state if applicable
        start_epoch = 0
        prior_train_elapsed = 0.0
        history: Dict[str, list] = {
            'train_losses': [], 'train_accs': [],
            'test_losses': [], 'test_accs': [],
        }

        if resume_phase == 'train':
            start_epoch = resume_ckpt['epoch'] + 1
            prior_train_elapsed = resume_ckpt.get('elapsed_train_seconds', 0.0)
            history = resume_ckpt.get('history', history)
            model.load_state_dict(resume_ckpt['model_state_dict'])
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            if resume_ckpt.get('scheduler_state_dict'):
                scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
            # Move optimizer tensors to device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print(f"\nResuming training from epoch {start_epoch}/{self.epochs}")
        else:
            print(f"\nTraining discovered sub-network for {self.epochs} epochs …")

        train_start = _time.time()
        stopped_training = False

        pbar = tqdm(range(start_epoch, self.epochs), desc="Training",
                    disable=False)
        for epoch in pbar:
            # Train one epoch
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer,
                self.device, masks, apply_fn,
            )

            # Evaluate
            test_loss, test_acc = evaluate(model, test_loader, criterion,
                                           self.device)

            if scheduler is not None:
                scheduler.step()

            history['train_losses'].append(train_loss)
            history['train_accs'].append(train_acc)
            history['test_losses'].append(test_loss)
            history['test_accs'].append(test_acc)

            pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.2f}%",
                test_acc=f"{test_acc:.2f}%",
            )

            # Periodic training checkpoint
            if (self.checkpoint_interval > 0
                    and (epoch + 1) % self.checkpoint_interval == 0):
                elapsed_train = prior_train_elapsed + (_time.time() - train_start)
                self._save_train_checkpoint(
                    model, optimizer, scheduler, epoch, masks,
                    initial_state, ga_stats, ga_time,
                    history, elapsed_train,
                )

            # Time-limit check during training
            if self._should_stop_training(wall_start, margin=180.0):
                elapsed_train = prior_train_elapsed + (_time.time() - train_start)
                self._save_train_checkpoint(
                    model, optimizer, scheduler, epoch, masks,
                    initial_state, ga_stats, ga_time,
                    history, elapsed_train,
                )
                remaining = self._time_remaining(wall_start)
                print(f"\n⏰ Time limit approaching ({remaining:.0f}s left). "
                      f"Training checkpoint saved at epoch {epoch + 1}/{self.epochs}.")
                stopped_training = True
                break

        train_time = prior_train_elapsed + (_time.time() - train_start)
        total_time = _time.time() - wall_start

        best_test = max(history['test_accs']) if history['test_accs'] else 0.0
        final_test = history['test_accs'][-1] if history['test_accs'] else 0.0

        self.results['training'] = {
            'train_losses': history['train_losses'],
            'train_accs': history['train_accs'],
            'test_losses': history['test_losses'],
            'test_accs': history['test_accs'],
            'training_time_seconds': train_time,
            'stopped_early': stopped_training,
        }
        self.results['final_results'] = {
            'best_test_accuracy': best_test,
            'final_test_accuracy': final_test,
            'overall_sparsity': overall_sparsity,
            'ga_time_seconds': ga_time,
            'training_time_seconds': train_time,
            'total_time_seconds': total_time,
            'total_generations': ga_stats['total_generations'],
            'completed': not stopped_training,
        }

        status = "INTERRUPTED (checkpoint saved)" if stopped_training else "Done"
        print(f"\n✓ {status} — best test acc: {best_test:.2f}%, "
              f"final test acc: {final_test:.2f}%")
        print(f"  GA time      : {ga_time:.2f}s")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Total time   : {total_time:.2f}s")
        print(f"  Sparsity     : {overall_sparsity:.2%}")

        # ---- Save ---- #
        self._save_results(model, masks)

        return self.results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _save_results(self, model: nn.Module, masks: Dict):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = (f"ga_{self.model_name}_{self.dataset_name}"
                f"_pop{self.ga_cfg.population_size}_seed{self.seed}")
        result_dir = self.save_dir / "genetic" / name / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        results_ser = _convert_to_serializable(self.results)
        with open(result_dir / "results.json", 'w') as f:
            json.dump(results_ser, f, indent=2)

        # Summary CSV
        self._save_summary_csv(result_dir / "summary.csv")

        # GA generation history CSV
        self._save_ga_history_csv(result_dir / "ga_history.csv")

        # Checkpoints
        torch.save(model.state_dict(), result_dir / "final_model.pt")
        masks_torch = {k: torch.from_numpy(v) for k, v in masks.items()}
        torch.save(masks_torch, result_dir / "masks.pt")

        print(f"Results saved to: {result_dir}")

    def _save_summary_csv(self, path: Path):
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc',
                             'test_loss', 'test_acc'])
            t = self.results['training']
            for i in range(len(t.get('train_losses', []))):
                writer.writerow([
                    i + 1,
                    f"{t['train_losses'][i]:.4f}",
                    f"{t['train_accs'][i]:.2f}",
                    f"{t['test_losses'][i]:.4f}" if i < len(t.get('test_losses', [])) else '',
                    f"{t['test_accs'][i]:.2f}" if i < len(t.get('test_accs', [])) else '',
                ])

    def _save_ga_history_csv(self, path: Path):
        import csv
        gens = self.results.get('ga', {}).get('generation_history', [])
        if not gens:
            return
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'best_perf', 'best_sparsity',
                'mean_perf', 'std_perf', 'mean_sparsity',
                'pop_before_select', 'cache_size', 'cache_hit_rate',
            ])
            for g in gens:
                writer.writerow([
                    g['generation'],
                    f"{g['best_perf']:.6f}",
                    f"{g['best_sparsity']:.6f}",
                    f"{g['mean_perf']:.6f}",
                    f"{g['std_perf']:.6f}",
                    f"{g['mean_sparsity']:.6f}",
                    g['pop_size_before_select'],
                    g['cache_size'],
                    f"{g['cache_hit_rate']:.4f}",
                ])


def run_ga_experiment(
    model_name: str = "resnet20",
    dataset_name: str = "cifar10",
    population_size: int = 100,
    min_generations: int = 100,
    max_generations: int = 200,
    stagnation_threshold: int = 50,
    use_loss_fitness: bool = True,
    max_eval_batches: Optional[int] = None,
    post_prune: bool = True,
    epochs: int = 160,
    learning_rate: float = 0.1,
    lr_milestones: List[int] = [80, 120],
    lr_gamma: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    save_dir: str = "./results",
) -> Dict[str, Any]:
    """Convenience function to run a single GA experiment.

    Args:
        model_name:  Model architecture ('resnet20', 'vgg16', …)
        dataset_name: Dataset ('cifar10', 'cifar100')
        population_size: GA population size
        min_generations: Minimum number of GA generations
        max_generations: Maximum number of GA generations
        stagnation_threshold: Stop after this many generations without improvement
        use_loss_fitness: If True, use negative loss as fitness; else accuracy
        max_eval_batches: Limit evaluation batches per individual (None=all)
        post_prune: Apply post-evolutionary pruning
        epochs: Training epochs after GA
        learning_rate: Initial learning rate
        lr_milestones: Epochs to decay LR
        lr_gamma: LR decay factor
        seed: Random seed
        device: Compute device
        save_dir: Results directory

    Returns:
        Experiment results dictionary.
    """
    num_classes = 10 if dataset_name.lower() == "cifar10" else 100

    experiment = GAExperiment(
        model_name=model_name,
        dataset_name=dataset_name,
        num_classes=num_classes,
        population_size=population_size,
        min_generations=min_generations,
        max_generations=max_generations,
        stagnation_threshold=stagnation_threshold,
        use_loss_fitness=use_loss_fitness,
        max_eval_batches=max_eval_batches,
        post_prune=post_prune,
        epochs=epochs,
        learning_rate=learning_rate,
        lr_milestones=lr_milestones,
        lr_gamma=lr_gamma,
        seed=seed,
        device=device,
        save_dir=save_dir,
    )
    return experiment.run()
# ====================================================================
# SynFlow Experiment
# ====================================================================

class SynFlowExperiment:
    """SynFlow (Synaptic Flow) experiment runner.

    SynFlow is a data-free, iterative pruning-at-initialisation method:
    1. Initialise model (random init, NO training)
    2. Iteratively compute SynFlow scores and prune (exponential schedule)
    3. Train the sparse network from the masked initialisation

    The pruning phase itself requires zero training data – it uses an
    all-ones surrogate input.
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        num_classes: int,
        rho: float,
        synflow_iters: int = 100,
        epochs: int = 160,
        batch_size: int = 128,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        lr_milestones: List[int] = [80, 120],
        lr_gamma: float = 0.1,
        seed: int = 42,
        device: str = "cuda",
        save_dir: str = "./results",
    ):
        """Initialise SynFlow experiment.

        Args:
            model_name: Architecture name ('resnet20', 'vgg16', …)
            dataset_name: Dataset name ('cifar10', 'cifar100')
            num_classes: Number of output classes
            rho: Compression ratio ρ ≥ 1.  ρ = 1 means no pruning,
                 ρ = 10 means keep 1/10 of weights (90 % sparsity).
            synflow_iters: Number of iterative SynFlow pruning rounds
            epochs: Training epochs after pruning
            batch_size: Training batch size
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: L2 weight decay
            lr_milestones: Epochs at which to multiply LR by lr_gamma
            lr_gamma: LR decay factor
            seed: Random seed
            device: 'cuda' or 'cpu'
            save_dir: Root directory for saved results
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.rho = rho
        self.synflow_iters = synflow_iters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_milestones = list(lr_milestones)
        self.lr_gamma = lr_gamma
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)

        self.results: Dict[str, Any] = {
            'config': self._get_config_dict(),
            'pruning': {},
            'training': {},
            'final_results': {},
        }

    def _get_config_dict(self) -> Dict:
        return {
            'algorithm': 'synflow',
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'num_classes': self.num_classes,
            'rho': self.rho,
            'synflow_iters': self.synflow_iters,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'lr_milestones': self.lr_milestones,
            'lr_gamma': self.lr_gamma,
            'seed': self.seed,
        }

    def run(self) -> Dict[str, Any]:
        """Run the complete SynFlow experiment.

        Returns:
            Dictionary containing pruning info, training history, and
            final accuracy / sparsity metrics.
        """
        import time as _time
        from torch.optim.lr_scheduler import MultiStepLR as _MultiStepLR

        print(f"\n{'='*60}")
        print(f"Starting SynFlow Experiment")
        print(f"Model: {self.model_name}, Dataset: {self.dataset_name}")
        print(f"Compression ratio ρ: {self.rho:.1f}  (sparsity {1 - 1/self.rho:.1%})")
        print(f"SynFlow iterations: {self.synflow_iters}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        total_start = _time.time()

        # Seed
        set_seed(self.seed)

        # Data
        loaders = get_dataloaders(
            self.dataset_name, batch_size=self.batch_size, num_workers=4
        )
        train_loader = loaders['train']
        test_loader = loaders['test']

        # Model (Kaiming init)
        model = get_model(self.model_name, num_classes=self.num_classes).to(self.device)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        param_info = count_parameters(model)
        print(f"Model parameters: {param_info['total']:,}")

        # Determine input shape from dataset
        input_shape = (3, 32, 32)  # CIFAR-10 / CIFAR-100

        # ---- SynFlow pruning (iterative, data-free) ----
        print(f"Running SynFlow pruning ({self.synflow_iters} iterations) …")
        prune_start = _time.time()

        masks = synflow_pruning(
            model,
            device=self.device,
            rho=self.rho,
            num_iters=self.synflow_iters,
            input_shape=input_shape,
        )

        prune_time = _time.time() - prune_start
        print(f"SynFlow pruning completed in {prune_time:.2f}s")

        # Apply masks to the model
        apply_synflow_masks(model, masks)

        layer_sp = get_synflow_sparsity(masks)
        print(f"Achieved overall sparsity: {layer_sp['overall']*100:.2f}%")
        for name, sp in layer_sp.items():
            if name != 'overall':
                print(f"  {name:>30s}: {sp*100:.2f}%")

        self.results['pruning'] = {
            'overall_sparsity': layer_sp['overall'],
            'layer_sparsities': {k: v for k, v in layer_sp.items() if k != 'overall'},
            'pruning_time_seconds': prune_time,
            'synflow_iters': self.synflow_iters,
        }

        # ---- Train ----
        apply_fn = create_mask_apply_fn(model)
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = _MultiStepLR(optimizer, milestones=self.lr_milestones,
                                 gamma=self.lr_gamma)
        criterion = nn.CrossEntropyLoss()

        print(f"Training for {self.epochs} epochs …")
        train_start = _time.time()

        history = train_epochs(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=self.epochs,
            device=self.device,
            scheduler=scheduler,
            masks=masks,
            apply_mask_fn=apply_fn,
            verbose=True,
        )

        train_time = _time.time() - train_start
        total_time = _time.time() - total_start

        best_test = max(history['test_accs']) if history['test_accs'] else 0.0
        final_test = history['final_test_acc']

        self.results['training'] = {
            'train_losses': history['train_losses'],
            'train_accs': history['train_accs'],
            'test_losses': history['test_losses'],
            'test_accs': history['test_accs'],
            'training_time_seconds': train_time,
        }
        self.results['final_results'] = {
            'best_test_accuracy': best_test,
            'final_test_accuracy': final_test,
            'rho': self.rho,
            'overall_sparsity': layer_sp['overall'],
            'pruning_time_seconds': prune_time,
            'training_time_seconds': train_time,
            'total_time_seconds': total_time,
        }

        print(f"\n✓ Done — best test acc: {best_test:.2f}%, "
              f"final test acc: {final_test:.2f}%")
        print(f"  Pruning time : {prune_time:.2f}s")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Total time   : {total_time:.2f}s")

        # ---- Save ----
        self._save_results(model, masks)

        return self.results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _save_results(self, model: nn.Module, masks: Dict):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = (f"synflow_{self.model_name}_{self.dataset_name}"
                f"_rho{self.rho}_seed{self.seed}")
        result_dir = self.save_dir / "synflow" / name / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        results_ser = _convert_to_serializable(self.results)
        with open(result_dir / "results.json", 'w') as f:
            json.dump(results_ser, f, indent=2)

        # Summary CSV
        self._save_summary_csv(result_dir / "summary.csv")

        # Checkpoints
        torch.save(model.state_dict(), result_dir / "final_model.pt")
        masks_torch = {k: torch.from_numpy(v) for k, v in masks.items()}
        torch.save(masks_torch, result_dir / "masks.pt")

        print(f"Results saved to: {result_dir}")

    def _save_summary_csv(self, path: Path):
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc',
                             'test_loss', 'test_acc'])
            t = self.results['training']
            for i in range(len(t.get('train_losses', []))):
                writer.writerow([
                    i + 1,
                    f"{t['train_losses'][i]:.4f}",
                    f"{t['train_accs'][i]:.2f}",
                    f"{t['test_losses'][i]:.4f}" if i < len(t.get('test_losses', [])) else '',
                    f"{t['test_accs'][i]:.2f}" if i < len(t.get('test_accs', [])) else '',
                ])


def run_synflow_experiment(
    model_name: str = "resnet20",
    dataset_name: str = "cifar10",
    rho: float = 10.0,
    synflow_iters: int = 100,
    epochs: int = 160,
    learning_rate: float = 0.1,
    lr_milestones: List[int] = [80, 120],
    lr_gamma: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    save_dir: str = "./results",
) -> Dict[str, Any]:
    """Convenience function to run a single SynFlow experiment.

    Args:
        model_name: Model architecture ('resnet20', 'vgg16', …)
        dataset_name: Dataset ('cifar10', 'cifar100')
        rho: Compression ratio ρ ≥ 1 (e.g. 10 = keep 10 % of weights)
        synflow_iters: Number of iterative SynFlow pruning rounds
        epochs: Training epochs after pruning
        learning_rate: Initial learning rate
        lr_milestones: Epochs to decay LR
        lr_gamma: LR decay factor
        seed: Random seed
        device: Compute device
        save_dir: Results directory

    Returns:
        Experiment results dictionary.
    """
    num_classes = 10 if dataset_name.lower() == "cifar10" else 100

    experiment = SynFlowExperiment(
        model_name=model_name,
        dataset_name=dataset_name,
        num_classes=num_classes,
        rho=rho,
        synflow_iters=synflow_iters,
        epochs=epochs,
        learning_rate=learning_rate,
        lr_milestones=lr_milestones,
        lr_gamma=lr_gamma,
        seed=seed,
        device=device,
        save_dir=save_dir,
    )
    return experiment.run()


def run_quick_synflow_test():
    """Run a quick SynFlow test with reduced parameters for debugging."""
    print("Running quick SynFlow test …")

    results = run_synflow_experiment(
        model_name="resnet20",
        dataset_name="cifar10",
        rho=2.0,
        synflow_iters=10,
        epochs=5,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="./results",
    )

    print("\nQuick SynFlow Test Results:")
    print(f"Overall Sparsity: {results['final_results']['overall_sparsity']:.2%}")
    print(f"Best Test Accuracy: {results['final_results']['best_test_accuracy']:.2f}%")
    print(f"Final Test Accuracy: {results['final_results']['final_test_accuracy']:.2f}%")
    print(f"Pruning Time: {results['final_results']['pruning_time_seconds']:.2f}s")
    print(f"Total Time: {results['final_results']['total_time_seconds']:.2f}s")
    return results


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


def _convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def _save_earlybird_search_csv(path: Path, history: Dict):
    """Save Early-Bird search phase history as CSV."""
    import csv
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            'epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'mask_distance', 'lr'
        ])
        # Data rows
        if isinstance(history.get('epoch'), list):
            for i in range(len(history['epoch'])):
                writer.writerow([
                    history['epoch'][i],
                    f"{history['train_loss'][i]:.4f}" if i < len(history.get('train_loss', [])) else '',
                    f"{history['train_acc'][i]:.2f}" if i < len(history.get('train_acc', [])) else '',
                    f"{history['test_loss'][i]:.4f}" if i < len(history.get('test_loss', [])) else '',
                    f"{history['test_acc'][i]:.2f}" if i < len(history.get('test_acc', [])) else '',
                    f"{history['mask_distance'][i]:.4f}" if i < len(history.get('mask_distance', [])) else '',
                    f"{history['lr'][i]:.6f}" if i < len(history.get('lr', [])) else '',
                ])


def _save_earlybird_finetune_csv(path: Path, history: Dict):
    """Save Early-Bird fine-tuning phase history as CSV."""
    import csv
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            'epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'
        ])
        # Data rows
        if isinstance(history.get('epoch'), list):
            for i in range(len(history['epoch'])):
                writer.writerow([
                    history['epoch'][i],
                    f"{history['train_loss'][i]:.4f}" if i < len(history.get('train_loss', [])) else '',
                    f"{history['train_acc'][i]:.2f}" if i < len(history.get('train_acc', [])) else '',
                    f"{history['test_loss'][i]:.4f}" if i < len(history.get('test_loss', [])) else '',
                    f"{history['test_acc'][i]:.2f}" if i < len(history.get('test_acc', [])) else '',
                ])


def run_experiment(
    algorithm: str,
    model: str,
    dataset: str,
    seed: int = 42,
    device: str = "cuda",
    **kwargs
) -> Dict[str, Any]:
    """Unified experiment runner supporting multiple algorithms.
    
    Args:
        algorithm: Algorithm to use ('imp', 'earlybird', 'earlybird_resnet')
        model: Model architecture ('resnet20', 'vgg16', etc.)
        dataset: Dataset name ('cifar10', 'mnist', etc.)
        seed: Random seed for reproducibility
        device: Device to use ('cuda' or 'cpu')
        **kwargs: Algorithm-specific arguments
        
    Algorithm-specific kwargs:
        IMP:
            - target_sparsity (float): Target final sparsity (default: 0.9)
            - num_iterations (int): Number of pruning iterations (default: 10)
            - epochs_per_iteration (int): Training epochs per round (default: 160)
            - batch_size (int): Batch size (default: 128)
            - learning_rate (float): Initial learning rate (default: 0.1)
            - momentum (float): SGD momentum (default: 0.9)
            - weight_decay (float): Weight decay (default: 5e-4)
            - use_global_pruning (bool): Global vs layerwise pruning (default: True)
            - warmup_epochs (int): Warmup epochs (default: 5)
            
        Early-Bird (VGG):
            - target_sparsity (float): Target channel sparsity (default: 0.5)
            - search_epochs (int): Max epochs for ticket search (default: 160)
            - finetune_epochs (int): Fine-tuning epochs (default: 160)
            - batch_size (int): Batch size (default: 256)
            - learning_rate (float): Initial learning rate (default: 0.1)
            - momentum (float): SGD momentum (default: 0.9)
            - weight_decay (float): Weight decay (default: 1e-4)
            - l1_coef (float): L1 regularization for BN gamma (default: 1e-4)
            - distance_threshold (float): Convergence threshold (default: 0.1)
            - patience (int): Convergence window (default: 5)
            - pruning_method (str): 'global' or 'layerwise' (default: 'global')
            
        Early-Bird ResNet:
            - Same as Early-Bird VGG (uses block-wise pruning)
    
    Returns:
        Dictionary containing experiment results
    """
    print(f"\n{'='*80}")
    print(f"Running Experiment")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    if algorithm.lower() == "imp":
        # IMP-specific parameters
        target_sparsity = kwargs.get('target_sparsity', 0.9)
        num_iterations = kwargs.get('num_iterations', 10)
        epochs_per_iteration = kwargs.get('epochs_per_iteration', 160)
        batch_size = kwargs.get('batch_size', 128)
        learning_rate = kwargs.get('learning_rate', 0.1)
        momentum = kwargs.get('momentum', 0.9)
        weight_decay = kwargs.get('weight_decay', 5e-4)
        use_global_pruning = kwargs.get('use_global_pruning', True)
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        
        # Determine number of classes from dataset
        num_classes = 10 if dataset.lower() in ['cifar10', 'mnist'] else 100
        
        experiment = IMPExperiment(
            model_name=model,
            dataset_name=dataset,
            num_classes=num_classes,
            target_sparsity=target_sparsity,
            num_iterations=num_iterations,
            epochs_per_iteration=epochs_per_iteration,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            seed=seed,
            device=device,
            use_global_pruning=use_global_pruning,
            warmup_epochs=warmup_epochs
        )
        return experiment.run()
    
    elif algorithm.lower() == "earlybird":
        # Early-Bird VGG parameters
        target_sparsity = kwargs.get('target_sparsity', 0.5)
        search_epochs = kwargs.get('search_epochs', 160)
        finetune_epochs = kwargs.get('finetune_epochs', 160)
        batch_size = kwargs.get('batch_size', 256)
        learning_rate = kwargs.get('learning_rate', 0.1)
        momentum = kwargs.get('momentum', 0.9)
        weight_decay = kwargs.get('weight_decay', 1e-4)
        l1_coef = kwargs.get('l1_coef', 1e-4)
        distance_threshold = kwargs.get('distance_threshold', 0.1)
        patience = kwargs.get('patience', 5)
        pruning_method = kwargs.get('pruning_method', 'global')
        
        num_classes = 10 if dataset.lower() in ['cifar10', 'mnist'] else 100
        
        experiment = EarlyBirdExperiment(
            model_name=model,
            dataset_name=dataset,
            num_classes=num_classes,
            target_sparsity=target_sparsity,
            search_epochs=search_epochs,
            finetune_epochs=finetune_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            l1_coef=l1_coef,
            distance_threshold=distance_threshold,
            patience=patience,
            pruning_method=pruning_method,
            seed=seed,
            device=device
        )
        return experiment.run()
    
    elif algorithm.lower() == "earlybird_resnet":
        # Early-Bird ResNet with block-wise pruning
        from src.train import train_resnet20_earlybird
        from src.data import get_dataloaders
        
        target_sparsity = kwargs.get('target_sparsity', 0.5)
        total_epochs = kwargs.get('total_epochs', 160)
        batch_size = kwargs.get('batch_size', 128)
        learning_rate = kwargs.get('learning_rate', 0.1)
        lr_milestones = kwargs.get('lr_milestones', [80, 120])
        lr_gamma = kwargs.get('lr_gamma', 0.1)
        momentum = kwargs.get('momentum', 0.9)
        weight_decay = kwargs.get('weight_decay', 1e-4)
        l1_coef = kwargs.get('l1_coef', 1e-4)
        distance_threshold = kwargs.get('distance_threshold', 0.1)
        patience = kwargs.get('patience', 5)
        pruning_method = kwargs.get('pruning_method', 'global')
        verbose = kwargs.get('verbose', True)
        
        # Determine number of classes from dataset
        num_classes = 10 if dataset.lower() in ['cifar10', 'mnist'] else 100
        
        # Set seed
        set_seed(seed)
        
        # Load data
        loaders = get_dataloaders(dataset, batch_size=batch_size, num_workers=4)
        train_loader = loaders['train']
        test_loader = loaders['test']
        
        # Run experiment
        device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
        results = train_resnet20_earlybird(
            train_loader=train_loader,
            test_loader=test_loader,
            device=device_obj,
            num_classes=num_classes,
            target_sparsity=target_sparsity,
            total_epochs=total_epochs,
            initial_lr=learning_rate,
            lr_milestones=lr_milestones,
            lr_gamma=lr_gamma,
            momentum=momentum,
            weight_decay=weight_decay,
            l1_coef=l1_coef,
            distance_threshold=distance_threshold,
            patience=patience,
            pruning_method=pruning_method,
            verbose=verbose
        )
        
        # Save results in same format as IMP
        save_dir = Path("./results")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"earlybird_resnet_{model}_{dataset}_s{target_sparsity:.2f}_seed{seed}"
        result_dir = save_dir / "earlybird_resnet" / experiment_name / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for saving (exclude model and tensors)
        save_results = {
            'config': {
                'algorithm': algorithm,
                'model': model,
                'dataset': dataset,
                'seed': seed,
                'target_sparsity': target_sparsity,
                'total_epochs': total_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'lr_milestones': lr_milestones,
                'lr_gamma': lr_gamma,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'l1_coef': l1_coef,
                'distance_threshold': distance_threshold,
                'patience': patience,
                'pruning_method': pruning_method,
            },
            'final_results': {
                'converged': results.get('converged', False),
                'convergence_epoch': results.get('convergence_epoch', None),
                'best_test_accuracy': results.get('best_test_acc', None),
                'test_accuracy_at_convergence': results.get('test_acc_pruned', None),
                'final_sparsity': results.get('actual_sparsity', None),
            },
            'search_history': _convert_to_serializable(results.get('history', {})),
            'finetune_history': _convert_to_serializable(results.get('finetune_history', {})),
            'eb_stats': _convert_to_serializable(results.get('eb_stats', {})),
        }
        
        # Save JSON results
        results_path = result_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        # Save summary CSV (search phase)
        if results.get('history'):
            summary_path = result_dir / "search_summary.csv"
            _save_earlybird_search_csv(summary_path, results['history'])
        
        # Save summary CSV (finetune phase)
        if results.get('finetune_history'):
            summary_path = result_dir / "finetune_summary.csv"
            _save_earlybird_finetune_csv(summary_path, results['finetune_history'])
        
        print(f"\nResults saved to: {result_dir}")
        return results
    
    elif algorithm.lower() == "grasp":
        # GraSP parameters
        target_sparsity = kwargs.get('target_sparsity', 0.9)
        epochs = kwargs.get('epochs', 160)
        batch_size = kwargs.get('batch_size', 128)
        learning_rate = kwargs.get('learning_rate', 0.1)
        momentum = kwargs.get('momentum', 0.9)
        weight_decay = kwargs.get('weight_decay', 1e-4)
        lr_milestones = kwargs.get('lr_milestones', [80, 120])
        lr_gamma = kwargs.get('lr_gamma', 0.1)
        samples_per_class = kwargs.get('samples_per_class', 10)
        grasp_T = kwargs.get('grasp_T', 200.0)
        grasp_iters = kwargs.get('grasp_iters', 1)

        num_classes = 10 if dataset.lower() in ['cifar10', 'mnist'] else 100

        experiment = GraSPExperiment(
            model_name=model,
            dataset_name=dataset,
            num_classes=num_classes,
            target_sparsity=target_sparsity,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            lr_milestones=lr_milestones,
            lr_gamma=lr_gamma,
            samples_per_class=samples_per_class,
            grasp_T=grasp_T,
            grasp_iters=grasp_iters,
            seed=seed,
            device=device,
        )
        return experiment.run()

    elif algorithm.lower() == "synflow":
        # SynFlow parameters
        rho = kwargs.get('rho', 10.0)
        synflow_iters = kwargs.get('synflow_iters', 100)
        epochs = kwargs.get('epochs', 160)
        batch_size = kwargs.get('batch_size', 128)
        learning_rate = kwargs.get('learning_rate', 0.1)
        momentum = kwargs.get('momentum', 0.9)
        weight_decay = kwargs.get('weight_decay', 1e-4)
        lr_milestones = kwargs.get('lr_milestones', [80, 120])
        lr_gamma = kwargs.get('lr_gamma', 0.1)

        num_classes = 10 if dataset.lower() in ['cifar10', 'mnist'] else 100

        experiment = SynFlowExperiment(
            model_name=model,
            dataset_name=dataset,
            num_classes=num_classes,
            rho=rho,
            synflow_iters=synflow_iters,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            lr_milestones=lr_milestones,
            lr_gamma=lr_gamma,
            seed=seed,
            device=device,
        )
        return experiment.run()

    elif algorithm.lower() == "genetic":
        # GA parameters
        population_size = kwargs.get('population_size', 100)
        rec_rate = kwargs.get('rec_rate', 0.3)
        mut_rate = kwargs.get('mut_rate', 0.1)
        mig_rate = kwargs.get('mig_rate', 0.1)
        par_rate = kwargs.get('par_rate', 0.3)
        min_generations = kwargs.get('min_generations', 100)
        max_generations = kwargs.get('max_generations', 200)
        stagnation_threshold = kwargs.get('stagnation_threshold', 50)
        use_adaptive_ab = kwargs.get('use_adaptive_ab', False)
        initial_ab_threshold = kwargs.get('initial_ab_threshold', 0.7)
        ab_decay_rate = kwargs.get('ab_decay_rate', 0.95)
        use_loss_fitness = kwargs.get('use_loss_fitness', True)
        max_eval_batches = kwargs.get('max_eval_batches', None)
        post_prune = kwargs.get('post_prune', True)
        epochs = kwargs.get('epochs', 160)
        batch_size = kwargs.get('batch_size', 128)
        learning_rate = kwargs.get('learning_rate', 0.1)
        momentum = kwargs.get('momentum', 0.9)
        weight_decay = kwargs.get('weight_decay', 1e-4)
        lr_milestones = kwargs.get('lr_milestones', [80, 120])
        lr_gamma = kwargs.get('lr_gamma', 0.1)
        resume_from = kwargs.get('resume_from', None)
        time_limit_seconds = kwargs.get('time_limit_seconds', None)
        checkpoint_interval = kwargs.get('checkpoint_interval', 10)

        num_classes = 10 if dataset.lower() in ['cifar10', 'mnist'] else 100

        experiment = GAExperiment(
            model_name=model,
            dataset_name=dataset,
            num_classes=num_classes,
            population_size=population_size,
            rec_rate=rec_rate,
            mut_rate=mut_rate,
            mig_rate=mig_rate,
            par_rate=par_rate,
            min_generations=min_generations,
            max_generations=max_generations,
            stagnation_threshold=stagnation_threshold,
            use_adaptive_ab=use_adaptive_ab,
            initial_ab_threshold=initial_ab_threshold,
            ab_decay_rate=ab_decay_rate,
            use_loss_fitness=use_loss_fitness,
            max_eval_batches=max_eval_batches,
            post_prune=post_prune,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            lr_milestones=lr_milestones,
            lr_gamma=lr_gamma,
            seed=seed,
            device=device,
            resume_from=resume_from,
            time_limit_seconds=time_limit_seconds,
            checkpoint_interval=checkpoint_interval,
        )
        return experiment.run()

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                         f"Choose from: imp, earlybird, earlybird_resnet, grasp, synflow, genetic")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run pruning experiments for Lottery Ticket Hypothesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run IMP on ResNet20 with CIFAR-10
  python -m src.experiments --algorithm imp --model resnet20 --dataset cifar10 --seed 42 \\
      --target_sparsity 0.9 --num_iterations 10 --epochs_per_iteration 160

  # Run Early-Bird on VGG16 with CIFAR-10
  python -m src.experiments --algorithm earlybird --model vgg16 --dataset cifar10 --seed 42 \\
      --target_sparsity 0.5 --search_epochs 160 --finetune_epochs 160

  # Run Early-Bird on ResNet20 with CIFAR-10 (block-wise pruning)
  python -m src.experiments --algorithm earlybird_resnet --model resnet20 --dataset cifar10 --seed 42 \\
      --target_sparsity 0.5 --total_epochs 160 --l1_coef 1e-4

  # Run GraSP on ResNet20 with CIFAR-10
  python -m src.experiments --algorithm grasp --model resnet20 --dataset cifar10 --seed 42 \\
      --target_sparsity 0.9 --epochs 160 --samples_per_class 10 --grasp_T 200

  # Run GraSP on ResNet20 with CIFAR-100
  python -m src.experiments --algorithm grasp --model resnet20 --dataset cifar100 --seed 42 \\
      --target_sparsity 0.9 --epochs 160 --samples_per_class 10 --grasp_T 200

  # Run SynFlow on ResNet20 with CIFAR-10
  python -m src.experiments --algorithm synflow --model resnet20 --dataset cifar10 --seed 42 \\
      --rho 10 --epochs 160 --synflow_iters 100

  # Run SynFlow on ResNet20 with CIFAR-100
  python -m src.experiments --algorithm synflow --model resnet20 --dataset cifar100 --seed 42 \\
      --rho 10 --epochs 160 --synflow_iters 100

  # Run GA on ResNet20 with CIFAR-10
  python -m src.experiments --algorithm genetic --model resnet20 --dataset cifar10 --seed 42 \\
      --population_size 100 --min_generations 100 --max_generations 200 --epochs 160

  # Run GA on ResNet20 with CIFAR-100
  python -m src.experiments --algorithm genetic --model resnet20 --dataset cifar100 --seed 42 \\
      --population_size 100 --min_generations 100 --max_generations 200 --epochs 160

  # Quick test modes
  python -m src.experiments --mode quick_imp
  python -m src.experiments --mode quick_earlybird
  python -m src.experiments --mode quick_grasp
  python -m src.experiments --mode quick_synflow
        """
    )
    
    # Main arguments
    parser.add_argument("--algorithm", type=str, default="imp",
                       choices=["imp", "earlybird", "earlybird_resnet", "grasp", "synflow", "genetic"],
                       help="Pruning algorithm to use")
    parser.add_argument("--model", type=str, default="resnet20",
                       help="Model architecture (resnet20, vgg16, etc.)")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       help="Dataset name (cifar10, cifar100, mnist)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    
    # Special modes
    parser.add_argument("--mode", type=str, default=None,
                       choices=["quick_imp", "quick_earlybird", "quick_grasp", "quick_synflow", "full"],
                       help="Quick test or full experiment mode")
    
    # IMP-specific arguments
    imp_group = parser.add_argument_group('IMP arguments')
    imp_group.add_argument("--target_sparsity", type=float, default=0.9,
                          help="Target final sparsity (IMP)")
    imp_group.add_argument("--num_iterations", type=int, default=10,
                          help="Number of pruning iterations (IMP)")
    imp_group.add_argument("--epochs_per_iteration", type=int, default=160,
                          help="Training epochs per iteration (IMP)")
    imp_group.add_argument("--use_global_pruning", action="store_true", default=True,
                          help="Use global magnitude pruning (IMP)")
    imp_group.add_argument("--warmup_epochs", type=int, default=5,
                          help="Warmup epochs (IMP)")
    
    # Early-Bird common arguments
    eb_group = parser.add_argument_group('Early-Bird arguments')
    eb_group.add_argument("--search_epochs", type=int, default=160,
                         help="Max search epochs (Early-Bird)")
    eb_group.add_argument("--finetune_epochs", type=int, default=160,
                         help="Fine-tuning epochs (Early-Bird)")
    eb_group.add_argument("--total_epochs", type=int, default=160,
                         help="Total epochs (Early-Bird ResNet)")
    eb_group.add_argument("--l1_coef", type=float, default=1e-4,
                         help="L1 coefficient for BN gamma (Early-Bird)")
    eb_group.add_argument("--distance_threshold", type=float, default=0.1,
                         help="Convergence threshold (Early-Bird)")
    eb_group.add_argument("--patience", type=int, default=5,
                         help="Convergence window (Early-Bird)")
    eb_group.add_argument("--pruning_method", type=str, default="global",
                         choices=["global", "layerwise"],
                         help="Pruning method (Early-Bird)")
    
    # GraSP-specific arguments
    grasp_group = parser.add_argument_group('GraSP arguments')
    grasp_group.add_argument("--samples_per_class", type=int, default=10,
                            help="Balanced samples per class for GraSP scoring batch")
    grasp_group.add_argument("--grasp_T", type=float, default=200.0,
                            help="Temperature scaling for GraSP logits")
    grasp_group.add_argument("--grasp_iters", type=int, default=1,
                            help="Number of balanced batches for GraSP gradient accumulation")
    grasp_group.add_argument("--epochs", type=int, default=160,
                            help="Training epochs after GraSP pruning")

    # GA-specific arguments
    ga_group = parser.add_argument_group('Genetic Algorithm arguments')
    ga_group.add_argument("--population_size", type=int, default=100,
                         help="GA population size (default: 100)")
    ga_group.add_argument("--rec_rate", type=float, default=0.3,
                         help="Recombination rate (default: 0.3)")
    ga_group.add_argument("--mut_rate", type=float, default=0.1,
                         help="Mutation rate (default: 0.1)")
    ga_group.add_argument("--mig_rate", type=float, default=0.1,
                         help="Migration rate (default: 0.1)")
    ga_group.add_argument("--par_rate", type=float, default=0.3,
                         help="Parent pool rate for mating (default: 0.3)")
    ga_group.add_argument("--min_generations", type=int, default=100,
                         help="Minimum number of GA generations (default: 100)")
    ga_group.add_argument("--max_generations", type=int, default=200,
                         help="Maximum number of GA generations (default: 200)")
    ga_group.add_argument("--stagnation_threshold", type=int, default=50,
                         help="Stop after N generations without improvement (default: 50)")
    ga_group.add_argument("--use_adaptive_ab", action="store_true", default=False,
                         help="Use adaptive accuracy bound for population init")
    ga_group.add_argument("--use_loss_fitness", action="store_true", default=True,
                         help="Use negative loss as fitness (default: True)")
    ga_group.add_argument("--max_eval_batches", type=int, default=None,
                         help="Max batches per fitness evaluation (None=all data)")
    ga_group.add_argument("--no_post_prune", action="store_true", default=False,
                         help="Disable post-evolutionary pruning")

    # Checkpoint / resume arguments (for GA on Kaggle etc.)
    ckpt_group = parser.add_argument_group('Checkpoint / resume arguments')
    ckpt_group.add_argument("--resume_from", type=str, default=None,
                           help="Path to a checkpoint file to resume from "
                                "(GA or training phase)")
    ckpt_group.add_argument("--time_limit", type=float, default=None,
                           help="Wall-clock time limit in seconds. The experiment "
                                "will auto-save a checkpoint before this deadline. "
                                "E.g. 39600 for Kaggle 11-hour safety margin")
    ckpt_group.add_argument("--checkpoint_interval", type=int, default=10,
                           help="Save a checkpoint every N GA generations / "
                                "training epochs (default: 10)")

    # SynFlow-specific arguments
    synflow_group = parser.add_argument_group('SynFlow arguments')
    synflow_group.add_argument("--rho", type=float, default=10.0,
                              help="Compression ratio ρ ≥ 1 (default: 10 = 90%% sparsity). "
                                   "ρ=1 no pruning, ρ=10 keep 10%%, ρ=100 keep 1%%")
    synflow_group.add_argument("--synflow_iters", type=int, default=100,
                              help="Number of iterative SynFlow pruning rounds (default: 100)")

    # Common training arguments
    train_group = parser.add_argument_group('Training arguments')
    train_group.add_argument("--batch_size", type=int, default=None,
                            help="Batch size (default: 128 for IMP, 256 for Early-Bird)")
    train_group.add_argument("--learning_rate", type=float, default=0.1,
                            help="Initial learning rate")
    train_group.add_argument("--momentum", type=float, default=0.9,
                            help="SGD momentum")
    train_group.add_argument("--weight_decay", type=float, default=None,
                            help="Weight decay (default: 5e-4 for IMP, 1e-4 for Early-Bird)")
    train_group.add_argument("--lr_milestones", type=int, nargs="+", default=[80, 120],
                            help="Learning rate milestones (Early-Bird ResNet)")
    train_group.add_argument("--lr_gamma", type=float, default=0.1,
                            help="Learning rate gamma (Early-Bird ResNet)")
    
    args = parser.parse_args()
    
    # Handle quick test modes
    if args.mode == "quick_imp":
        run_quick_imp_test()
    elif args.mode == "quick_earlybird":
        run_quick_earlybird_test()
    elif args.mode == "quick_grasp":
        run_quick_grasp_test()
    elif args.mode == "quick_synflow":
        run_quick_synflow_test()
    elif args.mode == "full":
        run_full_imp_study()
    else:
        # Build kwargs based on algorithm
        kwargs = {}
        
        if args.algorithm == "imp":
            kwargs['target_sparsity'] = args.target_sparsity
            kwargs['num_iterations'] = args.num_iterations
            kwargs['epochs_per_iteration'] = args.epochs_per_iteration
            kwargs['use_global_pruning'] = args.use_global_pruning
            kwargs['warmup_epochs'] = args.warmup_epochs
            kwargs['batch_size'] = args.batch_size if args.batch_size else 128
            kwargs['weight_decay'] = args.weight_decay if args.weight_decay else 5e-4
            
        elif args.algorithm == "earlybird":
            kwargs['target_sparsity'] = args.target_sparsity if args.target_sparsity != 0.9 else 0.5
            kwargs['search_epochs'] = args.search_epochs
            kwargs['finetune_epochs'] = args.finetune_epochs
            kwargs['l1_coef'] = args.l1_coef
            kwargs['distance_threshold'] = args.distance_threshold
            kwargs['patience'] = args.patience
            kwargs['pruning_method'] = args.pruning_method
            kwargs['batch_size'] = args.batch_size if args.batch_size else 256
            kwargs['weight_decay'] = args.weight_decay if args.weight_decay else 1e-4
            
        elif args.algorithm == "earlybird_resnet":
            kwargs['target_sparsity'] = args.target_sparsity if args.target_sparsity != 0.9 else 0.5
            kwargs['total_epochs'] = args.total_epochs
            kwargs['lr_milestones'] = args.lr_milestones
            kwargs['lr_gamma'] = args.lr_gamma
            kwargs['l1_coef'] = args.l1_coef
            kwargs['distance_threshold'] = args.distance_threshold
            kwargs['patience'] = args.patience
            kwargs['pruning_method'] = args.pruning_method
            kwargs['batch_size'] = args.batch_size if args.batch_size else 128
            kwargs['weight_decay'] = args.weight_decay if args.weight_decay else 1e-4

        elif args.algorithm == "grasp":
            kwargs['target_sparsity'] = args.target_sparsity
            kwargs['epochs'] = args.epochs
            kwargs['lr_milestones'] = args.lr_milestones
            kwargs['lr_gamma'] = args.lr_gamma
            kwargs['samples_per_class'] = args.samples_per_class
            kwargs['grasp_T'] = args.grasp_T
            kwargs['grasp_iters'] = args.grasp_iters
            kwargs['batch_size'] = args.batch_size if args.batch_size else 128
            kwargs['weight_decay'] = args.weight_decay if args.weight_decay else 1e-4

        elif args.algorithm == "synflow":
            kwargs['rho'] = args.rho
            kwargs['synflow_iters'] = args.synflow_iters
            kwargs['epochs'] = args.epochs
            kwargs['lr_milestones'] = args.lr_milestones
            kwargs['lr_gamma'] = args.lr_gamma
            kwargs['batch_size'] = args.batch_size if args.batch_size else 128
            kwargs['weight_decay'] = args.weight_decay if args.weight_decay else 1e-4

        elif args.algorithm == "genetic":
            kwargs['population_size'] = args.population_size
            kwargs['rec_rate'] = args.rec_rate
            kwargs['mut_rate'] = args.mut_rate
            kwargs['mig_rate'] = args.mig_rate
            kwargs['par_rate'] = args.par_rate
            kwargs['min_generations'] = args.min_generations
            kwargs['max_generations'] = args.max_generations
            kwargs['stagnation_threshold'] = args.stagnation_threshold
            kwargs['use_adaptive_ab'] = args.use_adaptive_ab
            kwargs['use_loss_fitness'] = args.use_loss_fitness
            kwargs['max_eval_batches'] = args.max_eval_batches
            kwargs['post_prune'] = not args.no_post_prune
            kwargs['epochs'] = args.epochs
            kwargs['lr_milestones'] = args.lr_milestones
            kwargs['lr_gamma'] = args.lr_gamma
            kwargs['batch_size'] = args.batch_size if args.batch_size else 128
            kwargs['weight_decay'] = args.weight_decay if args.weight_decay else 1e-4
            kwargs['resume_from'] = args.resume_from
            kwargs['time_limit_seconds'] = args.time_limit
            kwargs['checkpoint_interval'] = args.checkpoint_interval

        # Common arguments
        kwargs['learning_rate'] = args.learning_rate
        kwargs['momentum'] = args.momentum
        
        # Run experiment
        results = run_experiment(
            algorithm=args.algorithm,
            model=args.model,
            dataset=args.dataset,
            seed=args.seed,
            device=args.device,
            **kwargs
        )
        
        # Print summary
        print(f"\n{'='*80}")
        print("Experiment Complete!")
        if args.algorithm == "earlybird_resnet":
            if results.get('converged'):
                print(f"Converged at epoch: {results['convergence_epoch']}")
                print(f"Best test accuracy: {results['best_test_acc']:.2f}%")
                print(f"Actual sparsity: {results['actual_sparsity']:.2f}%")
            else:
                print("Did not converge within epoch limit")
        elif args.algorithm == "grasp":
            fr = results.get('final_results', {})
            print(f"Overall Sparsity: {fr.get('overall_sparsity', 0):.2%}")
            print(f"Best Test Accuracy: {fr.get('best_test_accuracy', 0):.2f}%")
            print(f"Final Test Accuracy: {fr.get('final_test_accuracy', 0):.2f}%")
        elif args.algorithm == "synflow":
            fr = results.get('final_results', {})
            print(f"Compression Ratio ρ: {fr.get('rho', 0):.1f}")
            print(f"Overall Sparsity: {fr.get('overall_sparsity', 0):.2%}")
            print(f"Best Test Accuracy: {fr.get('best_test_accuracy', 0):.2f}%")
            print(f"Final Test Accuracy: {fr.get('final_test_accuracy', 0):.2f}%")
            print(f"Pruning Time: {fr.get('pruning_time_seconds', 0):.2f}s")
            print(f"Total Time: {fr.get('total_time_seconds', 0):.2f}s")
        elif args.algorithm == "genetic":
            fr = results.get('final_results', {})
            print(f"Overall Sparsity: {fr.get('overall_sparsity', 0):.2%}")
            print(f"Best Test Accuracy: {fr.get('best_test_accuracy', 0):.2f}%")
            print(f"Final Test Accuracy: {fr.get('final_test_accuracy', 0):.2f}%")
            print(f"GA Time: {fr.get('ga_time_seconds', 0):.2f}s")
            print(f"Generations: {fr.get('total_generations', 0)}")
            print(f"Total Time: {fr.get('total_time_seconds', 0):.2f}s")
        print(f"{'='*80}\n")
