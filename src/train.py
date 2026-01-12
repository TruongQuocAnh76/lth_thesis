import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, List, Any, Callable


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    masks: Optional[Dict[str, Any]] = None,
    apply_mask_fn: Optional[Callable] = None
) -> tuple:
    """Train model for one epoch.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training (cuda/cpu)
        masks: Optional dictionary of pruning masks
        apply_mask_fn: Optional function to apply masks to model
    
    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Apply mask after gradient update if pruning is active
        if masks is not None:
            if apply_mask_fn is not None:
                apply_mask_fn(masks)
            elif hasattr(model, 'apply_mask'):
                model.apply_mask(masks)
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Evaluate model on test/validation set.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use for evaluation
    
    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return test_loss / len(test_loader), 100. * correct / total


def train_iterations(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_iterations: int,
    device: torch.device,
    masks: Optional[Dict[str, Any]] = None,
    eval_freq: int = 100,
    apply_mask_fn: Optional[Callable] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Train for a specified number of iterations.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer
        num_iterations: Total number of training iterations
        device: Device to use for training
        masks: Optional dictionary of pruning masks
        eval_freq: Frequency of evaluation (in iterations)
        apply_mask_fn: Optional function to apply masks to model
        verbose: Whether to show progress bar
    
    Returns:
        Dictionary containing training history:
        - 'train_losses': List of training losses per iteration
        - 'train_accs': List of training accuracies per iteration
        - 'test_accs': List of (iteration, test_accuracy) tuples
        - 'final_test_acc': Final test accuracy
    """
    model.train()
    iteration = 0
    epoch = 0
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    pbar = tqdm(total=num_iterations, desc="Training", disable=not verbose)
    
    while iteration < num_iterations:
        for batch_idx, (data, target) in enumerate(train_loader):
            if iteration >= num_iterations:
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Apply mask after gradient update if pruning is active
            if masks is not None:
                if apply_mask_fn is not None:
                    apply_mask_fn(masks)
                elif hasattr(model, 'apply_mask'):
                    model.apply_mask(masks)
            
            # Track metrics
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc = 100. * correct / target.size(0)
            
            train_losses.append(loss.item())
            train_accs.append(acc)
            
            # Evaluate periodically
            if iteration % eval_freq == 0:
                _, test_acc = evaluate(model, test_loader, criterion, device)
                test_accs.append((iteration, test_acc))
                pbar.set_postfix({'train_acc': f'{acc:.2f}%', 'test_acc': f'{test_acc:.2f}%'})
            
            iteration += 1
            pbar.update(1)
        
        epoch += 1
    
    pbar.close()
    
    # Final evaluation
    _, final_test_acc = evaluate(model, test_loader, criterion, device)
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'final_test_acc': final_test_acc
    }


def train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    scheduler: Optional[Any] = None,
    masks: Optional[Dict[str, Any]] = None,
    apply_mask_fn: Optional[Callable] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Train for a specified number of epochs.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Total number of training epochs
        device: Device to use for training
        scheduler: Optional learning rate scheduler
        masks: Optional dictionary of pruning masks
        apply_mask_fn: Optional function to apply masks to model
        verbose: Whether to show progress bar
    
    Returns:
        Dictionary containing training history:
        - 'train_losses': List of training losses per epoch
        - 'train_accs': List of training accuracies per epoch
        - 'test_losses': List of test losses per epoch
        - 'test_accs': List of test accuracies per epoch
        - 'final_test_acc': Final test accuracy
    """
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    pbar = tqdm(range(num_epochs), desc="Training", disable=not verbose)
    
    for epoch in pbar:
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, masks, apply_mask_fn
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.2f}%',
            'test_acc': f'{test_acc:.2f}%'
        })
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'final_test_acc': test_accs[-1] if test_accs else 0.0
    }

