import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os

def get_dataset(dataset_name, train=True, download=True, data_dir='./data'):
    """
    Get CIFAR dataset.

    Args:
        dataset_name (str): 'cifar10' or 'cifar100'
        train (bool): If True, return training set; else test set
        download (bool): Whether to download the dataset
        data_dir (str): Directory to store the dataset

    Returns:
        torchvision dataset
    """
    if dataset_name.lower() == 'cifar10':
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset_name.lower() == 'cifar100':
        dataset_class = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Define transforms
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    return dataset_class(root=data_dir, train=train, download=download, transform=transform)

def get_dataloaders(dataset_name, batch_size=128, num_workers=4, val_split=0.1, data_dir='./data'):
    """
    Get train, validation, and test data loaders for CIFAR datasets.

    Args:
        dataset_name (str): 'cifar10' or 'cifar100'
        batch_size (int): Batch size for loaders
        num_workers (int): Number of workers for data loading
        val_split (float): Fraction of training data to use for validation
        data_dir (str): Directory to store the dataset

    Returns:
        dict: {'train': train_loader, 'val': val_loader, 'test': test_loader}
    """
    # Get full training dataset
    train_dataset = get_dataset(dataset_name, train=True, data_dir=data_dir)

    # Split into train and validation
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size],
                                             generator=torch.Generator().manual_seed(42))

    # Get test dataset
    test_dataset = get_dataset(dataset_name, train=False, data_dir=data_dir)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def get_cifar10_dataloaders(batch_size=128, num_workers=4, data_dir='./data'):
    """
    Get train and test data loaders for CIFAR-10 (without validation split).
    
    Args:
        batch_size (int): Batch size for loaders
        num_workers (int): Number of workers for data loading
        data_dir (str): Directory to store the dataset
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Get datasets
    train_dataset = get_dataset('cifar10', train=True, data_dir=data_dir)
    test_dataset = get_dataset('cifar10', train=False, data_dir=data_dir)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader

# Example usage
if __name__ == "__main__":
    # Test the functions
    loaders = get_dataloaders('cifar10', batch_size=64)
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")

    # Get a batch
    images, labels = next(iter(loaders['train']))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")