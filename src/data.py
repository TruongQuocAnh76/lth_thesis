import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os

def get_dataset(dataset_name, train=True, download=True, data_dir='./data'):
    """
    Get dataset (CIFAR-10, CIFAR-100, or MNIST).

    Args:
        dataset_name (str): 'cifar10', 'cifar100', or 'mnist'
        train (bool): If True, return training set; else test set
        download (bool): Whether to download the dataset
        data_dir (str): Directory to store the dataset

    Returns:
        torchvision dataset
    """
    dataset = dataset_name.lower()
    if dataset == 'cifar10':
        dataset_class = torchvision.datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dataset == 'cifar100':
        dataset_class = torchvision.datasets.CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dataset == 'mnist':
        dataset_class = torchvision.datasets.MNIST
        mean = (0.1307,)
        std = (0.3081,)
        # For MNIST, keep augmentation minimal (random crop for train)
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform = train_transform if train else test_transform
    return dataset_class(root=data_dir, train=train, download=download, transform=transform)

def get_dataloaders(
    dataset_name,
    batch_size=128,
    num_workers=4,
    val_split=0.1,
    data_dir='./data',
    pin_memory: bool = True,
    persistent_workers: bool = False,
):
    """
    Get train, validation, and test data loaders for CIFAR-10, CIFAR-100, or MNIST datasets.

    Args:
        dataset_name (str): 'cifar10', 'cifar100', or 'mnist'
        batch_size (int): Batch size for loaders
        num_workers (int): Number of workers for data loading
        val_split (float): Fraction of training data to use for validation
        data_dir (str): Directory to store the dataset
        pin_memory (bool): Enable pinned (page-locked) memory for faster CPU→GPU transfers when using CUDA.
        persistent_workers (bool): Keep workers alive between epochs to avoid re-spawn overhead (requires num_workers > 0).

    Returns:
        dict: {'train': train_loader, 'val': val_loader, 'test': test_loader}
    """
    # persistent_workers requires at least one worker process
    _persistent = persistent_workers and num_workers > 0

    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    # Only rank 0 downloads; all others wait at the barrier
    if rank == 0:
        train_dataset = get_dataset(dataset_name, train=True, data_dir=data_dir, download=True)
        test_dataset  = get_dataset(dataset_name, train=False, data_dir=data_dir, download=True)

    if is_distributed:
        dist.barrier()  # block non-zero ranks until rank 0 finishes writing files

    if rank != 0:
        train_dataset = get_dataset(dataset_name, train=True, data_dir=data_dir, download=False)
        test_dataset  = get_dataset(dataset_name, train=False, data_dir=data_dir, download=False)

    # Split into train and validation
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size],
                                             generator=torch.Generator().manual_seed(42))

    # Use DistributedSampler under DDP so each GPU sees a unique shard
    if is_distributed:
        train_sampler = DistributedSampler(train_subset, shuffle=True)
        val_sampler   = DistributedSampler(val_subset, shuffle=False)
        test_sampler  = DistributedSampler(test_dataset, shuffle=False)
        train_shuffle = False  # sampler owns shuffling
    else:
        train_sampler = val_sampler = test_sampler = None
        train_shuffle = True

    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=train_shuffle,
        sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=_persistent,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        sampler=val_sampler, num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=_persistent,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        sampler=test_sampler, num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=_persistent,
    )

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