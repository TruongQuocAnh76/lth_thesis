import torch
import torch.nn as nn
import torch.nn.functional as F


# ResNet Implementation
class BasicBlock(nn.Module):
    """Basic residual block for ResNet-20/32/44/56."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture for CIFAR-10/100."""
    
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def get_block_info(self):
        """Get information about residual blocks for block-wise pruning.
        
        Returns block structure needed for Early-Bird:
        - Authority BN (bn2 in each block - the one before residual add)
        - Both convs in the block
        - Skip conv (if exists)
        """
        blocks = []
        
        for stage_name in ['layer1', 'layer2', 'layer3']:
            stage = getattr(self, stage_name)
            for block_idx, block in enumerate(stage):
                block_info = {
                    'name': f'{stage_name}.{block_idx}',
                    'authority_bn': f'{stage_name}.{block_idx}.bn2',  # BN before residual add
                    'conv1': f'{stage_name}.{block_idx}.conv1',
                    'bn1': f'{stage_name}.{block_idx}.bn1',
                    'conv2': f'{stage_name}.{block_idx}.conv2',
                    'bn2': f'{stage_name}.{block_idx}.bn2',
                    'has_shortcut': len(block.shortcut) > 0,
                }
                if block_info['has_shortcut']:
                    block_info['shortcut_conv'] = f'{stage_name}.{block_idx}.shortcut.0'
                    block_info['shortcut_bn'] = f'{stage_name}.{block_idx}.shortcut.1'
                blocks.append(block_info)
        
        return blocks


def resnet20(num_classes=10):
    """ResNet-20 for CIFAR datasets (3n+2 layers, n=3)."""
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet50(num_classes=10):
    """ResNet-50 for CIFAR datasets (adapted from ImageNet version).
    
    Note: This is a modified ResNet-50 for CIFAR with smaller input size.
    """
    return ResNet(Bottleneck, [3, 4, 6], num_classes=num_classes)


# VGG Implementation
class VGG(nn.Module):
    """VGG architecture for CIFAR-10/100."""
    
    def __init__(self, cfg, num_classes=10, batch_norm=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

    def _make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                if batch_norm:
                    layers.append(conv2d)
                    layers.append(nn.BatchNorm2d(x))
                    layers.append(nn.ReLU(inplace=True))
                else:
                    layers.append(conv2d)
                    layers.append(nn.ReLU(inplace=True))
                in_channels = x
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def vgg16(num_classes=10, batch_norm=True):
    """VGG-16 for CIFAR datasets with batch normalization."""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGG(cfg, num_classes=num_classes, batch_norm=batch_norm)


# Model Factory Function
def get_model(model_name, num_classes=10):
    """Factory function to get model by name.
    
    Args:
        model_name (str): Name of the model ('resnet20', 'resnet50', 'vgg16')
        num_classes (int): Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
    
    Returns:
        nn.Module: Initialized model
    
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = model_name.lower()
    
    if model_name == 'resnet20':
        return resnet20(num_classes=num_classes)
    elif model_name == 'resnet50':
        return resnet50(num_classes=num_classes)
    elif model_name == 'vgg16':
        return vgg16(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: resnet20, resnet50, vgg16")


def count_parameters(model):
    """Count total and trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        dict: Dictionary with 'total' and 'trainable' parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params
    }