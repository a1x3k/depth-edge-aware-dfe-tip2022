"""Multi-scale patch classifiers: the blur estimation network (BNet, Sec. III-A)
and the edge classification network (ENet, Sec. III-B).

Both networks look at a 41x41 RGB patch centred on an edge pixel together with
its central 27x27 and 15x15 crops. Each scale goes through its own convolutional
branch, the 7x7x64 feature maps are concatenated and reduced to a single
feature vector which is classified by three fully connected layers.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

PATCH_SIZE = 41

#: Blur levels (standard deviation of the Gaussian PSF) predicted by BNet.
BLUR_LEVELS = torch.arange(0.5, 6.25, 0.25)


def _conv(in_channels: int, out_channels: int, kernel_size: int = 3) -> list[nn.Module]:
    return [nn.Conv2d(in_channels, out_channels, kernel_size, bias=False), nn.ReLU(inplace=True)]


def _branch_41() -> nn.Sequential:
    """41x41 -> 7x7: conv, 3x3 max pooling, three convs."""
    return nn.Sequential(*_conv(3, 64), nn.MaxPool2d(3), *_conv(64, 64), *_conv(64, 64), *_conv(64, 64))


def _branch_27() -> nn.Sequential:
    """27x27 -> 7x7: three convs, 3x3 max pooling, 1x1 conv."""
    return nn.Sequential(*_conv(3, 64), *_conv(64, 64), *_conv(64, 64), nn.MaxPool2d(3), *_conv(64, 64, 1))


def _branch_15() -> nn.Sequential:
    """15x15 -> 7x7: four convs."""
    return nn.Sequential(*_conv(3, 64), *_conv(64, 64), *_conv(64, 64), *_conv(64, 64))


def _head(in_channels: int, num_classes: int) -> nn.Sequential:
    """7x7 -> 1x1 convolutions followed by the fully connected classifier."""
    return nn.Sequential(
        *_conv(in_channels, 64),
        *_conv(64, 64),
        *_conv(64, 64),
        *_conv(64, 64, 1),
        nn.Flatten(),
        nn.Linear(64, 300),
        nn.ReLU(inplace=True),
        nn.Linear(300, 150),
        nn.ReLU(inplace=True),
        nn.Linear(150, num_classes),
    )


class MultiScaleNet(nn.Module):
    """Common architecture of BNet and ENet.

    :param num_classes: Number of output classes
    :param num_wide_branches: Number of parallel branches on the 41x41 patch
    """

    def __init__(self, num_classes: int, num_wide_branches: int = 1):
        super().__init__()
        self.wide = nn.ModuleList(_branch_41() for _ in range(num_wide_branches))
        self.mid = _branch_27()
        self.narrow = _branch_15()
        self.head = _head(64 * (num_wide_branches + 2), num_classes)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        :param patches: (N, 3, 41, 41) RGB patches with values in [0, 1]
        :return: (N, num_classes) logits
        """
        mid = patches[:, :, 8:35, 8:35]
        narrow = patches[:, :, 14:29, 14:29]
        features = [branch(patches) for branch in self.wide]
        features += [self.mid(mid), self.narrow(narrow)]
        return self.head(torch.cat(features, dim=1))

    @classmethod
    def from_pretrained(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "MultiScaleNet":
        model = cls()
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)
        return model.eval()


class BNet(MultiScaleNet):
    """Blur estimation network: classifies a patch into one of ``BLUR_LEVELS``."""

    def __init__(self):
        super().__init__(num_classes=len(BLUR_LEVELS), num_wide_branches=1)


class ENet(MultiScaleNet):
    """Edge classification network: pattern edge (class 0) or depth edge (class 1)."""

    def __init__(self):
        super().__init__(num_classes=2, num_wide_branches=2)
