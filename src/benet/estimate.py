"""End-to-end defocus blur map estimation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import torch
from skimage import color, feature

from .domain_transform import propagate_blur_map
from .model import BLUR_LEVELS, PATCH_SIZE, BNet, ENet

WEIGHTS_DIR = resources.files("benet") / "weights"


@dataclass
class BlurEstimate:
    blur_map: np.ndarray
    """(H, W) dense blur map (Gaussian sigma in pixels)."""
    pattern_edges: np.ndarray
    """(H, W) boolean map of the detected edge pixels."""
    depth_edges: np.ndarray
    """(H, W) boolean map of the edge pixels classified as depth edges."""


def load_networks(
    weights_dir: str | Path | None = None, device: str | torch.device = "cpu"
) -> tuple[BNet, ENet]:
    """Load the pretrained BNet and ENet."""
    root = Path(weights_dir) if weights_dir is not None else WEIGHTS_DIR
    # NHWC convolutions are noticeably faster on CPU for these small patches.
    bnet = BNet.from_pretrained(root / "bnet.pth", map_location=device)
    enet = ENet.from_pretrained(root / "enet.pth", map_location=device)
    return (
        bnet.to(device, memory_format=torch.channels_last),
        enet.to(device, memory_format=torch.channels_last),
    )


def detect_edges(img: np.ndarray) -> np.ndarray:
    """Canny edges on the hue channel of an (H, W, 3) uint8 RGB image."""
    hue = color.rgb2hsv(img)[:, :, 0]
    return feature.canny(hue, sigma=np.sqrt(2), low_threshold=0.05, high_threshold=0.06)


def _patches(padded: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """Gather (N, 3, 41, 41) patches whose top-left corners in ``padded`` are (ys, xs)."""
    offsets = torch.arange(PATCH_SIZE, device=padded.device)
    rows = (ys[:, None] + offsets)[:, :, None]
    cols = (xs[:, None] + offsets)[:, None, :]
    return padded[:, rows, cols].permute(1, 0, 2, 3)


@torch.inference_mode()
def classify_edges(
    img: np.ndarray,
    edges: np.ndarray,
    bnet: BNet,
    enet: ENet,
    batch_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run BNet and ENet on the patches centred on every edge pixel.

    :return: (ys, xs, blur class indices, edge class indices)
    """
    device = next(bnet.parameters()).device
    half = PATCH_SIZE // 2
    # Mirror padding that repeats the border pixel ("fedcba|abcdef").
    padded = np.pad(img, ((half, half), (half, half), (0, 0)), mode="symmetric")
    padded = torch.from_numpy(padded).permute(2, 0, 1).to(device=device, dtype=torch.float32) / 255.0

    ys, xs = (torch.from_numpy(a).to(device) for a in np.nonzero(edges))
    blur_cls, edge_cls = [], []
    for start in range(0, len(ys), batch_size):
        batch = _patches(padded, ys[start:start + batch_size], xs[start:start + batch_size])
        batch = batch.contiguous(memory_format=torch.channels_last)
        blur_cls.append(bnet(batch).argmax(dim=1))
        edge_cls.append(enet(batch).argmax(dim=1))

    return ys.cpu(), xs.cpu(), torch.cat(blur_cls).cpu(), torch.cat(edge_cls).cpu()


def estimate_blur_map(
    img: np.ndarray,
    bnet: BNet,
    enet: ENet,
    edges: np.ndarray | None = None,
    edge_penalty: float = 20.0,
    batch_size: int = 2048,
) -> BlurEstimate:
    """Estimate a dense defocus blur map.

    :param img: (H, W, 3) uint8 RGB defocused image
    :param bnet: Blur estimation network
    :param enet: Edge classification network
    :param edges: Optional (H, W) boolean edge map; Canny on the hue channel if omitted
    :param edge_penalty: Weight of the depth edge term in the propagation (Eq. 2)
    :param batch_size: Number of patches per forward pass
    """
    if edges is None:
        edges = detect_edges(img)
    edges = edges.astype(bool)
    h, w = edges.shape

    ys, xs, blur_cls, edge_cls = classify_edges(img, edges, bnet, enet, batch_size)

    is_depth = (edge_cls == 1).double()
    pattern = torch.zeros(h, w, dtype=torch.float64)
    depth = torch.zeros(h, w, dtype=torch.float64)
    sparse_blur = torch.zeros(h, w, dtype=torch.float64)
    pattern[ys, xs] = 1 - is_depth
    depth[ys, xs] = is_depth
    # Blur values on depth edges are unreliable and discarded.
    sparse_blur[ys, xs] = BLUR_LEVELS.double()[blur_cls] * (1 - is_depth)

    image = torch.tensor(img, dtype=torch.float64)
    blur_map = propagate_blur_map(image, pattern, depth, sparse_blur, edge_penalty)

    return BlurEstimate(
        blur_map=blur_map.numpy(),
        pattern_edges=edges,
        depth_edges=depth.numpy().astype(bool),
    )
