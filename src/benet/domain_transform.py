"""Recursive-filter variant of the Domain Transform, with an extra penalty
term on depth edges, and the blur map propagation built on it.

E. Gastal, M. Oliveira, "Domain Transform for Edge-Aware Image and Video
Processing", ACM TOG 2011 (http://inf.ufrgs.br/~eslgastal/DomainTransform/).
"Edge-Based Defocus Blur Estimation with Adaptive Scale Selection",
IEEE TIP 2018.
"""

from __future__ import annotations

import math

import torch


def _horizontal_gradient(joint: torch.Tensor) -> torch.Tensor:
    """Sum over channels of |dJ/dx|, zero in the first column. ``joint`` is (H, W, C)."""
    grad = torch.zeros(joint.shape[:2], dtype=joint.dtype, device=joint.device)
    grad[:, 1:] = (joint[:, 1:] - joint[:, :-1]).abs().sum(dim=-1)
    return grad


def _recursive_filter_horizontal(img: torch.Tensor, dist: torch.Tensor, sigma: float) -> torch.Tensor:
    """One causal and one anti-causal recursive pass along the width.

    :param img: (H, W, C) image
    :param dist: (H, W) domain transform derivative
    :param sigma: Standard deviation of the filter for this iteration
    :return: Filtered (H, W, C) image
    """
    a = math.exp(-math.sqrt(2.0) / sigma)
    # Work column-major so each step reads and writes a contiguous slice.
    out = img.transpose(0, 1).clone(memory_format=torch.contiguous_format)
    weight = (a ** dist).transpose(0, 1).unsqueeze(-1)
    width = out.shape[0]

    for i in range(1, width):
        out[i] += weight[i] * (out[i - 1] - out[i])
    for i in range(width - 2, -1, -1):
        out[i] += weight[i + 1] * (out[i + 1] - out[i])

    return out.transpose(0, 1)


def domain_transform_filter(
    img: torch.Tensor,
    sigma_s: float,
    sigma_r: float,
    num_iterations: int = 3,
    joint_image: torch.Tensor | None = None,
    depth_edges: torch.Tensor | None = None,
    edge_penalty: float = 0.0,
) -> torch.Tensor:
    """Edge-aware recursive filtering.

    :param img: (H, W) or (H, W, C) image to be filtered
    :param sigma_s: Spatial standard deviation
    :param sigma_r: Range standard deviation
    :param num_iterations: Number of horizontal + vertical iterations
    :param joint_image: Optional (H, W) or (H, W, C) image for joint filtering
    :param depth_edges: Optional (H, W) depth edge map; adds ``edge_penalty``
        to the domain transform derivative on those pixels (Eq. 2)
    :param edge_penalty: Weight of the depth edge term
    :return: Filtered image, same shape as ``img``
    """
    squeeze = img.ndim == 2
    out = img.unsqueeze(-1) if squeeze else img
    joint = out if joint_image is None else joint_image
    if joint.ndim == 2:
        joint = joint.unsqueeze(-1)

    ratio = sigma_s / sigma_r
    penalty = 0.0 if depth_edges is None else edge_penalty * depth_edges
    dh_dx = 1 + ratio * _horizontal_gradient(joint) + penalty
    dv_dy = (1 + ratio * _horizontal_gradient(joint.transpose(0, 1)).T + penalty).T

    n = num_iterations
    for i in range(n):
        sigma_i = sigma_s * math.sqrt(3) * 2 ** (n - (i + 1)) / math.sqrt(4**n - 1)
        out = _recursive_filter_horizontal(out, dh_dx, sigma_i)
        out = _recursive_filter_horizontal(out.transpose(0, 1), dv_dy, sigma_i).transpose(0, 1)

    return out.squeeze(-1) if squeeze else out


def propagate_blur_map(
    img: torch.Tensor,
    pattern_edges: torch.Tensor,
    depth_edges: torch.Tensor,
    sparse_blur: torch.Tensor,
    edge_penalty: float = 20.0,
) -> torch.Tensor:
    """Propagate a sparse blur map to the full image (normalised joint filtering).

    :param img: (H, W, 3) RGB image with values in [0, 255]
    :param pattern_edges: (H, W) map, 1 on pattern edges
    :param depth_edges: (H, W) map, 1 on depth edges
    :param sparse_blur: (H, W) blur values on pattern edges, 0 elsewhere
    :param edge_penalty: Weight of the depth edge term (Eq. 2)
    :return: (H, W) dense blur map
    """
    h, w = pattern_edges.shape
    sigma_s = min(h, w) / 8.0
    sigma_r = 3.75
    num_iterations = 5

    reference = domain_transform_filter(img / 255.0, 7, 0.5, num_iterations)

    def filt(x: torch.Tensor) -> torch.Tensor:
        return domain_transform_filter(
            x, sigma_s, sigma_r, num_iterations, reference, depth_edges, edge_penalty
        )

    return filt(sparse_blur) / filt(pattern_edges)
