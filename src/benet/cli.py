"""Command line interface: ``benet -i image.png [-e edges.png]``."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .estimate import estimate_blur_map, load_networks

MAX_BLUR = 6.0


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth Edge Aware Defocus Blur Estimation")
    parser.add_argument("-i", "--image", required=True, type=Path, help="Defocused image")
    parser.add_argument("-e", "--edge-map", type=Path, help="Edge map of the defocused image (optional)")
    parser.add_argument("-o", "--output-dir", type=Path,
                        help="Where to write the results (default: next to the image)")
    parser.add_argument("--weights-dir", type=Path, help="Directory with bnet.pth and enet.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--edge-penalty", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = get_args(argv)

    img = np.asarray(Image.open(args.image).convert("RGB"))
    edges = None
    if args.edge_map:
        edges = np.asarray(Image.open(args.edge_map).convert("L")) == 255

    bnet, enet = load_networks(args.weights_dir, args.device)
    result = estimate_blur_map(img, bnet, enet, edges, args.edge_penalty, args.batch_size)

    out_dir = args.output_dir or args.image.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem

    bmap = np.clip(result.blur_map / MAX_BLUR * 255, 0, 255).astype(np.uint8)
    Image.fromarray(bmap).save(out_dir / f"{stem}_bmap.png")

    # All edge pixels in blue; depth edges additionally in red, so they show as magenta.
    edge_vis = np.zeros((*bmap.shape, 3), dtype=np.uint8)
    edge_vis[..., 2] = result.pattern_edges * 255
    edge_vis[..., 0] = result.depth_edges * 255
    Image.fromarray(edge_vis).save(out_dir / f"{stem}_edge.png")


if __name__ == "__main__":
    main()
