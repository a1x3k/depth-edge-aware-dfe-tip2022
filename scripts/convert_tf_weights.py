"""Convert the original TensorFlow 1.x weight exports (models_tf1x/*/N.mat)
into PyTorch state dicts (src/benet/weights/{bnet,enet}.pth).

    uv run --group convert scripts/convert_tf_weights.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch import nn

from benet.model import BNet, ENet, MultiScaleNet

ROOT = Path(__file__).resolve().parents[1]


def layers(*modules: nn.Module) -> list[nn.Module]:
    return [m for module in modules for m in module.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def load_mat(folder: Path, count: int) -> list[np.ndarray]:
    return [sio.loadmat(folder / f"{i}.mat")["weights"] for i in range(count)]


def convert(model: MultiScaleNet, ordered_layers: list[nn.Module], weights: list[np.ndarray]) -> dict:
    params = iter(weights)
    with torch.no_grad():
        for layer in ordered_layers:
            if isinstance(layer, nn.Conv2d):
                # TF (kh, kw, in, out) -> PyTorch (out, in, kh, kw)
                layer.weight.copy_(torch.from_numpy(next(params)).permute(3, 2, 0, 1))
            else:
                # TF (in, out) -> PyTorch (out, in). The flattened input is 1x1x64, so the
                # NHWC vs NCHW flattening order makes no difference.
                layer.weight.copy_(torch.from_numpy(next(params)).T)
                layer.bias.copy_(torch.from_numpy(next(params)).ravel())
    if next(params, None) is not None:
        raise ValueError("unused weights left")
    return model.state_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=ROOT / "models_tf1x")
    parser.add_argument("--dst", type=Path, default=ROOT / "src" / "benet" / "weights")
    args = parser.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)

    bnet = BNet()
    torch.save(
        convert(bnet, layers(bnet.wide[0], bnet.mid, bnet.narrow, bnet.head), load_mat(args.src / "blur", 22)),
        args.dst / "bnet.pth",
    )

    # The .mat files store the two 41x41 branches as A (0-3) then B (4-7), but the
    # features are concatenated as [B, A, 27x27, 15x15]; wide[0] is B.
    enet = ENet()
    torch.save(
        convert(enet, layers(enet.wide[1], enet.wide[0], enet.mid, enet.narrow, enet.head),
                load_mat(args.src / "edge", 26)),
        args.dst / "enet.pth",
    )
    print(f"wrote {args.dst / 'bnet.pth'} and {args.dst / 'enet.pth'}")


if __name__ == "__main__":
    main()
