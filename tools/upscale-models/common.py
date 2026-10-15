import os

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

TILE = 256  # blockSize used by Aidoku's MultiArrayModel


class ShrinkCrop(nn.Module):
    """Crops the border of the output the way Aidoku's `shrinkSize` expects.

    MultiArrayModel feeds blockSize x blockSize tiles that include `shrink` pixels of context on
    every side and expects `(blockSize - 2 * shrink) * scale` pixels back, so the model has to
    drop the context itself.
    """

    def __init__(self, net: nn.Module, shrink: int, scale: int = 2):
        super().__init__()
        self.net = net
        self.crop = shrink * scale

    def forward(self, x):
        y = self.net(x)
        if self.crop:
            y = y[:, :, self.crop : -self.crop, self.crop : -self.crop]
        return y


def convert(module: nn.Module, name: str, out_dir: str) -> str:
    module.eval()
    example = torch.rand(1, 3, TILE, TILE)
    with torch.no_grad():
        traced = torch.jit.trace(module, example)
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 3, TILE, TILE), dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.mlpackage")
    model.save(path)
    return path
