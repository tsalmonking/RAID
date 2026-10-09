import os
import torch
from typing import Union
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from external.omniaid.omniaid_arch import build_omniaid

from .omniaid_preprocess import OmniAIDPreProcess
from .omniaid_postprocess import OmniAIDPostProcess


def omniaid(checkpoint_path: str,
            device: Union[str, torch.device] = "cpu"):
    # MoE config (experts, rank, router) ships next to the checkpoint
    config_path = os.path.join(os.path.dirname(checkpoint_path),
                               "config_omniaid_genimage_paper.json")
    model = build_omniaid(checkpoint_path, config_path, device)
    return BasePytorchClassifier(
        model,
        preprocessing=OmniAIDPreProcess(),
        postprocessing=OmniAIDPostProcess(),
    )
