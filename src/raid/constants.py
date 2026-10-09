import os
from attacks.loss import AvgEnsembleLoss
from torch.nn import CrossEntropyLoss
from huggingface_hub import hf_hub_download
from models import (
    cavia2024,
    chen2024,
    corvi2023,
    effort,
    aide,
    d3,
    omniaid,
    dda,
    koutlis2024,
    ojha2023,
    wang2020,
    EnsembleModel,
    vit_lp14_dinov2,
    vit_lp14_reg_dinov2,
    vit_tp16_224_augreg_in21k,
)
from models.ensemble.ensemble_function import (
    RawEnsembleFunction,
    AvgEnsembleFunction,
    RandomEnsembleFunction
)


# Effort/AIDE/D3/OmniAID/DDA checkpoints (not on the RAID HF repo); machine-specific, see env_set.sh
_AIGI_CKPT_DIR = os.environ.get("AIGI_CKPT_DIR", "/storageC/heddoubi/guard/external/checkpoints")


def _get_raid_ckpt(filename):
    return hf_hub_download(repo_id="aimagelab/RAID_ckpt", filename=filename)


# models with default checkpoints
MODELS = {
    "cavia2024": (cavia2024, _get_raid_ckpt("cavia2024/model_best.pth.tar")), # retrained
    "chen2024_convnext": (
        chen2024, _get_raid_ckpt("chen2024_convnext/16_acc0.9993.pth")
    ),
    "chen2024_clip": (
        chen2024, _get_raid_ckpt("chen2024_clip/last_acc0.9112.pth")
    ),
    "corvi2023": (corvi2023, _get_raid_ckpt("corvi2023/model_best.pth.tar")),
    "koutlis2024": (
        koutlis2024, _get_raid_ckpt("koutlis2024/model_ldm_trainable.pth")
    ),
    "ojha2023": (ojha2023, _get_raid_ckpt("ojha2023/model_best.pth.tar")),
    "wang2020": (wang2020, _get_raid_ckpt("wang2020/model_best.pth.tar")),
    "effort": (effort, f"{_AIGI_CKPT_DIR}/effort/genimage_sd14_released.pth"),
    "aide": (aide, f"{_AIGI_CKPT_DIR}/aide/genimage_sd14_released.pth"),
    "d3": (d3, f"{_AIGI_CKPT_DIR}/d3/genimage_sd14_best.pth"),
    # OmniAID: official GenImage-SD1.4 release (HF Yunncheng/OmniAID), its MoE
    # config json alongside. DDA: official release (HF Junwei-Xi/Dual-Data-
    # Alignment), trained on DDA-aligned MSCOCO - no GenImage checkpoint exists.
    "omniaid": (omniaid, f"{_AIGI_CKPT_DIR}/omniaid/genimage_sd14_released.pth"),
    "dda": (dda, f"{_AIGI_CKPT_DIR}/dda/DDA_ckpt.pth"),

    "vit_lp14_dinov2": (
        vit_lp14_dinov2,
        _get_raid_ckpt("pretrained_linear/dinov2_retrain_d3/model_best.pth.tar")
    ),
    "vit_lp14_reg_dinov2": (
        vit_lp14_reg_dinov2,
        _get_raid_ckpt("pretrained_linear/dinov2_reg_retrain_d3/model_best.pth.tar")
    ),
    "vit_tp16_224_augreg_in21k": (
        vit_tp16_224_augreg_in21k,
        _get_raid_ckpt("pretrained_linear/vit_tiny_retrain_d3/model_best.pth.tar")
    ),
    "vit_tp16_224_code_augreg_in21k": (
        vit_tp16_224_augreg_in21k,
        _get_raid_ckpt("pretrained_linear/vit_tiny_code_retrain_d3/model_best.pth.tar")
    ),

    "ModelEnsemble" : EnsembleModel,

}
ENSEMBLING_STRATEGIES = {
    "raw": RawEnsembleFunction,
    "avg": AvgEnsembleFunction,
    "random": RandomEnsembleFunction
}
ENSEMBLE_LOSSES = {
    "avg_ce": AvgEnsembleLoss(),
    "ce" : CrossEntropyLoss(reduction="none"),
}
