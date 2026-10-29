"""DETR model used by training, evaluation, and realtime inference.

The architecture is intentionally unchanged: ResNet-50 backbone, 1x1
projection, a single encoder/decoder transformer layer, 25 learned queries, and
linear class/box heads. ``backbone_weights`` only controls how the backbone is
initialized; a task checkpoint replaces the full model state afterward.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50




def _get_1d_sincos_pos_embed(
    length: int, dim: int, temperature: float = 10000.0, device=None
) -> torch.Tensor:
    assert dim % 2 == 0
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(temperature) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def build_2d_sincos_position_embedding(
    height: int, width: int, dim: int, device=None
) -> torch.Tensor:
    """Create 2D sine-cos positional encoding of shape ``(1, H*W, dim)``."""
    assert dim % 2 == 0, "positional dim must be even"
    dim_half = dim // 2
    pe_y = _get_1d_sincos_pos_embed(height, dim_half, device=device)
    pe_x = _get_1d_sincos_pos_embed(width, dim_half, device=device)
    pos = torch.zeros(height, width, dim, device=device, dtype=torch.float32)
    pos[:, :, :dim_half] = pe_y[:, None, :].expand(-1, width, -1)
    pos[:, :, dim_half:] = pe_x[None, :, :].expand(height, -1, -1)
    return pos.view(1, height * width, dim)


class DETR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        num_queries: int = 25,
        *,
        backbone_weights: str | None = "imagenet",
        verbose: bool = True,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.nheads = int(nheads)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.num_queries = int(num_queries)
        self.backbone_weights_name = (
            "imagenet" if backbone_weights in ("imagenet", True) else "none"
        )
        if backbone_weights not in ("imagenet", "none", None, True, False):
            raise ValueError(
                "backbone_weights must be 'imagenet', 'none', or None; "
                f"got {backbone_weights!r}"
            )

        self.logger = None
        self.model_handler = None
        if verbose:
            from utils.logger import get_logger
            from utils.rich_handlers import ModelHandler

            self.logger = get_logger("model")
            self.model_handler = ModelHandler()
            self.model_handler.log_model_architecture(
                {
                    "Model Type": "DETR (Detection Transformer)",
                    "Number of Classes": self.num_classes,
                    "Hidden Dimension": self.hidden_dim,
                    "Attention Heads": self.nheads,
                    "Encoder Layers": self.num_encoder_layers,
                    "Decoder Layers": self.num_decoder_layers,
                    "Object Queries": self.num_queries,
                    "Backbone": (
                        "ResNet-50 (ImageNet initialization)"
                        if self.backbone_weights_name == "imagenet"
                        else "ResNet-50 (random initialization)"
                    ),
                }
            )

        weights = (
            ResNet50_Weights.IMAGENET1K_V1
            if self.backbone_weights_name == "imagenet"
            else None
        )
        try:
            self.backbone = resnet50(weights=weights)
        except Exception as exc:
            raise RuntimeError(
                "Could not create the ResNet-50 backbone. If the ImageNet weights "
                "are not cached and this machine is offline, pass "
                "backbone_weights=None or use --no-pretrained-backbone."
            ) from exc
        self.backbone.fc = nn.Identity()

        self.conv = nn.Conv2d(2048, self.hidden_dim, 1)
        self.transformer = nn.Transformer(
            self.hidden_dim,
            self.nheads,
            self.num_encoder_layers,
            self.num_decoder_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.linear_class = nn.Linear(self.hidden_dim, self.num_classes + 1)
        self.linear_bbox = nn.Linear(self.hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.randn(self.num_queries, self.hidden_dim))
        self.norm_src = nn.LayerNorm(self.hidden_dim)
        self.norm_tgt = nn.LayerNorm(self.hidden_dim)

    def architecture_config(self) -> dict[str, Any]:
        """Return the JSON-serializable parameters stored in checkpoints."""
        return {
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "nheads": self.nheads,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "num_queries": self.num_queries,
        }

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        feat = self.conv(x)
        bsz, d_model, height, width = feat.shape
        src = feat.flatten(2).permute(0, 2, 1)
        pos = build_2d_sincos_position_embedding(
            height, width, d_model, device=feat.device
        )
        src = self.norm_src(src + pos)

        tgt = torch.zeros(bsz, self.num_queries, d_model, device=feat.device)
        query_pos = self.query_pos.unsqueeze(0).expand(bsz, -1, -1)
        tgt = self.norm_tgt(tgt + query_pos)

        hs = self.transformer(src=src, tgt=tgt)
        return {
            "pred_logits": self.linear_class(hs),
            "pred_boxes": self.linear_bbox(hs).sigmoid(),
        }

    def log_model_info(self) -> None:
        if self.model_handler is None:
            return
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.model_handler.log_parameters_count(total_params, trainable_params)

    def load_pretrained(
        self,
        checkpoint_path: str,
        *,
        num_classes: int | None = None,
        classes: list[str] | None = None,
        map_location: str | torch.device = "cpu",
        restore_training: bool = False,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        allow_unsafe_load: bool = False,
    ) -> dict:
        """Strictly load a DETR checkpoint and return its metadata.

        This method never swallows a load error. Missing, corrupt, LFS-pointer,
        architecture-mismatched, or class-mismatched checkpoints raise
        :class:`utils.checkpoint.CheckpointError`.
        """
        from utils.checkpoint import load_model_checkpoint

        metadata = load_model_checkpoint(
            self,
            checkpoint_path,
            num_classes=self.num_classes if num_classes is None else num_classes,
            classes=classes,
            expected_model_config=self.architecture_config(),
            map_location=map_location,
            restore_training=restore_training,
            optimizer=optimizer,
            scheduler=scheduler,
            allow_unsafe_load=allow_unsafe_load,
        )
        if self.model_handler is not None:
            self.model_handler.log_model_loading(str(checkpoint_path), success=True)
        return metadata


if __name__ == "__main__":
    model = DETR(num_classes=3, backbone_weights=None)
    from torchinfo import summary

    summary(model, (5, 3, 224, 224))
