"""M1 frozen ImageNet feature baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from torchvision.models import ResNet18_Weights

from .models import ProtoNet, build_resnet18_encoder


@dataclass(frozen=True)
class FrozenImageNetProtocol:
    model_id: str
    model_name: str
    backbone: str
    way: int
    support_shots: tuple[int, ...]


def load_frozen_imagenet_protocol(path: Path) -> FrozenImageNetProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "model_id": "M1",
        "model_name": "FrozenImageNetPrototypeBaseline",
        "backbone": "resnet18_imagenet1k_v1",
        "use_eyelid_training_data": False,
        "freeze_backbone": True,
        "way": 3,
        "support_shots": [3, 5],
        "prototype": "arithmetic_mean",
        "distance": "squared_euclidean",
        "l2_normalize_features": False,
        "temperature": 1.0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"M1 configuration has an unsupported {key}")
    return FrozenImageNetProtocol(
        model_id=payload["model_id"],
        model_name=payload["model_name"],
        backbone=payload["backbone"],
        way=int(payload["way"]),
        support_shots=tuple(int(value) for value in payload["support_shots"]),
    )


def build_frozen_imagenet_model(*, load_pretrained_weights: bool = True) -> ProtoNet:
    """Build M1 with a permanently frozen ResNet18 encoder."""

    encoder = build_resnet18_encoder(
        pretrained=load_pretrained_weights,
        frozen=True,
    )
    model = ProtoNet(encoder, way=3)
    model.eval()
    return model


def imagenet_weights_metadata() -> dict[str, str]:
    weights = ResNet18_Weights.IMAGENET1K_V1
    return {
        "weights_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "weights_url": weights.url,
    }
