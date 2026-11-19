"""Prototype classification with focal loss and optional learned temperature."""

from __future__ import annotations

import torch

from torch import nn

from torch.nn import functional



class FocalLossProtoNet(nn.Module):
    """ProtoNet with focal loss instead of standard cross-entropy.

    Focal loss (Lin et al. 2017): FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t)
    """

    def __init__(self, encoder: nn.Module, *, way: int = 3, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        from .protonet import ProtoNet
        self.encoder = encoder
        self.way = int(way)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.head = ProtoNet(encoder, way=way)

    def forward(self, support_images, support_labels, query_images, query_labels=None, episode_class_ids=None):
        from .protonet import ProtoNetEpisodeOutput
        output = self.head(support_images, support_labels, query_images, query_labels, episode_class_ids=episode_class_ids)
        if query_labels is None:
            return output

        # Focal loss
        probs = output.probabilities  # [Q, way]
        targets = query_labels  # [Q]
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # p_t for true class
        focal_weight = self.alpha * (1 - pt).clamp(min=1e-8) ** self.gamma
        ce_loss = functional.cross_entropy(output.logits, targets, reduction="none")
        per_query_loss = focal_weight * ce_loss
        output.per_query_loss = per_query_loss
        output.loss = per_query_loss.mean()
        return output

class FocalTempProtoNet(nn.Module):
    """Prototype logits scaled by a learned temperature, trained with focal loss."""

    def __init__(self, encoder: nn.Module, *, way: int = 3, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        from .protonet import compute_prototypes, ProtoNetEpisodeOutput
        self.encoder = encoder
        self.way = int(way)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, support_images, support_labels, query_images, query_labels=None, episode_class_ids=None):
        from .protonet import compute_prototypes, squared_euclidean_distances, ProtoNetEpisodeOutput
        support_features = self.encoder(support_images)
        query_features = self.encoder(query_images)
        prototypes = compute_prototypes(support_features, support_labels, self.way)
        squared_distances = squared_euclidean_distances(query_features, prototypes)
        logits = -squared_distances / self.temperature
        probabilities = functional.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        per_query_loss = None
        loss = None
        if query_labels is not None:
            pt = probabilities.gather(1, query_labels.unsqueeze(1)).squeeze(1)
            focal_weight = self.alpha * (1 - pt).clamp(min=1e-8) ** self.gamma
            ce_loss = functional.cross_entropy(logits, query_labels, reduction="none")
            per_query_loss = focal_weight * ce_loss
            loss = per_query_loss.mean()

        return ProtoNetEpisodeOutput(
            support_features=support_features,
            query_features=query_features,
            prototypes=prototypes,
            squared_distances=squared_distances,
            logits=logits,
            probabilities=probabilities,
            predictions=predictions,
            per_query_loss=per_query_loss,
            loss=loss,
        )
