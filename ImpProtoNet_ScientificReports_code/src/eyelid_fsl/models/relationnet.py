"""Spatial Relation Network adapted to the shared ResNet18 experiment backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from .protonet import compute_prototypes, squared_euclidean_distances


@dataclass
class RelationNetEpisodeOutput:
    """Differentiable prediction outputs plus optional mechanism tensors."""

    support_features: torch.Tensor
    query_features: torch.Tensor
    prototypes: torch.Tensor
    squared_distances: torch.Tensor
    support_feature_maps: torch.Tensor
    query_feature_maps: torch.Tensor
    class_support_feature_maps: torch.Tensor
    relation_logits: torch.Tensor
    relation_scores: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    per_query_loss: torch.Tensor | None
    loss: torch.Tensor | None


class RelationModule(nn.Module):
    """Learn a scalar relation score from concatenated spatial feature maps."""

    def __init__(
        self,
        feature_channels: int,
        *,
        first_hidden_channels: int = 256,
        second_hidden_channels: int = 64,
        fully_connected_hidden: int = 8,
    ) -> None:
        super().__init__()
        values = (
            feature_channels,
            first_hidden_channels,
            second_hidden_channels,
            fully_connected_hidden,
        )
        if any(value < 1 for value in values):
            raise ValueError("All RelationNet channel dimensions must be positive")
        self.feature_channels = int(feature_channels)
        self.first_block = nn.Sequential(
            nn.Conv2d(
                2 * feature_channels,
                first_hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(first_hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.second_block = nn.Sequential(
            nn.Conv2d(
                first_hidden_channels,
                second_hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(second_hidden_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regressor = nn.Sequential(
            nn.Linear(second_hidden_channels, fully_connected_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fully_connected_hidden, 1),
        )

    def forward(self, relation_pairs: torch.Tensor) -> torch.Tensor:
        if relation_pairs.ndim != 4:
            raise ValueError("Relation pairs must be spatial feature maps")
        if relation_pairs.shape[1] != 2 * self.feature_channels:
            raise ValueError("Relation pair channel count does not match the module")
        if min(relation_pairs.shape[-2:]) < 2:
            raise ValueError("Relation feature maps must be at least 2x2")
        features = self.first_block(relation_pairs)
        features = self.second_block(features).flatten(1)
        return self.regressor(features).squeeze(1)


class RelationNet(nn.Module):
    """M5: original-style class-sum RelationNet with a ResNet spatial encoder."""

    def __init__(
        self,
        spatial_encoder: nn.Module,
        *,
        way: int = 3,
        first_hidden_channels: int = 256,
        second_hidden_channels: int = 64,
        fully_connected_hidden: int = 8,
        probability_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        if way < 2:
            raise ValueError("way must be at least two")
        feature_channels = getattr(spatial_encoder, "output_channels", None)
        if not isinstance(feature_channels, int) or feature_channels < 1:
            raise ValueError("Spatial encoder must declare a positive output_channels")
        if probability_epsilon <= 0:
            raise ValueError("Probability epsilon must be positive")
        self.spatial_encoder = spatial_encoder
        self.way = int(way)
        self.feature_channels = feature_channels
        self.probability_epsilon = float(probability_epsilon)
        self.relation_module = RelationModule(
            feature_channels,
            first_hidden_channels=first_hidden_channels,
            second_hidden_channels=second_hidden_channels,
            fully_connected_hidden=fully_connected_hidden,
        )

    def _validate_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor | None,
    ) -> None:
        if support_images.ndim != 4 or query_images.ndim != 4:
            raise ValueError("Support and query images must be four-dimensional")
        if support_labels.ndim != 1 or support_labels.shape[0] != support_images.shape[0]:
            raise ValueError("Support labels do not match support images")
        if support_labels.dtype != torch.long:
            raise ValueError("support_labels must use torch.long dtype")
        expected = torch.arange(self.way, device=support_labels.device)
        if not torch.equal(torch.unique(support_labels, sorted=True), expected):
            raise ValueError("Every episode class must appear in the support set")
        if query_labels is not None:
            if query_labels.ndim != 1 or query_labels.shape[0] != query_images.shape[0]:
                raise ValueError("Query labels do not match query images")
            if query_labels.dtype != torch.long:
                raise ValueError("query_labels must use torch.long dtype")
            if int(query_labels.min()) < 0 or int(query_labels.max()) >= self.way:
                raise ValueError("Query labels fall outside the episode class range")

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor | None = None,
    ) -> RelationNetEpisodeOutput:
        self._validate_episode(
            support_images, support_labels, query_images, query_labels
        )
        support_maps = self.spatial_encoder(support_images)
        query_maps = self.spatial_encoder(query_images)
        return self.forward_from_features(
            support_maps,
            support_labels,
            query_maps,
            query_labels,
        )

    def forward_from_features(
        self,
        support_maps: torch.Tensor,
        support_labels: torch.Tensor,
        query_maps: torch.Tensor,
        query_labels: torch.Tensor | None = None,
    ) -> RelationNetEpisodeOutput:
        """Apply the relation head to already encoded spatial feature maps."""

        if support_maps.ndim != 4 or query_maps.ndim != 4:
            raise ValueError("Support and query feature maps must be four-dimensional")
        if support_labels.ndim != 1 or support_labels.shape[0] != support_maps.shape[0]:
            raise ValueError("Support labels do not match support feature maps")
        if support_labels.dtype != torch.long:
            raise ValueError("support_labels must use torch.long dtype")
        expected = torch.arange(self.way, device=support_labels.device)
        if not torch.equal(torch.unique(support_labels, sorted=True), expected):
            raise ValueError("Every episode class must appear in the support set")
        if query_labels is not None:
            if query_labels.ndim != 1 or query_labels.shape[0] != query_maps.shape[0]:
                raise ValueError("Query labels do not match query feature maps")
            if query_labels.dtype != torch.long:
                raise ValueError("query_labels must use torch.long dtype")
            if int(query_labels.min()) < 0 or int(query_labels.max()) >= self.way:
                raise ValueError("Query labels fall outside the episode class range")
        if support_maps.shape[1:] != query_maps.shape[1:]:
            raise RuntimeError("Support and query spatial feature shapes differ")

        class_maps = torch.stack(
            [
                support_maps[support_labels == class_index].sum(dim=0)
                for class_index in range(self.way)
            ],
            dim=0,
        )
        query_count, channels, height, width = query_maps.shape
        repeated_classes = class_maps.unsqueeze(0).expand(
            query_count, -1, -1, -1, -1
        )
        repeated_queries = query_maps.unsqueeze(1).expand(
            -1, self.way, -1, -1, -1
        )
        relation_pairs = torch.cat(
            (repeated_classes, repeated_queries), dim=2
        ).reshape(query_count * self.way, 2 * channels, height, width)
        relation_logits = self.relation_module(relation_pairs).reshape(
            query_count, self.way
        )
        relation_scores = torch.sigmoid(relation_logits)
        probabilities = relation_scores / relation_scores.sum(
            dim=1, keepdim=True
        ).clamp_min(self.probability_epsilon)
        predictions = relation_scores.argmax(dim=1)

        per_query_loss = None
        loss = None
        if query_labels is not None:
            targets = functional.one_hot(
                query_labels, num_classes=self.way
            ).to(relation_scores.dtype)
            per_query_loss = functional.mse_loss(
                relation_scores, targets, reduction="none"
            ).mean(dim=1)
            loss = per_query_loss.mean()

        support_features = functional.adaptive_avg_pool2d(
            support_maps, (1, 1)
        ).flatten(1)
        query_features = functional.adaptive_avg_pool2d(
            query_maps, (1, 1)
        ).flatten(1)
        with torch.no_grad():
            prototypes = compute_prototypes(
                support_features, support_labels, self.way
            )
            squared_distances = squared_euclidean_distances(
                query_features, prototypes
            )
        return RelationNetEpisodeOutput(
            support_features=support_features,
            query_features=query_features,
            prototypes=prototypes,
            squared_distances=squared_distances,
            support_feature_maps=support_maps,
            query_feature_maps=query_maps,
            class_support_feature_maps=class_maps,
            relation_logits=relation_logits,
            relation_scores=relation_scores,
            logits=relation_logits,
            probabilities=probabilities,
            predictions=predictions,
            per_query_loss=per_query_loss,
            loss=loss,
        )
