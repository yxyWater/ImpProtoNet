"""Standard Prototypical Network episode computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional


class BalancedClassAwareProtoNet(nn.Module):
    """ProtoNet with rare-class-aware, compact-prototype reweighting.

    The episode remains leakage-free. For each episode class, the loss weight
    is larger when the class is rare in the training split and when its support
    features form a compact prototype. Both inputs are computed from the
    current episode and immutable split-level training counts only.
    """

    def __init__(self, encoder: nn.Module, *, way: int = 3, class_counts: Mapping[str, int]) -> None:
        super().__init__()
        if way < 2:
            raise ValueError("way must be at least two")
        if len(class_counts) < way:
            raise ValueError("class_counts must cover every training class")
        self.encoder = encoder
        self.way = int(way)
        self.class_counts = {str(key): int(value) for key, value in class_counts.items()}
        self.prototype_head = ProtoNet(encoder, way=way)

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor | None = None,
        episode_class_ids: Sequence[str] | None = None,
    ) -> ProtoNetEpisodeOutput:
        output = self.prototype_head(
            support_images,
            support_labels,
            query_images,
            query_labels,
        )
        if query_labels is None:
            return output
        # Validation/test episodes can contain unseen classes that are not in
        # the immutable training-count table. In that case, and whenever class
        # IDs are not supplied, use the ordinary unweighted loss.
        if episode_class_ids is None:
            output.per_query_loss = functional.cross_entropy(
                output.logits,
                query_labels,
                reduction="none",
            )
            output.loss = output.per_query_loss.mean()
            return output
        episode_class_ids = tuple(episode_class_ids)
        if len(episode_class_ids) != self.way:
            raise ValueError("episode_class_ids must match the episode way")
        support_variance = torch.ones(self.way, device=output.prototypes.device)
        class_features_all = []
        for class_index in range(self.way):
            class_features = output.support_features[support_labels == class_index]
            class_features_all.append(class_features)
            compactness = torch.tensor(1.0, device=support_variance.device)
            if class_features.shape[0] > 1:
                compactness = 1.0 / (1.0 + class_features.var(dim=0, unbiased=False).mean())
            support_variance[class_index] = compactness.clamp(max=1.0)
        episode_ids = tuple(str(class_id) for class_id in episode_class_ids)
        if not all(class_id in self.class_counts for class_id in episode_ids):
            output.per_query_loss = functional.cross_entropy(
                output.logits,
                query_labels,
                reduction="none",
            )
            output.loss = output.per_query_loss.mean()
            return output
        episode_counts = torch.tensor(
            [float(self.class_counts[class_id]) for class_id in episode_ids],
            device=support_variance.device,
            dtype=support_variance.dtype,
        )
        count_weights = 1.0 / torch.sqrt(torch.log(episode_counts.clamp(min=1.0)) + 1.0)
        class_weights = (support_variance * count_weights).detach()
        class_weights = class_weights / class_weights.sum().clamp(min=1e-8)
        class_weights = self.way * class_weights
        output.per_query_loss = functional.cross_entropy(
            output.logits,
            query_labels,
            reduction="none",
            weight=class_weights,
        )
        output.loss = output.per_query_loss.mean()
        output.support_features = torch.stack(class_features_all, dim=0)
        return output


@dataclass
class ProtoNetEpisodeOutput:
    """Complete differentiable outputs from one ProtoNet episode."""

    support_features: torch.Tensor
    query_features: torch.Tensor
    prototypes: torch.Tensor
    squared_distances: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    per_query_loss: torch.Tensor | None
    loss: torch.Tensor | None


def compute_prototypes(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    way: int,
) -> torch.Tensor:
    """Compute one arithmetic-mean prototype for every episode class."""

    if support_features.ndim != 2:
        raise ValueError("support_features must have shape [support_count, feature_dim]")
    if support_labels.ndim != 1:
        raise ValueError("support_labels must have shape [support_count]")
    if support_features.shape[0] != support_labels.shape[0]:
        raise ValueError("Support feature and label counts do not match")
    if way < 2:
        raise ValueError("way must be at least two")
    if support_labels.numel() == 0:
        raise ValueError("The support set cannot be empty")
    if support_labels.dtype != torch.long:
        raise ValueError("support_labels must use torch.long dtype")
    if int(support_labels.min()) < 0 or int(support_labels.max()) >= way:
        raise ValueError("Support labels fall outside the episode class range")

    prototypes = []
    for class_index in range(way):
        class_mask = support_labels == class_index
        if not bool(class_mask.any()):
            raise ValueError(f"Support set is missing episode class {class_index}")
        prototypes.append(support_features[class_mask].mean(dim=0))
    return torch.stack(prototypes, dim=0)


def squared_euclidean_distances(
    query_features: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """Return unnormalised squared Euclidean distances for all query/prototype pairs."""

    if query_features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("Query features and prototypes must both be two-dimensional")
    if query_features.shape[1] != prototypes.shape[1]:
        raise ValueError("Query and prototype feature dimensions do not match")
    return (query_features[:, None, :] - prototypes[None, :, :]).square().sum(dim=-1)


class ProtoNet(nn.Module):
    """M0: squared-distance ProtoNet without L2 normalisation or learned temperature."""

    def __init__(self, encoder: nn.Module, *, way: int = 3) -> None:
        super().__init__()
        if way < 2:
            raise ValueError("way must be at least two")
        self.encoder = encoder
        self.way = int(way)
        self.temperature = 1.0

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor | None = None,
        episode_class_ids: object | None = None,
    ) -> ProtoNetEpisodeOutput:
        if support_images.ndim != 4 or query_images.ndim != 4:
            raise ValueError("Support and query images must have shape [count, channels, height, width]")
        if episode_class_ids is not None:
            pass  # base ProtoNet ignores episode class ids; accepted for API compatibility
        if query_labels is not None:
            if query_labels.ndim != 1 or query_labels.shape[0] != query_images.shape[0]:
                raise ValueError("Query labels do not match the query image count")
            if query_labels.dtype != torch.long:
                raise ValueError("query_labels must use torch.long dtype")
            if int(query_labels.min()) < 0 or int(query_labels.max()) >= self.way:
                raise ValueError("Query labels fall outside the episode class range")

        support_features = self.encoder(support_images)
        query_features = self.encoder(query_images)
        return self.forward_from_features(
            support_features,
            support_labels,
            query_features,
            query_labels,
        )

    def forward_from_features(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        query_labels: torch.Tensor | None = None,
    ) -> ProtoNetEpisodeOutput:
        """Run the episode head from already encoded image features."""

        if support_features.ndim != 2 or query_features.ndim != 2:
            raise ValueError("Support and query features must be two-dimensional")
        if support_features.shape[1] != query_features.shape[1]:
            raise ValueError("Support and query feature dimensions do not match")
        if query_labels is not None:
            if query_labels.ndim != 1 or query_labels.shape[0] != query_features.shape[0]:
                raise ValueError("Query labels do not match the query feature count")
            if query_labels.dtype != torch.long:
                raise ValueError("query_labels must use torch.long dtype")
            if int(query_labels.min()) < 0 or int(query_labels.max()) >= self.way:
                raise ValueError("Query labels fall outside the episode class range")
        prototypes = compute_prototypes(support_features, support_labels, self.way)
        squared_distances = squared_euclidean_distances(query_features, prototypes)
        logits = -squared_distances
        probabilities = functional.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        per_query_loss = None
        loss = None
        if query_labels is not None:
            per_query_loss = functional.cross_entropy(
                logits,
                query_labels,
                reduction="none",
            )
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
