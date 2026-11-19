"""Deterministic multinomial L2 logistic regression for M2 episodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from .protonet import (
    ProtoNetEpisodeOutput,
    compute_prototypes,
    squared_euclidean_distances,
)


@dataclass(frozen=True)
class LogisticRegressionFit:
    weights: torch.Tensor
    bias: torch.Tensor
    objective: float


def fit_multinomial_l2_logistic_regression(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    way: int,
    regularization_c: float,
    maximum_iterations: int = 100,
    tolerance: float = 1e-7,
) -> LogisticRegressionFit:
    """Fit a softmax classifier using support features only.

    The objective matches the conventional C parameterisation after converting
    the summed data loss to a mean: mean cross-entropy plus
    ||W||^2 / (2 * C * support_count). The intercept is not penalised.
    """

    if features.ndim != 2:
        raise ValueError("features must have shape [sample_count, feature_dim]")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels do not match the feature count")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long dtype")
    if features.shape[0] == 0:
        raise ValueError("The support feature set cannot be empty")
    if way < 2:
        raise ValueError("way must be at least two")
    if int(labels.min()) < 0 or int(labels.max()) >= way:
        raise ValueError("labels fall outside the episode class range")
    if set(labels.detach().cpu().tolist()) != set(range(way)):
        raise ValueError("Every episode class must have at least one support sample")
    if regularization_c <= 0:
        raise ValueError("regularization_c must be positive")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")

    detached_features = features.detach().float()
    detached_labels = labels.detach()
    weights = torch.zeros(
        way,
        detached_features.shape[1],
        device=detached_features.device,
        dtype=detached_features.dtype,
        requires_grad=True,
    )
    bias = torch.zeros(
        way,
        device=detached_features.device,
        dtype=detached_features.dtype,
        requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        (weights, bias),
        lr=1.0,
        max_iter=maximum_iterations,
        tolerance_grad=tolerance,
        tolerance_change=tolerance,
        line_search_fn="strong_wolfe",
    )
    support_count = detached_features.shape[0]

    def objective() -> torch.Tensor:
        logits = functional.linear(detached_features, weights, bias)
        data_loss = functional.cross_entropy(logits, detached_labels)
        penalty = weights.square().sum() / (
            2.0 * regularization_c * support_count
        )
        return data_loss + penalty

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)
        final_objective = float(objective().detach().cpu())
    return LogisticRegressionFit(
        weights=weights.detach(),
        bias=bias.detach(),
        objective=final_objective,
    )


class LogisticRegressionEpisodeModel(nn.Module):
    """Frozen encoder plus a newly fitted L2 logistic head in every episode."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        way: int = 3,
        regularization_c: float,
        maximum_iterations: int = 100,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.way = int(way)
        self.regularization_c = float(regularization_c)
        self.maximum_iterations = int(maximum_iterations)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "LogisticRegressionEpisodeModel":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor | None = None,
    ) -> ProtoNetEpisodeOutput:
        if support_images.ndim != 4 or query_images.ndim != 4:
            raise ValueError(
                "Support and query images must have shape [count, channels, height, width]"
            )
        with torch.no_grad():
            support_features = self.encoder(support_images).float()
            query_features = self.encoder(query_images).float()
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
        """Refit the episode classifier from cached frozen-encoder features."""

        if support_features.ndim != 2 or query_features.ndim != 2:
            raise ValueError("Support and query features must be two-dimensional")
        if support_features.shape[1] != query_features.shape[1]:
            raise ValueError("Support and query feature dimensions do not match")
        support_features = support_features.float()
        query_features = query_features.float()
        # LBFGS is deliberately kept in float32 even when the ResNet encoder is
        # evaluated under CUDA mixed precision. Small support sets make the
        # optimisation cheap, while float32 avoids unstable line searches.
        with torch.amp.autocast(
            device_type=support_features.device.type,
            enabled=False,
        ):
            fit = fit_multinomial_l2_logistic_regression(
                support_features,
                support_labels,
                way=self.way,
                regularization_c=self.regularization_c,
                maximum_iterations=self.maximum_iterations,
            )
            logits = functional.linear(query_features, fit.weights, fit.bias)
        probabilities = functional.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        prototypes = compute_prototypes(
            support_features,
            support_labels,
            self.way,
        )
        squared_distances = squared_euclidean_distances(
            query_features,
            prototypes,
        )
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
