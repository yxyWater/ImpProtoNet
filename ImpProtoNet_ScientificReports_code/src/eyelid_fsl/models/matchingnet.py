"""Matching Networks with the paper's full-context embeddings (FCE)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from .protonet import compute_prototypes, squared_euclidean_distances


@dataclass
class MatchingNetEpisodeOutput:
    """Differentiable predictions and mechanism data from one episode."""

    support_features: torch.Tensor
    query_features: torch.Tensor
    prototypes: torch.Tensor
    squared_distances: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    per_query_loss: torch.Tensor | None
    loss: torch.Tensor | None
    base_support_features: torch.Tensor
    base_query_features: torch.Tensor
    cosine_similarities: torch.Tensor
    support_attention: torch.Tensor


class AugmentedHiddenLSTMCell(nn.Module):
    """LSTM cell whose recurrent input is the paper's concatenated [h, r]."""

    def __init__(self, feature_dimension: int) -> None:
        super().__init__()
        if feature_dimension < 1:
            raise ValueError("feature_dimension must be positive")
        self.feature_dimension = int(feature_dimension)
        self.input_projection = nn.Linear(feature_dimension, 4 * feature_dimension)
        self.recurrent_projection = nn.Linear(
            2 * feature_dimension,
            4 * feature_dimension,
            bias=False,
        )

    def forward(
        self,
        fixed_input: torch.Tensor,
        hidden_and_readout: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dimension = self.feature_dimension
        if fixed_input.ndim != 2 or fixed_input.shape[1] != dimension:
            raise ValueError("fixed_input has an invalid feature shape")
        if hidden_and_readout.shape != (fixed_input.shape[0], 2 * dimension):
            raise ValueError("hidden_and_readout must concatenate h and r")
        if cell.shape != fixed_input.shape:
            raise ValueError("cell state does not match the fixed input")
        gates = self.input_projection(fixed_input) + self.recurrent_projection(
            hidden_and_readout
        )
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=1)
        next_cell = (
            torch.sigmoid(forget_gate) * cell
            + torch.sigmoid(input_gate) * torch.tanh(candidate)
        )
        next_hidden = torch.sigmoid(output_gate) * torch.tanh(next_cell)
        return next_hidden, next_cell


class FullContextSupportEmbedding(nn.Module):
    """Paper Appendix A.2: bidirectional LSTM plus residual base feature."""

    def __init__(self, feature_dimension: int) -> None:
        super().__init__()
        self.feature_dimension = int(feature_dimension)
        self.bidirectional_lstm = nn.LSTM(
            input_size=feature_dimension,
            hidden_size=feature_dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, support_features: torch.Tensor) -> torch.Tensor:
        if support_features.ndim != 2:
            raise ValueError("support_features must be two-dimensional")
        if support_features.shape[1] != self.feature_dimension:
            raise ValueError("Unexpected support feature dimension")
        sequence_output, _ = self.bidirectional_lstm(
            support_features.unsqueeze(0)
        )
        forward_hidden, backward_hidden = sequence_output.squeeze(0).chunk(2, dim=1)
        return support_features + forward_hidden + backward_hidden


class FullContextQueryEmbedding(nn.Module):
    """Paper Appendix A.1: multi-step attention LSTM with a residual query."""

    def __init__(self, feature_dimension: int, processing_steps: int) -> None:
        super().__init__()
        if processing_steps < 1:
            raise ValueError("processing_steps must be positive")
        self.feature_dimension = int(feature_dimension)
        self.processing_steps = int(processing_steps)
        self.lstm_cell = AugmentedHiddenLSTMCell(feature_dimension)

    def forward(
        self,
        query_features: torch.Tensor,
        contextual_support: torch.Tensor,
    ) -> torch.Tensor:
        if query_features.ndim != 2 or contextual_support.ndim != 2:
            raise ValueError("Query and support features must be two-dimensional")
        if (
            query_features.shape[1] != self.feature_dimension
            or contextual_support.shape[1] != self.feature_dimension
        ):
            raise ValueError("Full-context feature dimensions do not match")
        if contextual_support.shape[0] == 0:
            raise ValueError("Contextual support cannot be empty")

        hidden = query_features
        cell = torch.zeros_like(query_features)
        for _ in range(self.processing_steps):
            content_scores = hidden @ contextual_support.transpose(0, 1)
            read_attention = functional.softmax(content_scores, dim=1)
            readout = read_attention @ contextual_support
            hidden_hat, cell = self.lstm_cell(
                query_features,
                torch.cat((hidden, readout), dim=1),
                cell,
            )
            hidden = hidden_hat + query_features
        return hidden


class MatchingNet(nn.Module):
    """M3 Matching Network with full context and support-label attention."""

    def __init__(
        self,
        support_encoder: nn.Module,
        *,
        way: int = 3,
        processing_steps: int,
        query_encoder: nn.Module | None = None,
        probability_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        if way < 2:
            raise ValueError("way must be at least two")
        feature_dimension = getattr(support_encoder, "output_dimension", None)
        if feature_dimension is None:
            raise ValueError("support_encoder must declare output_dimension")
        if query_encoder is not None and getattr(
            query_encoder, "output_dimension", None
        ) != feature_dimension:
            raise ValueError("Support and query encoders must have the same output dimension")
        if not 0.0 < probability_epsilon < 1.0:
            raise ValueError("probability_epsilon must lie between zero and one")
        self.support_encoder = support_encoder
        self.query_encoder = query_encoder
        self.way = int(way)
        self.processing_steps = int(processing_steps)
        self.temperature = 1.0
        self.feature_dimension = int(feature_dimension)
        self.probability_epsilon = float(probability_epsilon)
        self.support_fce = FullContextSupportEmbedding(self.feature_dimension)
        self.query_fce = FullContextQueryEmbedding(
            self.feature_dimension, self.processing_steps
        )

    @property
    def encoders_are_shared(self) -> bool:
        return self.query_encoder is None

    def _validate_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor | None,
    ) -> None:
        if support_images.ndim != 4 or query_images.ndim != 4:
            raise ValueError(
                "Support and query images must have shape [count, channels, height, width]"
            )
        if support_labels.ndim != 1 or support_labels.shape[0] != support_images.shape[0]:
            raise ValueError("Support labels do not match the support image count")
        if support_labels.dtype != torch.long:
            raise ValueError("support_labels must use torch.long dtype")
        if support_labels.numel() == 0:
            raise ValueError("The support set cannot be empty")
        if int(support_labels.min()) < 0 or int(support_labels.max()) >= self.way:
            raise ValueError("Support labels fall outside the episode class range")
        if set(support_labels.detach().cpu().tolist()) != set(range(self.way)):
            raise ValueError("Every episode class must occur in the support set")
        if query_labels is not None:
            if query_labels.ndim != 1 or query_labels.shape[0] != query_images.shape[0]:
                raise ValueError("Query labels do not match the query image count")
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
    ) -> MatchingNetEpisodeOutput:
        self._validate_episode(
            support_images, support_labels, query_images, query_labels
        )
        base_support = self.support_encoder(support_images)
        query_encoder = (
            self.support_encoder if self.query_encoder is None else self.query_encoder
        )
        base_query = query_encoder(query_images)
        return self.forward_from_features(
            base_support,
            support_labels,
            base_query,
            query_labels,
        )

    def forward_from_features(
        self,
        base_support: torch.Tensor,
        support_labels: torch.Tensor,
        base_query: torch.Tensor,
        query_labels: torch.Tensor | None = None,
    ) -> MatchingNetEpisodeOutput:
        """Apply full-context matching to already encoded base features."""

        if base_support.ndim != 2 or base_query.ndim != 2:
            raise ValueError("Support and query features must be two-dimensional")
        if (
            base_support.shape[1] != self.feature_dimension
            or base_query.shape[1] != self.feature_dimension
        ):
            raise ValueError("Base feature dimensions do not match MatchingNet")
        if support_labels.ndim != 1 or support_labels.shape[0] != base_support.shape[0]:
            raise ValueError("Support labels do not match support features")
        if support_labels.dtype != torch.long:
            raise ValueError("support_labels must use torch.long dtype")
        if set(support_labels.detach().cpu().tolist()) != set(range(self.way)):
            raise ValueError("Every episode class must occur in the support set")
        if query_labels is not None:
            if query_labels.ndim != 1 or query_labels.shape[0] != base_query.shape[0]:
                raise ValueError("Query labels do not match query features")
            if query_labels.dtype != torch.long:
                raise ValueError("query_labels must use torch.long dtype")
            if int(query_labels.min()) < 0 or int(query_labels.max()) >= self.way:
                raise ValueError("Query labels fall outside the episode class range")
        contextual_support = self.support_fce(base_support)
        contextual_query = self.query_fce(base_query, contextual_support)

        normalised_support = functional.normalize(contextual_support, p=2, dim=1)
        normalised_query = functional.normalize(contextual_query, p=2, dim=1)
        cosine_similarities = normalised_query @ normalised_support.transpose(0, 1)
        support_attention = functional.softmax(cosine_similarities, dim=1)
        support_one_hot = functional.one_hot(
            support_labels, num_classes=self.way
        ).to(dtype=support_attention.dtype)
        probabilities = support_attention @ support_one_hot
        logits = probabilities.clamp_min(self.probability_epsilon).log()
        predictions = probabilities.argmax(dim=1)

        # Prototype distances are retained only for cross-model mechanism
        # analysis. They do not define MatchingNet predictions, so building an
        # autograd graph for them would waste training memory.
        with torch.no_grad():
            prototypes = compute_prototypes(
                contextual_support.detach(), support_labels, self.way
            )
            squared_distances = squared_euclidean_distances(
                contextual_query.detach(), prototypes
            )
        per_query_loss = None
        loss = None
        if query_labels is not None:
            per_query_loss = functional.nll_loss(
                logits, query_labels, reduction="none"
            )
            loss = per_query_loss.mean()
        return MatchingNetEpisodeOutput(
            support_features=contextual_support,
            query_features=contextual_query,
            prototypes=prototypes,
            squared_distances=squared_distances,
            logits=logits,
            probabilities=probabilities,
            predictions=predictions,
            per_query_loss=per_query_loss,
            loss=loss,
            base_support_features=base_support,
            base_query_features=base_query,
            cosine_similarities=cosine_similarities,
            support_attention=support_attention,
        )
