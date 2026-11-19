"""Model components for the eyelid few-shot experiments."""

from .backbones import (
    ResNet18Encoder,
    ResNet18SpatialEncoder,
    build_resnet18_encoder,
    build_resnet18_spatial_encoder,
)
from .logistic import (
    LogisticRegressionEpisodeModel,
    LogisticRegressionFit,
    fit_multinomial_l2_logistic_regression,
)
from .matchingnet import (
    AugmentedHiddenLSTMCell,
    FullContextQueryEmbedding,
    FullContextSupportEmbedding,
    MatchingNet,
    MatchingNetEpisodeOutput,
)
from .protonet import (
    ProtoNet,
    ProtoNetEpisodeOutput,
    compute_prototypes,
    squared_euclidean_distances,
)
from .relationnet import RelationModule, RelationNet, RelationNetEpisodeOutput

__all__ = [
    "ProtoNet",
    "ProtoNetEpisodeOutput",
    "ResNet18Encoder",
    "ResNet18SpatialEncoder",
    "LogisticRegressionEpisodeModel",
    "LogisticRegressionFit",
    "AugmentedHiddenLSTMCell",
    "FullContextQueryEmbedding",
    "FullContextSupportEmbedding",
    "MatchingNet",
    "MatchingNetEpisodeOutput",
    "RelationModule",
    "RelationNet",
    "RelationNetEpisodeOutput",
    "build_resnet18_encoder",
    "build_resnet18_spatial_encoder",
    "compute_prototypes",
    "fit_multinomial_l2_logistic_regression",
    "squared_euclidean_distances",
]
