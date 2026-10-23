"""AI4S application development framework.

Top-level exports: the :class:`AI4SApplication` SDK (base types + registry)
and the built-in applications, each an ``AI4SApplication`` subclass that
implements the five-method interface (produce data / build model / make
dataset / train / infer) and inherits the full-stack pipeline for free.
"""

from .ai_les_app import AILesApp
from .base import (
    AI4SApplication,
    ApplicationRegistry,
    DataProduct,
    Prediction,
    RunReport,
    TrainingResult,
    registry,
)
from .flow_transformer_app import FlowTransformerApp
from .generative_flow import GenerativeFlow
from .inverse_problem import InverseProblem
from .mesh_gnn_flow import MeshGNNFlow
from .neural_operator_fno import NeuralOperatorFNO
from .physics_informed_lbm import PhysicsInformedLBM
from .suboff_app import SuboffSurrogateApp
from .uncertainty_quantification import UncertaintyQuantification

__all__ = [
    "AI4SApplication",
    "ApplicationRegistry",
    "DataProduct",
    "Prediction",
    "RunReport",
    "TrainingResult",
    "registry",
    "AILesApp",
    "FlowTransformerApp",
    "GenerativeFlow",
    "InverseProblem",
    "MeshGNNFlow",
    "NeuralOperatorFNO",
    "PhysicsInformedLBM",
    "SuboffSurrogateApp",
    "UncertaintyQuantification",
]
