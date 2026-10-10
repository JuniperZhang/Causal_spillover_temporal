"""
LSTM propensity model (full specification in combined_model.py).

Classes:
    GraphSAGEMeanAggregator: combines own and neighbour inputs into I_t
    LSTMBackbone: LSTM over processing times
    DecisionLatentHead_gZ: decision-time latents (Z^X, Z^D, Z^C)
    OwnTreatmentHead_fX, SpilloverHead_fD: assignment models
    ReconstructionHead_I: reconstructs I_t for the loss L_I
    ContinuousLSTMCausalModel (alias TemporalCausalModelSpillover): full model
"""

from .encoder import GraphSAGEMeanAggregator, LSTMBackbone
from .treatment_model import (
    DecisionLatentHead_gZ,
    OwnTreatmentHead_fX,
    SpilloverHead_fD,
)
from .reconstruction import ReconstructionHead_I
from .combined_model import ContinuousLSTMCausalModel, TemporalCausalModelSpillover

__all__ = [
    'GraphSAGEMeanAggregator',
    'LSTMBackbone',
    'DecisionLatentHead_gZ',
    'OwnTreatmentHead_fX',
    'SpilloverHead_fD',
    'ReconstructionHead_I',
    'ContinuousLSTMCausalModel',
    'TemporalCausalModelSpillover',
]
