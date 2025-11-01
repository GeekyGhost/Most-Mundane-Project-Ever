"""
ComfyUI Quantum Video Trainer
First-principles multimodal video training with VFE optimization
"""

from .nodes.input_nodes import (
    QVT_LoadVideo,
    QVT_LoadVideoBatch,
    QVT_LoadKnowledgeBase,
)

from .nodes.decomposition_nodes import (
    QVT_DecomposeVideo,
)

from .nodes.theory_nodes import (
    QVT_MapToCircleOfFifths,
    QVT_MapToPhysics,
)

from .nodes.vfe_nodes import (
    QVT_ComputeVFE_Gaussian,
    QVT_ComputeVFE_Categorical,
)

from .nodes.training_nodes import (
    QVT_IncrementalTrain,
)

from .nodes.state_nodes import (
    QVT_SaveKnowledgeBase,
)

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    # Input nodes
    "QVT_LoadVideo": QVT_LoadVideo,
    "QVT_LoadVideoBatch": QVT_LoadVideoBatch,
    "QVT_LoadKnowledgeBase": QVT_LoadKnowledgeBase,

    # Decomposition nodes
    "QVT_DecomposeVideo": QVT_DecomposeVideo,

    # Theory nodes
    "QVT_MapToCircleOfFifths": QVT_MapToCircleOfFifths,
    "QVT_MapToPhysics": QVT_MapToPhysics,

    # VFE nodes
    "QVT_ComputeVFE_Gaussian": QVT_ComputeVFE_Gaussian,
    "QVT_ComputeVFE_Categorical": QVT_ComputeVFE_Categorical,

    # Training nodes
    "QVT_IncrementalTrain": QVT_IncrementalTrain,

    # State nodes
    "QVT_SaveKnowledgeBase": QVT_SaveKnowledgeBase,
}

# Display names for ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    # Input nodes
    "QVT_LoadVideo": "📹 Load Video",
    "QVT_LoadVideoBatch": "📹 Load Video Batch",
    "QVT_LoadKnowledgeBase": "💾 Load Knowledge Base",

    # Decomposition nodes
    "QVT_DecomposeVideo": "🔍 Decompose Video",

    # Theory nodes
    "QVT_MapToCircleOfFifths": "🎵 Map to Circle of Fifths",
    "QVT_MapToPhysics": "⚛️ Map to Physics",

    # VFE nodes
    "QVT_ComputeVFE_Gaussian": "📊 Compute VFE (Gaussian)",
    "QVT_ComputeVFE_Categorical": "📊 Compute VFE (Categorical)",

    # Training nodes
    "QVT_IncrementalTrain": "🎓 Incremental Train",

    # State nodes
    "QVT_SaveKnowledgeBase": "💾 Save Knowledge Base",
}

__version__ = "0.1.0"
__author__ = "GeekyGhost"
__description__ = "First-principles multimodal video training with Variational Free Energy"

print(f"\n{'='*60}")
print(f"Quantum Video Trainer v{__version__} loaded")
print(f"{len(NODE_CLASS_MAPPINGS)} nodes registered")
print(f"{'='*60}\n")
