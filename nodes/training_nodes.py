"""
Training nodes for incremental learning
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.incremental_trainer import IncrementalTrainer


class QVT_IncrementalTrain:
    """Master training node - orchestrates full incremental learning"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "knowledge_base": ("KNOWLEDGE_BASE",),
                "video_batch": ("VIDEO_BATCH",),
                "batch_metadata": ("LIST",),
                "generations": ("INT", {"default": 50, "min": 10, "max": 500}),
                "beta": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("KNOWLEDGE_BASE", "STRING", "ARRAY")
    RETURN_NAMES = ("updated_kb", "training_report", "fitness_history")
    FUNCTION = "incremental_train"
    CATEGORY = "QuantumVideoTrainer/Training"

    def incremental_train(self, knowledge_base, video_batch, batch_metadata, generations, beta):
        trainer = IncrementalTrainer(
            knowledge_base=knowledge_base,
            generations=generations,
            beta=beta
        )

        # Run incremental training
        updated_kb, report, fitness_history = trainer.train_batch(
            videos=video_batch,
            metadata=batch_metadata
        )

        return (updated_kb, report, fitness_history)
