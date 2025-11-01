"""
State management nodes for saving/loading knowledge base
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.knowledge_base import KnowledgeBase


class QVT_SaveKnowledgeBase:
    """Save knowledge base to disk"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "knowledge_base": ("KNOWLEDGE_BASE",),
                "save_path": ("STRING", {"default": "./knowledge_base.pkl"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save_kb"
    CATEGORY = "QuantumVideoTrainer/State"

    def save_kb(self, knowledge_base, save_path):
        # Update save path if different
        knowledge_base.save_path = Path(save_path)

        # Save to disk
        knowledge_base.save()

        status = f"""
Knowledge Base Saved
────────────────────
Path: {save_path}
Total Videos: {knowledge_base.state['total_videos']}
Sessions: {knowledge_base.state['training_sessions']}
        """.strip()

        return (status,)
