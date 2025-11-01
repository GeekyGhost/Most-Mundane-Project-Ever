"""
Decomposition nodes for feature extraction
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.decomposer import VideoDecomposer


class QVT_DecomposeVideo:
    """Decompose video into multimodal features"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "metadata": ("DICT",),
                "sample_fps": ("INT", {"default": 2, "min": 1, "max": 30}),
                "extract_audio": ("BOOLEAN", {"default": True}),
                "extract_music": ("BOOLEAN", {"default": True}),
                "extract_motion": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("DECOMPOSED_VIDEO", "STRING")
    RETURN_NAMES = ("decomposed", "summary")
    FUNCTION = "decompose"
    CATEGORY = "QuantumVideoTrainer/Decomposition"

    def decompose(self, video, metadata, sample_fps, extract_audio, extract_music, extract_motion):
        decomposer = VideoDecomposer(
            sample_fps=sample_fps,
            extract_audio=extract_audio,
            extract_music=extract_music,
            extract_motion=extract_motion
        )

        decomposed = decomposer.decompose(video, metadata)

        summary = f"""
Decomposition Complete
─────────────────────
Frames: {len(decomposed['frames'])}
Audio: {'✓' if extract_audio else '✗'}
Music: {'✓' if extract_music else '✗'}
Motion: {'✓' if extract_motion else '✗'}
        """.strip()

        return (decomposed, summary)
