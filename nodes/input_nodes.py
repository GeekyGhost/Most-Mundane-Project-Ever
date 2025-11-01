"""
Input nodes for loading videos and knowledge base
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.video_loader import VideoLoader
from core.knowledge_base import KnowledgeBase
import glob


class QVT_LoadVideo:
    """Load a single video file"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "max_duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("VIDEO", "DICT")
    RETURN_NAMES = ("video", "metadata")
    FUNCTION = "load_video"
    CATEGORY = "QuantumVideoTrainer/Input"

    def load_video(self, video_path, max_duration, start_time):
        loader = VideoLoader()
        video_tensor, metadata = loader.load(
            path=video_path,
            max_duration=max_duration,
            start_time=start_time
        )

        return (video_tensor, metadata)


class QVT_LoadVideoBatch:
    """Load multiple videos for batch training"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_directory": ("STRING", {"default": "", "multiline": False}),
                "pattern": ("STRING", {"default": "*.mp4"}),
                "max_videos": ("INT", {"default": 10, "min": 1, "max": 100}),
                "max_duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 60.0}),
            }
        }

    RETURN_TYPES = ("VIDEO_BATCH", "LIST")
    RETURN_NAMES = ("video_batch", "batch_metadata")
    FUNCTION = "load_batch"
    CATEGORY = "QuantumVideoTrainer/Input"

    def load_batch(self, video_directory, pattern, max_videos, max_duration):
        loader = VideoLoader()
        video_dir = Path(video_directory)

        # Find matching videos
        video_paths = list(glob.glob(str(video_dir / pattern)))[:max_videos]

        if len(video_paths) == 0:
            raise ValueError(f"No videos found in {video_directory} matching {pattern}")

        # Load all videos
        videos = []
        metadata_list = []

        for video_path in video_paths:
            try:
                video, metadata = loader.load(video_path, max_duration=max_duration)
                videos.append(video)
                metadata['source_path'] = video_path
                metadata_list.append(metadata)
            except Exception as e:
                print(f"Warning: Failed to load {video_path}: {e}")
                continue

        return (videos, metadata_list)


class QVT_LoadKnowledgeBase:
    """Load or create knowledge base"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "knowledge_base_path": ("STRING", {"default": "./knowledge_base.pkl"}),
                "create_if_missing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("KNOWLEDGE_BASE", "STRING")
    RETURN_NAMES = ("knowledge_base", "summary")
    FUNCTION = "load_kb"
    CATEGORY = "QuantumVideoTrainer/State"

    def load_kb(self, knowledge_base_path, create_if_missing):
        kb = KnowledgeBase(save_path=knowledge_base_path)

        if kb.load():
            summary = f"""
Knowledge Base Loaded
─────────────────────
Total Videos: {kb.state['total_videos']}
Training Sessions: {kb.state['training_sessions']}
Last Updated: {kb.state['last_updated']}
            """.strip()
        elif create_if_missing:
            kb.initialize()
            summary = "New knowledge base created"
        else:
            raise FileNotFoundError(f"Knowledge base not found: {knowledge_base_path}")

        return (kb, summary)
