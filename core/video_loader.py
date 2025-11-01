"""
Video loading utilities for Quantum Video Trainer
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


class VideoLoader:
    """Loads videos and converts them to tensor format"""

    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    def load(
        self,
        path: str,
        max_duration: float = 5.0,
        start_time: float = 0.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Load video and convert to tensor format.

        Args:
            path: Path to video file
            max_duration: Maximum duration to load in seconds
            start_time: Start timestamp in seconds

        Returns:
            video_tensor: torch.Tensor of shape (T, H, W, C)
            metadata: dict with fps, duration, resolution
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {path.suffix}")

        cap = cv2.VideoCapture(str(path))

        # Get metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # Calculate frame range
        start_frame = int(start_time * fps)
        max_frames = int(max_duration * fps)
        end_frame = min(start_frame + max_frames, total_frames)

        # Set start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Load frames
        frames = []
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError("No frames loaded from video")

        # Convert to tensor (T, H, W, C)
        video_tensor = torch.from_numpy(np.array(frames)).float() / 255.0

        metadata = {
            'fps': fps,
            'duration': (end_frame - start_frame) / fps if fps > 0 else 0,
            'total_duration': duration,
            'resolution': (width, height),
            'num_frames': len(frames),
            'start_frame': start_frame,
            'end_frame': end_frame
        }

        return video_tensor, metadata
