"""
Video decomposition into multimodal features
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Optional


class VideoDecomposer:
    """Decomposes videos into multimodal features"""

    def __init__(
        self,
        sample_fps: int = 2,
        extract_audio: bool = True,
        extract_music: bool = True,
        extract_motion: bool = True
    ):
        self.sample_fps = sample_fps
        self.extract_audio = extract_audio
        self.extract_music = extract_music
        self.extract_motion = extract_motion

    def decompose(self, video: torch.Tensor, metadata: Dict) -> Dict:
        """
        Decompose video into multimodal features.

        Args:
            video: Tensor of shape (T, H, W, C)
            metadata: Video metadata dict

        Returns:
            decomposed: Dict with all extracted features
        """
        decomposed = {
            'frames': self._sample_frames(video, metadata),
            'metadata': metadata
        }

        if self.extract_audio:
            decomposed['audio'] = self._extract_audio(video, metadata)

        if self.extract_music and self.extract_audio:
            decomposed['music'] = self._analyze_music(decomposed['audio'])

        if self.extract_motion:
            decomposed['motion'] = self._analyze_motion(decomposed['frames'])

        return decomposed

    def _sample_frames(self, video: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """
        Sample frames at target fps.

        Args:
            video: (T, H, W, C)
            metadata: Must contain 'fps'

        Returns:
            sampled_frames: (T', H, W, C) where T' is reduced
        """
        original_fps = metadata['fps']
        num_frames = video.shape[0]

        # Calculate sampling indices
        frame_indices = np.arange(0, num_frames, original_fps / self.sample_fps).astype(int)
        frame_indices = frame_indices[frame_indices < num_frames]

        sampled_frames = video[frame_indices]

        return sampled_frames

    def _extract_audio(self, video: torch.Tensor, metadata: Dict) -> Dict:
        """
        Extract audio features from video.
        Note: This is a placeholder - actual implementation depends on video format.
        """
        # In practice, audio would be loaded separately from video file
        # For now, return placeholder structure

        return {
            'waveform': None,  # Would be actual audio tensor
            'sample_rate': 44100,
            'duration': metadata['duration']
        }

    def _analyze_music(self, audio: Dict) -> Dict:
        """
        Analyze musical features using librosa.

        Returns:
            music_features: Dict with key, tempo, chords, etc.
        """
        # Placeholder - would use actual audio with librosa
        # For demonstration, return structure with default values

        return {
            'key': 'C',  # Detected key
            'tempo': 120.0,  # BPM
            'chords': ['C', 'F', 'G', 'C'],  # Chord progression
            'chord_times': [0.0, 1.25, 2.5, 3.75],  # Timestamps
            'spectral_centroid': 0.5,  # Average spectral centroid
            'energy': 0.7  # RMS energy
        }

    def _analyze_motion(self, frames: torch.Tensor) -> Dict:
        """
        Compute optical flow and motion statistics.

        Args:
            frames: (T, H, W, C)

        Returns:
            motion_features: Dict with flow fields and statistics
        """
        num_frames = frames.shape[0]
        flow_fields = []

        # Convert to numpy and grayscale
        frames_np = (frames.numpy() * 255).astype(np.uint8)
        frames_gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_np]

        # Compute optical flow between consecutive frames
        for i in range(len(frames_gray) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                frames_gray[i],
                frames_gray[i+1],
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            flow_fields.append(flow)

        # Compute motion statistics
        flow_magnitudes = [np.sqrt(flow[..., 0]**2 + flow[..., 1]**2) for flow in flow_fields]
        avg_magnitude = np.mean([np.mean(mag) for mag in flow_magnitudes]) if flow_magnitudes else 0.0
        max_magnitude = np.max([np.max(mag) for mag in flow_magnitudes]) if flow_magnitudes else 0.0

        # Detect cuts (large motion discontinuities)
        if len(flow_magnitudes) > 1:
            magnitude_changes = [flow_magnitudes[i+1].mean() - flow_magnitudes[i].mean()
                                for i in range(len(flow_magnitudes)-1)]
            cut_threshold = np.std(magnitude_changes) * 3 if len(magnitude_changes) > 0 else 0
            cuts_detected = [i for i, change in enumerate(magnitude_changes)
                            if abs(change) > cut_threshold]
        else:
            cuts_detected = []

        return {
            'flow_fields': flow_fields,  # List of (H, W, 2) arrays
            'avg_magnitude': float(avg_magnitude),
            'max_magnitude': float(max_magnitude),
            'cuts_detected': cuts_detected,
            'smoothness_score': 1.0 / (1.0 + avg_magnitude)  # Higher = smoother
        }
