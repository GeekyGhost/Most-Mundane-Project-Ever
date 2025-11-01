"""
Physics-based motion theory mappers
"""

import numpy as np
from typing import Dict, Tuple


class PhysicsMapper:
    """Maps motion to physics constraint spaces"""

    def __init__(self, model: str = "smooth_motion"):
        self.model = model
        self.models = {
            'smooth_motion': self._map_smooth_motion,
            'lorenz_attractor': self._map_lorenz,
            'orbital': self._map_orbital,
            'chaos': self._map_chaos
        }

    def map(self, motion_features: Dict) -> Tuple[Dict, float]:
        """
        Map motion to physics model.

        Returns:
            physics_profile: Analysis dict
            conformity_score: How well motion matches model
        """
        if self.model not in self.models:
            raise ValueError(f"Unknown physics model: {self.model}")

        return self.models[self.model](motion_features)

    def _map_smooth_motion(self, motion: Dict) -> Tuple[Dict, float]:
        """
        Map to smooth motion model (temporal continuity).
        Higher smoothness score = better conformity.
        """
        smoothness = motion.get('smoothness_score', 0.5)
        avg_magnitude = motion.get('avg_magnitude', 0.0)
        cuts = motion.get('cuts_detected', [])

        # Smooth motion should have:
        # - High smoothness score
        # - Low average magnitude (slow motion)
        # - No cuts

        conformity_score = smoothness * 0.7
        if avg_magnitude < 1.0:
            conformity_score += 0.2
        if len(cuts) == 0:
            conformity_score += 0.1

        conformity_score = min(1.0, conformity_score)

        physics_profile = {
            'model': 'smooth_motion',
            'smoothness': smoothness,
            'avg_velocity': avg_magnitude,
            'num_cuts': len(cuts),
            'conformity_score': conformity_score,
            'interpretation': 'Camera motion is smooth and continuous' if conformity_score > 0.7 else 'Motion has discontinuities'
        }

        return physics_profile, conformity_score

    def _map_lorenz(self, motion: Dict) -> Tuple[Dict, float]:
        """
        Map to Lorenz attractor (chaotic but structured).
        """
        flow_fields = motion.get('flow_fields', [])

        if len(flow_fields) == 0:
            return {'error': 'No flow fields'}, 0.0

        # Analyze flow field structure
        # Lorenz attractor should show:
        # - Chaotic behavior (high variability)
        # - But with strange attractor structure (not random)

        # Compute flow statistics
        magnitudes = [np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).flatten()
                     for flow in flow_fields]

        # Measure chaos (high std)
        all_mags = np.concatenate(magnitudes)
        chaos_score = np.std(all_mags) / (np.mean(all_mags) + 1e-6)
        chaos_score = min(1.0, chaos_score / 2.0)  # Normalize

        # Measure structure (autocorrelation)
        if len(all_mags) > 10:
            autocorr = np.correlate(all_mags, all_mags, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            structure_score = float(np.mean(autocorr[:10]) / (autocorr[0] + 1e-6))
        else:
            structure_score = 0.0

        # Lorenz: high chaos + some structure
        conformity_score = (chaos_score * 0.6 + structure_score * 0.4)

        physics_profile = {
            'model': 'lorenz_attractor',
            'chaos_score': chaos_score,
            'structure_score': structure_score,
            'conformity_score': conformity_score,
            'interpretation': 'Motion shows chaotic attractor behavior' if conformity_score > 0.6 else 'Motion not chaotic'
        }

        return physics_profile, conformity_score

    def _map_orbital(self, motion: Dict) -> Tuple[Dict, float]:
        """
        Map to orbital mechanics (circular/elliptical paths).
        """
        flow_fields = motion.get('flow_fields', [])

        if len(flow_fields) == 0:
            return {'error': 'No flow fields'}, 0.0

        # Check for circular motion patterns
        # Orbital motion should show:
        # - Consistent magnitude (constant speed)
        # - Rotating direction (angular momentum)

        magnitudes = [np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                     for flow in flow_fields]

        # Consistency in magnitude
        magnitude_std = np.std(magnitudes)
        magnitude_mean = np.mean(magnitudes)
        consistency_score = 1.0 - min(1.0, magnitude_std / (magnitude_mean + 1e-6))

        # Check for rotation (average flow direction changes smoothly)
        angles = []
        for flow in flow_fields:
            angle = np.arctan2(flow[..., 1].mean(), flow[..., 0].mean())
            angles.append(angle)

        # Angular velocity consistency
        if len(angles) > 1:
            angular_diffs = np.diff(angles)
            angular_consistency = 1.0 - min(1.0, np.std(angular_diffs) / (np.pi/4))
        else:
            angular_consistency = 0.0

        conformity_score = (consistency_score * 0.5 + angular_consistency * 0.5)

        physics_profile = {
            'model': 'orbital',
            'magnitude_consistency': consistency_score,
            'angular_consistency': angular_consistency,
            'conformity_score': conformity_score,
            'interpretation': 'Motion shows orbital patterns' if conformity_score > 0.6 else 'Motion not orbital'
        }

        return physics_profile, conformity_score

    def _map_chaos(self, motion: Dict) -> Tuple[Dict, float]:
        """
        Map to pure chaos (unpredictable, no structure).
        """
        flow_fields = motion.get('flow_fields', [])
        cuts = motion.get('cuts_detected', [])

        if len(flow_fields) == 0:
            return {'error': 'No flow fields'}, 0.0

        # Pure chaos should have:
        # - High variability
        # - No consistent structure
        # - Frequent discontinuities

        magnitudes = [np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).flatten()
                     for flow in flow_fields]
        all_mags = np.concatenate(magnitudes)

        # High entropy = chaos
        variability_score = min(1.0, np.std(all_mags) / (np.mean(all_mags) + 1e-6))

        # Many cuts = chaos
        cut_score = min(1.0, len(cuts) / max(1, len(flow_fields) / 2))

        # Low autocorrelation = chaos
        if len(all_mags) > 10:
            autocorr = np.correlate(all_mags, all_mags, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            unpredictability = 1.0 - float(np.mean(autocorr[1:6]) / (autocorr[0] + 1e-6))
        else:
            unpredictability = 0.5

        conformity_score = (variability_score * 0.4 + cut_score * 0.3 + unpredictability * 0.3)

        physics_profile = {
            'model': 'chaos',
            'variability': variability_score,
            'discontinuities': cut_score,
            'unpredictability': unpredictability,
            'conformity_score': conformity_score,
            'interpretation': 'Motion is highly chaotic' if conformity_score > 0.7 else 'Motion has structure'
        }

        return physics_profile, conformity_score
