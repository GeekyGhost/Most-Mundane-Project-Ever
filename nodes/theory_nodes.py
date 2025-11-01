"""
Theory mapping nodes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from theories.musical_theories import CircleOfFifthsMapper
from theories.physics_theories import PhysicsMapper


class QVT_MapToCircleOfFifths:
    """Map musical features to circle of fifths theory"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decomposed": ("DECOMPOSED_VIDEO",),
                "strict_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("DICT", "FLOAT")
    RETURN_NAMES = ("harmonic_profile", "adherence_score")
    FUNCTION = "map_harmony"
    CATEGORY = "QuantumVideoTrainer/Theory"

    def map_harmony(self, decomposed, strict_mode):
        mapper = CircleOfFifthsMapper(strict_mode=strict_mode)
        harmonic_profile, adherence = mapper.map(decomposed['music'])

        return (harmonic_profile, adherence)


class QVT_MapToPhysics:
    """Map motion to physics constraint space"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decomposed": ("DECOMPOSED_VIDEO",),
                "physics_model": (["smooth_motion", "lorenz_attractor", "orbital", "chaos"],),
            }
        }

    RETURN_TYPES = ("DICT", "FLOAT")
    RETURN_NAMES = ("physics_profile", "conformity_score")
    FUNCTION = "map_physics"
    CATEGORY = "QuantumVideoTrainer/Theory"

    def map_physics(self, decomposed, physics_model):
        mapper = PhysicsMapper(model=physics_model)
        physics_profile, conformity = mapper.map(decomposed['motion'])

        return (physics_profile, conformity)
