"""
Musical theory mappers - Circle of Fifths and Jazz extensions
"""

import numpy as np
from typing import Dict, Tuple, List


class CircleOfFifthsMapper:
    """Maps chord progressions to circle of fifths theory space"""

    # Circle of fifths: C -> G -> D -> A -> E -> B -> F#/Gb -> Db -> Ab -> Eb -> Bb -> F -> C
    CIRCLE = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']

    # Common progressions and their "consonance" scores
    COMMON_PROGRESSIONS = {
        ('C', 'G'): 1.0,  # V-I (perfect)
        ('G', 'C'): 1.0,
        ('C', 'F'): 0.9,  # IV-I (strong)
        ('F', 'C'): 0.9,
        ('C', 'Am'): 0.95,  # Relative minor
        ('C', 'Dm'): 0.85,  # ii-I
        ('C', 'Em'): 0.85,  # iii-I
    }

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

    def map(self, music_features: Dict) -> Tuple[Dict, float]:
        """
        Map chord progression to circle of fifths analysis.

        Args:
            music_features: Dict with 'chords', 'key', etc.

        Returns:
            harmonic_profile: Analysis dict
            adherence_score: 0.0-1.0 score
        """
        chords = music_features.get('chords', [])
        key = music_features.get('key', 'C')

        if len(chords) == 0:
            return {'error': 'No chords detected'}, 0.0

        # Normalize chords to root notes
        root_notes = [self._extract_root(chord) for chord in chords]

        # Map to circle positions
        circle_positions = [self._get_circle_position(note) for note in root_notes]

        # Analyze transitions
        transitions = []
        transition_scores = []

        for i in range(len(circle_positions) - 1):
            pos1, pos2 = circle_positions[i], circle_positions[i+1]
            distance = self._circle_distance(pos1, pos2)

            # Score transition (closer = better)
            if distance <= 1:
                score = 1.0  # Perfect (adjacent on circle)
            elif distance <= 2:
                score = 0.8  # Good
            elif distance <= 3:
                score = 0.6  # Okay
            else:
                score = 0.3  # Distant (unless jazz)

            transitions.append({
                'from': root_notes[i],
                'to': root_notes[i+1],
                'distance': distance,
                'score': score
            })
            transition_scores.append(score)

        # Overall adherence
        if len(transition_scores) > 0:
            adherence_score = float(np.mean(transition_scores))
        else:
            adherence_score = 0.5  # Neutral for single chord

        # Penalize if strict mode and has distant jumps
        if self.strict_mode and adherence_score < 0.7:
            adherence_score *= 0.5

        harmonic_profile = {
            'key': key,
            'chords': chords,
            'root_notes': root_notes,
            'circle_positions': circle_positions,
            'transitions': transitions,
            'avg_transition_distance': float(np.mean([t['distance'] for t in transitions])) if transitions else 0.0,
            'adherence_score': adherence_score,
            'strict_mode': self.strict_mode
        }

        return harmonic_profile, adherence_score

    def _extract_root(self, chord: str) -> str:
        """Extract root note from chord symbol (e.g., 'Cmaj7' -> 'C')"""
        # Simple parser - just take first 1-2 characters
        if len(chord) > 1 and chord[1] in ['#', 'b']:
            return chord[:2]
        return chord[0]

    def _get_circle_position(self, note: str) -> int:
        """Get position on circle of fifths (0-11)"""
        try:
            return self.CIRCLE.index(note)
        except ValueError:
            # Handle enharmonic equivalents
            enharmonic_map = {
                'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb',
                'G#': 'Ab', 'A#': 'Bb'
            }
            if note in enharmonic_map:
                return self.CIRCLE.index(enharmonic_map[note])
            return 0  # Default to C if unknown

    def _circle_distance(self, pos1: int, pos2: int) -> int:
        """Calculate shortest distance on circle"""
        forward = (pos2 - pos1) % 12
        backward = (pos1 - pos2) % 12
        return min(forward, backward)


class JazzHarmonyMapper:
    """
    Extension of CircleOfFifths that allows jazz progressions.
    """

    # Jazz allows tritone substitutions, altered dominants, etc.
    JAZZ_PROGRESSIONS = {
        ('C', 'Db'): 0.8,  # Tritone sub
        ('C', 'Ab'): 0.7,  # bVI (common in jazz)
        ('C', 'Eb'): 0.7,  # bIII
    }

    def __init__(self):
        self.circle_mapper = CircleOfFifthsMapper(strict_mode=False)

    def map(self, music_features: Dict) -> Tuple[Dict, float]:
        """
        Map with jazz theory extensions.
        Increases scores for jazz-appropriate distant jumps.
        """
        harmonic_profile, base_adherence = self.circle_mapper.map(music_features)

        # Adjust scores for jazz transitions
        if 'transitions' in harmonic_profile:
            adjusted_scores = []
            for trans in harmonic_profile['transitions']:
                base_score = trans['score']

                # Check if this is a known jazz progression
                transition_key = (trans['from'], trans['to'])
                if transition_key in self.JAZZ_PROGRESSIONS:
                    # Boost score for valid jazz move
                    adjusted_score = max(base_score, self.JAZZ_PROGRESSIONS[transition_key])
                else:
                    adjusted_score = base_score

                adjusted_scores.append(adjusted_score)

            if adjusted_scores:
                adherence_score = float(np.mean(adjusted_scores))
            else:
                adherence_score = base_adherence
        else:
            adherence_score = base_adherence

        harmonic_profile['jazz_adjusted'] = True
        harmonic_profile['adherence_score'] = adherence_score

        return harmonic_profile, adherence_score
