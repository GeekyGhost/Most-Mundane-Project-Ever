"""
Knowledge base for persistent storage of training state
"""

import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, Any, List


class KnowledgeBase:
    """Manages persistent storage of training state"""

    def __init__(self, save_path='./knowledge_base.pkl'):
        self.save_path = Path(save_path)
        self.state = None

    def initialize(self) -> Dict:
        """Create new knowledge base with default priors"""
        self.state = {
            'version': '1.0.0',
            'created_at': datetime.now(),
            'last_updated': datetime.now(),
            'total_videos': 0,
            'training_sessions': 0,

            # Theory weights (genome)
            'theory_weights': {
                'circle_of_fifths_weight': 0.8,
                'jazz_extensions_weight': 0.2,
                'continuity_editing_weight': 0.7,
                'montage_editing_weight': 0.3,
                'motion_smoothness_weight': 0.8,
                'chaos_weight': 0.2,
            },

            # VFE priors (Gaussian)
            'vfe_priors_gaussian': {
                'motion_smoothness': {
                    'mu': 0.7,
                    'sigma': 0.3
                },
                'audio_visual_sync': {
                    'mu': 0.8,
                    'sigma': 0.2
                }
            },

            # VFE priors (Categorical)
            'vfe_priors_categorical': {
                'editing_style': {
                    'continuity': 0.6,
                    'montage': 0.3,
                    'experimental': 0.1
                },
                'music_theory_type': {
                    'circle_of_fifths': 0.7,
                    'jazz': 0.2,
                    'experimental': 0.1
                }
            },

            # Video fingerprints (compressed)
            'video_fingerprints': [],

            # Evolution history
            'evolution_history': [],

            # Best fitness scores
            'best_fitness': -np.inf,
            'best_fitness_history': []
        }

        return self.state

    def load(self) -> bool:
        """Load existing knowledge base"""
        if not self.save_path.exists():
            return False

        try:
            with open(self.save_path, 'rb') as f:
                self.state = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return False

    def save(self):
        """Save current state to disk"""
        self.state['last_updated'] = datetime.now()

        # Ensure directory exists
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.save_path, 'wb') as f:
            pickle.dump(self.state, f)

        print(f"💾 Knowledge base saved: {self.state['total_videos']} videos")

    def add_video_fingerprint(self, fingerprint: Dict):
        """Add a new video fingerprint to history"""
        self.state['video_fingerprints'].append(fingerprint)
        self.state['total_videos'] += 1

    def update_theory_weights(self, new_weights: Dict):
        """Update theory weights after evolution"""
        self.state['theory_weights'] = new_weights

    def update_vfe_priors(self, new_priors_gaussian: Dict, new_priors_categorical: Dict):
        """Update VFE priors after Bayesian update"""
        self.state['vfe_priors_gaussian'] = new_priors_gaussian
        self.state['vfe_priors_categorical'] = new_priors_categorical

    def add_evolution_record(self, generation: int, fitness: float, genome: Dict):
        """Record evolution history"""
        self.state['evolution_history'].append({
            'generation': generation,
            'fitness': fitness,
            'genome': genome,
            'timestamp': datetime.now()
        })

        if fitness > self.state['best_fitness']:
            self.state['best_fitness'] = fitness

        self.state['best_fitness_history'].append(fitness)

    def get_summary(self) -> str:
        """Return text summary of current state"""
        return f"""
Knowledge Base Summary
══════════════════════
Videos Trained: {self.state['total_videos']}
Sessions: {self.state['training_sessions']}
Best Fitness: {self.state['best_fitness']:.4f}
Last Updated: {self.state['last_updated'].strftime('%Y-%m-%d %H:%M')}

Theory Weights:
{self._format_dict(self.state['theory_weights'])}

VFE Priors (Gaussian):
{self._format_dict(self.state['vfe_priors_gaussian'])}

VFE Priors (Categorical):
{self._format_dict(self.state['vfe_priors_categorical'])}
        """.strip()

    def _format_dict(self, d: Dict, indent: int = 2) -> str:
        """Format dict for display"""
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{' '*indent}{k}:")
                lines.append(self._format_dict(v, indent+2))
            else:
                if isinstance(v, float):
                    lines.append(f"{' '*indent}{k}: {v:.4f}")
                else:
                    lines.append(f"{' '*indent}{k}: {v}")
        return '\n'.join(lines)
