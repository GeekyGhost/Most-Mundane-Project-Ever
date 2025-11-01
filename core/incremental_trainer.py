"""
Incremental training orchestration
Coordinates decomposition, theory mapping, VFE computation, and evolution
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from .decomposer import VideoDecomposer
from .knowledge_base import KnowledgeBase
from ..theories.musical_theories import CircleOfFifthsMapper, JazzHarmonyMapper
from ..theories.physics_theories import PhysicsMapper
from ..pmc_vfe.vfe_gaussian import vfe_gaussian
from ..pmc_vfe.vfe_categorical import vfe_categorical


class IncrementalTrainer:
    """
    Orchestrates incremental training:
    1. Decompose videos
    2. Map to theories
    3. Compute VFE
    4. Evolve theory weights
    5. Update priors
    """

    def __init__(self, knowledge_base: KnowledgeBase, generations: int = 50, beta: float = 0.7):
        self.kb = knowledge_base
        self.generations = generations
        self.beta = beta

        # Initialize components
        self.decomposer = VideoDecomposer(sample_fps=2)
        self.circle_mapper = CircleOfFifthsMapper()
        self.jazz_mapper = JazzHarmonyMapper()
        self.physics_mapper = PhysicsMapper(model='smooth_motion')

    def train_batch(
        self,
        videos: List[torch.Tensor],
        metadata: List[Dict]
    ) -> Tuple[KnowledgeBase, str, np.ndarray]:
        """
        Main training loop for a batch of videos.

        Args:
            videos: List of video tensors
            metadata: List of metadata dicts

        Returns:
            updated_kb: Updated knowledge base
            report: Training summary string
            fitness_history: Array of fitness scores per generation
        """
        print(f"\n{'='*60}")
        print(f"INCREMENTAL TRAINING SESSION")
        print(f"{'='*60}")
        print(f"Videos in batch: {len(videos)}")
        print(f"Current total: {self.kb.state['total_videos']}")
        print(f"Generations: {self.generations}")
        print(f"Beta: {self.beta}")
        print(f"{'='*60}\n")

        # Step 1: Decompose all videos
        print("Step 1: Decomposing videos...")
        decomposed_videos = []
        for i, (video, meta) in enumerate(zip(videos, metadata)):
            print(f"  Decomposing video {i+1}/{len(videos)}...")
            decomposed = self.decomposer.decompose(video, meta)
            decomposed_videos.append(decomposed)
        print("✓ Decomposition complete\n")

        # Step 2: Map to theory spaces
        print("Step 2: Mapping to theory spaces...")
        theory_fingerprints = []
        for i, decomposed in enumerate(decomposed_videos):
            print(f"  Mapping video {i+1}/{len(videos)}...")
            fingerprint = self._create_theory_fingerprint(decomposed)
            theory_fingerprints.append(fingerprint)
        print("✓ Theory mapping complete\n")

        # Step 3: Evolve theory weights
        print("Step 3: Evolving theory weights...")
        best_genome, evolution_history = self._evolve_theory_weights(
            theory_fingerprints,
            decomposed_videos
        )
        print(f"✓ Evolution complete (best fitness: {evolution_history[-1]['fitness']:.4f})\n")

        # Step 4: Update VFE priors
        print("Step 4: Updating VFE priors...")
        self._update_vfe_priors(theory_fingerprints)
        print("✓ Priors updated\n")

        # Step 5: Save to knowledge base
        print("Step 5: Saving to knowledge base...")
        for fingerprint in theory_fingerprints:
            self.kb.add_video_fingerprint(fingerprint)

        self.kb.update_theory_weights(best_genome)

        for record in evolution_history:
            self.kb.add_evolution_record(
                generation=record['generation'],
                fitness=record['fitness'],
                genome=record['genome']
            )

        self.kb.state['training_sessions'] += 1
        self.kb.save()
        print("✓ Knowledge base saved\n")

        # Generate report
        report = self._generate_report(evolution_history)

        # Extract fitness history
        fitness_history = np.array([r['fitness'] for r in evolution_history])

        print(f"{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}\n")

        return (self.kb, report, fitness_history)

    def _create_theory_fingerprint(self, decomposed: Dict) -> Dict:
        """
        Create compressed theory-space representation of video.
        """
        # Map music to circle of fifths
        if 'music' in decomposed:
            harmonic_profile, harmonic_score = self.circle_mapper.map(decomposed['music'])
        else:
            harmonic_profile, harmonic_score = {}, 0.5

        # Map motion to physics
        if 'motion' in decomposed:
            physics_profile, physics_score = self.physics_mapper.map(decomposed['motion'])
        else:
            physics_profile, physics_score = {}, 0.5

        fingerprint = {
            'harmonic': harmonic_profile,
            'harmonic_score': harmonic_score,
            'physics': physics_profile,
            'physics_score': physics_score,
            'duration': decomposed['metadata']['duration'],
            'num_frames': len(decomposed['frames']),
        }

        return fingerprint

    def _evolve_theory_weights(
        self,
        fingerprints: List[Dict],
        decomposed_videos: List[Dict]
    ) -> Tuple[Dict, List[Dict]]:
        """
        Run evolutionary algorithm to optimize theory weights.
        """
        # Initialize population from current best
        current_weights = self.kb.state['theory_weights']
        population_size = max(10, len(fingerprints))
        population = self._initialize_population(current_weights, size=population_size)

        history = []

        for gen in range(self.generations):
            # Evaluate fitness for each genome
            fitness_scores = []

            for genome in population:
                fitness = self._compute_fitness(
                    genome,
                    fingerprints,
                    decomposed_videos
                )
                fitness_scores.append((genome, fitness))

            # Sort by fitness (higher is better)
            fitness_scores.sort(key=lambda x: x[1], reverse=True)

            # Track best
            best_genome, best_fitness = fitness_scores[0]
            history.append({
                'generation': gen,
                'fitness': best_fitness,
                'genome': best_genome.copy()
            })

            if (gen + 1) % 10 == 0 or gen == 0:
                print(f"    Generation {gen+1}/{self.generations}: fitness = {best_fitness:.4f}")

            # Selection: keep top 30%
            num_survivors = max(2, len(population) // 3)
            survivors = [g for g, f in fitness_scores[:num_survivors]]

            # Create next generation
            new_population = survivors.copy()

            while len(new_population) < len(population):
                parent1 = np.random.choice(len(survivors))
                parent2 = np.random.choice(len(survivors))

                child = self._crossover(survivors[parent1], survivors[parent2])
                child = self._mutate(child, mutation_rate=0.15)

                new_population.append(child)

            population = new_population

        return history[-1]['genome'], history

    def _initialize_population(self, base_genome: Dict, size: int) -> List[Dict]:
        """Create initial population around current best genome"""
        population = [base_genome.copy()]  # Keep current best

        # Create variants
        for _ in range(size - 1):
            variant = self._mutate(base_genome.copy(), mutation_rate=0.2)
            population.append(variant)

        return population

    def _compute_fitness(
        self,
        genome: Dict,
        fingerprints: List[Dict],
        decomposed_videos: List[Dict]
    ) -> float:
        """
        Compute fitness of genome using VFE + EFE.

        Fitness = -VFE (lower free energy = higher fitness)
        """
        total_vfe = 0.0

        for fingerprint in fingerprints:
            # Compute VFE for harmonic features
            if 'harmonic_score' in fingerprint:
                # Simple VFE based on conformity to theory
                harmonic_vfe = (1.0 - fingerprint['harmonic_score']) * genome['circle_of_fifths_weight']
                total_vfe += harmonic_vfe

            # Compute VFE for motion
            if 'physics_score' in fingerprint:
                motion_vfe = (1.0 - fingerprint['physics_score']) * genome['motion_smoothness_weight']
                total_vfe += motion_vfe

        # Average across videos
        if len(fingerprints) > 0:
            total_vfe /= len(fingerprints)

        # Add regularization to prevent extreme weights
        weight_variance = np.var(list(genome.values()))
        regularization = 0.1 * weight_variance

        total_free_energy = total_vfe + regularization

        # Fitness is negative free energy
        fitness = -total_free_energy

        return fitness

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Combine two parent genomes"""
        child = {}
        for key in parent1.keys():
            alpha = np.random.random()
            child[key] = alpha * parent1[key] + (1 - alpha) * parent2[key]

        # Normalize within categories
        return self._normalize_genome(child)

    def _mutate(self, genome: Dict, mutation_rate: float = 0.15) -> Dict:
        """Randomly perturb genome"""
        mutated = genome.copy()

        for key in genome.keys():
            if np.random.random() < mutation_rate:
                mutated[key] += np.random.normal(0, 0.05)
                mutated[key] = max(0.0, min(1.0, mutated[key]))  # Clamp to [0, 1]

        return self._normalize_genome(mutated)

    def _normalize_genome(self, genome: Dict) -> Dict:
        """Normalize theory weights to sum to 1 within categories"""
        normalized = genome.copy()

        # Normalize music theory weights
        music_keys = ['circle_of_fifths_weight', 'jazz_extensions_weight']
        music_total = sum(normalized[k] for k in music_keys)
        if music_total > 0:
            for k in music_keys:
                normalized[k] /= music_total

        # Normalize editing weights
        editing_keys = ['continuity_editing_weight', 'montage_editing_weight']
        editing_total = sum(normalized[k] for k in editing_keys)
        if editing_total > 0:
            for k in editing_keys:
                normalized[k] /= editing_total

        # Normalize physics weights
        physics_keys = ['motion_smoothness_weight', 'chaos_weight']
        physics_total = sum(normalized[k] for k in physics_keys)
        if physics_total > 0:
            for k in physics_keys:
                normalized[k] /= physics_total

        return normalized

    def _update_vfe_priors(self, fingerprints: List[Dict]):
        """
        Bayesian update of VFE priors using new observations.
        """
        # Update Gaussian priors (motion smoothness)
        motion_observations = [f['physics_score'] for f in fingerprints if 'physics_score' in f]

        if motion_observations:
            for obs in motion_observations:
                # Bayesian update
                current_mu = self.kb.state['vfe_priors_gaussian']['motion_smoothness']['mu']
                current_sigma = self.kb.state['vfe_priors_gaussian']['motion_smoothness']['sigma']
                observation_noise = 0.1

                # Posterior variance
                posterior_sigma_sq = 1.0 / (1.0 / current_sigma**2 + 1.0 / observation_noise**2)
                posterior_sigma = np.sqrt(posterior_sigma_sq)

                # Posterior mean
                posterior_mu = posterior_sigma_sq * (
                    current_mu / current_sigma**2 + obs / observation_noise**2
                )

                # Update in place
                self.kb.state['vfe_priors_gaussian']['motion_smoothness']['mu'] = float(posterior_mu)
                self.kb.state['vfe_priors_gaussian']['motion_smoothness']['sigma'] = float(posterior_sigma)

        # Update categorical priors (music theory types)
        theory_counts = {'circle_of_fifths': 0, 'jazz': 0, 'experimental': 0}

        for fingerprint in fingerprints:
            if 'harmonic_score' in fingerprint:
                score = fingerprint['harmonic_score']
                if score > 0.8:
                    theory_counts['circle_of_fifths'] += 1
                elif score > 0.6:
                    theory_counts['jazz'] += 1
                else:
                    theory_counts['experimental'] += 1

        # Bayesian update with Dirichlet prior
        alpha = 1.0  # Pseudo-count strength
        current_priors = self.kb.state['vfe_priors_categorical']['music_theory_type']

        total_counts = sum(theory_counts.values())
        if total_counts > 0:
            for theory_type in theory_counts.keys():
                prior_prob = current_priors[theory_type]
                count = theory_counts[theory_type]

                posterior_prob = (prior_prob * alpha + count) / (alpha + total_counts)
                current_priors[theory_type] = float(posterior_prob)

            # Renormalize
            total = sum(current_priors.values())
            for key in current_priors.keys():
                current_priors[key] /= total

    def _generate_report(self, evolution_history: List[Dict]) -> str:
        """Generate human-readable training report"""
        initial_fitness = evolution_history[0]['fitness']
        final_fitness = evolution_history[-1]['fitness']
        improvement = final_fitness - initial_fitness
        improvement_pct = (improvement / abs(initial_fitness)) * 100 if initial_fitness != 0 else 0

        best_genome = evolution_history[-1]['genome']

        report = f"""
╔═══════════════════════════════════════════════════════════╗
║           INCREMENTAL TRAINING REPORT                     ║
╚═══════════════════════════════════════════════════════════╝

Session Statistics:
───────────────────
  Videos in Batch:      {len(evolution_history)}
  Total Videos Trained: {self.kb.state['total_videos']}
  Training Sessions:    {self.kb.state['training_sessions']}
  Generations:          {self.generations}

Fitness Evolution:
──────────────────
  Initial Fitness: {initial_fitness:.4f}
  Final Fitness:   {final_fitness:.4f}
  Improvement:     {improvement:+.4f} ({improvement_pct:+.1f}%)

Best Theory Weights:
────────────────────
  Circle of Fifths: {best_genome['circle_of_fifths_weight']:.3f}
  Jazz Extensions:  {best_genome['jazz_extensions_weight']:.3f}
  Continuity Edit:  {best_genome['continuity_editing_weight']:.3f}
  Montage Edit:     {best_genome['montage_editing_weight']:.3f}
  Smooth Motion:    {best_genome['motion_smoothness_weight']:.3f}
  Chaos:            {best_genome['chaos_weight']:.3f}

VFE Priors (Gaussian):
──────────────────────
  Motion Smoothness:
    μ = {self.kb.state['vfe_priors_gaussian']['motion_smoothness']['mu']:.4f}
    σ = {self.kb.state['vfe_priors_gaussian']['motion_smoothness']['sigma']:.4f}

VFE Priors (Categorical):
─────────────────────────
  Music Theory Type:
    Circle of Fifths: {self.kb.state['vfe_priors_categorical']['music_theory_type']['circle_of_fifths']:.3f}
    Jazz:             {self.kb.state['vfe_priors_categorical']['music_theory_type']['jazz']:.3f}
    Experimental:     {self.kb.state['vfe_priors_categorical']['music_theory_type']['experimental']:.3f}

  Editing Style:
    Continuity:   {self.kb.state['vfe_priors_categorical']['editing_style']['continuity']:.3f}
    Montage:      {self.kb.state['vfe_priors_categorical']['editing_style']['montage']:.3f}
    Experimental: {self.kb.state['vfe_priors_categorical']['editing_style']['experimental']:.3f}

═══════════════════════════════════════════════════════════

Next Steps:
  1. Add more videos to continue refining
  2. Visualize evolution progress
  3. Use learned weights for video generation
  4. Save knowledge base for future sessions

        """.strip()

        return report
