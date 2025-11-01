"""
Strategy scorer combining VFE and EFE for decision making
"""

import numpy as np
from typing import List, Dict, NamedTuple
from .vfe_gaussian import vfe_gaussian
from .vfe_categorical import vfe_categorical
from .efe import expected_free_energy


class StrategyInput(NamedTuple):
    """Input data for strategy scoring"""
    strategy_id: str
    description: str
    theory_weights: Dict[str, float]
    predicted_outcomes: Dict[str, float]


def score_strategy(
    strategy: StrategyInput,
    priors_gaussian: Dict,
    priors_categorical: Dict,
    observations: Dict,
    beta: float = 0.7
) -> Dict:
    """
    Score a single strategy using VFE + EFE.

    Args:
        strategy: Strategy to evaluate
        priors_gaussian: Gaussian priors from knowledge base
        priors_categorical: Categorical priors from knowledge base
        observations: Observed data
        beta: Balance between VFE (exploitation) and EFE (exploration)
              beta=1.0: Pure VFE (exploit known patterns)
              beta=0.0: Pure EFE (explore for information)

    Returns:
        score_dict: Contains VFE, EFE, combined score, and fitness
    """
    # Compute VFE (how well does strategy match current beliefs)
    vfe_total = 0.0

    # Example: Compute VFE for continuous features
    if 'motion_smoothness' in strategy.predicted_outcomes:
        predicted_mu = np.array([strategy.predicted_outcomes['motion_smoothness']])
        predicted_sigma = np.array([[0.1]])  # Uncertainty in prediction

        prior_mu = np.array([priors_gaussian['motion_smoothness']['mu']])
        prior_sigma = np.array([[priors_gaussian['motion_smoothness']['sigma']**2]])

        vfe_motion = vfe_gaussian(
            mu_q=predicted_mu,
            Sigma_q=predicted_sigma,
            mu_p=prior_mu,
            Sigma_p=prior_sigma
        )
        vfe_total += vfe_motion

    # Compute EFE (how informative/valuable is this strategy for future)
    # Simplified: Use strategy diversity as proxy for epistemic value
    efe_total = -strategy_diversity_score(strategy)  # Negative because diverse = lower EFE

    # Combine VFE and EFE
    combined_free_energy = beta * vfe_total + (1 - beta) * efe_total

    # Fitness is negative free energy (lower free energy = higher fitness)
    fitness = -combined_free_energy

    return {
        'strategy_id': strategy.strategy_id,
        'vfe': vfe_total,
        'efe': efe_total,
        'combined_free_energy': combined_free_energy,
        'fitness': fitness,
        'beta': beta
    }


def strategy_diversity_score(strategy: StrategyInput) -> float:
    """
    Compute diversity score for a strategy.
    Higher diversity = more exploratory = more informative.
    """
    weights = list(strategy.theory_weights.values())

    # Compute entropy of weight distribution
    weights_arr = np.array(weights)
    weights_arr = weights_arr / (np.sum(weights_arr) + 1e-12)

    entropy = -np.sum(weights_arr * np.log(weights_arr + 1e-12))

    # Normalize by max entropy
    max_entropy = np.log(len(weights))
    diversity = entropy / max_entropy if max_entropy > 0 else 0.0

    return diversity


def score_batch(
    strategies: List[StrategyInput],
    priors_gaussian: Dict,
    priors_categorical: Dict,
    observations: Dict,
    beta: float = 0.7
) -> List[Dict]:
    """
    Score a batch of strategies.

    Args:
        strategies: List of strategies to evaluate
        priors_gaussian: Gaussian priors
        priors_categorical: Categorical priors
        observations: Observed data
        beta: VFE/EFE balance

    Returns:
        scores: List of score dicts, sorted by fitness (highest first)
    """
    scores = []

    for strategy in strategies:
        score = score_strategy(
            strategy=strategy,
            priors_gaussian=priors_gaussian,
            priors_categorical=priors_categorical,
            observations=observations,
            beta=beta
        )
        scores.append(score)

    # Sort by fitness (highest first)
    scores.sort(key=lambda x: x['fitness'], reverse=True)

    return scores
