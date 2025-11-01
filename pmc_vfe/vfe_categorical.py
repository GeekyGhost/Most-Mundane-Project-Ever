"""
Variational Free Energy for Categorical distributions
"""

import numpy as np
from typing import Optional


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Safe logarithm to avoid log(0)"""
    return np.log(np.clip(x, eps, None))


def normalize(p: np.ndarray) -> np.ndarray:
    """Normalize distribution to sum to 1"""
    p = np.array(p)
    total = np.sum(p)
    if total > 0:
        return p / total
    else:
        return np.ones_like(p) / len(p)


def vfe_categorical(
    q_h: np.ndarray,
    p_h: np.ndarray,
    p_y_given_h: Optional[np.ndarray] = None,
    y: Optional[int] = None
) -> float:
    """
    Compute Variational Free Energy for categorical distributions.

    VFE = KL(q(h) || p(h)) + E_q[-log p(y | h)]

    Where:
    - q(h) is the approximate posterior over hidden states
    - p(h) is the prior over hidden states
    - p(y | h) is the likelihood matrix (optional)
    - y is the observed outcome index (optional)

    Args:
        q_h: Posterior distribution over states (N,)
        p_h: Prior distribution over states (N,)
        p_y_given_h: Likelihood matrix (M, N) where M is number of observations, N is number of states (optional)
        y: Observed outcome index (0 to M-1) (optional)

    Returns:
        vfe: Scalar VFE value
    """
    # Normalize distributions
    q = normalize(q_h)
    p = normalize(p_h)

    # Compute KL divergence: KL(q || p) = sum_h q(h) * log(q(h) / p(h))
    kl = np.sum(q * (safe_log(q) - safe_log(p)))

    # Compute expected negative log-likelihood (if observation provided)
    if p_y_given_h is not None and y is not None:
        # p_y_given_h[y, :] is p(y | h) for each h
        # E_q[-log p(y | h)] = -sum_h q(h) * log p(y | h)

        likelihood_y = p_y_given_h[y, :]  # Extract likelihood for observation y
        nll = -np.sum(q * safe_log(likelihood_y))
    else:
        nll = 0.0

    vfe = kl + nll

    return float(vfe)
