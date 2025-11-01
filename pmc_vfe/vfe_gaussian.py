"""
Variational Free Energy for Gaussian distributions
"""

import numpy as np
from typing import Optional


def vfe_gaussian(
    mu_q: np.ndarray,
    Sigma_q: np.ndarray,
    mu_p: np.ndarray,
    Sigma_p: np.ndarray,
    y: Optional[np.ndarray] = None,
    Sigma_y: Optional[np.ndarray] = None
) -> float:
    """
    Compute Variational Free Energy for Gaussian distributions.

    VFE = KL(q || p) + E_q[-log p(y | h)]

    Where:
    - q(h) = N(mu_q, Sigma_q) is the approximate posterior
    - p(h) = N(mu_p, Sigma_p) is the prior
    - p(y | h) = N(h, Sigma_y) is the likelihood (optional)

    Args:
        mu_q: Posterior mean (d,)
        Sigma_q: Posterior covariance (d, d)
        mu_p: Prior mean (d,)
        Sigma_p: Prior covariance (d, d)
        y: Observation (optional) (d,)
        Sigma_y: Observation noise covariance (optional) (d, d)

    Returns:
        vfe: Scalar VFE value
    """
    # Ensure inputs are numpy arrays
    mu_q = np.array(mu_q).flatten()
    mu_p = np.array(mu_p).flatten()
    Sigma_q = np.array(Sigma_q)
    Sigma_p = np.array(Sigma_p)

    d = mu_q.shape[0]

    # Compute KL divergence: KL(q || p)
    # KL = 0.5 * [tr(Sigma_p^-1 @ Sigma_q) + (mu_p - mu_q)^T @ Sigma_p^-1 @ (mu_p - mu_q) - d + log(det(Sigma_p)/det(Sigma_q))]

    inv_Sigma_p = np.linalg.inv(Sigma_p + np.eye(d) * 1e-8)  # Add small regularization

    # Trace term
    tr_term = np.trace(inv_Sigma_p @ Sigma_q)

    # Quadratic term
    diff = (mu_p - mu_q).reshape(-1, 1)
    quad_term = float(diff.T @ inv_Sigma_p @ diff)

    # Log determinant term
    logdet_p = np.log(np.linalg.det(Sigma_p) + 1e-12)
    logdet_q = np.log(np.linalg.det(Sigma_q) + 1e-12)

    kl = 0.5 * (tr_term + quad_term - d + logdet_p - logdet_q)

    # Compute expected negative log-likelihood (if observation provided)
    if y is not None and Sigma_y is not None:
        y = np.array(y).flatten()
        Sigma_y = np.array(Sigma_y)

        inv_Sigma_y = np.linalg.inv(Sigma_y + np.eye(d) * 1e-8)

        # E_q[-log p(y | h)] = 0.5 * [tr(Sigma_y^-1 @ Sigma_q) + (y - mu_q)^T @ Sigma_y^-1 @ (y - mu_q) + d*log(2*pi) + log(det(Sigma_y))]

        tr_term_y = np.trace(inv_Sigma_y @ Sigma_q)

        diff_y = (y - mu_q).reshape(-1, 1)
        quad_term_y = float(diff_y.T @ inv_Sigma_y @ diff_y)

        logdet_y = np.log(np.linalg.det(Sigma_y) + 1e-12)

        nll = 0.5 * (tr_term_y + quad_term_y + d * np.log(2 * np.pi) + logdet_y)
    else:
        nll = 0.0

    vfe = kl + nll

    return float(vfe)
