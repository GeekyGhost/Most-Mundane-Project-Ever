"""
Expected Free Energy (EFE) for planning and strategy selection
"""

import numpy as np
from .vfe_categorical import safe_log, normalize


def expected_free_energy(
    q_h_future: np.ndarray,
    p_h_future: np.ndarray,
    p_y_given_h: np.ndarray,
    p_y_preferred: np.ndarray
) -> float:
    """
    Compute Expected Free Energy for a future state.

    EFE = E_q[KL(p(y|h) || p_preferred(y))] - E_q[H[p(y|h)]]

    This balances:
    - Pragmatic value: Achieving preferred outcomes (negative utility)
    - Epistemic value: Reducing uncertainty (information gain)

    Args:
        q_h_future: Expected posterior over future hidden states (N,)
        p_h_future: Prior over future hidden states (N,)
        p_y_given_h: Likelihood matrix (M, N)
        p_y_preferred: Preferred distribution over outcomes (M,)

    Returns:
        efe: Scalar EFE value (lower is better)
    """
    q = normalize(q_h_future)
    p_pref = normalize(p_y_preferred)

    num_outcomes = p_y_given_h.shape[0]

    # Compute expected distribution over outcomes: p(y) = sum_h q(h) * p(y|h)
    p_y = np.zeros(num_outcomes)
    for h_idx in range(len(q)):
        p_y += q[h_idx] * p_y_given_h[:, h_idx]

    p_y = normalize(p_y)

    # Pragmatic value: KL(p(y) || p_preferred(y))
    # This is the "cost" of not achieving preferred outcomes
    pragmatic_value = np.sum(p_y * (safe_log(p_y) - safe_log(p_pref)))

    # Epistemic value: -H[p(y)] = sum_y p(y) * log p(y)
    # Negative entropy - we want to reduce uncertainty
    # Higher entropy = more uncertainty = higher EFE
    epistemic_value = np.sum(p_y * safe_log(p_y))  # Negative of entropy

    # EFE combines both
    # Note: We want LOW EFE, so high utility = low EFE
    efe = pragmatic_value - epistemic_value

    return float(efe)


def expected_free_energy_policy(
    policy_states: list,
    q_h_current: np.ndarray,
    transition_matrix: np.ndarray,
    p_y_given_h: np.ndarray,
    p_y_preferred: np.ndarray
) -> float:
    """
    Compute EFE for an entire policy (sequence of actions).

    Args:
        policy_states: List of state indices representing the policy
        q_h_current: Current posterior over states (N,)
        transition_matrix: State transition probabilities (N, N)
        p_y_given_h: Likelihood matrix (M, N)
        p_y_preferred: Preferred distribution over outcomes (M,)

    Returns:
        efe: Total EFE for the policy
    """
    total_efe = 0.0
    q_h = normalize(q_h_current)

    for state in policy_states:
        # Predict next state distribution
        q_h_next = transition_matrix @ q_h

        # Compute EFE for this step
        efe_step = expected_free_energy(
            q_h_future=q_h_next,
            p_h_future=q_h_next,  # Simplified - could use separate prior
            p_y_given_h=p_y_given_h,
            p_y_preferred=p_y_preferred
        )

        total_efe += efe_step

        # Update state for next iteration
        q_h = q_h_next

    return total_efe
