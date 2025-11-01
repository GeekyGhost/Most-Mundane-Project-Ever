"""
VFE computation nodes
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pmc_vfe.vfe_gaussian import vfe_gaussian
from pmc_vfe.vfe_categorical import vfe_categorical


class QVT_ComputeVFE_Gaussian:
    """Compute VFE for Gaussian distributions"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mu_q": ("ARRAY",),
                "Sigma_q": ("ARRAY",),
                "mu_p": ("ARRAY",),
                "Sigma_p": ("ARRAY",),
            },
            "optional": {
                "observation": ("ARRAY",),
                "observation_noise": ("ARRAY",),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("vfe", "kl_term", "nll_term")
    FUNCTION = "compute_vfe"
    CATEGORY = "QuantumVideoTrainer/VFE"

    def compute_vfe(self, mu_q, Sigma_q, mu_p, Sigma_p, observation=None, observation_noise=None):
        # Convert to numpy if needed
        mu_q = np.array(mu_q)
        Sigma_q = np.array(Sigma_q)
        mu_p = np.array(mu_p)
        Sigma_p = np.array(Sigma_p)

        if observation is not None:
            observation = np.array(observation)
            observation_noise = np.array(observation_noise) if observation_noise is not None else None

        # Compute VFE
        vfe_total = vfe_gaussian(
            mu_q=mu_q,
            Sigma_q=Sigma_q,
            mu_p=mu_p,
            Sigma_p=Sigma_p,
            y=observation,
            Sigma_y=observation_noise
        )

        # Compute components separately for interpretability
        inv_Sp = np.linalg.inv(Sigma_p)
        tr_term = np.trace(inv_Sp @ Sigma_q)
        diff = (mu_p - mu_q).reshape(-1, 1)
        quad_term = float(diff.T @ inv_Sp @ diff)
        d = mu_q.shape[0]
        logdet = np.log(np.linalg.det(Sigma_p) + 1e-12) - np.log(np.linalg.det(Sigma_q) + 1e-12)
        kl_term = 0.5 * (tr_term + quad_term - d + logdet)

        # NLL term
        nll_term = 0.0
        if observation is not None and observation_noise is not None:
            inv_Sy = np.linalg.inv(observation_noise)
            diffy = (observation - mu_q).reshape(-1, 1)
            nll_term = 0.5 * (float(diffy.T @ inv_Sy @ diffy) + np.trace(inv_Sy @ Sigma_q))

        return (float(vfe_total), float(kl_term), float(nll_term))


class QVT_ComputeVFE_Categorical:
    """Compute VFE for categorical distributions"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "q_h": ("ARRAY",),
                "p_h": ("ARRAY",),
            },
            "optional": {
                "p_y_given_h": ("ARRAY",),
                "observation": ("INT",),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("vfe", "kl_term", "nll_term")
    FUNCTION = "compute_vfe"
    CATEGORY = "QuantumVideoTrainer/VFE"

    def compute_vfe(self, q_h, p_h, p_y_given_h=None, observation=None):
        from pmc_vfe.vfe_categorical import safe_log, normalize

        q_h = np.array(q_h)
        p_h = np.array(p_h)

        # Compute VFE
        vfe_total = vfe_categorical(
            q_h=q_h,
            p_h=p_h,
            p_y_given_h=p_y_given_h,
            y=observation
        )

        # Compute KL component
        q = normalize(q_h)
        p = normalize(p_h)
        kl_term = np.sum(q * (safe_log(q) - safe_log(p)))

        # Compute NLL component
        nll_term = vfe_total - kl_term

        return (float(vfe_total), float(kl_term), float(nll_term))
