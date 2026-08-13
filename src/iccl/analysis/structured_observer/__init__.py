"""Generator-aware Gaussian-process observers for frozen ICCL sequences.

These observers integrate over latent task schedules while matching the
HyperTeacher module prior. They are approximate reference algorithms, not
certified bounds for the finite-width teacher or trained GDN.
"""

from iccl.analysis.structured_observer.gp import BatchedOnlineGP
from iccl.analysis.structured_observer.kernel import FeatureBank, sample_feature_bank

__all__ = ["BatchedOnlineGP", "FeatureBank", "sample_feature_bank"]
