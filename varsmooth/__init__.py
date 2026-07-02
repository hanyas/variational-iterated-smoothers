"""varsmooth: variational iterated Gaussian smoothing in state-space models.

Implements the proximal / entropic trust-region smoothers of "Recursive
Entropic Variational Inference for Nonlinear State-Space Models"
(arXiv:2511.15409): iterated
KL-constrained updates over a Gauss-Markov posterior, with the model expanded
through generalized statistical linear regression or Fourier-Hermite moment
matching. See varsmooth/objects.py for the shape and notation conventions.
"""

__version__ = "0.1.0"

from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import AffineGaussian
from varsmooth.objects import ConditionalMomentsModel
from varsmooth.objects import GaussMarkov
from varsmooth.objects import Gaussian
from varsmooth.smoothers import forward_markov_smoother
from varsmooth.smoothers import hybrid_markov_smoother
from varsmooth.smoothers import iterated_forward_markov_smoother
from varsmooth.smoothers import iterated_hybrid_markov_smoother
from varsmooth.smoothers import iterated_reverse_markov_smoother
from varsmooth.smoothers import reverse_markov_smoother

__all__ = [
    "__version__",
    "Gaussian",
    "AffineGaussian",
    "GaussMarkov",
    "AdditiveGaussianModel",
    "ConditionalMomentsModel",
    "forward_markov_smoother",
    "iterated_forward_markov_smoother",
    "reverse_markov_smoother",
    "iterated_reverse_markov_smoother",
    "hybrid_markov_smoother",
    "iterated_hybrid_markov_smoother",
]
