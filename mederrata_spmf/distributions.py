import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import (
    distribution, kullback_leibler,
    TransformedDistribution, JointDistributionNamed)

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util

tfd = tfp.distributions
tfb = tfp.bijectors

convert_nonref_to_tensor = tensor_util.convert_nonref_to_tensor


__all__ = [
    'LogHalfCauchy',
    'SqrtCauchy',
    'SqrtInverseGamma',
    'FactorizedDistributionMoments',
    'SoftplusHorseshoe',
    'AbsHorseshoe'
]


class SqrtCauchy(TransformedDistribution):
    def __init__(self, loc, scale, validate_args=False,
                 allow_nan_stats=True, name="SqrtCauchy"):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(SqrtCauchy, self).__init__(
                distribution=tfd.HalfCauchy(loc=loc, scale=scale),
                bijector=tfb.Invert(tfb.Square()),
                validate_args=validate_args,
                parameters=parameters,
                name=name)

    @classmethod
    def _params_event_ndims(cls):
        return dict(loc=0, scale=0)

    @property
    def loc(self):
        """Distribution parameter for the pre-transformed mean."""
        return self.distribution.loc

    @property
    def scale(self):
        """Distribution parameter for the
            pre-transformed standard deviation."""
        return self.distribution.scale


class SoftplusHorseshoe(TransformedDistribution):
    def __init__(self, scale, validate_args=False,
                 allow_nan_stats=True, name="SoftplusHorseshoe"):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(SoftplusHorseshoe, self).__init__(
                distribution=tfd.Horseshoe(scale=scale),
                bijector=tfb.Softplus(),
                validate_args=validate_args,
                parameters=parameters,
                name=name)

    @classmethod
    def _params_event_ndims(cls):
        return dict(scale=0)

    @property
    def scale(self):
        """Distribution parameter for the
            pre-transformed standard deviation."""
        return self.distribution.scale


class AbsHorseshoe(TransformedDistribution):
    def __init__(self, scale, validate_args=False,
                 allow_nan_stats=True, name="AbsHorseshoe"):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(AbsHorseshoe, self).__init__(
                distribution=tfd.Horseshoe(scale=scale),
                bijector=tfb.AbsoluteValue(),
                validate_args=validate_args,
                parameters=parameters,
                name=name)

    @classmethod
    def _params_event_ndims(cls):
        return dict(scale=0)

    @property
    def scale(self):
        """Distribution parameter for the
            pre-transformed standard deviation."""
        return self.distribution.scale


class SqrtInverseGamma(TransformedDistribution):
    def __init__(self, concentration, scale, validate_args=False,
                 allow_nan_stats=True, name="SqrtInverseGamma"):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(SqrtInverseGamma, self).__init__(
                distribution=tfd.InverseGamma(
                    concentration=concentration, scale=scale),
                bijector=tfb.Invert(tfb.Square()),
                validate_args=validate_args,
                parameters=parameters,
                name=name)

    @classmethod
    def _params_event_ndims(cls):
        return dict(loc=0, scale=0)

    @property
    def loc(self):
        """Distribution parameter for the pre-transformed mean."""
        return self.distribution.loc

    @property
    def scale(self):
        """Distribution parameter for the
            pre-transformed standard deviation."""
        return self.distribution.scale


class LogHalfCauchy(TransformedDistribution):
    """Exponent of RV follows a HalfCauchy distribution

    log(x) ~ HalfCauchy

    Arguments:
        TransformedDistribution {[type]} -- [description]

    Returns:
        tfp.distribution -- [description]
    """

    def __init__(self, loc, scale, validate_args=False,
                 allow_nan_stats=True, name="LogHalfCauchy"):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(LogHalfCauchy, self).__init__(
                distribution=tfd.HalfCauchy(loc=loc, scale=scale),
                bijector=tfb.Invert(tfb.Exp()),
                validate_args=validate_args,
                parameters=parameters,
                name=name)

    @classmethod
    def _params_event_ndims(cls):
        return dict(loc=0, scale=0)

    @property
    def loc(self):
        """Distribution parameter for the pre-transformed mean."""
        return self.distribution.loc

    @property
    def scale(self):
        """Distribution parameter for the pre-transformed
            standard deviation."""
        return self.distribution.scale


"""
def FactorizedDistributionMoments(distribution, exclude=[]):
    means = {}
    variances = {}
    for k, v in distribution.model.items():
        if k in exclude:
            continue
        if callable(v):
            raise AttributeError("Need factorized nameddistribution object")
        else:
            test_distribution = v
        mean = test_distribution.mean()
        variance = test_distribution.variance()
        means[k] = mean
        variances[k] = variance
    return means, variances
"""


def FactorizedDistributionMoments(distribution, samples=250, exclude=[]):
    """ Compute the mean in a memory-safe manner
    """
    means = {}
    variances = {}
    for k, v in distribution.model.items():
        if k in exclude:
            continue
        if callable(v):
            raise AttributeError("Need factorized nameddistribution object")
        else:
            test_distribution = v
        try:
            mean = test_distribution.mean()
            variance = test_distribution.variance()
        except NotImplementedError:
            sum_1 = test_distribution.sample()
            sum_2 = sum_1**2
            for _ in range(samples-1):
                s = test_distribution.sample()
                sum_1 = sum_1 + s
                sum_2 = sum_2 + s**2
            mean = sum_1/samples
            variance = sum_2/samples - mean**2
        means[k] = mean
        variances[k] = variance
    return means, variances
