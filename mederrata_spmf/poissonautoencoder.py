from mederrata_spmf.dense import DenseHorseshoe

from itertools import cycle
import functools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow_probability import distributions as tfd

from mederrata_spmf.model import BayesianModel
from mederrata_spmf.distributions import SqrtInverseGamma
from mederrata_spmf.util import (
    build_trainable_InverseGamma_dist, build_trainable_normal_dist,
    run_chain, clip_gradients, build_surrogate_posterior,
    fit_surrogate_posterior)

tfb = tfp.bijectors


class PoissonAutoencoder(DenseHorseshoe):
    var_list = []

    def __init__(
            self, data, data_transform_fn=None, latent_dim=None,
            scale_columns=True, column_norms=None, encoder_layers=1,
            decoder_layers=1, activation_function=tf.nn.softplus,
            strategy=None, dtype=tf.float64, **kwargs):
        """Instantiate unconstrained dense Poisson autoencoder

        Args:
            data ([type]): [description]
            data_transform_fn ([type], optional): [description]. Defaults to None.
            latent_dim ([type], optional): [description]. Defaults to None.
            scale_columns (bool, optional): [description]. Defaults to True.
            column_norms ([type], optional): [description]. Defaults to None.
            strategy ([type], optional): [description]. Defaults to None.
            dtype ([type], optional): [description]. Defaults to tf.float64.
        """
        super(DenseHorseshoe, self).__init__(
            data, data_transform_fn, strategy=strategy, dtype=dtype)
        self.dtype = dtype
        record = next(iter(data))
        indices = record['indices']
        data = record['data']
        self.column_norm_factor = 1.

        if scale_columns:
            if column_norms is not None:
                self.column_norm_factor = tf.cast(
                    column_norms, self.dtype)
            else:
                self.column_norm_factor = tf.reduce_mean(
                    tf.cast(data, self.dtype), axis=0, keepdims=True)

        if 'normalization' in record.keys():
            norm = record['normalization']
        data = tf.cast(data, self.dtype)
        self.feature_dim = data.shape[-1]
        self.latent_dim = self.feature_dim if (
            latent_dim) is None else latent_dim

        self.neural_network_model = DenseHorseshoe(
            self.feature_dim,
            [self.feature_dim]*encoder_layers + [self.latent_dim] +
            [self.feature_dim]*decoder_layers + [self.feature_dim],
            dtype=self.dtype)

        var_list = self.neural_network_model.var_list
        self.var_list = var_list
        # rewrite the log_likelihood signature with the variable names
        # function_string = f"lambda self, data, {', '.join(var_list)}: self.log_likelihood(data, {', '.join([str(v) + '=' + str(v) for v in var_list])})"
        # self.log_likelihood = eval(function_string, globals(), self.__dict__)
        self.joint_prior = self.neural_network_model.joint_prior
        self.surrogate_distribution = build_surrogate_posterior(
            self.joint_prior, self.neural_network_model.bijectors,
            dtype=self.dtype,
            strategy=self.strategy)
        self.surrogate_vars = self.surrogate_distribution.variables

        self.var_list = list(self.surrogate_distribution.variables)

        self.set_calibration_expectations()

    def log_likelihood(self, data, **params):
        neural_networks = self.neural_network_model.assemble_networks(params)
        rates = tf.math.exp(
            neural_networks(
                tf.cast(
                    data['data'],
                    self.neural_network_model.dtype)
                / tf.cast(
                    self.column_norm_factor,
                    self.neural_network_model.dtype)
            )
        )
        rates = tf.cast(rates, self.dtype)
        rates *= self.column_norm_factor
        rv_poisson = tfd.Poisson(rate=rates)
        log_lik = rv_poisson.log_prob(
            tf.cast(data['data'], self.dtype)[tf.newaxis, ...])
        log_lik = tf.reduce_sum(log_lik, axis=-1)
        log_lik = tf.reduce_sum(log_lik, axis=-1)
        return log_lik

    def unormalized_log_prob_parts(self, data=None, **params):
        if data is None:
            #  use self.data, taking the next batch
            try:
                data = next(self.dataset_cycler)
            except tf.errors.OutOfRangeError:
                self.dataset_iterator = cycle(iter(self.data))
                data = next(self.dataset_iterator)

        prior_parts = self.neural_network_model.joint_prior.log_prob_parts(
            params)
        log_likelihood = self.log_likelihood(data, **params)
        prior_parts['x'] = log_likelihood
        return prior_parts

    def unormalized_log_prob(self, data=None, **params):
        prob_parts = self.unormalized_log_prob_parts(
            data, **params)
        value = tf.add_n(
            list(prob_parts.values()))
        return value
