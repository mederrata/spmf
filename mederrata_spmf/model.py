import inspect
from itertools import cycle
import dill
import arviz as az

import tensorflow as tf

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow.python.distribute.input_lib import DistributedDataset

from mederrata_spmf.util import (
    clip_gradients, fit_surrogate_posterior)
from mederrata_spmf.distributions import FactorizedDistributionMoments


class BayesianModel(object):
    surrogate_distribution = None
    surrogate_sample = None
    prior_distribution = None
    data = None
    var_list = []
    bijectors = []

    def __init__(self, data=None, data_transform_fn=None,
                 strategy=None, *args, **kwargs):
        """Instatiate Model object based on tensorflow dataset
        Arguments:
            data {[type]} -- [description]
        Keyword Arguments:
            data_transform_fn {[type]} -- [description] (default: {None})
            strategy {[type]} -- [description] (default: {None})
        Raises:
            AttributeError: [description]
        """
        super(BayesianModel, self).__init__()
        if data is not None:
            self.set_data(data, data_transform_fn)

        self.strategy = strategy

    def set_data(self, data, data_transform_fn=None):
        if isinstance(
            data, (np.ndarray, np.generic)) or isinstance(
                data, pd.DataFrame):

            if isinstance(data, pd.DataFrame):
                data = data.to_numpy()
                samples = data.shape[0]
                data = tf.data.Dataset.zip((
                    tf.data.Dataset.from_tensor_slices(
                        data
                    ),
                    tf.data.Dataset.from_tensor_slices(
                        np.arange(samples)
                    ))
                )
                data = data.batch(samples)

        elif not isinstance(data, tf.data.Dataset):
            raise AttributeError("Need numpy/dataframe or tf.dataset")

        self.data = data
        self.data_transform_fn = data_transform_fn

    #  @tf.function
    def calibrate_advi(
            self, num_epochs=100, learning_rate=0.1,
            opt=None, abs_tol=1e-10, rel_tol=1e-8,
            clip_value=5., max_decay_steps=25, lr_decay_factor=0.99,
            check_every=25, set_expectations=True, sample_size=4,
            data=None, data_batches=25, prefetch_batches=2,
            **kwargs):
        """Calibrate using ADVI

        Args:
            num_epochs (int, optional): [description]. Defaults to 100.
            learning_rate (float, optional): [description]. Defaults to 0.1.
            opt ([type], optional): [description]. Defaults to None.
            abs_tol ([type], optional): [description]. Defaults to 1e-10.
            rel_tol ([type], optional): [description]. Defaults to 1e-8.
            clip_value ([type], optional): [description]. Defaults to 5..
            max_decay_steps (int, optional): [description]. Defaults to 25.
            lr_decay_factor (float, optional): [description]. Defaults to 0.99.
            check_every (int, optional): [description]. Defaults to 25.
            set_expectations (bool, optional): [description]. Defaults to True.
            sample_size (int, optional): [description]. Defaults to 4.
            data ([type], optional): [description]. Defaults to None.
            data_batches (int, optional): Ignored if data is already batched. 
                Defaults to 25.
        """
        data = self.data if data is None else data
        # check if data is batched
        batched = False
        root = False
        up = data
        while (not batched) and (not root):
            if hasattr(up, "._batch_size"):
                batch_size = up._batch_size
                batched = True
                break
            if hasattr(up, "._input_dataset"):
                up = up._input_dataset
            else:
                root = True

        if not batched:
            card = tf.data.experimental.cardinality(data)
            batch_size = int(np.floor(card/data_batches))
            data = data.batch(batch_size, drop_remainder=True)
            # data = data.batch

        data = data.prefetch(2)

        def run_approximation(num_epochs):
            losses = fit_surrogate_posterior(
                target_log_prob_fn=self.unormalized_log_prob,
                surrogate_posterior=self.surrogate_distribution,
                num_epochs=num_epochs,
                sample_size=sample_size,
                learning_rate=learning_rate,
                max_decay_steps=max_decay_steps,
                decay_rate=lr_decay_factor,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                clip_value=clip_value,
                check_every=check_every,
                strategy=self.strategy,
                batched_dataset=data
            )
            return(losses)

        losses = run_approximation(num_epochs)
        if set_expectations:
            if (not np.isnan(losses[-1])) and (not np.isinf(losses[-1])):
                self.surrogate_sample = self.surrogate_distribution.sample(100)
                self.set_calibration_expectations()
        return(losses)

    def set_calibration_expectations(self, samples=50, variational=True):
        if variational:
            mean, var = FactorizedDistributionMoments(
                self.surrogate_distribution, samples=samples)
            self.calibrated_expectations = {
                k: tf.Variable(v) for k, v in mean.items()
            }
            self.calibrated_sd = {
                k: tf.Variable(tf.math.sqrt(v)) for k, v in var.items()
            }
        else:
            self.calibrated_expectations = {
                k: tf.Variable(tf.reduce_mean(v, axis=0))
                for k, v in self.surrogate_sample.items()
            }

            self.calibrated_sd = {
                k: tf.Variable(tf.math.reduce_std(v, axis=0))
                for k, v in self.surrogate_sample.items()
            }

    def calibrate_mcmc(self, num_steps=1000, burnin=500,
                       init_state=None, step_size=1e-1, nuts=True,
                       num_leapfrog_steps=10, clip=None):
        """Calibrate using HMC/NUT
        Keyword Arguments:
            num_chains {int} -- [description] (default: {1})
        """

        if init_state is None:
            init_state = self.calibrated_expectations

        step_size = tf.cast(step_size, self.dtype)

        initial_list = [init_state[v] for v in self.var_list]
        bijectors = [self.bijectors[k] for k in self.var_list]

        samples, sampler_stat = run_chain(
            init_state=initial_list,
            step_size=step_size,
            target_log_prob_fn=(
                self.unormalized_log_prob_list if clip is None
                else clip_gradients(self.unormalized_log_prob_list, clip)),
            unconstraining_bijectors=bijectors,
            num_steps=num_steps,
            burnin=burnin,
            num_leapfrog_steps=num_leapfrog_steps,
            nuts=nuts
        )
        self.surrogate_sample = {
            k: sample for k, sample in zip(self.var_list, samples)
        }
        self.set_calibration_expectations()

        return samples, sampler_stat

    def log_likelihood(self, *args, **kwargs):
        pass

    def psis_loo(self, data=None, params=None, num_samples=100, num_splits=20):
        data = self.data if data is None else data
        likelihood_vars = inspect.getfullargspec(
            self.log_likelihood).args[1:]

        # split param samples
        params = self.surrogate_sample if params is None else params
        if 'data' in likelihood_vars:
            likelihood_vars.remove('data')
        params = self.surrogate_distribution.sample(num_samples) if (
            params is None) else params
        if len(likelihood_vars) == 0:
            likelihood_vars = params.keys()
        if 'data' in likelihood_vars:
            likelihood_vars.remove('data')
        splits = [
            tf.split(
                params[v],
                num_splits) for v in likelihood_vars]

        # reshape the splits
        splits = [
            {
                k: v for k, v in zip(
                    likelihood_vars, split)} for split in zip(*splits)]

    def waic(
            self, data=None, params=None, num_samples=100,
            num_splits=20, data_batches=25):
        data = self.data if data is None else data
        # check if data is batched
        batched = False
        root = False
        up = data
        while (not batched) and (not root):
            if hasattr(up, "._batch_size"):
                batch_size = up._batch_size
                batched = True
                break
            if hasattr(up, "._input_dataset"):
                up = up._input_dataset
            else:
                root = True

        if not batched:
            card = tf.data.experimental.cardinality(data)
            batch_size = int(np.floor(card/data_batches))
            data = data.batch(batch_size, drop_remainder=True)
            # data = data.batch

        data = data.prefetch(2)
        
        likelihood_vars = inspect.getfullargspec(
            self.log_likelihood).args[1:]

        # split param samples
        params = self.surrogate_sample if params is None else params
        if 'data' in likelihood_vars:
            likelihood_vars.remove('data')
        params = self.surrogate_distribution.sample(num_samples) if (
            params is None) else params
        if len(likelihood_vars) == 0:
            likelihood_vars = params.keys()
        if 'data' in likelihood_vars:
            likelihood_vars.remove('data')
        splits = [
            tf.split(
                params[v],
                num_splits) for v in likelihood_vars]

        # reshape the splits
        splits = [
            {
                k: v for k, v in zip(
                    likelihood_vars, split)} for split in zip(*splits)]
        """
        if not isinstance(data, BatchDataset):
            N = list(data.shape)[0]
            data = tf.data.Dataset.from_tensor_slices(data)
            data = data.batch(int(N/num_splits))
        """

        sum_S_log_likelihoods = 0.
        sum_N_log_likelihoods = 0.
        sum_S_sq_log_likes = 0.
        sum_N_sq_log_likes = 0.
        sum_S_likelihoods = 0.
        sum_N_likelihoods = 0.
        N = 0
        S = tf.shape(tf.nest.flatten(params)[0])[0]

        for batch in data:
            # This should have shape S x N, where S is the number of param
            # samples and N is the batch size
            batch_log_likelihoods = [
                self.log_likelihood(
                    **this_split, data=batch
                )
                for this_split in splits
            ]
            batch_log_likelihoods = tf.concat(
                batch_log_likelihoods, axis=0
            )
            finite_part = tf.where(
                tf.math.is_finite(batch_log_likelihoods),
                batch_log_likelihoods,
                tf.zeros_like(batch_log_likelihoods))
            min_val = tf.math.reduce_min(finite_part)
            #  batch_ll = tf.clip_by_value(batch_ll, min_val-1000, 0.)
            batch_log_likelihoods = tf.where(
                tf.math.is_finite(batch_log_likelihoods),
                batch_log_likelihoods,
                tf.ones_like(batch_log_likelihoods)*min_val - 1000.
            )
            batch_log_likelihoods = tf.cast(batch_log_likelihoods, tf.float64)
            sum_S_log_likelihoods = sum_S_log_likelihoods + tf.reduce_sum(
                batch_log_likelihoods, axis=0)
            sum_N_log_likelihoods = sum_N_log_likelihoods + tf.reduce_sum(
                batch_log_likelihoods, axis=1)
            sum_S_sq_log_likes = sum_S_sq_log_likes + tf.reduce_sum(
                batch_log_likelihoods**2, axis=0)
            sum_N_sq_log_likes = sum_N_sq_log_likes + tf.reduce_sum(
                batch_log_likelihoods**2, axis=1)
            sum_S_likelihoods = sum_S_likelihoods + tf.reduce_sum(
                tf.math.exp(batch_log_likelihoods), axis=0)
            sum_N_likelihoods = sum_N_likelihoods + tf.reduce_sum(
                tf.math.exp(batch_log_likelihoods), axis=1)
            N += tf.shape(tf.nest.flatten(batch)[0])[0]

        # These stats are over the samples
        mean_S_likelihood = (
            sum_S_likelihoods/tf.cast(N, sum_S_likelihoods.dtype))
        lppdi = tf.math.log(
            tf.cast(mean_S_likelihood, tf.float64)
        )
        lppd = tf.reduce_sum(lppdi).numpy()

        mean_sq_S_log_likelihood = (
            sum_S_sq_log_likes/tf.cast(N, sum_S_sq_log_likes.dtype))
        mean__S_log_likelihood = (
            sum_S_log_likelihoods/tf.cast(N, sum_S_log_likelihoods.dtype))
        pwaici = mean_sq_S_log_likelihood - mean__S_log_likelihood**2

        pwaic = tf.reduce_sum(pwaici).numpy()

        elpdi = lppdi-pwaici

        waic = 2*(-lppd + pwaic)

        se = 2.0*tf.math.sqrt(
            tf.cast(N, tf.float64) *
            tf.cast(tf.math.reduce_variance(elpdi), dtype=tf.float64))

        return {
            'waic': waic, 'se': se.numpy(), 'lppd': lppd, 'pwaic': pwaic}

    def save(self, filename="model_save.pkl"):
        with open(filename, 'wb') as file:
            dill.dump(self, file)

    def __getstate__(self):
        state = self.__dict__.copy()
        keys = self.__dict__.keys()

        for k in keys:
            # print(k)
            if isinstance(
                    state[k], tf.Tensor) or isinstance(state[k], tf.Variable):
                state[k] = state[k].numpy()
            elif isinstance(state[k], dict) or isinstance(state[k], list):
                flat = tf.nest.flatten(state[k])
                new = []
                for t in flat:
                    if isinstance(t, tf.Tensor) or isinstance(t, tf.Variable):
                        # print(k)
                        new += [t.numpy()]
                    elif hasattr(inspect.getmodule(t), "__name__"):
                        if inspect.getmodule(
                                t).__name__.startswith("tensorflow"):
                            if not isinstance(t, tf.dtypes.DType):
                                new += [None]
                            else:
                                new += [None]
                        else:
                            new += [t]
                    else:
                        new += [t]
                state[k] = tf.nest.pack_sequence_as(state[k], new)
            elif hasattr(inspect.getmodule(state[k]), "__name__"):
                if inspect.getmodule(
                        state[k]).__name__.startswith("tensorflow"):
                    if not isinstance(state[k], tf.dtypes.DType):
                        del state[k]
        state['strategy'] = None
        return(state)

    def unormalized_log_prob_list(self, params):
        dict_params = {k: p for k, p in zip(self.var_list, params)}
        return self.unormalized_log_prob(**dict_params)

    def unormalized_log_prob(self, *args, **kwargs):
        """Generic method for the unormalized log probability function
        """
        return

    def reconstitute(self, state):
        pass

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        #  self.dtype = tf.float64
        self.reconstitute(state)
        self.saved_state = state
        self.set_calibration_expectations()
