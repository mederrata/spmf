import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import dill as pickle

from mederrata_spmf import PoissonAutoencoder

import matplotlib.pyplot as plt


def main():
    N = 50000
    D = 30
    P = 4

    # Test taking in from tf.dataset, don't pre-batch
    data = tf.data.Dataset.from_tensor_slices(
        {
            'data': np.random.poisson(1.0, size=(N, D)),
            'indices': np.arange(N),
            'normalization': np.ones(N)
        })

    data = data.batch(1000)
    # strategy = tf.distribute.MirroredStrategy()
    strategy = None
    factor = PoissonAutoencoder(
        data, latent_dim=P, strategy=strategy,
        dtype=tf.float64)
    # Test to make sure sampling works
    sample = factor.joint_prior.sample()
    # Compute the joint log probability of the sample
    probs = factor.joint_prior.log_prob(sample)
    sample_surrogate = factor.surrogate_distribution.sample(77)
    probs_parts = factor.unormalized_log_prob_parts(
        **sample_surrogate, data=next(iter(data)))
    prob = factor.unormalized_log_prob(
        **sample_surrogate,  data=next(iter(data)))

    losses = factor.calibrate_advi(
        num_epochs=50, rel_tol=1e-4, learning_rate=.0001)

    waic = factor.waic()
    print(waic)


if __name__ == "__main__":
    main()
