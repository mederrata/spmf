import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import dill as pickle

from mederrata_spmf import PoissonFactorization

# import matplotlib.pyplot as plt


def main():
    N = 50000
    D = 3500
    P = 250

    # Test taking in from tf.dataset, don't pre-batch
    data = tf.data.Dataset.from_tensor_slices(
        {
            'counts': np.random.poisson(1.0, size=(N, D)),
            'indices': np.arange(N),
            'normalization': np.ones(N)
        }).batch(1000)

    # data = data.batch(1000)
    # strategy = tf.distribute.MirroredStrategy()
    strategy = None
    factor = PoissonFactorization(
        data, latent_dim=P, feature_dim=D,
        strategy=strategy,  # horseshoe_plus=False,
        dtype=tf.float64)
    # Test to make sure sampling works
    sample = factor.sample()
    # Compute the joint log probability of the sample
    probs = factor.prior_distribution.log_prob(sample)
    sample_surrogate = factor.surrogate_distribution.sample(77)
    probs_parts = factor.unormalized_log_prob_parts(
        **sample_surrogate, data=next(iter(data)))
    prob = factor.unormalized_log_prob(
        **sample_surrogate,  data=next(iter(data)))

    losses = factor.calibrate_advi(
        num_epochs=20, rel_tol=1e-4, learning_rate=.005)

    waic = factor.waic()
    print(waic)

if __name__ == "__main__":
    main()
