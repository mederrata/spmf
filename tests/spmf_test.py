import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import dill as pickle

from mederrata_spmf import PoissonFactorization

# import matplotlib.pyplot as plt


def main():
    N = 500000
    D = 350
    P = 50

    # Test taking in from tf.dataset, don't pre-batch
    data = tf.data.Dataset.from_tensor_slices(
        {
            'counts': np.random.poisson(1.0, size=(N, D)),
            'indices': np.arange(N),
            'normalization': np.ones(N)
        })

    def data_factory(batch_size=10000):
        ds = data.shuffle(10000)
        ds = ds.batch(batch_size)
        return ds

    factor = PoissonFactorization(
        latent_dim=P, feature_dim=D, u_tau_scale=1.0/np.sqrt(N*D),
        dtype=tf.float64)

    factor.compute_scales(data_factory=data_factory)

    losses = factor.fit(
        batched_data_factory=data_factory,
        dataset_size=500000,
        batch_size=10000,
        sample_size=20,
        sample_batches=4,
        num_epochs=20,
        rel_tol=1e-4,
        learning_rate=.01)
    print(losses)


if __name__ == "__main__":
    main()
