#!/usr/bin/env python3
import os
import sys
import argparse
import csv
import itertools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from mederrata_spmf import PoissonMatrixFactorization

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description='Train PMF on CSV-formatted count matrix')
    parser.add_argument(
        '-f', '--csv-file', nargs='?', type=str,
        help="Enter the CSV file"
    )
    parser.add_argument(
        '-e', '--epoch', nargs='?', type=int, default=50,
        help='Enter Epoch value: Default: 50'
    )
    parser.add_argument(
        '-d', '--dimension', nargs='?', type=int, default=2,
        help='Enter embedding dimension. Default: 2'
    )
    parser.add_argument(
        '-b', '--batch-size', nargs='?', type=int, default=5000,
        help='Enter batch size. Default: 5000'
    )

    parser.add_argument(
        '-lr', '--learning-rate', nargs='?', type=float, default=0.05,
        help='Enter float. Default: 0.05'
    )

    args = parser.parse_args(sys.argv[1:])
    if args.csv_file is None:
        sys.exit("You need to specify a csv file")
    elif not os.path.exists(args.csv_file):
        sys.exit("File doesn't exist")
    else:
        _FILENAME = args.csv_file

    _BATCH_SIZE = args.batch_size

    _EPOCH_NUMBER = args.epoch
    _DIMENSION = args.dimension
    _LEARNING_RATE = args.learning_rate

    with open(_FILENAME) as f:
        csv_file = csv.reader(f)
        columns = len(next(csv_file))

    N = sum(1 for line in open(_FILENAME))

    csv_data = tf.data.experimental.CsvDataset(_FILENAME, [tf.float64]*columns)
    csv_data = csv_data.enumerate()
    csv_data = csv_data.map(
        lambda j, *x: {'indices': j, 'data': tf.squeeze(tf.stack(x, axis=-1))})

    csv_data_batched = csv_data.batch(_BATCH_SIZE, drop_remainder=True)

    factor = PoissonMatrixFactorization(
        csv_data_batched, latent_dim=_DIMENSION, strategy=None,
        encoder_function=lambda x: x, decoder_function=lambda x: x,
        scale_rates=True,
        u_tau_scale=1.0/_DIMENSION/columns/np.sqrt(N),
        dtype=tf.float64)

    factor.calibrate_advi(
        num_epochs=_EPOCH_NUMBER,
        rel_tol=1e-4,
        learning_rate=_LEARNING_RATE)

    print("Saving the encoding matrix")

    filename = f"{_FILENAME}_{_DIMENSION}D_encoding.csv"
    with open(filename, "w") as f:
        writer = csv.writer(f)
        encoding = factor.encoding_matrix().numpy().T
        for row in range(encoding.shape[0]):
            writer.writerow(encoding[row, :])

    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    D = factor.feature_dim
    pcm = ax[0].imshow(factor.encoding_matrix().numpy()
                       [::-1, :], vmin=0, cmap="Blues")
    ax[0].set_yticks(np.arange(_DIMENSION))
    ax[0].set_yticklabels(np.arange(_DIMENSION))
    ax[0].set_ylabel("item")
    ax[0].set_xlabel("factor dimension")
    ax[0].set_xticks(np.arange(_DIMENSION))
    ax[0].set_xticklabels(np.arange(_DIMENSION))

    fig.colorbar(pcm, ax=ax[0], orientation="vertical")
    az.plot_forest(intercept_data, ax=ax[1])
    ax[1].set_xlabel("background rate")
    ax[1].set_ylim((-0.014, .466))
    ax[1].set_title("65% and 95% CI")
    ax[1].axvline(1.0, linestyle='dashed', color="black")
    plt.savefig(f"{_FILENAME}_{_DIMENSION}D_encoding.pdf", bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    main()
