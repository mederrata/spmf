#!/usr/bin/env python3
import os
import sys
import argparse
import csv
import itertools
import pandas as pd
import arviz as az

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
        '-e', '--epoch', nargs='?', type=int, default=300,
        help='Enter Epoch value: Default: 300'
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
        '-lr', '--learning-rate', nargs='?', type=float, default=0.01,
        help='Enter float. Default: 0.01'
    )
    
    parser.add_argument(
        '-c', '--clip-value', nargs='?', type=float, default=3.,
        help='Gradient clip value. Default: 3.0'
    )

    parser.add_argument(
        '-lt', '--log-transform',
        help='Log-transform?', action='store_true'
    )

    parser.add_argument(
        '-rn', '--row-normalize',
        help='Row normalize based on counts?', action='store_true'
    )

    args = parser.parse_args(sys.argv[1:])
    if args.csv_file is None:
        sys.exit("You need to specify a csv file")
    elif not os.path.exists(args.csv_file):
        sys.exit("File doesn't exist")
    else:
        _FILENAME = args.csv_file

    _BATCH_SIZE = args.batch_size
    _LOG_TRANSFORM = args.log_transform
    _EPOCH_NUMBER = args.epoch
    _DIMENSION = args.dimension
    _LEARNING_RATE = args.learning_rate
    _ROW_NORMALIZE = args.row_normalize
    _CLIP_VALUE = args.clip_value

    with open(_FILENAME) as f:
        csv_file = csv.reader(f)
        columns = len(next(csv_file))

    csv_data0 = tf.data.experimental.CsvDataset(
        _FILENAME, [tf.float64]*columns)
    csv_data0 = csv_data0.enumerate()

    csv_data = csv_data0.map(
        lambda j, *x: {
            'indices': j,
            'data': tf.squeeze(tf.stack(x, axis=-1))
        })

    # Grab a batch to compute statistics
    colsums = []
    batch_sizes = []
    N = 0
    for batch in iter(csv_data.batch(_BATCH_SIZE, drop_remainder=False)):
        colsums += [tf.reduce_sum(batch['data'], axis=0, keepdims=True)]
        N += batch['data'].shape[0]

    colsums = tf.add_n(colsums)
    colmeans = colsums/N
    rowmean = tf.reduce_sum(colmeans)

    if _ROW_NORMALIZE:
        csv_data = csv_data0.map(
            lambda j, *x: {
                'indices': j,
                'data': tf.squeeze(tf.stack(x, axis=-1)),
                'normalization': tf.reduce_max([
                    tf.reduce_sum(x), 1.])/rowmean
            })

    csv_data_batched = csv_data.batch(_BATCH_SIZE, drop_remainder=True)
    csv_data_batched = csv_data_batched.prefetch(
        tf.data.experimental.AUTOTUNE)

    factor = PoissonMatrixFactorization(
        csv_data_batched, latent_dim=_DIMENSION, strategy=None,
        scale_columns=True, log_transform=_LOG_TRANSFORM,
        column_norms=colmeans,
        u_tau_scale=1.0/np.sqrt(columns*N),
        dtype=tf.float64)

    factor.calibrate_advi(
        num_epochs=_EPOCH_NUMBER,
        rel_tol=1e-4, clip_value=_CLIP_VALUE,
        learning_rate=_LEARNING_RATE)

    print("Saving the encoding matrix")

    filename = f"{_FILENAME}_{_DIMENSION}D_encoding"
    filename += f"_lt_{_LOG_TRANSFORM}_rn_{_ROW_NORMALIZE}.csv"
    with open(filename, "w") as f:
        writer = csv.writer(f)
        encoding = factor.encoding_matrix().numpy().T
        for row in range(encoding.shape[0]):
            writer.writerow(encoding[row, :])

    print("Saving the trained model object")
    filename = f"{_FILENAME}_{_DIMENSION}D_model"
    filename += f"_lt_{_LOG_TRANSFORM}_rn_{_ROW_NORMALIZE}.pkl"
    factor.save(filename)

    print("Saving figure with the encodings")

    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    D = factor.feature_dim
    pcm = ax[0].imshow(
        factor.encoding_matrix().numpy()[::-1, :],
        vmin=0, cmap="Blues")
    ax[0].set_yticks(np.arange(factor.feature_dim))
    ax[0].set_yticklabels(np.arange(factor.feature_dim))
    ax[0].set_ylabel("item")
    ax[0].set_xlabel("factor dimension")
    ax[0].set_xticks(np.arange(_DIMENSION))
    ax[0].set_xticklabels(np.arange(_DIMENSION))

    surrogate_samples = factor.surrogate_distribution.sample(250)
    if 's' in surrogate_samples.keys():
        weights = surrogate_samples['s'] / \
            tf.reduce_sum(surrogate_samples['s'], -2, keepdims=True)
        intercept_data = az.convert_to_inference_data(
            {
                r"":
                    (
                        tf.squeeze(surrogate_samples['w'])
                        * weights[:, -1, :]
                        * factor.column_norm_factor
                    ).numpy().T})
    else:
        intercept_data = az.convert_to_inference_data(
            {
                r"":
                    (
                        tf.squeeze(surrogate_samples['w'])
                        * factor.column_norm_factor).numpy().T})

    fig.colorbar(pcm, ax=ax[0], orientation="vertical")
    az.plot_forest(intercept_data, ax=ax[1])
    ax[1].set_xlabel("background rate")
    ax[1].set_ylim((-0.014, .466))
    ax[1].set_title("65% and 95% CI")
    ax[1].axvline(1.0, linestyle='dashed', color="black")
    filename = f"{_FILENAME}_{_DIMENSION}D_encoding_"
    filename += f"lt_{_LOG_TRANSFORM}_rn_{_ROW_NORMALIZE}.pdf"
    plt.savefig(
        filename,
        bbox_inches='tight')

    print("Generating representations")
    filename = f"{_FILENAME}_{_DIMENSION}D_representation"
    filename += f"_lt_{_LOG_TRANSFORM}_rn_{_ROW_NORMALIZE}.csv"

    csv_data_batched = csv_data.batch(_BATCH_SIZE, drop_remainder=False)
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for record in iter(csv_data_batched):
            z = factor.encode(tf.cast(record['data'], factor.dtype)).numpy()
            if _ROW_NORMALIZE:
                z *= (record['normalization'].numpy())[:, np.newaxis]
            ind = record['indices'].numpy()
            for row in range(z.shape[0]):
                writer.writerow(np.concatenate([[ind[row]], z[row, :]]))


if __name__ == "__main__":
    main()
