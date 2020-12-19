# TODO: batching errors? Have weird cell numbers
# TODO: create X for scRNAseq examples {small easy, medium (preselect HVGs), and full genome}
# TODO: plot scatter of latent vars (like UMAP??)
# TODO: output plot of top20 features per latent loading???
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
 
import scanpy as sc  # for interface to single cell stuff
from mederrata_spmf import PoissonMatrixFactorization
import matplotlib.font_manager as fm
from matplotlib import rcParams
import matplotlib.pyplot as plt
import arviz as az
import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import numpy as np


rcParams['font.family'] = 'sans-serif'
rcParams.update({'figure.autolayout': True})

sys.path.append('../')

# load the data
dataset_name = 'pbmc3k'
datapath = "C:\\data\\scRNAseq\\" + dataset_name + "\\"
X = np.load(datapath + dataset_name + '_counts.npy')
gene_names = np.load(datapath + dataset_name +
                     '_genenames.npy', allow_pickle=True)
UMAP = np.load(datapath + dataset_name + '_UMAP_scanpy.npy')

#remove genes with very low column sums (cells per gene)
X_column_sums = (X>0).sum(0)
# X=X[:,X_column_sums>20]

P = 3
panels=(1,P)

D = X.shape[1]
# N_BATCHES = 7
# BATCH_SIZE = int(np.floor(X.shape[0]/N_BATCHES))
BATCH_SIZE=256


after = np.median(np.array(X.sum(1)))
# after=1
row_size_factors = X.sum(1) / after

# keep the first D genes
X = X[:, :D]
gene_names = gene_names[:D]
X_row_means = X.mean(1)
X_col_means = X.mean(0)

after = np.median(np.array(X.sum(0)))
# after=1
col_size_factors = X.sum(0) / after

row_norm=row_size_factors
# col_norm=col_size_factors
# row_norm=X_row_means
col_norm=X_col_means

# alternative: specify a list of known cell type markers
# marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
#                 'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
#                 'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']
# hvgix = pd.Series(gene_names).isin(marker_genes)
# X = X[:, hvgix]
# gene_names = gene_names[hvgix]

N, D = X.shape
print(X.shape)
print(
    f"Total observations={X.shape[0]}, Batch size={BATCH_SIZE}: dropping {X.shape[0]%BATCH_SIZE} observations.")

# Test taking in from tf.dataset, don't pre-batch
#'normalization': row_size_factors
data = tf.data.Dataset.from_tensor_slices(
    {
        'data': X,
        'indices': np.arange(N)
    })

data = data.shuffle(buffer_size=N)
data = data.batch(BATCH_SIZE, drop_remainder=True)

# strategy = tf.distribute.MirroredStrategy()
# column_norms=col_norm
factor = PoissonMatrixFactorization(
    data, data_transform_fn=None, latent_dim=P,
    u_tau_scale=1.0/np.sqrt(D*N), s_tau_scale=1., symmetry_breaking_decay=0.5,
    strategy=None, encoder_function=None, decoder_function=None,
    scale_columns=True, column_norms=None, scale_rows=True,
    with_s=True, with_w=True, log_transform=False,
    dtype=tf.float64
    )

# num_epochs=500, learning_rate=0.01,
# abs_tol=1e-3, rel_tol=1e-3,
# clip_value=10.,
losses = factor.calibrate_advi(
    num_epochs=500, learning_rate=0.01,
    opt=None, abs_tol=1e-10, rel_tol=1e-8,
    clip_value=5., max_decay_steps=25, lr_decay_factor=0.99,
    check_every=25, set_expectations=True, sample_size=4
    )

# waic = factor.waic()
# print(waic)

# encoding matrix
A = factor.encoding_matrix().numpy()

# intercept
Phi = factor.intercept_matrix().numpy()
intercept_score = Phi * col_norm[np.newaxis,:]

# Latent representation and cell score
Z = factor.encode(X).numpy()
cell_score = Z * row_norm[:,np.newaxis]
# cell_score = Z * row_norm[:,np.newaxis]

# gene score 
B = factor.decoding_matrix().numpy()
gene_score = B * col_norm[np.newaxis,:]

np.save(datapath + dataset_name + '_A_'+f"{P}"+'.npy', A)
np.save(datapath + dataset_name + '_B_'+f"{P}"+'.npy', B)
np.save(datapath + dataset_name + '_Phi_'+f"{P}"+'.npy', Phi)
np.save(datapath + dataset_name + '_Z_'+f"{P}"+'.npy', Z)
np.save(datapath + dataset_name + '_cellscore_'+f"{P}"+'.npy', cell_score)
np.save(datapath + dataset_name + '_genescore_'+f"{P}"+'.npy', gene_score)
np.save(datapath + dataset_name + '_interceptscore_'+f"{P}"+'.npy', intercept_score)