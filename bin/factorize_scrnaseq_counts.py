#TODO: batching errors? Have weird cell numbers
#TODO: create X for scRNAseq examples {small easy, medium (preselect HVGs), and full genome}
#TODO: plot scatter of latent vars (like UMAP??)
#TODO: output plot of top20 features per latent loading???
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys

import numpy as np
import pandas as pd
import scanpy as sc #for interface to single cell stuff

import tensorflow as tf
import tensorflow_probability as tfp
import arviz as az

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm

rcParams['font.family'] = 'sans-serif'

sys.path.append('../')
from mederrata_spmf import PoissonMatrixFactorization

# load the data
dataset_name = 'pbmc3k'
datapath="C:\\data\\scRNAseq\\" + dataset_name + "\\"
X = np.load(datapath + dataset_name + '_counts.npy')
gene_names = np.load(datapath + dataset_name + '_genenames.npy', allow_pickle=True)
UMAP =  np.load(datapath + dataset_name + '_UMAP.npy')

P = 3
D = 300
N_BATCHES = 4
BATCH_SIZE = int(np.floor(X.shape[0]/N_BATCHES))

# normalization for cells, computed using all genes
counts_per_cell = X.sum(1)
after = np.median(np.array(counts_per_cell))
# after=1e4
size_factors = counts_per_cell / after
norm_vals = size_factors

# keep the first D genes
# X = X[:, :D]
# gene_names=gene_names[:D]

#alternative: specify a list of known cell type markers
marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
                'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
                'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']
hvgix = pd.Series(gene_names).isin(marker_genes)
X = X[:, hvgix]
gene_names=gene_names[hvgix]

N, D = X.shape
# print(X.shape)
print(f"Total observations={X.shape[0]}, Batch size={BATCH_SIZE}: dropping {X.shape[0]%BATCH_SIZE} observations.")

# Test taking in from tf.dataset, don't pre-batch
data = tf.data.Dataset.from_tensor_slices(
    {
        'data': X,
        'indices': np.arange(N),
        'normalization': norm_vals
    })

# data = data.shuffle(buffer_size=N+1)
data = data.batch(BATCH_SIZE, drop_remainder=True)

# strategy = tf.distribute.MirroredStrategy()
    # encoder_function=lambda x: tf.math.log(x+1), decoder_function=lambda x: tf.math.exp(x)-1,
    # encoder_function=lambda x: x, decoder_function=lambda x: x,
strategy = None
factor = PoissonMatrixFactorization(
    data, latent_dim=P, strategy=strategy,
    encoder_function=lambda x: x, decoder_function=lambda x: x,
    scale_rates=True, 
    u_tau_scale=1.0/P/D/np.sqrt(N),
    dtype=tf.float64)

losses = factor.calibrate_advi(
    num_epochs=50, learning_rate=.1)

waic = factor.waic()
print(waic)

surrogate_samples = factor.surrogate_distribution.sample(1000)
weights = surrogate_samples['s']/tf.reduce_sum(surrogate_samples['s'],-2,keepdims=True)

encoding_matrix=factor.encoding_matrix().numpy()

# use all genes (with makers)
topix = range(min(len(gene_names),20))

# genes with highest dispersion
# topD = 20
# topix=range(topD)

#try to extract the topD features loaded onto each latent dimension for a plot
# topD=5
# topix=[]
# for d in range(P):
#     thisix=np.argsort(encoding_matrix[:,d])[::-1][:topD]
#     topix+=thisix.tolist()


fig, ax = plt.subplots(1,2, figsize=(14,8))
pcm = ax[0].imshow(encoding_matrix[topix,:], vmin=0, cmap="Blues")
ax[0].set_yticks(np.arange(len(topix)))
ax[0].set_yticklabels(gene_names[topix])
ax[0].set_ylabel("gene")
ax[0].set_xlabel("factor dimension")
ax[0].set_xticks(np.arange(P))
ax[0].set_xticklabels(np.arange(P))

fig.colorbar(pcm, ax=ax[0], orientation = "vertical")
ID=(tf.squeeze(surrogate_samples['w'])*weights[:,-1,:]*factor.norm_factor).numpy().T
ID=ID[topix,:]
intercept_data = az.convert_to_inference_data({r"$w_d$": ID})
az.plot_forest(intercept_data, ax=ax[1])
ax[1].set_xlabel("background rate")
ax[1].set_ylim((-0.014,.466))
ax[1].set_title("65% and 95% CI")
ax[1].axvline(1.0, linestyle='dashed', color="black")
# plt.savefig('mix_factorization_sepmf.pdf', bbox_inches='tight')
plt.show()

nploss = np.array(losses)
fig = plt.figure(figsize=(7,4))
plt.subplot(111)
plt.plot(nploss)
# lastloss = nploss[-1]
# plt.ylim((lastloss*0.9, lastloss*1.25))
# plt.savefig(f"./cache/losses.png", bbox_inches='tight', transparent=False)
plt.show()

# factor.save()
print("done")