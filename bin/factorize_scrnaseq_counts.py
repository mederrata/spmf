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
UMAP = np.load(datapath + dataset_name + '_UMAP.npy')

#remove genes with very low column sums (cells per gene)
X_column_sums = (X>0).sum(0)
X=X[:,X_column_sums>10]

P = 7
panels=(4,2)

D = X.shape[1]
N_BATCHES = 6
BATCH_SIZE = int(np.floor(X.shape[0]/N_BATCHES))


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
# print(X.shape)
print(
    f"Total observations={X.shape[0]}, Batch size={BATCH_SIZE}: dropping {X.shape[0]%BATCH_SIZE} observations.")

# Test taking in from tf.dataset, don't pre-batch
data = tf.data.Dataset.from_tensor_slices(
    {
        'data': X,
        'indices': np.arange(N),
        'normalization': row_size_factors
    })

# data = data.shuffle(buffer_size=N)
data = data.batch(BATCH_SIZE, drop_remainder=True)

# strategy = tf.distribute.MirroredStrategy()
strategy = None
factor = PoissonMatrixFactorization(
    data, latent_dim=P, strategy=strategy,
    scale_rates=True, column_norms=col_norm,
    u_tau_scale=1.0/np.sqrt(D*N),
    dtype=tf.float64, 
    encoder_function=lambda x: tf.math.log(x/col_norm+1), decoder_function=lambda x: tf.math.exp(x*col_norm)-1,
    )
    # encoder_function=lambda x: tf.math.log(x+1), decoder_function=lambda x: tf.math.exp(x)-1,
    # encoder_function=lambda x: x, decoder_function=lambda x: x,

losses = factor.calibrate_advi(
    num_epochs=200, learning_rate=0.05,
    abs_tol=1e-3, rel_tol=1e-3,
    clip_value=1.,
    )

# waic = factor.waic()
# print(waic)

# encoding matrix
U = factor.encoding_matrix().numpy()

# cell score
Z = factor.encode(X).numpy()
cell_score = Z * row_norm[:,np.newaxis]
# cell_score = Z * X_row_means[:,np.newaxis]

# gene score 
V = factor.decoding_matrix().numpy()
gene_score = V * col_norm[np.newaxis,:]

np.save(datapath + dataset_name + '_U.npy', U)
np.save(datapath + dataset_name + '_cellscore.npy', cell_score)
np.save(datapath + dataset_name + '_genescore.npy', gene_score)

####
# plotting

# use all genes (with makers)
# topix = range(min(len(gene_names), 20))

# genes with highest dispersion
# topD = 20
# topix=range(topD)

# try to extract the topD features loaded onto each latent dimension for a plot
topD=5
topix=[]
for p in range(P):
    thisix=np.argsort(gene_score[p,:])[::-1][:topD]
    topix+=thisix.tolist()


# fig= plt.figure(figsize=(7, 4))
# pcm = plt.imshow(gene_score[:,topix], vmin=0, cmap="Blues")
# ax = plt.gca()
# ax.xaxis.set_ticks_position('top')
# plt.xticks(np.arange(len(topix)), gene_names[topix], rotation=90, )
# plt.xlabel("gene")
# plt.ylabel("factor dimension")
# plt.yticks(np.arange(P),np.arange(P))
# # # plt.savefig('encoding_matrix.pdf', bbox_inches='tight')
# # fig.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.25, aspect=10)
# plt.show()

fig= plt.figure(figsize=(2.5, 5))
pcm = plt.imshow(np.transpose(gene_score[:,topix]), vmin=0, cmap="Blues")
ax = plt.gca()
plt.yticks(np.arange(len(topix)), gene_names[topix] )
plt.ylabel("gene")
plt.xlabel("factor dimension")
plt.xticks(np.arange(P),np.arange(P))
# # plt.savefig('encoding_matrix.pdf', bbox_inches='tight')
# fig.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.25, aspect=10)
plt.tight_layout()
# plt.show()

# plot the top-loaded gene for each factor on UMAP
topgix=np.argmax(gene_score, axis=1)

Xn=X/row_size_factors[:,np.newaxis]
Xl=np.log10(Xn[:,topgix]+1)

fig, AX = plt.subplots(panels[0], panels[1], figsize=(5, 5))
AX = AX.flat
for i in range(P):
    idx = Xl[:,i].argsort()
    plt.sca(AX[i])
    hs = plt.scatter(UMAP[idx,0], UMAP[idx,1], c=Xl[idx, i], s=5)
    plt.title(gene_names[topgix[i]])
    plt.axis('off')
plt.tight_layout()
# plt.show()

# cell scores
fig, AX = plt.subplots(panels[0], panels[1], figsize=(5, 5))
AX = AX.flat
for i in range(P):
    idx = cell_score[:,i].argsort()
    plt.sca(AX[i])
    hs = plt.scatter(UMAP[idx,0], UMAP[idx,1], c=cell_score[idx, i], s=5)
    plt.title(f"factor {i}")
    plt.axis('off')
plt.tight_layout()
# plt.show()


nploss = np.array(losses)
fig = plt.figure(figsize=(7, 4))
plt.subplot(111)
plt.plot(nploss)
# lastloss = nploss[-1]
# plt.ylim((lastloss*0.9, lastloss*1.25))
# plt.savefig(f"./cache/losses.png", bbox_inches='tight', transparent=False)
plt.show()



# factor.save()
print("done")


# encoding_matrix = factor.encoding_matrix().numpy()

# # use all genes (with makers)
# # topix = range(min(len(gene_names), 20))

# # genes with highest dispersion
# # topD = 20
# # topix=range(topD)

# # try to extract the topD features loaded onto each latent dimension for a plot
# topD=5
# topix=[]
# for p in range(P):
#     thisix=np.argsort(encoding_matrix[:,p])[::-1][:topD]
#     topix+=thisix.tolist()


# fig, ax = plt.subplots(1, 1, figsize=(5, 10))
# pcm = ax.imshow(encoding_matrix[topix, :], vmin=0, cmap="Blues")
# ax.set_yticks(np.arange(len(topix)))
# ax.set_yticklabels(gene_names[topix])
# ax.set_ylabel("gene")
# ax.set_xlabel("factor dimension")
# ax.set_xticks(np.arange(P))
# ax.set_xticklabels(np.arange(P))
# # # plt.savefig('encoding_matrix.pdf', bbox_inches='tight')
# fig.colorbar(pcm, ax=ax, orientation="vertical")
# plt.show()

# # fig, ax = plt.subplots(1, 2, figsize=(14, 8))
# # pcm = ax[0].imshow(encoding_matrix[topix, :], vmin=0, cmap="Blues")
# # ax[0].set_yticks(np.arange(len(topix)))
# # ax[0].set_yticklabels(gene_names[topix])
# # ax[0].set_ylabel("gene")
# # ax[0].set_xlabel("factor dimension")
# # ax[0].set_xticks(np.arange(P))
# # ax[0].set_xticklabels(np.arange(P))
# # plt.show()

# # surrogate_samples = factor.surrogate_distribution.sample(1000)
# # weights = surrogate_samples['s'] / \
# #     tf.reduce_sum(surrogate_samples['s'], -2, keepdims=True)

# # fig.colorbar(pcm, ax=ax[0], orientation="vertical")
# # ID = (tf.squeeze(surrogate_samples['w']) *
# #       weights[:, -1, :]*factor.column_norm_factor).numpy().T
# # ID = ID[topix, :]
# # intercept_data = az.convert_to_inference_data({r"$w_d$": ID})
# # az.plot_forest(intercept_data, ax=ax[1])
# # ax[1].set_xlabel("background rate")
# # ax[1].set_ylim((-0.014, .466))
# # ax[1].set_title("65% and 95% CI")
# # ax[1].axvline(1.0, linestyle='dashed', color="black")
# # # plt.savefig('mix_factorization_sepmf.pdf', bbox_inches='tight')
# # plt.show()