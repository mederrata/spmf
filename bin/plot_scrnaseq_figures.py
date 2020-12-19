import matplotlib.font_manager as fm
from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

rcParams['font.family'] = 'sans-serif'
rcParams.update({'figure.autolayout': True})

P=4

# load the data
dataset_name = 'pbmc3k'
datapath = "C:\\data\\scRNAseq\\" + dataset_name + "\\"

X = np.load(datapath + dataset_name + '_counts.npy')
gene_names = np.load(datapath + dataset_name +
                     '_genenames.npy', allow_pickle=True)
UMAP = np.load(datapath + dataset_name + '_UMAP_scanpy.npy')
# UMAP = np.load(datapath + dataset_name + '_UMAP.npy')
U = np.load(datapath + dataset_name + '_U_'+f"{P}"+'.npy')
V = np.load(datapath + dataset_name + '_V_'+f"{P}"+'.npy')
W = np.load(datapath + dataset_name + '_W_'+f"{P}"+'.npy')
Z = np.load(datapath + dataset_name + '_Z_'+f"{P}"+'.npy')
cell_score = np.load(datapath + dataset_name + '_cellscore_'+f"{P}"+'.npy')
gene_score = np.load(datapath + dataset_name + '_genescore_'+f"{P}"+'.npy')
intercept_score = np.load(datapath + dataset_name + '_interceptscore_'+f"{P}"+'.npy')

#for now, rename vars:
A = U
B = V
Phi = W

D = X.shape[1]

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


# alternative: specify a list of known cell type markers
marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8E', 'LYZ', 'CD14',
                'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
                'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']
# hvgix = pd.Series(gene_names).isin(marker_genes)
# X = X[:, hvgix]
# gene_names = gene_names[hvgix]


rcParams.update({'font.size': 8})
###############
# heatmap plots

# histogram of loading coefficients in A
# plt.hist(np.ndarray.flatten(A),bins=100)
# plt.show()


# topix = [i for i, v in enumerate(pd.Series(gene_names).isin(marker_genes)) if v==True]

# Gene score (rescaled decoder: latent x genes)
topD=5
topix=[]
for p in range(P):
    # thisix=np.argsort(B[p,:])[::-1][:topD]
    thisix=np.argsort(gene_score[p,:])[::-1][:topD]
    topix+=thisix.tolist()


fig= plt.figure(figsize=(5.5, 2))
# pcm = plt.imshow(B[:,topix], vmin=0, cmap="Oranges")
pcm = plt.imshow(gene_score[:,topix], vmin=0, cmap="Oranges", aspect='auto')
ax = plt.gca()
ax.xaxis.set_ticks_position('top')
plt.xticks(np.arange(len(topix)), gene_names[topix], rotation=90, )
plt.ylabel("k")
plt.yticks(np.arange(P),np.arange(1,P+1))
fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.1, shrink=0.5, aspect=30)
plt.tight_layout()
# plt.savefig(datapath+'B_'+f"{P}"+'.pdf')
plt.savefig(datapath+'gene_score_'+f"{P}"+'.pdf')
# plt.show()

# Encoding matrix (genes x latent)
# A = X_col_means[:,np.newaxis] * A
# topD=5
# topix=[]
# for p in range(P):
#     thisix=np.argsort(A[:,p])[::-1][:topD]
#     topix+=thisix.tolist()

# alt: just find the overall top loadings (in any latent dim)
# maxA = np.amax(A,axis=1)
# topix=np.argsort(maxA)[::-1][:topD]

#alt use same genes as decoder

fig= plt.figure(figsize=(2, 5.5))
pcm = plt.imshow(A[topix,:], vmin=0, cmap="Blues", aspect='auto')
# pcm = plt.imshow(A, vmin=0, cmap="Blues")
ax = plt.gca()
plt.xlabel("k")
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top') 
plt.xticks(np.arange(P),np.arange(1,P+1))
# plt.ylabel("gene")
plt.yticks(np.arange(len(topix)), gene_names[topix])
fig.colorbar(pcm, ax=ax, orientation="vertical", pad=0.1, shrink=0.5, aspect=30)
plt.tight_layout()
plt.savefig(datapath+'A_'+f"{P}"+'.pdf')
# plt.show()

# intercept
topI=P*topD
thisix=np.argsort(W[0,:])[::-1][:topI]
# thisix=np.argsort(intercept_score[0,:])[::-1][:topI]

fig= plt.figure(figsize=(5.5, 1.5))
pcm = plt.imshow(W[0,thisix][np.newaxis,:], vmin=0, cmap="Oranges")
# pcm = plt.imshow(intercept_score[0,thisix][np.newaxis,:], vmin=0, cmap="Oranges")
ax = plt.gca()
ax.xaxis.set_ticks_position('top')
plt.xticks(np.arange(len(thisix)), gene_names[thisix], rotation=90, )
plt.yticks(range(1),[R"$\varphi_i$"])
fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.1, shrink=0.5, aspect=30)
plt.tight_layout()
plt.savefig(datapath+'varphi_'+f"{P}"+'.pdf')
# plt.show()

#############
# UMAP plots
panels=(1,P)

# alternative: specify a list of known cell type markers
# marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
#                 'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
#                 'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']
# marker_genes = ['S100A8', 'GNLY', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
#                 'LGALS3',   'NKG7', 'KLRB1',
#                 'FCGR3A', 'MS4A7', 'FCER1A', , 'PPBP']
# hvgix = pd.Series(gene_names).isin(marker_genes)

# plot the top-loaded gene for each factor on UMAP
topgix=np.argmax(gene_score, axis=1)

Xn=X/row_size_factors[:,np.newaxis]
Xl=np.log10(Xn[:,topgix]+1)

fig, AX = plt.subplots(panels[0], panels[1], figsize=(5.5, 1.5))
AX = AX.flat
for i in range(P):
    idx = Xl[:,i].argsort()
    plt.sca(AX[i])
    hs = plt.scatter(UMAP[idx,0], UMAP[idx,1], c=Xl[idx, i], s=1, cmap="cividis")
    plt.title(gene_names[topgix[i]])
    plt.axis('off')
    # plt.axis('equal')
plt.tight_layout()
plt.savefig(datapath+'top_genes_'+f"{P}"+'.pdf') #, bbox_inches='tight'
# plt.show()

# cell scores
fig, AX = plt.subplots(panels[0], panels[1], figsize=(5.5, 1.5))
AX = AX.flat
for i in range(P):
    idx = Z[:,i].argsort()
    # idx = cell_score[:,i].argsort()
    plt.sca(AX[i])
    hs = plt.scatter(UMAP[idx,0], UMAP[idx,1], c=Z[idx, i], s=1, cmap="copper")
    # hs = plt.scatter(UMAP[idx,0], UMAP[idx,1], c=cell_score[idx, i], s=3, cmap="copper")
    plt.title(f"k={i+1}")
    plt.axis('off')
    # plt.axis('equal')
plt.tight_layout()
plt.savefig(datapath+'cell_scores_'+f"{P}"+'.pdf')

# plt.show()