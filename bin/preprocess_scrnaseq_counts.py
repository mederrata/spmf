import numpy as np
import pandas as pd
import scanpy as sc  # for interface to single cell stuff
import umap
import matplotlib.pyplot as plt

sc.settings.verbosity = 2
# sc.logging.print_versions()

# load a dataset
dataset_name = 'pbmc3k'
datapath = "C:\\data\\scRNAseq\\" + dataset_name + "\\"
savefile_counts = dataset_name + '_counts.npy'
savefile_genenames = dataset_name + '_genenames.npy'
savefile_UMAP_scanpy = dataset_name + '_UMAP_scanpy.npy'
savefile_UMAP = dataset_name + '_UMAP.npy'

# uncomment below to download the data matrix
# !mkdir ~/data
# !wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz -O data/pbmc3k_filtered_gene_bc_matrices.tar.gz
# !cd data; tar -xzf pbmc3k_filtered_gene_bc_matrices.tar.gz
# datapath='~/data'

# custom path (Windows)

adata = sc.read_10x_mtx(
    datapath,  # the directory with the `.mtx` file
    # use gene symbols for the variable names (variables-axis index)
    var_names='gene_symbols',
    cache=True)                              # write a cache file for faster subsequent reading


# begin preprocessing
adata.var_names_make_unique()

# filter out poor cells and uninformative genes
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# annotate the group of mitochondrial genes as 'mt'
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(
    adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
# sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
#              jitter=0.4, multi_panel=True)

# Remove cells that have too many mitochondrial genes expressed or too many total counts.
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

# drop cells manually before computing hvg's or else some cells end up with all zero counts.
sc.pp.filter_genes(adata, min_cells=3)

# raw counts
X = adata.X.todense()

# this is to get HVGs
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata,n_top_genes=n_top_genes)
# sc.pp.highly_variable_genes(adata, min_mean=0.25, max_mean=5, min_disp=1.5)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# sc.pl.highly_variable_genes(adata)

# hvgix = adata.var.highly_variable
# 
# disp_norm = adata.var.dispersions_norm
# disp_norm.index = np.arange(len(disp_norm))
# disp_norm_sorted = disp_norm.sort_values(ascending=False)
# dispix = disp_norm_sorted.index
# all_gene_names = adata.var_names
#
## #order: hvgix, then remaining sorted by disp
## hvgix=hvgix[dispix]
## newix=pd.concat(disp_norm_sorted.index[hvgix==True], disp_norm_sorted.index[hvgix==False])
#
# sort the whole matrix by normalized dispersion, then select desired number
# X = X[:, dispix]
# all_gene_names_sorted = all_gene_names[dispix]

# store raw and do the hvg filtering
adata.raw = adata
adata = adata[:, adata.var.highly_variable]

# regress out
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

# scale
sc.pp.scale(adata, max_value=10)

# Reduce the dimensionality of the data by running principal component analysis (PCA), which reveals the main axes of variation and denoises the data.
sc.tl.pca(adata, svd_solver='arpack')

# Let us compute the neighborhood graph of cells using the PCA representation of the data matrix. You might simply use default values here. For the sake of reproducing Seurat's results, let's take the following values.
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

## Embedding the neighborhood graph
sc.tl.leiden(adata)
sc.tl.paga(adata)
sc.pl.paga(adata)  # remove `plot=False` if you want to see the coarse-grained graph
sc.tl.umap(adata, init_pos='paga')
sc.tl.umap(adata)

sc.pl.umap(adata)

# save the UMAP coords
X_umap_scanpy=adata.obsm['X_umap']

# save the data
# np.save(datapath+savefile_counts, X)
# np.save(datapath+savefile_genenames, all_gene_names_sorted.to_numpy())
np.save(datapath+savefile_UMAP_scanpy, X_umap_scanpy)

umap_obj = umap.UMAP(n_neighbors=10, n_components=2, n_epochs=1000, 
                     min_dist=0.1, spread=1.0, verbose=True)

X_umap=umap_obj.fit_transform(X)

plt.figure()
hs = plt.scatter(X_umap_scanpy[:,0], X_umap_scanpy[:,1], s=3)
plt.figure()
hs = plt.scatter(X_umap[:,0], X_umap[:,1], s=3)
plt.show()

np.save(datapath+savefile_UMAP, X_umap)
