from __future__ import division

import json
from glob import glob
from tqdm import tqdm
import sys, os
import pandas as pd
import scanpy as sc
import numpy as np
import scipy
from scipy import stats
import itertools
import warnings
from collections import OrderedDict 
from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_sklearn
import pickle

#plotting
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import seaborn as sns

warnings.simplefilter(action='ignore', category=FutureWarning)


def pickling(filename, data_structure):
    with open(filename, 'wb') as p:
        pickle.dump(data_structure, p)


def unpickling(pickle_file):
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


def count_features(df, groupby):
    #df might adata.obs
    cnts = df.groupby(groupby)[groupby[0]].count().reset_index(name='cnt')
    # filter cells below 3
    cnts = cnts[cnts['cnt']>3]
    cnts['pct'] = cnts.groupby([groupby[0]])['cnt'].transform(lambda x : np.round(100*x/x.sum(), 1))
    return cnts

def plotbar(df, color, ax, kind='barh', **kwargs):
    df.plot(kind=kind, stacked=True, color=color, alpha=0.9, edgecolor = "black", ax=ax, **kwargs)
    
def summarize(df, x, y, aggr, order=None):
    summ = count_features(df, [x, y]).sort_values(by=[aggr], ascending=False)
    summ = pd.pivot_table(summ, values=aggr, index=[y], columns=[x])
    if order != None:
        summ = summ[order]
        #order_clusters = [el for el in order if el in summ.columns]
    return summ

def plotbarcomplete(df, x, y, colors, ax, order, kind='barh', show_numbers=True, **kwargs):
    df1 = summarize(df, x, y, 'pct', order)
    df2 = summarize(df, x, y, 'cnt', order)
    df2 = df2.sum(axis=0).values.tolist()
    plotbar(df1.T, colors, kind=kind, ax=ax, width=0.8)
    #ax.set_title(f'Pct cells in clusters according to {y}')
    if show_numbers == True:
        for el in range(len(df2)):
            if kind=='barh':
                ax.text(105, el-0.2, str(int(df2[el])))
            elif kind=='bar':
                ax.text(el-0.3, 105, str(int(df2[el])))
    sns.despine()
    ax.set_ylabel(x)
    ax.set_xlabel('Percent')
    ax.legend(loc='best', bbox_to_anchor=(1.3, 1), frameon=False)
    
def cosine_similarity(adata, comparison_field, embedding):
    embeddings = []
    genes = []

    gene_ids = adata.obs[comparison_field].values
    for gene in adata.obs[comparison_field].unique():
        if 'rand' not in gene:
            genes.append(gene)
            embeddings.append(np.mean(adata.obsm[embedding][gene_ids==gene,:], axis=0))
    
    embedding = np.stack(embeddings)
    genes = np.array(genes)
    return pd.DataFrame(data=cosine_similarity_sklearn(embedding), index=genes, columns=genes) 

def order_df_based_linkage(df):
    c_link = linkage(df.T, 'complete')
    ordered_cols = dendrogram(c_link, labels=df.columns, no_plot=True)['ivl']
    return ordered_cols

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# MinMax scale so the scores are from 0 - 1

def minMax(x):
    mi = x.min()
    ma = x.max()
    return (x - mi)/(ma - mi)

def map_genes_set(marker_dict):
    map_genes = pd.DataFrame()

    for el in marker_dict:
        tmp = pd.DataFrame.from_dict(marker_dict[el])
        tmp['combined_clusters'] = el
        map_genes = pd.concat([map_genes, tmp])

    map_genes = map_genes.set_index(0)
    return map_genes


def basic_proc_for_velocity(adata):
    adata.obsm['X_umap_integrated'] = adata.obsm['X_umap'].copy()
    scv.pp.filter_genes(adata, min_shared_counts=20)
    # Recompute neighbors
    scv.pp.neighbors(adata, n_neighbors=30)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    sc.tl.umap(adata, spread=1., min_dist=.5, random_state=11)
    scv.tl.recover_dynamics(adata, n_jobs=40)
    scv.tl.velocity(adata, mode="dynamical")
    scv.tl.velocity_graph(adata, n_jobs=40)
    # compute confidence
    print("Computing velocity confidence")
    scv.tl.velocity_confidence(adata)
    # calculate pseudotime
    print("Computing pseudotime")
    scv.tl.velocity_pseudotime(adata)
    # caluclate latent time
    print("Computing latent time")
    scv.tl.latent_time(adata)
    return adata

def proc(adata, n_top_genes='auto', key=None, norm=True, scale=False, regress=False, embedding=True, n_pcs=30, n_neighbors=10, regress_cell_cycle=False, **hvg_kwargs):
    if norm == True:
        print('Putting back counts layers as X')
        adata.X = adata.layers["counts"].copy()
        print('normalizing')
        sc.pp.normalize_total(adata, target_sum=10000)
        sc.pp.log1p(adata)
    if n_top_genes=='auto':
        print('selecting highly variable genes')
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, **hvg_kwargs)
        hvg=adata[:, adata.var.highly_variable].X.shape[1]
        print('Done selecting ' + str(hvg) + ' highly variable genes')
    else:
        print('selecting '+ str(n_top_genes) +' highly variable genes')
        sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top_genes, min_mean=0.01, max_mean=5, min_disp=0.5, **hvg_kwargs)
    if regress==True:
        print("regressing")
        sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    if regress_cell_cycle==True:
        print("regressing cell cycle")
        sc.pp.regress_out(adata, ['S_score', 'G2M_score'])
    if scale==True:
        print("scaling")
        sc.pp.scale(adata, max_value=10)
    if embedding==True:
        print("computing PCA")
        sc.tl.pca(adata)
        if key!=None:
            print('batch correcting')
            sc.external.pp.harmony_integrate(adata, basis='X_pca', adjusted_basis='X_pca_harmony', key=key, max_iter_harmony=50)
            sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_pcs=n_pcs, n_neighbors=30, random_state=42)
        else:
            sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors, random_state=42)
        print("computing UMAP")
        sc.tl.umap(adata, spread=1., min_dist=.5, random_state=11)
    return adata


def local_correlation_plot(local_correlation_z, modules, mod_cmap='tab20', vmin=-8, vmax=8, z_cmap='RdBu_r', yticklabels=False, 
                           savepath=None, higlight_modules=None, **kwargs):
    """
    This assumes local_correlation and modules are already ordered in the way that we want it. Modified from hotspot.
    """
    row_colors = None
    colors = list(plt.get_cmap(mod_cmap).colors)
    module_colors = {i: colors[(i-1) % len(colors)] for i in modules.unique()}
    module_colors[-1] = '#ffffff'

    row_colors1 = pd.Series(
        [module_colors[i] for i in modules],
        index=local_correlation_z.index,
    )

    row_colors = pd.DataFrame({
        "Modules": row_colors1,
    })

    cm = sns.clustermap(
        local_correlation_z,
        col_cluster=False,
        row_cluster=False,
        vmin=vmin,
        vmax=vmax,
        cmap=z_cmap,
        xticklabels=False,
        yticklabels=yticklabels,
        row_colors=row_colors,
        rasterized=True,
        **kwargs
    )

    fig = plt.gcf()
    plt.sca(cm.ax_heatmap)
    plt.ylabel("")
    plt.xlabel("")

    cm.ax_row_dendrogram.remove()

    mod_reordered = modules.loc[local_correlation_z.columns.tolist()]

    mod_map = {}
    y = np.arange(modules.size)

    for x in mod_reordered.unique():
        if x == -1:
            continue

        mod_map[x] = y[mod_reordered == x].mean()

    plt.sca(cm.ax_row_colors)
    for mod, mod_y in mod_map.items():
        plt.text(-.5, y=mod_y, s="Module {}".format(mod),
                 horizontalalignment='right',
                 verticalalignment='center')
    plt.xticks([])

    # Find the colorbar 'child' and modify
    min_delta = 1e99
    min_aa = None
    for aa in fig.get_children():
        try:
            bbox = aa.get_position()
            delta = (0-bbox.xmin)**2 + (1-bbox.ymax)**2
            if delta < min_delta:
                delta = min_delta
                min_aa = aa
        except AttributeError:
            pass

    min_aa.set_ylabel('Z-Scores')
    min_aa.yaxis.set_label_position("left")

    
    if higlight_modules!=None:
        for pos, col in zip(higlight_modules.keys(), higlight_modules.values()):
            rect = patches.Rectangle((pos, pos), col, col, linewidth=2, edgecolor='black', facecolor='none')
            cm.ax_heatmap.add_patch(rect)
    
    if savepath != None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        
def highlight(adata, color, grp, title, ax):
    tmp = adata.copy()
    tmp.obs['new'] = tmp.obs[color].astype(str)
    tmp.obs.loc[tmp.obs['new']!=grp, 'new'] = 'other'
    sc.pl.umap(tmp, color='new', groups=grp, show=False, title=title, legend_loc='', ax=ax)
    
    
def plot_cdf(s, cluster, genesig, ax):
    sns.ecdfplot(s, ax=ax)
    
    
def ks_2stat(sample1, sample2):
    s = stats.ks_2samp(sample1, sample2)
    stat, pval = s[0], s[1]
    if pval == 0:
        pval = 1E-100
        
    return stat, pval


def plot_cluster(adata1, adata2, adata3, cluster, genesig, xtext, ytext, ax):
    s1 = adata1[adata1.obs['combined_clusters']==cluster].obs[genesig]
    s2 = adata2[adata2.obs['combined_clusters']==cluster].obs[genesig]
    s3 = adata3[adata3.obs['combined_clusters']==cluster].obs[genesig]
    stat1, pval1 = ks_2stat(s1, s2)
    stat2, pval2 = ks_2stat(s1, s3)
    stat3, pval3 = ks_2stat(s3, s1)
    sns.ecdfplot(s1, ax=ax)
    sns.ecdfplot(s2, ax=ax)
    sns.ecdfplot(s3, ax=ax)
    
    
def data_integration(concat_adata):
    concat_adata = concat_adata[concat_adata.obs['combined_clusters'].isin(['I1', 'I2', 'I3', 'I4'])].copy()
    concat_adata.obs['label_study_concise_annotation'] = concat_adata.obs['combined_clusters'].astype(str) + '_' + concat_adata.obs['study_concise_annotation'].astype(str)
    concat_adata = proc(concat_adata, norm=True, key='study')
    concat_adata.obs['hue'] = concat_adata.obs['concise_annotation'].transform(lambda x:x.split('_')[0])
    concat_adata.obs['hue'] = concat_adata.obs['hue'].map({
        'control':'control', 'N2':'N2', 'bleomycin':'insult', 'injury':'insult', 'bleomycin+nintedanib':'insult', 'MI':'insult', 'HFD':'insult', 
                                   'western diet':'insult', 'asbestos':'insult', 'control_bleomycin':'insult', 'N2_systemic_bleomycin':'N2 bleo'})
    concat_adata.obs.loc[concat_adata.obs['concise_annotation']=='control_bleomycin', 'hue'] = 'insult'
    concat_adata.obs.loc[concat_adata.obs['concise_annotation']=='N2_systemic_bleomycin', 'hue'] = 'N2 bleo'

    concat_adata.obs.loc[concat_adata.obs['concise_annotation']=='N2_systemic', 'hue'] = 'N2 IP'
    concat_adata.obs.loc[concat_adata.obs['concise_annotation']=='N2_IT', 'hue'] = 'N2 IT'

    concat_adata.obs['study_concise_annotation'] = concat_adata.obs['study_concise_annotation'].transform(lambda x:x.replace('_', ' '))
    return concat_adata

def process_cosine_plot(adata, cosine, anchor):
    def cnt_grp(grp):
        counts = adata.obs.groupby(grp)['concise_annotation'].count().reset_index(name='cnts')
        counts = counts[counts['cnts']>0]
        return counts
    
    df = cosine[[anchor]]
    
    cnts = cnt_grp(['author', 'concise_annotation', 'study_concise_annotation', 'combined_clusters', 'hue', 'tissue'])
    cnts['pct'] = cnts.groupby(['author', 'study_concise_annotation'])['cnts'].transform(lambda x:np.round(100*(x/x.sum()), 1))

    for el in ['author', 'concise_annotation','study_concise_annotation', 'combined_clusters', 'hue', 'tissue']:
        cnts[el] = cnts[el].astype(str)

    cnts = pd.pivot(cnts, 
                    index=['author', 'study_concise_annotation', 'hue', 'tissue'], 
                    columns='combined_clusters', values='pct').reset_index().fillna(0)

    cnts = cnts.set_index('study_concise_annotation')

    cnts = pd.merge(cnts, df, left_index=True, right_index=True)
    cnts = cnts.sort_values(by=anchor, ascending=False)
    cnts['tissue'] = cnts['tissue'].astype("category").cat.reorder_categories(['Adipose', 'Heart', 'Liver', 'Lung', 'Skeletal muscle'])
    return cnts