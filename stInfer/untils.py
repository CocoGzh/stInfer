from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import sklearn
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import torch
from scipy.sparse import csr_matrix


def gene_impute(ref, query, rep='X_scCorrect', n_neighbors=30):
    if not isinstance(ref.raw.var, pd.DataFrame):
        print('please set ref.raw.var')
        raise NotImplementedError

    if isinstance(ref.raw.X, csr_matrix):
        ref_raw = ref.raw.X.toarray()
    else:
        ref_raw = ref.raw.X
    X_train = ref.obsm[rep]
    X_test = query.obsm[rep]

    # impute
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='cosine').fit(X_train)
    distances, indices = knn.kneighbors(X_test)
    # weight = (1 - (distances[distances < 1]) / (np.sum(distances[distances < 1]))).reshape(distances.shape)
    weight = 1 - distances / (np.sum(distances))
    weight = weight / (weight.shape[1])
    query_impute = []

    # method 2 fast
    print('method 2 fast')
    ref_raw_t = torch.tensor(ref_raw, dtype=torch.float32)  # float32 or float64
    weight_t = torch.tensor(weight.reshape(weight.shape[0], 1, -1), dtype=torch.float32)
    indices_t = torch.tensor(indices, dtype=torch.int64)
    query_impute_t = torch.bmm(weight_t, ref_raw_t[indices_t])
    query_impute = query_impute_t.squeeze().detach().cpu().numpy()

    # store
    adata_impute = sc.AnnData(X=np.array(query_impute), obs=query.obs, var=ref.raw.var, obsm=query.obsm, uns=query.uns)
    return adata_impute


def cal_pearsonr(adata_pred, adata_true, dim=1, func=pearsonr):
    """

    :param adata_pred:
    :param adata_true:
    :param dim: 0 menas cell, 1 means gene
    :param func:
    :return:
    """
    data_pred = adata_pred.X
    data_true = adata_true.X
    r1, p1 = [], []
    for g in range(data_pred.shape[dim]):
        if dim == 1:
            r, pv = func(data_pred[:, g], data_true[:, g])
        elif dim == 0:
            r, pv = func(data_pred[g, :], data_true[g, :])
        r1.append(r)
        p1.append(pv)
    r1 = np.array(r1)  # 相关性
    p1 = np.array(p1)
    return r1, p1


def super_resolution(adata, rad_cutoff_high=450, rad_cutoff_low=50, k_cutoff=None,
                     max_neigh=50, model='Radius', verbose=True):
    """

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is
        less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    adata.X = adata.X.toarray() if isinstance(adata.X, sparse.csr_matrix) else adata.X
    adata.obs.index = np.array(list(range(0, len(adata.obs)))).astype(np.str)

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')

    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    nbrs = NearestNeighbors(n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff_high,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    # 生成伪adata
    Spatial_Net_array = Spatial_Net.to_numpy()
    fake_expression = []
    fake_x = []
    fake_y = []
    for item in Spatial_Net_array:
        x_temp = int((adata.obs.loc[item[0], 'pixel_x'] + adata.obs.loc[item[1], 'pixel_x']) / 2)
        fake_x.append(x_temp)
        y_temp = int((adata.obs.loc[item[0], 'pixel_y'] + adata.obs.loc[item[1], 'pixel_y']) / 2)
        fake_y.append(y_temp)
        expression_temp = (adata[item[0], :].X + adata[item[1], :].X) / 2
        fake_expression.append(expression_temp)

    adata_fake = sc.AnnData(X=np.concatenate(fake_expression, axis=0), var=adata.var)
    adata_fake.obs.loc[:, 'pixel_x'] = fake_x
    adata_fake.obs.loc[:, 'pixel_y'] = fake_y
    adata_fake.obs.index = np.array(list(range(len(adata), len(adata) + len(adata_fake)))).astype(np.str)  # TODO
    adata_fake.obsm['spatial'] = adata_fake.obs.loc[:, ['pixel_x', 'pixel_y']].to_numpy()
    adata_fake.obs.loc[:, 'cell_type'] = 'simulated_spot'
    adata_new = sc.concat([adata, adata_fake])
    adata_new.uns = adata.uns

    adata_fake2 = adata_fake.copy()
    adata_fake2.X = np.zeros(adata_fake.X.shape)
    adata_new2 = sc.concat([adata, adata_fake2])
    adata_new2.uns = adata.uns

    # 将 adata_new 中barcode之间小于rad_cutoff_low的删掉一个
    coor = pd.DataFrame(adata_new.obsm['spatial'])
    coor.index = adata_new.obs.index
    coor.columns = ['imagerow', 'imagecol']
    nbrs = NearestNeighbors(n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Spatial_Net = KNN_df.copy()

    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    distance_min = Spatial_Net[Spatial_Net.loc[:, 'Distance'] < rad_cutoff_low]
    distance_min = distance_min.loc[distance_min.loc[:, 'Cell1'] != distance_min.loc[:, 'Cell2'], :].to_numpy()
    barcode_dict = {}
    # 把临近的spot合并成为较大的
    for item in distance_min:
        if item[1] > item[0]:
            barcode_dict[item[1]] = item[0]
    adata_new.obs.loc[:, 'barcode'] = adata_new.obs.index
    adata_new.obs.loc[:, 'barcode'] = adata_new.obs.loc[:, 'barcode'].apply(
        lambda x: x if x not in barcode_dict.keys() else barcode_dict[x])
    # 去重复
    condition = adata_new.obs.drop_duplicates(keep='first', subset=['barcode'], inplace=False)
    adata_new = adata_new[condition.index, :].copy()
    adata_new2 = adata_new2[adata_new.obs.index, :].copy()

    return adata_new, adata_new2
