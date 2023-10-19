import torch
import cv2
import numpy as np
import scanpy as sc
import pandas as pd
import torchvision.transforms.functional as TF
import random
from PIL import Image
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from .config import config


def preprocess_adata(adata):
    # sparse to array and uniform gene
    if isinstance(adata.X, csr_matrix):
        adata.X = adata.X.toarray()
    # norm and log1p
    sc.pp.log1p(adata)
    sc.pp.normalize_total(adata)
    return adata


def data_aug(patch_image):
    # Random flipping and rotations
    image = Image.fromarray(patch_image)
    if random.random() > 0.5:
        image = TF.hflip(image)
    if random.random() > 0.5:
        image = TF.vflip(image)
    angle = random.choice([180, 90, 0, -90])
    image = TF.rotate(image, angle)
    return np.array(image)


def normalize_image(full_image):
    from normalizer import IterativeNormaliser, scale_img
    full_image = Image.fromarray(full_image.astype("uint8"))
    normaliser = IterativeNormaliser(normalisation_method='vahadane', standardise_luminosity=True)
    normaliser.fit_source(scale_img(full_image))
    normaliser.fit_target(scale_img(full_image))
    normaliser.transform_tile(full_image)
    full_image.save(f"./img.png")
    return full_image


class TenXDataset(torch.utils.data.Dataset):
    def __init__(self,
                 adata: sc.AnnData,
                 dataset_name: str,
                 patch_size=(112, 112)
                 ):

        self.dataset_name = dataset_name
        self.expression = adata.X
        self.obs = adata.obs
        if config.expression_normalize:
            preprocess_adata(adata)
        self.barcode = adata.obs.index.tolist()
        self.width, self.height = patch_size
        if config.expression_normalize:
            image_temp = normalize_image(adata.uns.data['spatial'][self.dataset_name]['images']['full'])
        else:
            image_temp = adata.uns.data['spatial'][self.dataset_name]['images']['full']
        self.image = image_temp
        print(f"Finished loading all files {dataset_name}")

    def __getitem__(self, idx):
        # prepare
        # get pixel position
        barcode = self.barcode[idx]
        x = int(self.obs.loc[barcode, 'pixel_y'])  # 坐标x  TODO 交换xy坐标，符合python中cv2图片的习惯
        y = int(self.obs.loc[barcode, 'pixel_x'])  # 坐标x
        # get patch TODO 可能会裁剪到外面，小于224*224
        image = self.image[(x - self.width):(x + self.width), (y - self.height):(y + self.height)]
        # data augment
        # image = data_aug(image)  # TODO 测试时不需要做

        # out put
        item = {}
        item['dataset_name'] = self.dataset_name
        item['barcode'] = barcode
        item['dataset_barcode'] = self.dataset_name + '_' + barcode
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()  # color channel first, then XY
        if isinstance(self.expression, csr_matrix):
            item['expression'] = torch.tensor(self.expression[idx, :].toarray()[0]).float()
        else:
            item['expression'] = torch.tensor(self.expression[idx, :]).float()

        return item

    def __len__(self):
        return len(self.barcode)


def get_TenX_dataloader(
        data_path,
        data_name,
        genes,
        patch_size=(28, 28)
):
    dataset = []
    for name in data_name:
        print(f'loading {name} file')
        adata_temp = sc.read_h5ad(f'{data_path}/{name}.h5ad')[:, genes].copy()
        dataset_temp = TenXDataset(adata_temp, name, patch_size=patch_size)
        dataset.append(dataset_temp)
    dataset = torch.utils.data.ConcatDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0,
                            pin_memory=True, drop_last=False)
    return dataloader


def get_Anndata_dataloader(
        adata_list,
        name_list,
        genes,
        patch_size=(28, 28)
):
    dataset = []
    for i, adata_temp in enumerate(adata_list):
        adata_temp = adata_temp[:, genes]
        dataset_temp = TenXDataset(adata_temp, name_list[i], patch_size=patch_size)
        dataset.append(dataset_temp)
    dataset = torch.utils.data.ConcatDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0,
                            pin_memory=True, drop_last=False)
    return dataloader


class OldSTDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_path,
                 spatial_pos_path,
                 expression_path,
                 dataset_name,
                 genes,
                 patch_size):
        self.dataset_name = dataset_name
        self.image_path = image_path
        self.genes = genes
        self.reduced_matrix = pd.read_csv(expression_path, index_col=0, header=0, sep='\t').loc[:, self.genes]
        # self.whole_image = cv2.cvtColor(cv2.imread(image_path).transpose(1, 0, 2), cv2.COLOR_BGR2RGB)  # 旋转+通道转换
        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # 旋转+通道转换
        meta = pd.read_csv(spatial_pos_path, sep="\t", header=0, index_col=None)
        meta.index = meta.loc[:, 'x'].astype(str) + 'x' + meta.loc[:, 'y'].astype(str)
        meta = meta.loc[self.reduced_matrix.index, :]
        self.spatial_pos_csv = meta
        self.barcode_tsv = meta.index.tolist()
        self.width, self.height = patch_size
        print(f"Finished loading all files {dataset_name}")

    def __getitem__(self, idx):
        # barcode = self.barcode_tsv.values[idx, 0]
        barcode = self.barcode_tsv[idx]
        x = int(self.spatial_pos_csv.loc[barcode, 'pixel_y'])  # 坐标x
        y = int(self.spatial_pos_csv.loc[barcode, 'pixel_x'])  # 坐标y
        image = self.image[(x - self.width):(x + self.width), (y - self.height):(y + self.height)]  #
        # image = data_aug(image)

        item = {}
        item['dataset_name'] = self.dataset_name
        item['barcode'] = barcode
        item['dataset_barcode'] = self.dataset_name + '_' + barcode
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()  # color channel first, then XY
        item['expression'] = torch.tensor(
            self.reduced_matrix.loc[barcode, :].to_numpy()).float()  # cell x features (3467)
        item['spatial_coords'] = [x, y]

        return item

    def __len__(self):
        return len(self.barcode_tsv)


def get_OldST_dataloader(data_path, data_name, genes, patch_size=(112, 112)):
    import os
    dataset = []
    for temp in data_name:
        pre = f'{data_path}/ST-imgs/{temp[0]}/{temp}/'
        fig_name = os.listdir(pre)[0]
        image_path = pre + '/' + fig_name
        spatial_pos_path = f'./{data_path}/ST-spotfiles/{temp}_selection.tsv'
        reduced_mtx_path = f'./{data_path}/ST-cnts/ut_{temp}_stdata_filtered.tsv'
        dataset_temp = OldSTDataset(image_path=image_path, spatial_pos_path=spatial_pos_path,
                                    expression_path=reduced_mtx_path, dataset_name=temp, genes=genes,
                                    patch_size=patch_size)
        dataset.append(dataset_temp)
    dataset = torch.utils.data.ConcatDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0,
                            pin_memory=True, drop_last=False)
    return dataloader
