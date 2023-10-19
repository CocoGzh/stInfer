import os
import sys
import torch
import datetime
import shutil

import scanpy as sc
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from .dataset import TenXDataset, OldSTDataset
from .model import ClassModel
from .config import config
from .dataset import get_TenX_dataloader, get_OldST_dataloader


def print_log(info):
    time_now = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    print("\n" + "==========" * 8 + "%s" % time_now)
    print(str(info) + "\n")


class Trainer(object):

    def __init__(self,
                 genes=None,
                 train_dataloader=None,
                 test_dataloader=None,
                 result_path=f"./log/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
                 model_name='model'
                 ):

        # dataloader
        self.genes = genes
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # model & train
        self.model = ClassModel(len(self.genes), config.model_name, pretrained=config.pretrained)
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(
            [{'params': [item[1] for item in list(self.model.named_parameters()) if 'image_header' in item[0]],
              'lr': config.header_lr},
             {'params': [item[1] for item in list(self.model.named_parameters()) if 'image_header' not in item[0]],
              'lr': config.encoder_lr}])
        self.StepLR = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=10)
        self.early_stop = None
        self.epoch = config.epoch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_switch = config.train_switch

        # result & log
        # time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        # self.result_path = f"./log/{time_now}"
        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        shutil.copyfile('./stInfer/config.py', f'{self.result_path}/config.py')
        self.log = pd.DataFrame()
        self.model_name = model_name

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, name=''):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(1, self.epoch + 1):
            print_log(f"Epoch {epoch} / {self.epoch}")
            # 1，train -------------------------------------------------
            total_loss, step = 0, 0
            loop = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), file=sys.stdout)
            expression_list = []
            image_list = []
            pred_expr_list = []
            dataset_barcode_list = []
            for i, batch in loop:
                features, labels = batch['image'], batch['expression']
                # =========================移动数据到GPU上==============================
                features = features.to(self.device)
                labels = labels.to(self.device)
                # ====================================================================
                # forward
                hidden, pred = self.model(features)
                # backward
                loss = self.loss(pred, labels)

                if self.train_switch:
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()

                total_loss += loss.item()
                step += 1

                expression_list.append(batch['expression'].detach().cpu().numpy())
                dataset_barcode_list.extend(batch['dataset_barcode'])
                image_list.append(hidden.detach().cpu().numpy())
                pred_expr_list.append(pred.detach().cpu().numpy())

            self.log.loc[epoch, "train_loss"] = total_loss

        adata_train = sc.AnnData(X=np.concatenate(expression_list, axis=0))
        adata_train.obs.index = dataset_barcode_list
        adata_train.var.index = self.genes
        adata_train.obsm['image_embedding'] = np.concatenate(image_list, axis=0)
        adata_train.obsm['pred_expr'] = np.concatenate(pred_expr_list, axis=0)

        self.log.to_csv(f'{self.result_path}/log.csv', header=True, index=True)
        torch.save(self.model.state_dict(), f'{self.result_path}/{self.model_name}.pth')
        adata_train.obs.loc[:, 'batch'] = [item.split('_')[0] for item in adata_train.obs.index]
        adata_train.obs.loc[:, 'barcode'] = [item.split('_')[1] for item in adata_train.obs.index]
        adata_train.var.loc[:, 'symbol'] = adata_train.var.index.tolist()
        adata_train.write_h5ad(f'{self.result_path}/adata_train_{name}.h5ad', compression='gzip')
        return adata_train

    def test(self, name=''):
        # 2，test -------------------------------------------------
        self.model.to(self.device)
        self.model.eval()
        total_loss, step = 0, 0
        loop = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), file=sys.stdout)
        # test_pred_dict = {name: [] for name, adata in self.test_dict.items()}  # TODO 如何构建dataloader，写一个sampler

        expression_list = []
        dataset_barcode_list = []
        image_list = []
        pred_expr_list = []

        with torch.no_grad():
            for i, batch in loop:
                features, labels = batch['image'], batch['expression']
                dataset_name, barcode = batch['dataset_name'], batch['barcode']
                # =========================移动数据到GPU上==============================
                features = features.to(self.device)
                labels = labels.to(self.device)
                # ====================================================================
                # forward
                hidden, pred = self.model(features)
                loss = self.loss(pred, labels)
                # log
                total_loss += loss.item()
                step += 1

                expression_list.append(batch['expression'].detach().cpu().numpy())
                dataset_barcode_list.extend(batch['dataset_barcode'])
                image_list.append(hidden.detach().cpu().numpy())
                pred_expr_list.append(pred.detach().cpu().numpy())

        self.log.to_csv(f'{self.result_path}/log.csv', header=True, index=True)

        adata_test = sc.AnnData(X=np.concatenate(expression_list, axis=0))
        adata_test.obs.index = dataset_barcode_list
        adata_test.var.index = self.genes
        adata_test.obsm['image_embedding'] = np.concatenate(image_list, axis=0)
        adata_test.obsm['pred_expr'] = np.concatenate(pred_expr_list, axis=0)

        adata_test.obs.loc[:, 'batch'] = [item.split('_')[0] for item in adata_test.obs.index]
        adata_test.obs.loc[:, 'barcode'] = [item.split('_')[1] for item in adata_test.obs.index]
        adata_test.var.loc[:, 'symbol'] = adata_test.var.index.tolist()
        adata_test.write_h5ad(f'{self.result_path}/adata_test_{name}.h5ad', compression='gzip')
        return adata_test
