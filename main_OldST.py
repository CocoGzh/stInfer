import pandas as pd
import scanpy as sc
import numpy as np
from stInfer.trainer import Trainer
from stInfer.untils import cal_pearsonr
from stInfer.dataset import get_OldST_dataloader, get_TenX_dataloader

if __name__ == '__main__':
    all_name = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6',
                'B1', 'B2', 'B3', 'B4', 'B5', 'B6',  # error B3
                'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
                'D1', 'D2', 'D3', 'D4', 'D5', 'D6',
                'E1', 'E2', 'E3',  # error
                'F1', 'F2', 'F3',  # error
                'G1', 'G2', 'G3',
                'H1', 'H2', 'H3', ]  # error
    # all_name = ['A1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1']
    # all_name = ['A1']
    test_name = ['B1']
    train_name = list(set(all_name) - set(test_name))

    genes = pd.read_csv('./data/TenX_HVG.csv', index_col=None, header=None).iloc[:, 0].to_numpy()
    # genes = pd.read_csv('./data/OldST_HVG.csv', index_col=None, header=None).iloc[:, 0].to_numpy()

    data_path_train = './data/breast_OldST'
    train_dataloader = get_OldST_dataloader(data_path=data_path_train, data_name=train_name, genes=genes, patch_size=(112, 112))
    trainer = Trainer(genes=genes, train_dataloader=train_dataloader, result_path='./log/OldST_All_B1_train')
    adata_train = trainer.train(name='All')

    data_path_test = './data/breast_OldST'
    test_dataloader = get_OldST_dataloader(data_path=data_path_test, data_name=test_name, genes=genes, patch_size=(112, 112))
    trainer.test_dataloader = test_dataloader
    adata_test = trainer.test(name='B1')
