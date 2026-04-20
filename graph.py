import pandas as pd
import numpy as np
import torch
import re

from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

def create_edges(n_wavelengths):
    edge_index = []
    for i in range(n_wavelengths - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() #перестраивает в формат ребер, где
                                                                             #первый массив - начало ребра.
                                                                             # второй - конец
    return edge_index

def create_graphs(dataset):
    n_wavelengths = 106
    src_cols = [col for col in dataset.columns if 'leafs_src_mean_' in col]
    minmax_cols = [col for col in dataset.columns if 'leafs_minmax_mean_' in col]
    std_cols = [col for col in dataset.columns if 'leafs_std_mean_' in col]

    graphs = []
    le = LabelEncoder()
    dataset["class"] = le.fit_transform(dataset["class"])

    for idx, row in dataset.iterrows():
        node_features = []
        for i in range(n_wavelengths):
            src_val = row[src_cols[i]]
            minmax_val = row[minmax_cols[i]]
            std_val = row[std_cols[i]]
            node_features.append([src_val, minmax_val, std_val])

        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor([row['class']], dtype=torch.long)  # целевая переменная

        graph = Data(x=x, y=y)
        graphs.append(graph)

    return graphs


class Graph:
    def __init__(self, path):
        n_wavelengths = 106
        self.dataset = pd.read_csv(path,index_col=0).drop(columns=['dai', 'box_i', 'class_generalized','file_path',
                                                                   'sort'])

        unique_ds = self.dataset['ds_name'].unique()
        data_train = self.dataset.loc[self.dataset['ds_name'].isin(unique_ds[2:])]
        data_test = self.dataset.loc[self.dataset['ds_name'].isin(unique_ds[:2])]

        self.edge_index = create_edges(n_wavelengths)
        self.graphs_test = create_graphs(data_test)
        self.graphs_train = create_graphs(data_train)
        self.list_graph = self.graphs_train + self.graphs_test


# grap_a = Graph("C:\ML\hyperspectr\Spectr\Data\hyper_wheat_ds_ch_norm_prep_mode=dai.csv")
# print(grap_a.list_graph[0])
# print(grap_a.list_graph[1].x[0])
# print(grap_a.list_graph[14])
# print(grap_a.edge_index[:, :10])
# print(grap_a.list_graph[0].y)
# print(grap_a.graphs_test[0])
