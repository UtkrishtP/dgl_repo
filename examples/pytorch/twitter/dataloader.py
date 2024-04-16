import argparse
import numpy as np
import torch
import os.path as osp

import dgl
from dgl.data import DGLDataset
import warnings
warnings.filterwarnings("ignore")

        
class Twitter(object):
    def __init__(self, root: str, size: str, in_memory: int, \
        classes: int, synthetic: int):
        self.dir = root
        self.size = size
        self.synthetic = synthetic
        self.in_memory = in_memory
        self.num_classes = classes

    def num_nodes(self):
        # return no of nodes
        return 41652230

    @property
    def node_feat(self) -> np.ndarray:
        num_nodes = self.num_nodes()
        emb = np.random.rand(num_nodes, 380).astype('f')
        return emb
    
    @property
    def node_label(self) -> np.ndarray:
        path = '/data/twitter/node_label_64.npy'
        node_labels = np.load(path)
        return node_labels
    
    @property
    def node_edge(self) -> np.ndarray:
        path = '/data/twitter/edge_idx.npy'
        return np.load(path)
    
class TwitterDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='TwitterDataset')

    def process(self):
        dataset = Twitter(root=self.dir, size=self.args.dataset_size, in_memory=self.args.in_memory, \
            classes=self.args.num_classes, synthetic=self.args.synthetic)

        node_features = torch.from_numpy(dataset.node_feat)
        node_edges = torch.from_numpy(dataset.node_edge)
        node_labels = torch.from_numpy(dataset.node_label).to(torch.long)

        self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        self.graph = dgl.remove_self_loop(self.graph)
        self.graph = dgl.add_self_loop(self.graph)

        n_nodes = node_features.shape[0]
        n_train = int(n_nodes * 0.025)
        n_val   = int(n_nodes * 0.025)
        
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)