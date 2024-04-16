import argparse
import numpy as np
import torch
import os.path as osp

import dgl
from dgl.data import DGLDataset
import warnings
warnings.filterwarnings("ignore")


class Friendster(object):
    def __init__(self, root: str, size: str, in_memory: int, \
        classes: int, synthetic: int):
        self.dir = root
        self.size = size
        self.synthetic = synthetic
        self.in_memory = in_memory
        self.num_classes = classes

    def num_nodes(self):
        # return no of nodes
        return 65608366

    @property
    def node_feat(self) -> np.ndarray:
        num_nodes = self.num_nodes()
        emb = np.random.rand(num_nodes, 256).astype('f')
        return emb
    
    @property
    def node_label(self) -> np.ndarray:
        path = '/data/friendster-dataset/node_label_64.npy'
        node_labels = np.load(path)
        return node_labels
    
    @property
    def node_edge(self) -> np.ndarray:
        path = '/data/friendster-dataset/edge_idx_reassigned.npy'
        return np.load(path)
    
class FriendsterDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='FriendsterDataset')

    def process(self):
        dataset = Friendster(root=self.dir, size=self.args.dataset_size, in_memory=self.args.in_memory, \
            classes=self.args.num_classes, synthetic=self.args.synthetic)

        node_features = torch.from_numpy(dataset.node_feat)
        node_edges = torch.from_numpy(dataset.node_edge)
        node_labels = torch.from_numpy(dataset.node_label).to(torch.long)

        self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        self.graph = dgl.remove_self_loop(self.graph)
        self.graph = dgl.add_self_loop(self.graph)

        # np.random.seed(20)
        n_nodes = node_features.shape[0]
        n_train = int(n_nodes * 0.016)
        n_val   = int(n_nodes * 0.025)
        # mask_indices = np.random.permutation(n_nodes)[:n_train]
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[:n_train] = True
        # train_mask[mask_indices] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)

