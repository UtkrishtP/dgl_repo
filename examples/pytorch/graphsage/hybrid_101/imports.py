import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.backend as F_
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    DataLoaderCGG,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
    LaborSampler,
    SAINTSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
import time, psutil
from concurrent.futures import ThreadPoolExecutor
from dgl.utils.internal import recursive_apply, set_num_threads
from dgl.createshm import create_shmarray, create_shmoffset, reset_shm, get_shm_ptr, print_offset, read_offset
import ctypes
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
from dgl.convert import hetero_from_shared_memory
from dgl.utils.pin_memory import pin_memory_inplace
from custom_dl import FriendsterDataset, TwitterDataset, IGB260MDGLDataset
from args import get_args
from load_dataset import fetch_train_graph, fetch_all, fetch_shapes
from queue import Empty, Full, Queue
from pynvml import *
import subprocess, os
from datetime import datetime
import math
from tabulate import tabulate
from mps_utils import *
from torch.utils.data._utils.worker import _IterableDatasetStopIteration
import utils as util
os.environ["DGL_BENCH_DEVICE"] = "gpu"
# @util.benchmark("time")
# sys.path.append("/media/utkrisht/deepgraph/")
# from benchmarks.benchmarks import utils
SHARED_MEM_METAINFO_SIZE_MAX = 1024 * 64
SHARED_MEM_GPU_METAINFO_SIZE_MAX = (1024 * 48) + (3 * 64)
MIN_VALUE_LONG = -(2 ** (ctypes.sizeof(ctypes.c_long) * 8 - 1))
class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        for _ in range(num_layers-2):
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, aggregator_type='mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y

def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )

def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )