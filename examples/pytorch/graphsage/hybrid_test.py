import argparse
from platform import node
import queue
from socket import timeout
from turtle import done
from dgl import cuda, data
from dgl.ndarray import cpu

# import dgl
import dgl.nn as dglnn
from dgl.utils.pin_memory import pin_memory_inplace
from numpy import block
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
import torch.profiler
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
    LaborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from timeit import default_timer as timer
import threading
import os
import sys
sys.path.append("../../..")
from dgl.utils.internal import recursive_apply
from dgl.heterograph import *
import time
import torch.multiprocessing
# try:
#     torch.multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass
import torch.utils.data._utils.pin_memory as pm
from pynvml import *
import psutil
import multiprocessing

# from dgl.dataloading.dataloader import *

# import os.path.join(os.getcwd(), "..", "..", "..", "python", "dgl", "utils")
# print(util_path)
# import util_path
torch.multiprocessing.set_sharing_strategy('file_system')

th = 0
def util():
    f = open("./dram.txt", "a")
    while th == 0:
        f.write(f'{psutil.virtual_memory()[3] / (1024**3):.2f},')
        time.sleep(1)
    f.write('\n')
    f.close()

def process_func(g):
    # g.create_formats_()
    return
    # g.pin_memory_()
    # torch.cuda.synchronize()

def process_func2(g): 
    # print(g.ndata['label'].device)
    return
    # g.pin_memory_() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default = "ogbn-arxiv",
        )
    

    args = parser.parse_args()
    print(f'Used DRAM: {psutil.virtual_memory()[3] / (1024**3):.2f} GB')
    # t = threading.Thread(target=util)
    # t.start()
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root="/storage/utk/dgl/examples/pytorch/graphsage/dataset/")
                                    , save_dir="/disk1/tmp/")    
    # end = timer()
    
    g = dataset[0]
    # print(g)
    # print(g._graph)
    # print(g._node_frames)
    print("Data loaded")
    
    print(f'Used DRAM: {psutil.virtual_memory()[3] / (1024**3):.2f} GB')
    g.create_formats_()
    print("formats created")
    exit(0)
    # g.pin_memory_()
    # device = torch.device('cuda')

    # in_size = g.ndata["label"].to(device)
    out_size = dataset.num_classes
    # model = SAGE(in_size, 256, out_size).to(device)

    print(f'Used DRAM: {psutil.virtual_memory()[3] / (1024**3):.2f} GB')
    '''
        g._graph.share_memory_()
        Create a custom function to share memory for all adjacency matrices.
        Check unit_graph.cc : 1341
    '''
    # for frames in g._node_frames:
    #     for key in frames.keys():
    #         frames[key] = frames[key].share_memory_()
    print("memory shared")
    print(f'Used DRAM: {psutil.virtual_memory()[3] / (1024**3):.2f} GB')
    # th = 1
    # t.join()
    # exit(0)
    # Launch two processes
    processes = []
    func = [process_func, process_func2]
    for _p in func:
        
        p = multiprocessing.Process(target=_p, args=(g,))
        print("start", p)
        p.start()
        processes.append(p)
    
    # Wait for the processes to finish
    for p in processes:
        p.join()
    # th = 1
    # t.join()