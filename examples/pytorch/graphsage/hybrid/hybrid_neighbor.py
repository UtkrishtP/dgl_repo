import argparse

import dgl
import dgl.nn as dglnn
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
# from dgl.dataloading.dataloader import *

# import os.path.join(os.getcwd(), "..", "..", "..", "python", "dgl", "utils")
# print(util_path)
# import util_path

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
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
'''
    prof :              global varibale to signal the end of the CPU sampler thread
    cpu_sampled_blocks: List of dictionary to store how many CPU samples have been generated
    cpu_samples :       Variable to keep track of total epochs samples by cpu

    cpu_sample():
        Function to continuously sample using CPU MP. The dataloader object passed has all the
        optimizations turned on. We have specifically turned off the feature prefetch part.
        MFGs are still being transferred as and when being created using cudaMemCpyAsync().
'''
prof = 0
cpu_sampled_blocks = {}
cpu_samples = 0

def cpu_sample(dataloader):
    global prof, cpu_samples
    while prof == 0:
        for it, (input_nodes, output_nodes, blocks) in enumerate(
                dataloader
            ):
                cpu_sampled_blocks[cpu_samples] = []
                cpu_sampled_blocks[cpu_samples].append(blocks)
                '''
                    Create a shared array where we store all the MFG's
                    The GPU dataloader will use these to train on it.
                '''
        cpu_samples = cpu_samples + 1
    done = 1

def hybrid(cpu_dataloader, gpu_dataloader, epoch_, mfg) :
    cpu_samples_processed = 0
    global prof
    prof = 0
    global cpu_samples
    cpu_samples = 0

    th = threading.Thread(target=cpu_sample, args=(cpu_dataloader,))
    th.start()

    epoch = epoch_
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    end1 = start1 = end2 = count = count1 = 0
    start = timer()
    while epoch > 0:

        model.train()
        total_loss = 0
        
        # Checking if new samples for an epoch have been produced, if yes then train this epoch directly
        while cpu_samples_processed < cpu_samples:
            
            for blocks in cpu_sampled_blocks[cpu_samples_processed]:

                if mfg == 0:
                    blocks = recursive_apply(
                        blocks, lambda x: x.to("cuda", non_blocking=True))
                    
                x = blocks[0].srcdata["feat"]

                start2 = timer()
                y = blocks[-1].dstdata["label"].type(torch.LongTensor).to(device)
                # y = blocks[-1].dstdata["label"]
                end2 = timer()

                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                count1 = count1 + (end2 - start2)
            
            cpu_samples_processed = cpu_samples_processed + 1
            epoch = epoch - 1
            
        # TODO: Overlap GPU sampling with the above process.
        # GPU UVA pipeline

        for it, (input_nodes, output_nodes, blocks) in enumerate(
            gpu_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            start2 = timer()
            y = blocks[-1].dstdata["label"].type(torch.LongTensor).to(device)
            # y = blocks[-1].dstdata["label"]
            end2 = timer()
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            count = count + (end2 - start2)
            # continue

        epoch = epoch - 1
        end1 = end1 + count
        # print("GPU trained : ", count)

    end = timer()
    prof = 1
    f.write(str(end - start - count - count1) + "," +
    str(cpu_samples) + "," + str(cpu_samples_processed) + "\n")
    th.join()    
    

def train(args, device, g, dataset, model, num_classes, f, batch_size, use_uva, _pin, _alternate, _prefetch_th, epoch_):
    # create sampler & dataloader

    '''
        Creating two different training INDICES, since we are using use_uva=True and False for both CPU and GPU training
        To initialize dataloaders 
            train indices and graph need to be on CPU for CPU sampling
            train_indices (GPU) and graph(CPU) for GPU sampling
        
        This is for the scenario of inter epoch sampling
    '''

    gpu_train_idx = dataset.train_idx.to(device)
    cpu_train_idx = dataset.train_idx

    gpu_neighbor_sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )

    cpu_neighbor_sampler_prefetch = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        fused=False,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )

    cpu_neighbor_sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        fused=False,
    )

    gpu_neighbor_dataloader = DataLoader(
        g,
        gpu_train_idx,
        gpu_neighbor_sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )

    cpu_neighbor_dataloader_prefetch_nfeats_mfg = DataLoader(
        g,
        cpu_train_idx,
        cpu_neighbor_sampler_prefetch,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=32,
        use_prefetch_thread = True,
        pin_prefetcher = True,
        use_alternate_streams = True,
        persistent_workers=True,
        skip_mfg=0,
    )

    cpu_neighbor_dataloader_prefetch_nfeats = DataLoader(
        g,
        cpu_train_idx,
        cpu_neighbor_sampler_prefetch,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=32,
        use_prefetch_thread = True,
        pin_prefetcher = True,
        use_alternate_streams = True,
        persistent_workers=True,
        skip_mfg=1,
    )

    cpu_neighbor_dataloader_prefetch_mfg = DataLoader(
        g,
        cpu_train_idx,
        cpu_neighbor_sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=32,
        use_prefetch_thread = True,
        pin_prefetcher = True,
        use_alternate_streams = True,
        persistent_workers=True,
        skip_mfg=0,
    )

    cpu_neighbor_dataloader = DataLoader(
        g,
        cpu_train_idx,
        cpu_neighbor_sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=32,
        use_prefetch_thread = True,
        pin_prefetcher = True,
        use_alternate_streams = False,
        persistent_workers=True,
        skip_mfg=1,
    )


    '''
        Looping over 1 iteration of CPU based MP sampling to compensate for process launch overhead.
        Once the processes are launched they will be re-used from the pool.
    '''
    for it, (input_nodes, output_nodes, blocks) in enumerate(
            cpu_neighbor_dataloader
        ):
            break
    
    for it, (input_nodes, output_nodes, blocks) in enumerate(
            cpu_neighbor_dataloader_prefetch_mfg
        ):
            break
    
    for it, (input_nodes, output_nodes, blocks) in enumerate(
            cpu_neighbor_dataloader_prefetch_nfeats
        ):
            break
    
    for it, (input_nodes, output_nodes, blocks) in enumerate(
            cpu_neighbor_dataloader_prefetch_nfeats_mfg
        ):
            break
    
    '''
        Profiling hybrid sampling using all 4 combinations of prefetch:
            - MFG + Nfeat
            - Nfeat
            - MFG
            - None
    '''

    hybrid(cpu_neighbor_dataloader, gpu_neighbor_dataloader, epoch_, 0)
    hybrid(cpu_neighbor_dataloader_prefetch_mfg, gpu_neighbor_dataloader, epoch_, 1)
    hybrid(cpu_neighbor_dataloader_prefetch_nfeats, gpu_neighbor_dataloader, epoch_, 0)
    hybrid(cpu_neighbor_dataloader_prefetch_nfeats_mfg, gpu_neighbor_dataloader, epoch_, 1)

if __name__ == "__main__":
    # f = open('/mnt/utk/data/dgl-latest/sage_node_class_product.txt', "a")
    f = open('/disk/results/hybrid/neighbor.txt', "a")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
      "--uva",
      type=bool,
        default = True,
        choices = [True,False])
    
    parser.add_argument(
      "--pin_prefetcher",
      type=bool,
        default = False,
        choices = [True,False])

    parser.add_argument(
      "--alternate_streams",
      type=bool,
        default = False,
        choices = [True,False])
    
    parser.add_argument(
      "--prefetch_thread",
      type=bool,
        default = False,
        choices = [True,False])

    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")
    # load and preprocess dataset
    print("Loading data")
    start = timer()
    
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-papers100M")) #, root="/storage/utk/dgl/examples/pytorch/graphsage/dataset/"))
    
    end = timer()
    

    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    g.pin_memory_()
    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # model training
    print("Training...")
    
    # train(args, device, g, dataset, model, num_classes, f, 1024, args.uva, args.pin_prefetcher, args.alternate_streams, args.prefetch_thread)
    b = [512, 1024, 2048, 4096, 8192]
    epoch_ = [10, 50]
    for batch in reversed(b):
        for epoch in epoch_:
            f.write(str(batch) + "," + str(epoch) + ",")
            train(args, device, g, dataset, model, num_classes, f, batch, args.uva, args.pin_prefetcher, args.alternate_streams, args.prefetch_thread, epoch)
        
    # # test the model
    # print("Testing...")
    # acc = layerwise_infer(
    #     device, g, dataset.test_idx, model, num_classes, batch_size=4096
    # )
    # print("Test Accuracy {:.4f}".format(acc.item()))
    # f.close()
