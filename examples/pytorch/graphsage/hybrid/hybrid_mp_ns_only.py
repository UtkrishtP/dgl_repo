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

def transfer_mfg(sample):
    global cpu_sampled_blocks
    # stream = torch.cuda.Stream("cuda")
    # current_stream = torch.cuda.current_stream()
    # current_stream.wait_stream(stream)

    # with torch.cuda.stream(stream):
    for blocks in cpu_sampled_blocks[sample]:
        blocks = recursive_apply(
                    blocks, lambda x: x.to("cuda", non_blocking=True))
        # blocks = recursive_apply(blocks, _record_stream, current_stream)

def train(args, device, g, dataset, model, num_classes, f, batch_size, use_uva, _pin, _alternate, _prefetch_th, epoch_):
    # create sampler & dataloader

    '''
        Creating two different training INDICES, since we are using use_uva=True and False for both CPU and GPU training
        To initialize dataloaders 
            train indices and graph need to be on CPU for CPU sampling
            train_indices (GPU) and graph(CPU) for GPU sampling
        
        This is for the scenario of inter epoch sampling
    '''

    cpu_train_idx = dataset.train_idx

    # val_idx = dataset.val_idx.to(device)

    # Two different samplers and dataloaders have been created for GPU and CPU counterparts.
    cpu_fused_sampler_pre = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )

    cpu_fused_sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        # prefetch_node_feats=["feat"],
        # prefetch_labels=["label"],
    )

    cpu_neighbor_sampler_pre = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        fused=False,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )

    cpu_neighbor_sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        fused=False,
        # prefetch_node_feats=["feat"],
        # prefetch_labels=["label"],
    )

    cpu_labor_sampler_pre = LaborSampler(
        [10, 10, 10],
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )

    cpu_labor_sampler = LaborSampler(
        [10, 10, 10],
        # prefetch_node_feats=["feat"],
        # prefetch_labels=["label"],
    )

    cpu_fused_dataloader_prefetch_nfeats_mfg = DataLoader(
        g,
        cpu_train_idx,
        cpu_neighbor_sampler_pre,
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

    cpu_fused_dataloader_prefetch_nfeats = DataLoader(
        g,
        cpu_train_idx,
        cpu_neighbor_sampler_pre,
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

    cpu_fused_dataloader_prefetch_mfg = DataLoader(
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

    cpu_fused_dataloader = DataLoader(
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
        skip_mfg=1,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
   
    for it, (input_nodes, output_nodes, blocks) in enumerate(
            cpu_fused_dataloader_prefetch_nfeats
        ):
        break

    for it, (input_nodes, output_nodes, blocks) in enumerate(
        cpu_fused_dataloader_prefetch_nfeats_mfg
    ):
        break

    for it, (input_nodes, output_nodes, blocks) in enumerate(
            cpu_fused_dataloader
        ):
        break

    for it, (input_nodes, output_nodes, blocks) in enumerate(
        cpu_fused_dataloader_prefetch_mfg
    ):
        break

    '''
        Profiling timers for GPU UVA based neighbor/labor sampling based training
    '''
    end1 = 0
    start = timer()
    for epoch in range(epoch_):

        model.train()
        total_loss = 0
        end1 = start2 = end2 = count = 0

        for it, (input_nodes, output_nodes, blocks) in enumerate(
            cpu_fused_dataloader
        ):
            blocks = recursive_apply(
                        blocks, lambda x: x.to("cuda", non_blocking=True))
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            # count = count + (end2 - start2)
            # continue
    # end1 = end1 + count
    end = timer()
    # print(end - start)
    f.write(str(end - start - end1) + ",")

    start = timer()
    for epoch in range(epoch_):

        model.train()
        total_loss = 0
        end1 = start2 = end2 = count = 0

        for it, (input_nodes, output_nodes, blocks) in enumerate(
            cpu_fused_dataloader_prefetch_mfg
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            # count = count + (end2 - start2)
            # continue
    # end1 = end1 + count
    end = timer()
    # print(end - start)
    f.write(str(end - start - end1) + ",")

    start = timer()
    for epoch in range(epoch_):

        model.train()
        total_loss = 0
        end1 = start2 = end2 = count = 0

        for it, (input_nodes, output_nodes, blocks) in enumerate(
            cpu_fused_dataloader_prefetch_nfeats
        ):
            blocks = recursive_apply(
                        blocks, lambda x: x.to("cuda", non_blocking=True))
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            # count = count + (end2 - start2)
            # continue
    # end1 = end1 + count
    end = timer()
    # print(end - start)
    f.write(str(end - start - end1) + ",")

    start = timer()
    for epoch in range(epoch_):

        model.train()
        total_loss = 0
        end1 = start2 = end2 = count = 0

        for it, (input_nodes, output_nodes, blocks) in enumerate(
            cpu_fused_dataloader_prefetch_nfeats_mfg
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            # count = count + (end2 - start2)
            # continue
    # end1 = end1 + count
    end = timer()
    # print(end - start)
    f.write(str(end - start - end1) + "\n")

    
if __name__ == "__main__":
    # f = open('/mnt/utk/data/dgl-latest/sage_node_class_product.txt', "a")
    f = open('/disk/results/hybrid/fused_mp_ns_only.txt', "a")
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
    
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-papers100M"), save_dir="/disk1/tmp/") #, root="/storage/utk/dgl/examples/pytorch/graphsage/dataset/"))
    
    end = timer()

    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    g.pin_memory_()
    # create GraphSAGE model
    g.ndata["label"] = g.ndata["label"].type(torch.LongTensor)
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # model training
    print("Training...")
    
    # train(args, device, g, dataset, model, num_classes, f, 1024, args.uva, args.pin_prefetcher, args.alternate_streams, args.prefetch_thread)
    b = [512, 2048, 8192]
    # b = [8192]
    epoch_ = [10, 20]
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
