import argparse

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
    LaborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
import time
import torch.cuda as cuda
import sys
sys.path.append("../../..")
from dgl.utils.internal import recursive_apply
from dgl.heterograph import *
from queue import Queue
# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.profiler import profile, record_function, ProfilerActivity

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


def train(args, device, g, sampler_, batch, dataset, model, num_classes, workers, cgg, cache_size):
    # create sampler & dataloader
    device = torch.device("cuda")
    # train_idx = dataset.train_idx.to(device)
    train_idx = dataset.train_idx
    mini_batch = (train_idx.shape[0] + batch - 1) // batch
    
    sampler = ""
    if sampler_ == 0:
        # print("fns")
        sampler = NeighborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    if sampler_ == 1:
        # print("nbr")
        sampler = NeighborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            fused = False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    if sampler_ == 2:
        # print("lbr")
        sampler = LaborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )
    use_uva = args.mode == "mixed"
    # cgg = False

    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        num_workers=workers,
        use_uva=False,
        persistent_workers=True if workers > 0 else False,
        gpu_cache={"node": {"feat": cache_size}},
        # skip_mfg=True,
        # mfg_buffer=Queue(mini_batch * 4),
        cgg_on_demand=True,
        pin_prefetcher=False,
        # gather_pin_only=True,
    )
    # print(torch.multiprocessing.get_sharing_strategy())
    # torch.set_num_threads(64)
    print("Dataloaded")
    # if workers > 0:
    #     for it, (input_nodes, output_nodes, blocks) in enumerate(
    #             train_dataloader
    #         ):
    #         break

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    file = open("./cggcache.txt", "a")
    
    start_ = time.time()
    tot = 0
    # with profile(activities=[
    #     ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for epoch in range(5):
        model.train()
        total_loss = 0
        start = time.time()
        count = count1 = 0
        step = 0
        total_nodes = 0
        print("start")
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            total_nodes += input_nodes.shape[0]
            start1 = time.time()
            if train_dataloader.cgg_on_demand == True:
                x = train_dataloader._cgg_on_demand("feat", "_N", blocks[0].srcdata["_ID"])
                y = train_dataloader._cgg_on_demand("label", "_N", blocks[-1].dstdata["_ID"])
            else:
                x = blocks[0].srcdata["feat"]
                y = blocks[-1].dstdata["label"]

            end1 = time.time()
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            end2 = time.time()
            step = step + 1
            
            count = count + end2 - start1
            count1 = count1 + end1 - start1
            continue
        
        end = time.time()
        tot += end - start
        # print(end - start, total_nodes)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        file.write(str(batch) + "," + str(sampler_) + "," + str(cache_size) + "," + str(workers) + "," + 
                    str(end - start) + "\n")
    # print(tot)
    

def train_ggg(args, device, g, sampler_, batch, dataset, model, num_classes, workers, cgg, cache_size):
    # create sampler & dataloader
    device = torch.device("cuda")
    # train_idx = dataset.train_idx.to(device)
    train_idx = dataset.train_idx
    
    sampler = ""
    if sampler_ == 0:
        # print("fns")
        sampler = NeighborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )
  
    if sampler_ == 2:
        # print("lbr")
        sampler = LaborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )
    use_uva = args.mode == "mixed"
    # cgg = False
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
        persistent_workers=True if workers > 0 else False,
        gpu_cache={"node": {"feat": cache_size}},
        # skip_mfg=True,
        # cgg_on_demand=True,
        # gather_pin_only=True,
    )
    print("Dataloaded")
    if workers > 0:
        for it, (input_nodes, output_nodes, blocks) in enumerate(
                train_dataloader
            ):
            break

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    file = open("./ggg_nbr_cache.txt", "a")
    file.write(str(batch) + "," + str(sampler_) + "," + str(cache_size) + "," + str(workers) + ",")
    start_ = time.time()
    tot = 0
    for epoch in range(10):
        model.train()
        total_loss = 0
        start = time.time()
        count = count1 = 0
        step = 0
        print("start")
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            start1 = time.time()
            if train_dataloader.cgg_on_demand == True:
                x = train_dataloader._cgg_on_demand("feat", "_N", blocks[0].srcdata["_ID"])
                y = train_dataloader._cgg_on_demand("label", "_N", blocks[-1].dstdata["_ID"])
            else:
                x = blocks[0].srcdata["feat"]
                y = blocks[-1].dstdata["label"]

            end1 = time.time()
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            end2 = time.time()
            step = step + 1
            
            count = count + end2 - start1
            count1 = count1 + end1 - start1
            continue
        # print(step)
        tot += end - start
        end = time.time()
        file.write(str(end - start) + ",")
    print(tot)
    file.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )

    parser.add_argument(
      "--cgg",
      type=bool,
        default = True,
        choices = [True,False])
    
    parser.add_argument(
      "--sampler",
      type=int,
        default = 0,
        )
    
    parser.add_argument(
      "--workers",
      type=int,
        default = 0,
        )
    
    parser.add_argument(
      "--batch_size",
      type=int,
        default = 8192,
        )
    
    parser.add_argument(
      "--cache_size",
      type=int,
        default = 10000000,
        )
    
    parser.add_argument(
          "--hid_size",
          type=int,
            default = 256,
            )
    
    parser.add_argument(
      "--dataset",
      type=str,
        default = "ogbn-papers100M",
        )
    
    args = parser.parse_args()
    # if not torch.cuda.is_available():
    #     args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")
    start = time.time()
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root="/data/")
                                , save_dir="/data/tmp/")
    g = dataset[0]
    g.ndata["label"] = g.ndata["label"].type(torch.LongTensor)
    # g = g.to("cuda" if args.mode == "puregpu" else "cpu")

    
    num_classes = dataset.num_classes
    # device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    device = ""
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, args.hid_size, out_size)
    # print("Sharing memory")
    g_ = g.shared_memory("g")
    g_.ndata["label"] = g.ndata["label"].share_memory_()
    g_.ndata["feat"] = g.ndata["feat"].share_memory_()
    # print("Memory shared")
    # del g
    # processes = []
    # func = [train]
    # g.create_formats_()
    g_.pin_memory_()
    print(args.dataset, args.batch_size, time.time() - start)
    train(args, device, g_, args.sampler, args.batch_size, dataset, model, num_classes, args.workers, args.cgg, args.cache_size)
    # process_func(g, model, dataset.train_idx, 0, 8192, 0, 10000000)
    # exit(0)
    