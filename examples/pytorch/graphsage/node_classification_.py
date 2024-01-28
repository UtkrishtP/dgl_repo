import argparse
from unittest import skip

import dgl
import dgl.nn as dglnn
from numpy import block
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
from dgl.dataloading.dataloader import _record_stream
from dgl.heterograph import *
import threading
import torch.multiprocessing
# try:
#     torch.multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass
from pympler import asizeof
from pynvml import *

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

stream_event = None
blocks = []
prof = 0
def transfer_mfg(queue_, queue_read_event):

    while not queue_.empty():
        queue_read_event.set()
        blocks = queue_.get()
        if train_dataloader.cgg_on_demand == True:
                x = train_dataloader._cgg_on_demand("feat","_N",blocks[0].srcdata["_ID"])
                y = train_dataloader._cgg_on_demand("label","_N",blocks[-1].dstdata["_ID"])
        else:
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
        y_hat = model(blocks, x)
        loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        end2 = time.time()
        queue_read_event.clear()
        
    # stream = torch.cuda.Stream()
    # current_stream = torch.cuda.current_stream()
    # stream.wait_stream(current_stream)
    # global blocks
    # with torch.cuda.stream(stream):
    #     blocks = recursive_apply(
    #                         blocks, lambda x: x.to("cuda", non_blocking=True))
    
    #     blocks = recursive_apply(blocks, _record_stream, current_stream)
    # global stream_event
    # stream_event = stream.record_event() 
    # queue_.put(blocks)
    # return blocks

def train(args, device, g, sampler_, batch, dataset, model, num_classes, 
          workers, skip_mfg_, cache_size, epoch_):
    # create sampler & dataloader
    train_idx = dataset.train_idx
    sampler = ""
    global stream_event, blocks
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
        device="cuda",
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        num_workers=workers,
        use_uva=False,
        persistent_workers=True if workers > 0 else False,
        gpu_cache={"node": {"feat": cache_size}},
        skip_mfg=True,
        # cgg=True,
        pin_prefetcher=True,
        use_prefetch_thread=True,
        cgg_on_demand=True,
        # gather_pin_only=True,
    )

    if workers > 0:
        for it, (input_nodes, output_nodes, blocks) in enumerate(
                train_dataloader
            ):
            break

    # th = threading.Thread(target=util)
    # th.start()
    queue_read_event = torch.multiprocessing.Event()

    queue_ = torch.multiprocessing.Queue()
    p = torch.multiprocessing.Process(target=transfer_mfg, args=(queue_, queue_read_event))
    p.start()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    file = open("./profiler/cgg_cache_on_demand.txt", "a")
    start_ = time.time()
    for epoch in range(epoch_):
        model.train()
        total_loss = 0
        start = time.time()
        count = count1 = 0
        step = 0
        mfg_time = 0
        k = 0
        # while queue_read_event.is_set():
        #     continue
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            start1 = time.time()
            # if train_dataloader.skip_mfg == True:
            #     blocks = recursive_apply(
            #             blocks, lambda x: x.to("cuda", non_blocking=True))
               
            stream = torch.cuda.Stream()
            current_stream = torch.cuda.current_stream()
            stream.wait_stream(current_stream)
            global blocks
            with torch.cuda.stream(stream):
                blocks = recursive_apply(
                                    blocks, lambda x: x.to("cuda", non_blocking=True))
            
                blocks = recursive_apply(blocks, _record_stream, current_stream)
            global stream_event
            stream_event = stream.record_event()
            while queue_read_event.is_set():
                continue
            queue_.put(blocks)
            start2 = time.time() 
            mfg_time = mfg_time + start2 - start1  
            # continue       
            # if train_dataloader.cgg_on_demand == True:
            #     x = train_dataloader._cgg_on_demand("feat","_N",blocks[0].srcdata["_ID"])
            #     y = train_dataloader._cgg_on_demand("label","_N",blocks[-1].dstdata["_ID"])
            # else:
            #     x = blocks[0].srcdata["feat"]
            #     y = blocks[-1].dstdata["label"]
            # y_hat = model(blocks, x)
            # loss = F.cross_entropy(y_hat, y)
            # opt.zero_grad()
            # loss.backward()
            # opt.step()
            # total_loss += loss.item()
            # end2 = time.time()
            # step = step + 1
            
            # count = count + end2 - end1
            # count1 = count1 + end2 - start1
            # mfg_time = mfg_time + start2 - start1 

        # print(step)
        end = time.time()
        print(mfg_time)
        
        file.write(str(batch) + "," +
                   str(sampler_) + "," +
                   str(epoch) + "," + 
                   str(workers) + "," + 
                   str(end - start) + "," +
                   str(mfg_time) + "\n"
        )
    file.close()
    # end_ = time.time()
    # global prof
    # prof = 1
    # th.join()
    
def util():
    nvmlInit()
    gpu = 0 #GPU 0
    f = open('./gpu_mem_usage.txt', "a+")
    # f_ = open('/storage/utk/data/pure-sampling-mp/pcie_rx.txt', "a+")
    handle = nvmlDeviceGetHandleByIndex(gpu)
    #nvmlUtilization_t util
    # f.write("TS,HBM,PCI_TX,PCI_RX,DRAM,CPU\n")
    # f.write(str(stage) + ",")
    # f_.write(str(stage) + ",")
    global prof
    while prof == 0:
        # util = nvmlDeviceGetUtilizationRates(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        # f.write(str(timer()) + ","  + str(mem_info.used >> 20) + ',' +
        #         str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)) + "," +
        #         str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)) + "," +
        #         str(psutil.virtual_memory()[3] / (1 << 30)) + ',' + 
        #         str(psutil.cpu_percent()) + ",\n")
        # time.sleep(0.1)
        # f.write(str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)) + ",")
        # f_.write(str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)) + ",")
        f.write(str(mem_info.used >> 20) + ",")
        time.sleep(0.1)
        # f.write(str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_COUNT) / (10 ** 6)) + ",")
    
    f.write("\n")
    # f_.write("\n")
    nvmlShutdown()
    f.close()
def train_bg(args, device, g, sampler_, batch, dataset, model, num_classes, workers, skip_mfg_, cache_size, epoch_):
    # create sampler & dataloader
    train_idx = dataset.train_idx
    
    sampler = ""
    if sampler_ == 0:
        # print("fns")
        if skip_mfg_ == 1:
            sampler = NeighborSampler(
                [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
                prefetch_node_feats=["feat"],
                prefetch_labels=["label"],
            )
        else:   
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
        # gpu_cache={"node": {"feat": 10000000}},
        # skip_mfg=True,
        # cgg=cgg,
        # gather_pin_only=True,
    )

    if workers > 0:
        for it, (input_nodes, output_nodes, blocks) in enumerate(
                train_dataloader
            ):
            break

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    # file = open("./cgg_mfg_off.txt", "a")
    s = time.time()
    for epoch in range(1):
        model.train()
        total_loss = 0
        start = time.time()
        count = count1 = 0
        step = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            # start1 = time.time()
            # if train_dataloader.skip_mfg == True:
            #     blocks = recursive_apply(
            #             blocks, lambda x: x.to("cuda", non_blocking=True))
            # x = train_dataloader.cgg_on_demand("feat","_N",input_nodes)
            # y = train_dataloader.cgg_on_demand("label","_N",output_nodes)
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            # x = x.to(device)
            # y = y.to(device)
            # cuda.synchronize()
            # end1 = time.time()
            # y_hat = model(blocks, x)
            # loss = F.cross_entropy(y_hat, y)
            # opt.zero_grad()
            # loss.backward()
            # opt.step()
            # total_loss += loss.item()
            # end2 = time.time()
            # step = step + 1
            break
            # count = count + end2 - start1
            # count1 = count1 + end1 - start1
        # print(step)
        end = time.time()
        
        
    # e = time.time()
    # file.write(str(batch) + "," +
    #            str(e - s) + "\n")
    # file.close()

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
      "--skip_mfg",
      type=int,
        default = 0,
        # choices = [True,False]
    )
    
    parser.add_argument(
      "--sampler",
      type=int,
        default = 0,
        )
    
    parser.add_argument(
      "--epoch",
      type=int,
        default = 1,
        )
    
    parser.add_argument(
      "--workers",
      type=int,
        default = 0,
        )
    
    parser.add_argument(
      "--batch_size",
      type=int,
        default = 0,
        )
    
    parser.add_argument(
      "--cache_size",
      type=int,
        default = 5000000,
        )

    parser.add_argument(
      "--dataset",
      type=str,
        default = "ogbn-papers100M",
        )
    
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root="/storage/utk/dgl/examples/pytorch/graphsage/dataset/")
                                , save_dir="/disk1/tmp/")
    g = dataset[0]
    g.ndata["label"] = g.ndata["label"].type(torch.LongTensor)
    # g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    # print(g.formats())
    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size)

    # model training
    print("Training...")
    # train(args, device, g, 0, args.batch_size, dataset, model, num_classes, 0, args.skip_mfg, args.cache_size, args.epoch)
    
    train(args, device, g, args.sampler, args.batch_size, dataset, model, num_classes, args.workers, args.skip_mfg, args.cache_size, args.epoch)
    # train(args, device, g, 2, args.batch_size, dataset, model, num_classes, 16, args.skip_mfg, args.cache_size, args.epoch)
    exit(0)
    b = [2048, 1024, 512]
    for batch in reversed(b):
        print("batch size: ", batch)
    #     # train(args, device, g, args.sampler, batch, dataset, model, num_classes, args.workers, True)
        # train(args, device, g, 0, batch, dataset, model, num_classes, 0, True)
        train(args, device, g, 1, batch, dataset, model, num_classes, 16, True)
    
    for batch in reversed(b):
        print("batch size: ", batch)
    #     # train(args, device, g, args.sampler, batch, dataset, model, num_classes, args.workers, True)
        # train_bg(args, device, g, 0, batch, dataset, model, num_classes, 0, True)
        train_bg(args, device, g, 1, batch, dataset, model, num_classes, 16, True)