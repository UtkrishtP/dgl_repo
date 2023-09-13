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
)
from ogb.nodeproppred import DglNodePropPredDataset
from timeit import default_timer as timer
from pynvml import *
import threading
import psutil
import time
prof = 0
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

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


def train(args, device, g, f, f1, gpu, batch, workers, pin, prefetch, stream, dataset, model, num_classes):
    
    # create sampler & dataloader
    use_uva = False
    # train_idx = dataset.train_idx.to(device)
    # val_idx = dataset.val_idx.to(device)
    # use_uva = True    

    train_idx = dataset.train_idx
    val_idx = dataset.val_idx
   
    # train_idx = dataset.train_idx.to(device)
    sampler = ""
    if gpu == 0:
        sampler = NeighborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            # fused = False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    else:
        sampler = NeighborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            fused = False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
    )
    # use_uva = args.mode == "mixed"
    start = timer()
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=False,
        pin_prefetcher=True,
        use_prefetch_thread=True,
        use_alternate_streams=True,
        # persistent_workers=True,    
    )
    end = timer()
    # f.write("\nTrain Dataloader object : " + str(end - start))
    s = str(batch) + "_" + str(workers)
    # for it, (input_nodes, output_nodes, blocks) in enumerate(
    #         train_dataloader
    #     ):
    #     break
    # th1 = threading.Thread(target=util, args=(str(batch) + "_" + str(workers),))
    # global prof
    # prof = 0
    # th1.start()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sample = 0

    for epoch in range(2):
        count = 0
        count1 = 0
        step = 0
        model.train()
        total_loss = 0
        start = timer()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            # step = step + 1
            # start1 = timer()
            # x = blocks[0].srcdata["feat"]
            # start2 = timer()
            # y = blocks[-1].dstdata["label"].type(torch.LongTensor).to(device)
            # # y = blocks[-1].dstdata["label"]
            # end2 = timer()
            # y_hat = model(blocks, x)
            # loss = F.cross_entropy(y_hat, y)
            # opt.zero_grad()
            # loss.backward()
            # opt.step()
            # total_loss += loss.item()
            continue
            
            # end1 = timer()
            # # print("Training," + str(it) + "," + str(start1) + "," + str(end1))
            # count = count + (end1 - start1)
            # count1 = count1 + (end2 - start2)
            

        end = timer()
        print(s + "_" + str(gpu) + "_" + str(epoch) + "," + str((end - start)) + ",")
        # f1.write(s + "_" + str(gpu) + "_" + str(epoch) + "," + str((end - start)) + ","
        # + str(end - start) + ","
        # + str(count - count1) + ","
        # + str(count1) + ","
        # + str(train_dataloader.pre)
        # str(sampler.nsc) + "," +
        # str(sampler.mfg) + "," +
        # + str(step)
        # + str(train_dataloader.sample) 
        # + ",\n")
        # sampler.mfg = sampler.nsc = 0
    # global prof
    # prof = 1
    # th1.join()
    return

def train_(args, device, g, f, f1, gpu, batch, workers, pin, prefetch, stream, dataset, model, num_classes):
    
    # create sampler & dataloader
    # use_uva = True
    # train_idx = dataset.train_idx.to(device)
    # val_idx = dataset.val_idx.to(device)
    # use_uva = True    

    train_idx = dataset.train_idx
    # val_idx = dataset.val_idx
   
    # train_idx = dataset.train_idx.to(device)
   
    sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        # fused = False,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    
    # use_uva = args.mode == "mixed"
    start = timer()
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=False,
        pin_prefetcher=True,
        use_prefetch_thread=True,
        use_alternate_streams=True,
        # persistent_workers=True,    
    )
    end = timer()
    # f.write("\nTrain Dataloader object : " + str(end - start))
    s = str(batch) + "_" + str(workers)
    # th1 = threading.Thread(target=util, args=(str(batch) + "_" + str(workers),))
    # global prof
    # prof = 0
    # th1.start()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sample = 0
    for epoch in range(2):
        count = 0
        count1 = 0
        step = 0
        model.train()
        total_loss = 0
        start = timer()
        # print("Starting Epoch,", start)
        # for it, (input_nodes, output_nodes, blocks) in enumerate(
        #     train_dataloader
        # ):
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            # step = step + 1
            start1 = timer()
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
            # continue
            
            end1 = timer()
            # print("Training," + str(it) + "," + str(start1) + "," + str(end1))
            count = count + (end1 - start1)
            count1 = count1 + (end2 - start2)
            

        end = timer()
        f1.write(s + "_" + str(epoch) + "_" + str(gpu) + "," + str((end - start)) + ","
        # + str(train_dataloader.sample) + ","
        + str(end - start - count - count1) + ","
        + str(count - count1) + ","
        + str(count1) + ","
        + str(train_dataloader.pre)
        # str(sampler.nsc) + "," +
        # str(sampler.mfg) + "," +
        # + str(step)
        # + str(train_dataloader.sample) 
        + ",\n")
        # sampler.mfg = sampler.nsc = 0
        # train_dataloader.pre = 0
        # train_dataloader.sample = 0
    # global prof
    # prof = 1
    # th1.join()
    return
def util(stage):
    nvmlInit()
    gpu = 0 #GPU 0
    f = open('/storage/utk/data/pure-sampling-mp/pcie_count.txt', "a+")
    # f_ = open('/storage/utk/data/pure-sampling-mp/pcie_rx.txt', "a+")
    handle = nvmlDeviceGetHandleByIndex(gpu)
    #nvmlUtilization_t util
    # f.write("TS,HBM,PCI_TX,PCI_RX,DRAM,CPU\n")
    f.write(str(stage) + ",")
    # f_.write(str(stage) + ",")
    global prof
    while prof == 0:
        util = nvmlDeviceGetUtilizationRates(handle)
        # mem_info = nvmlDeviceGetMemoryInfo(handle)
        # f.write(str(timer()) + ","  + str(mem_info.used >> 20) + ',' +
        #         str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)) + "," +
        #         str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)) + "," +
        #         str(psutil.virtual_memory()[3] / (1 << 30)) + ',' + 
        #         str(psutil.cpu_percent()) + ",\n")
        # time.sleep(0.1)
        # f.write(str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)) + ",")
        # f_.write(str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)) + ",")
        time.sleep(0.1)
        f.write(str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_COUNT) / (10 ** 6)) + ",")
    
    f.write("\n")
    # f_.write("\n")
    nvmlShutdown()
    f.close()
    # f_.close()
    

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
      "--uva",
      type=bool,
        default = True,
        choices = [True,False])

    parser.add_argument(
      "--pin_prefetcher",
      type=int,
        default = 0,
        )

    parser.add_argument(
      "--alternate_streams",
      type=int,
        default = 1,
        )

    parser.add_argument(
      "--prefetch_thread",
      type=int,
        default = 0,
        )

    parser.add_argument(
      "--batch_size",
      type=int,
        default = 1024)

    parser.add_argument(
      "--workers",
      type=int,
        default = 0)
    
    parser.add_argument(
        "--dir",
        default=[""],
        choices=["0000", "1000", "1101", "0010", "0101"]
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0
    )

    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"

    f = open('/disk/results/pure-sampling-uva/arxiv.txt', "a+")
    f1 = open('/disk/results/pure-sampling-uva/fused/arxiv.txt', "a+")
    
    start = timer()
    #dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-papers100M"), save_dir="./dataset/ogbn_papers100M")
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-papers100M", root="/storage/utk/dgl/examples/pytorch/graphsage/dataset/"))
    end = timer()
    # f.write(str(end - start) + ",")
    
    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    # g = torch.tensor(g).to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)
    g.pin_memory_()

    # b = [512, 1024, 2048, 4096, 8192]
    b = [2048, 4096]
    # b = [16384, 32768, 65536, 128000, 256000]
    # step = [8, 16, 32, 64]
    # step = [4, 8, 16, 32]
    step = [0]
    # train(args, device, g, f, f1, args.sample, args.batch_size, args.workers, bool(args.pin_prefetcher), 
    #       bool(args.prefetch_thread), bool(args.alternate_streams), dataset, model, num_classes)
    for batch in reversed(b):
      for workers in step:
        train(args, device, g, f, f1, 0, batch, workers, bool(args.pin_prefetcher), 
          bool(args.prefetch_thread), bool(args.alternate_streams), dataset, model, num_classes)
        
        train(args, device, g, f, f1, 1, batch, workers, bool(args.pin_prefetcher), 
          bool(args.prefetch_thread), bool(args.alternate_streams), dataset, model, num_classes)
        # train(args, device, g, f, f1, 1, batch, workers, bool(args.pin_prefetcher), 
        #   bool(args.prefetch_thread), bool(args.alternate_streams), dataset, model, num_classes)
        
        # train_(args, device, g, f, f1, 2, batch, workers, bool(args.pin_prefetcher), 
        #   bool(args.prefetch_thread), bool(args.alternate_streams), dataset, model, num_classes)
    # val = 1
    # sample = 0
    # while val == 1 :
    #     train(args, device, g, f, f1, sample, args.batch_size, args.workers, bool(args.pin_prefetcher), 
    #       bool(args.prefetch_thread), bool(args.alternate_streams), dataset, model, num_classes)
        
    #     sample = int(input("Prefetch?"))
    #     val = int(input("Continue?"))
    # th1.join()
    # print("\n==================================================================================\n")
    # test the model
    # print("Testing...")
    # acc = layerwise_infer(
    #     device, g, dataset.test_idx, model, num_classes, batch_size=4096
    # )
    # print("Test Accuracy {:.4f}".format(acc.item()))
