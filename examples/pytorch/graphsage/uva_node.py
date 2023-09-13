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

from timeit import default_timer as timer
from pynvml import *
import threading
import subprocess
prof = 0
import psutil, time
from ogb.lsc import MAG240MDataset
import sys
sys.path.append("../../..")
from dgl.heterograph import *
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


def train(args, device, g, f, gpu, batch, workers, pin, prefetch, dataset, model, num_classes):
    
    # create sampler & dataloader
    use_uva = True
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    
    # train_idx = dataset.train_idx
    # val_idx = dataset.val_idx
    
    # use_uva = True    

    # if gpu == 0:
    #     train_idx = dataset.train_idx
    #     val_idx = dataset.val_idx
    # else:
    #     train_idx = dataset.train_idx.to(device)
    #     val_idx = dataset.val_idx.to(device)
    #     use_uva = True
    sampler = ""
    if gpu == 1:
        sampler = NeighborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            # fused = True,
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )
    else:
        sampler = NeighborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            # fused = True,
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
        use_uva=use_uva,
        # use_prefetch_thread=True,
    )
    end = timer()
    # f.write("\nTrain Dataloader object : " + str(end - start))
    # y = input("loading done")
    # th1 = threading.Thread(target=util, args=(str(batch),))
    # global prof
    # prof = 0
    # th1.start()

    # opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(3):
        count = 0
        count1 = 0
        step = 0
        # start = timer()
        model.train()
        total_loss = 0
        # print("Start,",timer())
        start = timer()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            
            # break
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
            # if step == 2:
            # break
            continue
            # end1 = timer()
            # count = count + (end1 - start1)
            # count1 = count1 + (end2 - start2)
            
        end = timer()
        
        
        f.write(str(batch) + "_" + str(gpu) + "_" + str(epoch) + "," + 
                str(train_dataloader.sample) + "," +
                str((end - start)) + "," +
                # str(sampler.nsc) + "," +
                # str(sampler.mfg) + "," +
                # str(step) + "," +
        #         str(count - count1) + "," +
        #         str(count1) + "," +
                str(train_dataloader.pre) + "\n")
        # sampler.mfg = sampler.nsc = 0
        
    # prof = 1
    # th1.join()

def train_(args, device, g, f, gpu, batch, workers, pin, prefetch, dataset, model, num_classes):
    
    # create sampler & dataloader
    use_uva = True
    train_idx = dataset.train_idx.to(device)
    # val_idx = dataset.val_idx.to(device)

    # train_idx = dataset.train_idx
    # use_uva = True    

    # if gpu == 0:
    #     train_idx = dataset.train_idx
    #     val_idx = dataset.val_idx
    # else:
    #     train_idx = dataset.train_idx.to(device)
    #     val_idx = dataset.val_idx.to(device)
    #     use_uva = True

    sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        # fused = True,
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
        use_uva=use_uva,
        # use_prefetch_thread=True,
        
    )
    end = timer()
    # f.write("\nTrain Dataloader object : " + str(end - start))

    # th1 = threading.Thread(target=util_, args=(str(batch),))
    # global prof
    # prof = 0
    # th1.start()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    tmp = 0
    for epoch in range(3):
        count = 0
        count1 = 0
        step = 0
        # start = timer()
        model.train()
        total_loss = 0
        # print("Start,",timer())
        start = timer()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            
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
            # break
            end1 = timer()
            count = count + (end1 - start1)
            count1 = count1 + (end2 - start2)
            
        end = timer()
    
        
        f.write(str(batch) + "_" + str(gpu) + "_" + str(epoch) + "," + 
                str(train_dataloader.sample) + "," +
                str((end - start)) + "," +
                # str(sampler.nsc) + "," +
                # str(sampler.mfg) + "," +
                # str(step) + "," +
                str(count - count1) + "," +
                str(count1) + "," +
                str(train_dataloader.pre) + "\n")
        tmp = train_dataloader.sample
        # sampler.mfg = sampler.nsc = 0
        # start = timer()
        # end = timer()
        # f.write("\nAccuracy time: " + str(end - start))
        
    # prof = 1
    # th1.join()

def util(stage):
    nvmlInit()
    gpu = 0 #GPU 0
    # f = open('/storage/utk/data/pipeline-mp/_gpu-stats/results-papers-' + str(stage) + '.txt', "w")
    # f = open('/storage/utk/data/pure-sampling-mp/cpu-util' + str(stage) + '.txt', "w")
    f = open('/disk/results/pure-sampling-uva/labor_pcie_rx.txt', "a+")
    f_ = open('/disk/results/pure-sampling-uva/labor_pcie_tx.txt', "a+")
    f__ = open('/disk/results/pure-sampling-uva/labor_gpu_mem.txt', "a+")
    # f = open('/media/data/utkrisht/graphsage/node_class/mean/util-' + stage + '.txt', "w")
    handle = nvmlDeviceGetHandleByIndex(gpu)
    #nvmlUtilization_t util
    # f.write("TS,HBM,PCI_TX,PCI_RX,DRAM,CPU\n")
    f.write(str(stage) + ",")
    f_.write(str(stage) + ",")
    f__.write(str(stage) + ",")
    global prof
    while prof == 0:
        util = nvmlDeviceGetUtilizationRates(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        # f.write(str(timer()) + ","  + str(mem_info.used >> 20) + ',' +
        #         str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)) + "," +
        #         str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)) + "," +
        #         str(psutil.virtual_memory()[3] / (1 << 30)) + ',' + 
        #         str(psutil.cpu_percent()) + ",\n")
        time.sleep(0.1)
        f.write(str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)) + ",")
        f_.write(str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)) + ",")
        f__.write(str(mem_info.used >> 20) + ",")
    
    f.write("\n")
    f_.write("\n")
    f__.write("\n")
    nvmlShutdown()
    f.close()
    f_.close()
    f__.close()

def util_(stage):
    nvmlInit()
    gpu = 0 #GPU 0
    # f = open('/storage/utk/data/pipeline-mp/_gpu-stats/results-papers-' + str(stage) + '.txt', "w")
    # f = open('/storage/utk/data/pure-sampling-mp/cpu-util' + str(stage) + '.txt', "w")
    f = open('/disk/results/pipeline-uva/labor_pcie_rx.txt', "a+")
    f_ = open('/disk/results/pipeline-uva/labor_pcie_tx.txt', "a+")
    f__ = open('/disk/results/pipeline-uva/labor_gpu_mem.txt', "a+")
    # f = open('/media/data/utkrisht/graphsage/node_class/mean/util-' + stage + '.txt', "w")
    handle = nvmlDeviceGetHandleByIndex(gpu)
    #nvmlUtilization_t util
    # f.write("TS,HBM,PCI_TX,PCI_RX,DRAM,CPU\n")
    f.write(str(stage) + ",")
    f_.write(str(stage) + ",")
    f__.write(str(stage) + ",")
    global prof
    while prof == 0:
        util = nvmlDeviceGetUtilizationRates(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        # f.write(str(timer()) + ","  + str(mem_info.used >> 20) + ',' +
        #         str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)) + "," +
        #         str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)) + "," +
        #         str(psutil.virtual_memory()[3] / (1 << 30)) + ',' + 
        #         str(psutil.cpu_percent()) + ",\n")
        time.sleep(0.1)
        # f.write(str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_COUNT)) + ",")
        f.write(str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)) + ",")
        f_.write(str(nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)) + ",")
        f__.write(str(mem_info.used >> 20) + ",")
    
    f.write("\n")
    f_.write("\n")
    f__.write("\n")
    nvmlShutdown()
    f.close()
    f_.close()
    f__.close()

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
    # print(f"Training in {args.mode} mode.")

    # f = open('/storage/utk/data/tmp/papers-' + str(args.batch_size) + '.txt', "a")
    f = open('/disk/results/pure-sampling-uva/fused/gpu_101010.txt', "a+")
    # if args.sample == 0:
    #     f.write("\n\n\nCPU based sampling")
    #     f.write("\n Num of workers" + str(args.workers))
    # else:
    #     f.write("\n\n\nGPU based sampling")
    # f.write("\nBatch Size : " + str(args.batch_size))

    # load and preprocess dataset
    # print("Sample : ", args.sample)
    # print("Batch Size : ", args.batch_size)
    # print("Loading data")
    start = timer()
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-papers100M", root="/storage/utk/dgl/examples/pytorch/graphsage/dataset/"))
    # dataset = MAG240MDataset(root = "/disk/dataset/")

    end = timer()
    # f.write("\nData Loading : " + str(end - start))

    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    # g = torch.tensor(g).to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 128, out_size).to(device)

    # model training
    # print("Training...")
    b = [512, 1024, 2048, 4096, 8192]
    # b = [8192]
    # train(args, device, g, f, args.sample, args.b, args.workers, args.pin_prefetcher, args.prefetch_thread, dataset, model, num_classes)
    # # step = [1, 2, 4, 8, 16, 32, 64, 128]
    # for batch in b:
    # if args.sample == 0:
    # val = 1
    # while val == 1:
    #     val = input("Start the pcm-pcie?")
    #     batch = input("Batch Size")
    #     train(args, device, g, f, args.sample, batch, args.workers, args.pin_prefetcher, args.prefetch_thread, dataset, model, num_classes)
    #     p = subprocess.run(["pkill", "-f", "pcm-pcie"], shell=True, capture_output=True, text=True)
    # else:
        # train_(args, device, g, f, args.sample, args.batch_size, args.workers, args.pin_prefetcher, args.prefetch_thread, dataset, model, num_classes)
    # f.close()
    for batch in b:
        train(args, device, g, f, 0, batch, args.workers, args.pin_prefetcher, args.prefetch_thread, dataset, model, num_classes)
        train(args, device, g, f, 1, batch, args.workers, args.pin_prefetcher, args.prefetch_thread, dataset, model, num_classes)
        train_(args, device, g, f, 2, batch, args.workers, args.pin_prefetcher, args.prefetch_thread, dataset, model, num_classes)
    
    # if args.sample == 0:
    #     train(args, device, g, f, args.sample, args.batch_size, args.workers, args.pin_prefetcher, args.prefetch_thread, dataset, model, num_classes)
    # elif args.sample == 1:
    #     train(args, device, g, f, args.sample, args.batch_size, args.workers, args.pin_prefetcher, args.prefetch_thread, dataset, model, num_classes)
    # else:
    #     train_(args, device, g, f, args.sample, args.batch_size, args.workers, args.pin_prefetcher, args.prefetch_thread, dataset, model, num_classes)
    # f = open('/storage/utk/data/pipeline-uva/test.txt', "a+")
    # for batch in b:
    # train_(args, device, g, f, args.sample, args.batch_size, args.workers, args.pin_prefetcher, args.prefetch_thread, dataset, model, num_classes)

        # train_(args, device, g, f, args.sample, args.batch_size, args.workers, args.pin_prefetcher, args.prefetch_thread, dataset, model, num_classes)
    # test the model
    # print("Testing...")
    # acc = layerwise_infer(
    #     device, g, dataset.test_idx, model, num_classes, batch_size=4096
    # )
    # print("Test Accuracy {:.4f}".format(acc.item()))
