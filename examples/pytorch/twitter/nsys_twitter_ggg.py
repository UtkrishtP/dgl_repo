import argparse
import dgl.dataloading as dd
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
from dataloader import TwitterDataset

from ogb.nodeproppred import DglNodePropPredDataset
from timeit import default_timer as timer
# from pynvml import *
import threading
import subprocess
prof = 0
import psutil, time
from ogb.lsc import MAG240MDataset
from pynvml import *
import csv
import sys
sys.path.append("../../..")
from dgl.utils.internal import recursive_apply
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers=2):
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

def cpu_sample(args, device, g, f, gpu, batch, workers, pin, prefetch, dataset, model, num_classes, cache_size):
    
    # create sampler & dataloader
    use_uva = False
    # train_idx = dataset.train_idx.to(device)
    train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    # train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    # temp_f = open('/media/yash/results/twitter/check_time_2.txt', "a+")
    train_idx = train_idx.to(device)
    num_nodes = g.num_nodes()
    print("Number of nodes: ", num_nodes)
    # access_frequency = {node_id: 0 for node_id in range(num_nodes)}
    sampler = ""
    if gpu == 0:
        sampler = NeighborSampler(
            [15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
            # fused = False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )

    if gpu == 2:
        sampler = LaborSampler(
            [15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
            # fused = True,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )
    
    if gpu == 1:
        sampler = NeighborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            # fused = False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )

    if gpu == 3:
        sampler = LaborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            # fused = True,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )
    
    if gpu == 4:
        sampler = NeighborSampler(
            [15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
            fused = False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )

    if gpu == 5:
        sampler = NeighborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            fused = False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    
    
    start = timer()
    train_dataloader = ""
   
    
    if workers == 0:
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
            # use_presistent_workers=True,
        )
    else:
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
            # use_prefetch_thread=True,
            persistent_workers=True,
        )
        end = timer()
        # f.write("\nTrain Dataloader object : " + str(end - start))
        # y = input("loading done")
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            break
        
    # th1 = threading.Thread(target=util, args=(str(batch),))
    # th1 = threading.Thread(target=util, args=(str(batch) + "---" + str(cache_size),))
    # global prof
    # prof = 0
    # th1.start()
    loss_fcn = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    print("Entering loop")
    for epoch in range(5):
        count = 0
        count1 = 0
        step = 1
        # start = timer()
        model.train()
        total_loss = 0
        start = timer()
        access_counter = 0
        mfg_time = 0
        # access_frequency = bytearray(num_nodes)
        # access_frequency = torch.zeros(num_nodes, dtype=torch.bool)
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            # start_access = timer()
            # # for node_id in input_nodes.tolist():
            # #     if(access_frequency[node_id] == 0):
            # #         access_frequency[node_id] = 1
            # #         counter=counter+1
            # # for node_id in input_nodes.tolist():
            # #     access_frequency[node_id] = 1
            # access_frequency[input_nodes] = True

            # end_access = timer()
            # access_counter = access_counter + (end_access - start_access)
            # for_start = time.time()

            ######
            # start1 = timer()
            # # x_start = time.time()
            # x = blocks[0].srcdata["feat"]
            # # # x_end = time.time()
            # y = blocks[-1].dstdata["label"]
            # # # end2 = timer()
            # # # model_start = time.time()
            # y_hat = model(blocks, x)
            # # # model_end = time.time()
            # loss = F.cross_entropy(y_hat, y)
            # # # loss_end = time.time()
            # opt.zero_grad()
            # # # opt_end = time.time()
            # loss.backward()
            # # # back_end = time.time()
            # opt.step()
            # # # step_end = time.time()
            # total_loss += loss.item()
            # # total_end = time.time()
            
            # # # if step == 1:
            # # #     break
            # end1 = timer()
            # count = count + (end1 - start1)
            start1 = timer()
            blocks = recursive_apply(blocks, lambda x : x.to("cuda", non_blocking=True))
            start2 = timer()
            mfg_time += start2 - start1
            continue

            ####
            # temp_f.write(str(for_start) + "," + str(x_start) + "," + str(x_end) + "," + str(model_start) + "," + str(model_end)  + "," + str(loss_end) + "," + str(opt_end) + "," + str(back_end) + "," + str(step_end) + "," + str(total_end) + "\n")
        end = timer()
        
        # temp_f.write("------------------------\n")
        f.write(str(batch) + ","  + str(epoch) + "," + str(gpu) + "," +
                str((end - start - mfg_time)) + "," + str(mfg_time) + "\n")
        
    # with open(f'/disk1/access_frequency_files/twitter_{gpu}_{batch}.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['NodeID', 'AccessCount'])

    #     for node_id, count in access_frequency.items():
    #         writer.writerow([node_id, count])
        
    # prof = 1
    # th1.join()



def train(args, device, g, f, sample_mode, batch, workers, pin, prefetch, dataset, model, num_classes, cache_size):
    
    # create sampler & dataloader
    use_uva = True
    # train_idx = dataset.train_idx.to(device)
    train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    # train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    # temp_f = open('/media/yash/results/twitter/check_time_2.txt', "a+")
    train_idx = train_idx.to(device)
    num_nodes = g.num_nodes()
    print("Number of nodes: ", num_nodes)
    # access_frequency = {node_id: 0 for node_id in range(num_nodes)}
    sampler = ""
    if sample_mode == 0:
        sampler = NeighborSampler(
            [15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
            # fused = False,
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )

    if sample_mode == 2:
        sampler = LaborSampler(
            [15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
            # fused = True,
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )
    
    if sample_mode == 1:
        sampler = NeighborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            # fused = False,
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )

    if sample_mode == 3:
        sampler = LaborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            # fused = True,
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )
    
    start = timer()
    train_dataloader = ""
   
    
    if workers == 0:
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
            gpu_cache = {"node": {"feat": cache_size}},
            # use_prefetch_thread=True,
            # use_presistent_workers=True,
        )
    else:
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
            # use_prefetch_thread=True,
            persistent_workers=True,
        )
        end = timer()
        # f.write("\nTrain Dataloader object : " + str(end - start))
        # y = input("loading done")
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            break
        
    
    # th1 = threading.Thread(target=util, args=(str(batch) + "---" + str(cache_size),))
    # global prof
    # prof = 0
    # th1.start()
    loss_fcn = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    print("Entering loop")
    for epoch in range(1):
        count = 0
        count1 = 0
        step = 1
        # start = timer()
        model.train()
        total_loss = 0
        start = timer()
        # access_frequency = bytearray(num_nodes)
        
        # access_counter = 0
        # access_frequency = torch.zeros(num_nodes, dtype=torch.bool)
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            # start_access = timer()
            # # for node_id in input_nodes.tolist():
            # #     if(access_frequency[node_id] == 0):
            # #         access_frequency[node_id] = 1
            # #         counter=counter+1
            # # for node_id in input_nodes.tolist():
            # #     access_frequency[node_id] = 1
            # access_frequency[input_nodes] = True

            # end_access = timer()
            # access_counter = access_counter + (end_access - start_access)
            # for_start = time.time()
            start1 = timer()
            # x_start = time.time()
            x = blocks[0].srcdata["feat"]
            # # x_end = time.time()
            y = blocks[-1].dstdata["label"]
            # # end2 = timer()
            # # model_start = time.time()
            y_hat = model(blocks, x)
            # # model_end = time.time()
            loss = F.cross_entropy(y_hat, y)
            # # loss_end = time.time()
            opt.zero_grad()
            # # opt_end = time.time()
            loss.backward()
            # # back_end = time.time()
            opt.step()
            # # step_end = time.time()
            total_loss += loss.item()
            # total_end = time.time()
            
            # # if step == 1:
            # #     break
            end1 = timer()
            count = count + (end1 - start1)
            # temp_f.write(str(for_start) + "," + str(x_start) + "," + str(x_end) + "," + str(model_start) + "," + str(model_end)  + "," + str(loss_end) + "," + str(opt_end) + "," + str(back_end) + "," + str(step_end) + "," + str(total_end) + "\n")
        end = timer()
        
        # temp_f.write("------------------------\n")
        # f.write(str(batch) + ","  + str(epoch) + "," + str(train_dataloader.get_cache_time())+ "," +
        #         str((end - start)) + "," +
        #         str(end - start - count) + "," +
        #         str(count) + "," + str(cache_size/1000000) + "M" + ","+ str(torch.sum(access_frequency).item())+ "," + str(access_counter) + ","+str(sample_mode) + "\n")
        
        # f.write(str(batch) + ","  + str(epoch) + "," +
        #         str((end - start)) + "," +
        #         str(end - start - count) + "," +
        #         str(count) + "," + str(cache_size/1000000) + "M" +  "," + str(sample_mode) + "\n")
    # with open(f'/disk1/access_frequency_files/twitter_{gpu}_{batch}.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['NodeID', 'AccessCount'])

    #     for node_id, count in access_frequency.items():
    #         writer.writerow([node_id, count])
        
    # prof = 1
    # th1.join()

def util(stage):
    nvmlInit()
    gpu = 0

    f = open('/media/yash/results/twitter/del_gpu_usage_15105.txt', "a+")
    handle = nvmlDeviceGetHandleByIndex(gpu)
    f.write(str(stage) + ",")
    global prof

    while prof == 0:
        util = nvmlDeviceGetUtilizationRates(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        f.write(str(mem_info.used >> 20) + ',')
        time.sleep(0.1)
    
    f.write("\n\nEND\n\n")
    nvmlShutdown()
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # '/disk/igb/igb_large/'
    # '/storage/yash/dataset/'
    # '/disk1/twitter-dataset/'
    parser.add_argument('--path', type=str, default='/data/twitter', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='medium',
        choices=['tiny', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=64, 
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=1, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=1,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')

    # Model
    parser.add_argument('--model_type', type=str, default='sage',
                        choices=['gat', 'sage', 'gcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Model parameters 
    parser.add_argument('--fan_out', type=str, default='15,10,5')
    parser.add_argument('--batch_size', type=int, default=10240)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--learning_rate', type=int, default=0.01)
    parser.add_argument('--decay', type=int, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--log_every', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
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

    # parser.add_argument(
    #   "--batch_size",
    #   type=int,
    #     default = 1024)

    parser.add_argument(
      "--sample_mode",
      type=int,
        default = 0)
    
    parser.add_argument(
      "--cache_size",
      type=int,
        default = 0)
    
    parser.add_argument(
      "--workers",
      type=int,
        default = 0)
    
    parser.add_argument(
        "--dir",
        type=int,
        default=0
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=0
    )
    args = parser.parse_args()

    # f = open('/disk/results/igb_medium/gpu-ns.txt', "a+")
    # f = open('/media/data/yashp/results/neighbour.txt', "a+")   # Change file
    # f = open('/media/yash/results/igb_medium/del_gpu_neighbour.txt', "a+")
    # f1 = open('/disk/results/igb_medium/gpu-labor.txt', "a+")
    # f1 = open('/media/yash/results/igb_medium/del_gpu_labor.txt', "a+")      # Change file
    # f2 = open('/disk/results/igb_medium/ccg_sync/labor_cpu.txt', "a+")
    start = timer()
   
    dataset = TwitterDataset(args)
    g = dataset[0]
    num_classes = 64
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    in_size = g.ndata["feat"].shape[1]
    out_size = 64
    model = SAGE(in_size, 128, out_size, 6).to(device)

    # train(args, device, g, f, args.sample, args.batch_size, 0, 1, 1, dataset, model, num_classes)
    # exit(0)

    # b = [8192, 4096, 2048, 1024, 512]
    b = [8192, 2048, 4096, 1024]
    # s = [0, 4, 8, 16, 32]

    # f = open("/media/yash/results/twitter_ccc.txt", "a+")
    f = open("/media/yash/results/twitter/execTime/Neighbor/pure_ggg_neighbor.txt", "a+")
    if(args.sample_mode == 2 or args.sample_mode == 3):
         f = open("/media/yash/results/twitter/execTime/Labor/pure_ggg_labor.txt", "a+")

    # for batch in b:
    #    print("Training batch: ", batch)
    #    for s in [0, 1, 2, 3, 4, 5]:
    #        print("Sampling mode: ", s)
    #        if(s<2):
    #            cpu_sample(args, device, g, f, s, batch, 0, 1, 1, dataset, model, num_classes, args.cache_size)
    #        else:
    #            cpu_sample(args, device, g, f, s, batch, 16, 1, 1, dataset, model, num_classes, args.cache_size)

    train(args, device, g, f, args.sample_mode, args.batch_size, 0, 1, 1, dataset, model, num_classes, args.cache_size)

    exit()
