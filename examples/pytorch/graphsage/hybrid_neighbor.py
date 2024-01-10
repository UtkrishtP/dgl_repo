import argparse
from platform import node
from turtle import done
from dgl import data
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
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
import torch.utils.data._utils.pin_memory as pm
# from dgl.dataloading.dataloader import *

# import os.path.join(os.getcwd(), "..", "..", "..", "python", "dgl", "utils")
# print(util_path)
# import util_path
torch.multiprocessing.set_sharing_strategy('file_system')


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
        Function to continuously sample using CPU. The dataloader object passed has all the
        optimizations turned on. 
'''
prof = 0
cpu_sampled_blocks = {}
# cpu_sampled_blocks.share_memory_()
cpu_samples = 0

def pin_samples(sample_queue, data_queue, done_event, cpu_samples, mini_batches):
    count = 0
    sample_count = 0
    global cpu_sampled_blocks
    while not done_event.is_set() :

        while cpu_samples.value <= sample_count:
            continue
        
        for block in cpu_sampled_blocks[sample_count]:
            count = count + 1
            # data_queue.cancel_join_thread()
            # block = sample_queue.get()
            pm.pin_memory(block)
            data_queue.put(block)
            if mini_batches == count:
                count = 0
                # cpu_samples.value = cpu_samples.value + 1
        sample_count = sample_count + 1
        # key = key + 1

# def cpu_sample(dataset, batch, workers, sampler, cpu_train_idx, skip_mfg, done_event, queue):
def cpu_sample(dataset, batch, workers, sampler, cpu_train_idx, skip_mfg, queue_, done_event, cpu_samples):
    g = dataset[0]
    g.ndata["label"] = g.ndata["label"].type(torch.LongTensor)
    cpu_neighbor_sampler = ""
    f = open("./cpu_sample.txt", "a")
    mini_batches = (dataset.train_idx.shape[0] // batch) + 1
    # print("CPU sampling....")
    if sampler == 0:
        cpu_neighbor_sampler = NeighborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            # fused=False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    elif sampler == 1:
        cpu_neighbor_sampler = NeighborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            fused=False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    elif sampler == 2:
        cpu_neighbor_sampler = LaborSampler(
            [10, 10, 10],
            layer_dependency=True,
            importance_sampling=-1,
        )

    cgg_train_dataloader = DataLoader(
        g,
        cpu_train_idx,
        cpu_neighbor_sampler,
        # device="cuda" if torch.cuda.is_available() else "cpu",
        device="cuda",
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        num_workers=workers,
        use_uva=False,
        persistent_workers=True if workers > 0 else False,
        # gpu_cache=ggg_train_dataloader.gpu_cache,
        skip_mfg=False if skip_mfg == 0 else True,
        # dataloader=ggg_train_dataloader,
        # cgg_on_demand=True,
        # cgg=cgg,
    )

    if workers > 0:
        for it, (input_nodes, output_nodes, blocks) in enumerate(
                cgg_train_dataloader
            ):
            break

    sample_queue = torch.multiprocessing.Queue()
    sample_queue.cancel_join_thread()
    done_event_ = torch.multiprocessing.Event()
    cpu_samples.value = 0
    th = threading.Thread(target=pin_samples, args=(sample_queue, queue_, done_event_, cpu_samples, mini_batches))

    th.start()
    global prof, cpu_sampled_blocks
    cpu_samples_ = 0
    while not done_event.is_set():
    # while prof == 0:
        
        cpu_sampled_blocks[cpu_samples_] = []
        for it, (input_nodes, output_nodes, blocks) in enumerate(
                cgg_train_dataloader
            ):
                cpu_sampled_blocks[cpu_samples_].append(blocks)
                # queue_.put(blocks)
                
        cpu_samples.value = cpu_samples.value + 1
        cpu_samples_ = cpu_samples_ + 1
        # print("CPU samples : ", cpu_samples.value)
        f.write("CPU_sample(" + str(time.time()) + ")," )
    print('ho')
    done_event_.set()
    th.join()
    f.write("\n")

# def hybrid_(gpu_dataloader, epoch_, th, skip_mfg, done_event, queue, mini_batches) :
def hybrid_(gpu_dataloader, epoch_, th, skip_mfg, mini_batches) :
    cpu_samples_processed = 0
    global prof
    prof = 0
    global cpu_samples
    cpu_samples = 0
    
    # th = threading.Thread(target=cpu_sample, args=(cpu_dataloader,))
    # th = torch.multiprocessing.Process(target=cpu_sample, args=(cpu_dataloader,))
    
    th.start()
    print("Process started")
    epoch = epoch_
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    end1 = start1 = end2 = count = count1 = 0
    start = timer()
    while epoch > 0:

        model.train()

        total_loss = 0
        mini_batch_counter = 0
        print("Inside training")
        # Checking if new samples for an epoch have been produced, if yes then train this epoch directly
        # while cpu_samples_processed < cpu_samples and epoch > 0:
        # while not queue.empty() and epoch > 0:
        while cpu_samples_processed < cpu_samples and epoch > 0:
            print("Inside CPU training")
            f.write("cgg(" + str(time.time()) + "),")
            for blocks in cpu_sampled_blocks[cpu_samples_processed]:
                # blocks = queue.get()
                # mini_batch_counter = mini_batch_counter + 1

                if skip_mfg == True:
                    blocks = recursive_apply(
                        blocks, lambda x: x.to("cuda", non_blocking=True))
                    
                # x = blocks[0].srcdata["feat"]
                # y = blocks[-1].dstdata["label"]
                x = gpu_dataloader._cgg_on_demand("feat","_N",blocks[0].srcdata["_ID"])
                y = gpu_dataloader._cgg_on_demand("label","_N",blocks[-1].dstdata["_ID"])
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                # count1 = count1 + (end2 - start2)
                # if mini_batch_counter == mini_batches:
                #     mini_batch_counter = 0
            cpu_sampled_blocks[cpu_samples_processed] = []
            cpu_samples_processed = cpu_samples_processed + 1
            epoch = epoch - 1
            
        # TODO: Overlap GPU sampling with the above process.
        # GPU UVA pipeline
        if epoch == 0:
            break

        f.write("ggg(" + str(time.time()) + "),")

        for it, (input_nodes, output_nodes, blocks) in enumerate(
            gpu_dataloader
        ):
            print("GPU training")
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            # count = count + )
            # continue
        epoch = epoch - 1
        # end1 = end1 + count
        # print("GPU trained : ", count)

    end = timer()
    done_event.set()
    print("Process ended")
    f.write("\n" + str(end - start - count - count1) + "," +
    str(cpu_samples) + "," + str(cpu_samples_processed) + "\n")
    prof = 1

# def hybrid(dataset, epoch_, skip_mfg, done_event, queue, mini_batches, sampler, batch, model) :
def hybrid(dataset, epoch_, skip_mfg, mini_batches, sampler, batch, model, queue_, done_event, cpu_samples) :
    cpu_samples_processed = 0
    
    f = open("./cpu_samples.txt", "a")
    g = dataset[0]
    g.ndata["label"] = g.ndata["label"].type(torch.LongTensor)
    epoch = epoch_
    gpu_train_idx = dataset.train_idx.to("cuda")
    print("Hybrid")
    gpu_neighbor_sampler = ""
    if sampler == 2:
        gpu_neighbor_sampler = LaborSampler(
            [10, 10, 10],
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )
    else:
        gpu_neighbor_sampler = NeighborSampler(
            [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )

    ggg_train_dataloader = DataLoader(
        g,
        gpu_train_idx,
        gpu_neighbor_sampler,
        device="cuda",
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
        # persistent_workers=True if workers > 0 else False,
        gpu_cache={"node": {"feat": 10000000}},
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    end1 = start1 = end2 = count = count1 = 0
    start = timer()

    while epoch > 0:

        model.train()

        total_loss = 0
        mini_batch_counter = 0
        bl = 0
        # print("Inside training")
        # Checking if new samples for an epoch have been produced, if yes then train this epoch directly
        # while cpu_samples_processed < cpu_samples and epoch > 0:
        while not queue_.empty() and epoch > 0 and cpu_samples_processed < cpu_samples.value:
            # print("Inside CPU training")
            # print("CPU samples hybrid: ", cpu_samples.value)
            f.write("cgg(" + str(time.time()) + "),")
            while mini_batch_counter != mini_batches:
            # while cpu_samples_processed < cpu_samples and epoch > 0:
                
                # for blocks in cpu_sampled_blocks[cpu_samples_processed]:
                s = time.time()
                blocks = queue_.get()
                e = time.time()
                bl = bl + (e - s)
                mini_batch_counter = mini_batch_counter + 1

                if skip_mfg == True:
                    blocks = recursive_apply(
                        blocks, lambda x: x.to("cuda", non_blocking=True))
                    
                # x = blocks[0].srcdata["feat"]
                # y = blocks[-1].dstdata["label"]
                x = ggg_train_dataloader._cgg_on_demand("feat","_N",blocks[0].srcdata["_ID"])
                y = ggg_train_dataloader._cgg_on_demand("label","_N",blocks[-1].dstdata["_ID"])
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                # count1 = count1 + (end2 - start2)
            epoch = epoch - 1
            cpu_samples_processed = cpu_samples_processed + 1
            mini_batch_counter = 0
            
        # TODO: Overlap GPU sampling with the above process.
        # GPU UVA pipeline
        if epoch == 0:
            break

        f.write("ggg(" + str(time.time()) + "),")

        for it, (input_nodes, output_nodes, blocks) in enumerate(
            ggg_train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            # count = count + )
            # continue
        epoch = epoch - 1
        # end1 = end1 + count
        # print("GPU trained : ", count)

    end = timer()
    done_event.set()
    print("Process ended")
    f.write("\n" + str(end - start - count - count1) + "," + str(bl) + "," +
    str(cpu_samples) + "," + str(cpu_samples_processed) + "\n")
    
def train(args, device, g, dataset, model, f, batch, epoch_, workers, sampler, skip_mfg):
    # create sampler & dataloader

    '''
        Creating two different training INDICES, since we are using use_uva=True and False for both CPU and GPU training
        To initialize dataloaders 
            train indices and graph need to be on CPU for CPU sampling
            train_indices (GPU) and graph(CPU) for GPU sampling
        
        This is for the scenario of inter epoch sampling
    '''
    cpu_train_idx = dataset.train_idx
    mini_batches = (cpu_train_idx.shape[0] // batch) + 1
    f = open("./cpu_samples.txt", "a")
    try:
        with torch.multiprocessing.Pool(processes=2) as pool:
            with torch.multiprocessing.Manager() as manager:
                queue = manager.Queue()
                done_event = manager.Event()
                cpu_samples = manager.Value('i', 0)
                cpu_args = (dataset, batch, workers, sampler, cpu_train_idx, skip_mfg, queue
                            , done_event, cpu_samples)
                gpu_args = (dataset, epoch_, skip_mfg, mini_batches, sampler, batch, model, queue,
                             done_event, cpu_samples)
                f.write("Process Launch : " + str(time.time()) + "\n")
                tasks = [(cpu_sample, cpu_args), (hybrid, gpu_args)]
                results = [pool.apply_async(func, args=args) for func, args in tasks]
                for result in results:
                    result.get()  # Wait for the task to complete
    except Exception:
        pool.terminate()
        raise
    finally:
        pool.close()
        pool.join()

    print("Hello")

if __name__ == "__main__":

    # f = open('/mnt/utk/data/dgl-latest/sage_node_class_product.txt', "a")
    f = open('./hybrid_fused_cgg_on_demand.txt', "a")
    f = open('./tmp.txt', "a")
    # torch.multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    f1 = open('./hybrid_nbr_cgg_on_demand.txt', "a")
    f2 = open('./hybrid_labor_cgg_on_demand.txt', "a")
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
      "--dataset",
      type=str,
        default = "ogbn-papers100M",
        )
    
    parser.add_argument(
      "--batch_size",
      type=int,
        default = 8192,
        )
    
    parser.add_argument(
      "--epoch",
      type=int,
        default = 10,
        )
    
    parser.add_argument(
      "--workers",
      type=int,
        default = 0,
        )
    
    parser.add_argument(
        "--sampler",
        type=int,
        default=0,
        help="0: Fused, 1: Neighbor, 2: Labor"
    )

    parser.add_argument(
        "--skip_mfg",
        type=int,
    )
    
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")
    # load and preprocess dataset
    print("Loading data")
    start = timer()
    
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root="/storage/utk/dgl/examples/pytorch/graphsage/dataset/")
                                , save_dir="/disk1/tmp/")    
    end = timer()

    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    g.pin_memory_()
    # create GraphSAGE model
    # g.ndata["label"] = g.ndata["label"].type(torch.LongTensor)
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # model training
    print("Training...")
    
    # f = f if args.sampler == 0 else f1 if args.sampler == 1 else f2
    
    f.write(str(args.batch_size) + "," + str(args.epoch) + ",")
    train(args, device, g, dataset, model, f, args.batch_size, args.epoch, 
          args.workers, args.sampler, args.skip_mfg)

    exit(0) 
    # train(args, device, g, dataset, model, num_classes, f, 1024, args.uva, args.pin_prefetcher, args.alternate_streams, args.prefetch_thread)
    b = [8192, 2048, 512]
    epoch_ = [10]
    for batch in reversed(b):
        for epoch in epoch_:
            f.write(str(batch) + "," + str(epoch) + ",")
            train(args, device, g, dataset, model, num_classes, f, batch, args.uva, args.pin_prefetcher, args.alternate_streams, args.prefetch_thread, epoch, 0)
            f1.write(str(batch) + "," + str(epoch) + ",")
            train(args, device, g, dataset, model, num_classes, f1, batch, args.uva, args.pin_prefetcher, args.alternate_streams, args.prefetch_thread, epoch, 8)
        
    # # test the model
    # print("Testing...")
    # acc = layerwise_infer(
    #     device, g, dataset.test_idx, model, num_classes, batch_size=4096
    # )
    # print("Test Accuracy {:.4f}".format(acc.item()))
    # f.close()
