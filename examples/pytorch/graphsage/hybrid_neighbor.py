import argparse
from platform import node
import queue
from socket import timeout
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
# try:
#     torch.multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass
import torch.utils.data._utils.pin_memory as pm
from pynvml import *

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
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--mode",
#     default="mixed",
#     choices=["cpu", "mixed", "puregpu"],
#     help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
#     "'puregpu' for pure-GPU training.",
# )
# f1 = open('./hybrid_nbr_cgg_on_demand.txt', "a")
# f2 = open('./hybrid_labor_cgg_on_demand.txt', "a")
# parser.add_argument(
#     "--uva",
#     type=bool,
#     default = True,
#     choices = [True,False])

# parser.add_argument(
#     "--pin_prefetcher",
#     type=bool,
#     default = False,
#     choices = [True,False])

# parser.add_argument(
#     "--alternate_streams",
#     type=bool,
#     default = False,
#     choices = [True,False])

# parser.add_argument(
#     "--prefetch_thread",
#     type=bool,
#     default = False,
#     choices = [True,False])

# parser.add_argument(
#     "--dataset",
#     type=str,
#     default = "ogbn-papers100M",
#     )

# parser.add_argument(
#     "--batch_size",
#     type=int,
#     default = 8192,
#     )

# parser.add_argument(
#     "--epoch",
#     type=int,
#     default = 10,
#     )

# parser.add_argument(
#     "--workers",
#     type=int,
#     default = 0,
#     )

# parser.add_argument(
#     "--sampler",
#     type=int,
#     default=0,
#     help="0: Fused, 1: Neighbor, 2: Labor"
# )

# parser.add_argument(
#     "--skip_mfg",
#     type=int,
#     default=1,
# )

# args = parser.parse_args()
# dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root="/storage/utk/dgl/examples/pytorch/graphsage/dataset/")
#                                 , save_dir="/disk1/tmp/")    
# end = timer()

# g = dataset[0]
# g = g.to("cuda" if args.mode == "puregpu" else "cpu")
# num_classes = dataset.num_classes
# # device = torch.device("cpu" if args.mode == "cpu" else "cuda")
# # g.pin_memory_()
# # create GraphSAGE model
# # g.ndata["label"] = g.ndata["label"].type(torch.LongTensor)
# in_size = g.ndata["feat"].shape[1]
# out_size = dataset.num_classes
# # model = SAGE(in_size, 256, out_size).to(device)

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

def util(event):
    nvmlInit()
    gpu = 0 #GPU 0
    handle = nvmlDeviceGetHandleByIndex(gpu)
    #nvmlUtilization_t util
    global prof
    while prof == 0:
        # util = nvmlDeviceGetUtilizationRates(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        if (mem_info.total - mem_info.used) < (1<<30 * 10):
            event.set()
        else:
            event.clear()

    nvmlShutdown()

def cpu_sample(g, batch, workers, sampler, cpu_train_idx, skip_mfg, queue1,
               done_event, queue_read_event, process_launch_event, cpu_samples, epoch_):
    # global g
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # torch.cuda.set_device(0) 
    # torch.cuda.init()
    # print("CPU sampling ",torch.__path__)
    # print("Sampler : ", torch.cuda.is_available())
    # torch._C_.cuda_init()
    # os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    cpu_neighbor_sampler = ""
    f = open("./cpu_sample.txt", "a")
    mini_batches = (dataset.train_idx.shape[0] // batch) + 1
    # device = torch.device("cuda")
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
    print("Initialize dataloader")
    cgg_train_dataloader = DataLoader(
        g,
        cpu_train_idx,
        cpu_neighbor_sampler,
        device="cpu",
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        num_workers=workers,
        use_uva=False,
        persistent_workers=True if workers > 0 else False,
        # gpu_cache=ggg_train_dataloader.gpu_cache,
        skip_mfg=True,
        # dataloader=ggg_train_dataloader,
        # cgg_on_demand=True,
        # cgg=True,
        # pin_prefetcher=True,
        # use_prefetch_thread=True,
    )
    print("Initialized dataloader...")

    if workers > 0:
        for it, (input_nodes, output_nodes, blocks) in enumerate(
                cgg_train_dataloader
            ):
            break

    gpu_mem_event = torch.multiprocessing.Event()
    cpu_samples.value = 0
    th = threading.Thread(target=util, args=(gpu_mem_event,))
    print("Starting controller thread")
    th.start()
    print("Controller thread started")
    while process_launch_event.is_set():
        continue

    print("CPU sampling starts")
    epoch = 0
    global prof
    while not done_event.is_set() and epoch < epoch_ - 1:
        t = 0
        # cpu_sampled_blocks[cpu_samples] = []
        for it, (input_nodes, output_nodes, blocks) in enumerate(
                cgg_train_dataloader
            ):
                while gpu_mem_event.is_set() or queue_read_event.is_set():
                    if done_event.is_set():
                        break
                    continue
                if done_event.is_set():
                        break
                s = time.time()

                blocks = recursive_apply(
                        blocks, lambda x: x.to("cuda", non_blocking=True))
                queue1.put(blocks)
                del blocks
                e = time.time()
                t += (e - s)
        cpu_samples.value = cpu_samples.value + 1
        epoch += 1
        # print("CPU samples : ", cpu_samples.value)
        f.write("CPU_sample(" + str(time.time()) + ", " + str(t) + "), ")

    # done_event.set()
    prof = 1
    th.join()
    f.write("\n")
    print('Cpu samples ended')

    f.close()

def hybrid(g, epoch_, skip_mfg, mini_batches, sampler, batch, queue1,
            done_event, queue_read_event, process_launch_event, cpu_samples) :
    # time.sleep(5)
    # try:
    #     while not queue1.empty():
    #         print("hi")
    #         print(queue1.get(timeout=1))
    # except Exception as e:
    #     print(e)
    # print("Hybrid Ended")
    # return
    # torch.cuda.init()
    # os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    f = open('./env_child.txt', "a+")
    # torch.cuda.set_device(0)
    
    print("Hybrid : ", torch.cuda.is_available())
    # torch._C_.cuda_init()
    device = torch.device("cuda")
    cpu_samples_processed = 0
    model = SAGE(in_size, 256, out_size).to(device)
    f = open("./timeline.txt", "a")
    # global g
    g.ndata["label"] = g.ndata["label"].type(torch.LongTensor)
    epoch = epoch_
    gpu_train_idx = dataset.train_idx.to("cuda")
    
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

    print("Initialize dataloader")
    ggg_train_dataloader = DataLoader(
        g,
        gpu_train_idx,
        gpu_neighbor_sampler,
        device=device,
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        use_uva=True,
        gpu_cache={"node": {"feat": 10000000}},
    )
    print("Initialized dataloader...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    end1 = start1 = end2 = count = count1 = 0
    start = timer()
    process_launch_event.clear()
    print("Hybrid")

    time.sleep(5)
    done_event.set()
    return
    while epoch > 0:

        model.train()

        total_loss = 0
        mini_batch_counter = 0
        bl = 0
        while not queue1.empty() and epoch > 0 and cpu_samples_processed < cpu_samples.value:
            f.write("cgg(" + str(time.time()))
            queue_read_event.set()
            bl = 0
            while mini_batch_counter != mini_batches:
            # while cpu_samples_processed < cpu_samples and epoch > 0:
                
            # for blocks in cpu_sampled_blocks[cpu_samples_processed]:
                s = time.time()
                blocks = queue1.get()
                e = time.time()
                bl = bl + (e - s)
                mini_batch_counter += 1

                    # if skip_mfg == True:
                    #     blocks = recursive_apply(
                    #         blocks, lambda x: x.to("cuda", non_blocking=True))
                        
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
            epoch -= 1
            cpu_samples_processed += 1
            mini_batch_counter = 0
            f.write(", " + str(bl) + "), ")

        if queue_read_event.is_set():
            queue_read_event.clear() 
        # TODO: Overlap GPU sampling with the above process.
        # GPU UVA pipeline
        if epoch == 0:
            break

        f.write("ggg(" + str(time.time()) + "), ")
        print("GPU training")

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
        epoch -= 1
        # end1 = end1 + count
        # print("GPU trained : ", count)

    end = timer()
    done_event.set()
    print("Process ended")
    f.write("\n" + str(end - start - count - count1) + "," + str(bl) + "," +
    str(cpu_samples) + "," + str(cpu_samples_processed) + "\n")
    f.close()

def train(args, device, g, dataset, batch, epoch_, workers, sampler, skip_mfg):
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
    # g.shared_memory()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # print("Parent ",torch.__path__)
    # ctx = torch.multiprocessing.get_context('spawn')
    # f = open("./cpu_samples.txt", "a")
    try:
        queue1 = torch.multiprocessing.Queue()
        queue1.cancel_join_thread()
        process_launch_event = torch.multiprocessing.Event()
        process_launch_event.set()
        done_event = torch.multiprocessing.Event()
        queue_read_event = torch.multiprocessing.Event()
        cpu_samples = torch.multiprocessing.Value('i', 0)

        cpu_args = (g, batch, workers, sampler, cpu_train_idx, skip_mfg, queue1
                    , done_event, queue_read_event, process_launch_event, cpu_samples, epoch_)
        gpu_args = (g, epoch_, skip_mfg, mini_batches, sampler, batch, queue1
                    , done_event, queue_read_event, process_launch_event, cpu_samples)
        print("Process Launch : " + str(time.time()))
        tasks = [(cpu_sample, cpu_args), (hybrid, gpu_args)]
        # tasks = [(hybrid, gpu_args)]
        processes = []
        for func, args in tasks:
            p = torch.multiprocessing.Process(target=func, args=args)
            p.start()
            print(p)
            processes.append(p)
        while not done_event.is_set():
            continue
        # queue1.close()
        # queue1.join_thread()
        time.sleep(5)
        for p in processes:
            p.terminate()
    except Exception as e:
        print("Error: ", e)
    
    # done_event = torch.multiprocessing.Event()
    # th = threading.Thread(target=cpu_sample, args=(g, batch, workers, sampler, cpu_train_idx, skip_mfg
    #                         , done_event, epoch_))
    # th = threading.Thread(target=hybrid, args=(g, epoch_, skip_mfg, mini_batches, sampler, batch, model, done_event))
    # th.start()
    # hybrid(g, epoch_, skip_mfg, mini_batches, sampler, batch, model,
                            #   done_event)
    
    # while not done_event.is_set():
    #     continue
    # th.join()
    # print("Hello")

# model training

if __name__ == "__main__":

    # f = open('/mnt/utk/data/dgl-latest/sage_node_class_product.txt', "a")
    # f = open('./hybrid_fused_cgg_on_demand.txt', "a")
    # f = open('./tmp.txt', "a")
    # torch.multiprocessing.set_start_method('spawn')
    # print("Main : ", torch.cuda.is_available())
    # exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    # f1 = open('./hybrid_nbr_cgg_on_demand.txt', "a")
    # f2 = open('./hybrid_labor_cgg_on_demand.txt', "a")
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
        default=1,
    )

    args = parser.parse_args()
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root="/storage/utk/dgl/examples/pytorch/graphsage/dataset/")
                                    , save_dir="/disk1/tmp/")    
    end = timer()
    
    g = dataset[0]
    # print(g)
    # print(g._graph)
    # print(g._node_frames)
    # for frames in g._node_frames:
    #     for key, tensor in frames.items():
    #         tensor.share_memory_()
    g.create_formats_()
    g.pin_memory_()
    # print(g)
    # exit(0)
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    # g.pin_memory_()
    # create GraphSAGE model
    g.ndata["label"] = g.ndata["label"].type(torch.LongTensor)

    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    # model = SAGE(in_size, 256, out_size).to(device)
    
    # if not torch.cuda.is_available():
    #     args.mode = "cpu"
    print(f"Training in {args.mode} mode.")
    # load and preprocess dataset
    print("Loading data")
    start = timer()
    device = "cuda"
    # f = f if args.sampler == 0 else f1 if args.sampler == 1 else f2
    # f.write(str(args.batch_size) + "," + str(args.epoch) + ",")
    train(args, device, g, dataset, args.batch_size, args.epoch, 
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
        
