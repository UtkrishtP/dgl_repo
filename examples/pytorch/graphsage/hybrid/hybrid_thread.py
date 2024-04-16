
import psutil
import argparse
import gc
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
from queue import Empty, Full, Queue
from dgl.utils.internal import recursive_apply
from dgl.dataloading.dataloader import _record_stream
from dgl.dataloading.dataloader import _put_if_event_not_set
from dgl.heterograph import *
from pynvml import *
import faulthandler
faulthandler.enable()
from concurrent.futures import ThreadPoolExecutor
# torch.multiprocessing.set_sharing_strategy('file_system')
# try:
#     torch.multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass
th = 0
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

def memAvail():
    with open('/proc/meminfo', 'r') as mem:
        print(mem.readlines()[2])

import ctypes
def set_affinity(mask):
    """Set CPU affinity for the current thread on Linux."""
    libc = ctypes.CDLL("libc.so.6")
    pthread_self = libc.pthread_self
    pthread_self.restype = ctypes.c_void_p
    pthread_setaffinity_np = libc.pthread_setaffinity_np
    pthread_setaffinity_np.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ulong)]
    cpuset = ctypes.c_ulong(mask)
    pthread_setaffinity_np(pthread_self(), ctypes.sizeof(cpuset), ctypes.byref(cpuset))
cpu_samples = 0
epoch = 0
def ggg(f, g, model, train_idx, sampler_, batch, workers, cache_size, gpu_buffer_queue,
         ggg_launched, insert_mfg_gpu, transfer_mfg_gpu):
    
    set_affinity(1 << 63)
    device = torch.device("cuda")
    sampler = ""
    if sampler_ == 0 or sampler_ == 1:
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
            # layer_dependency=True,
            # importance_sampling=-1,
        )

    train_idx = train_idx.to(device)
    s = time.time()
    ggg_train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        use_uva=True,
        gpu_cache={"node": {"feat": cache_size}},
    )
    print("Dataloader : ", time.time() - s)
    model.to(device)

    s1 = time.time()
    print("GGG started ", time.time())
    ggg_launched.set()
    tot = 0
    cgg = 0
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    global cpu_samples, epoch
    try:
        while epoch > 0:
            model.train()
            total_loss = 0
            mini_batch = (train_idx.shape[0] + batch - 1) // batch
            while gpu_buffer_queue.qsize() >= mini_batch and epoch > 0:
                # tot = 0
                # f.write("CGG : " + str(gpu_buffer_queue.qsize()) + " " + str(mini_batch) 
                #         + " " + str(time.time()) + "\n")
                cgg += 1
                s_ = time.time()
                while mini_batch != 0:
                    insert_mfg_gpu.clear()
                    s = time.time()
                    blocks = gpu_buffer_queue.get(block=False)
                    e = time.time()
                    
                    x = ggg_train_dataloader._cgg_on_demand("feat", "_N", blocks[0].srcdata["_ID"])
                    y = ggg_train_dataloader._cgg_on_demand("label", "_N", blocks[-1].dstdata["_ID"])                
                    y_hat = model(blocks, x)
                    loss = F.cross_entropy(y_hat, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                    tot += e - s
                    mini_batch -= 1
                    # insert_mfg_gpu.set()
                    # del blocks

                mini_batch = (train_idx.shape[0] + batch - 1) // batch
                epoch -= 1
                insert_mfg_gpu.set()

                f.write("CGG : " + str(time.time() - s_) + " " + str(epoch) + " " + str(time.time()) + "\n")
                # print("Inside CGG : ", gpu_buffer_queue.qsize(), time.time())

            if epoch == 0:
                break
            
            if cpu_samples > 0:
                cpu_samples -= 1
            epoch -= 1
            
            # print("GPU training ", gpu_buffer_queue.qsize(), time.time())
            s = time.time()
            transfer_mfg_gpu.clear()
            for it, (input_nodes, output_nodes, blocks) in enumerate(
                ggg_train_dataloader
            ):
                transfer_mfg_gpu.set()
                x = blocks[0].srcdata["feat"]
                y = blocks[-1].dstdata["label"]
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                transfer_mfg_gpu.clear()
            # print("GGG : ", time.time() - s, epoch)
            transfer_mfg_gpu.set()
            f.write("ggg :" + str(time.time() - s) + " " + str(time.time())+  "\n")
            
            # epoch -= 1
    except Exception as e:
        print(e)
        pass

    print("&&&&&&&&&&&&&&&&& ", str(time.time() - s1))
    f.write("Training Time : " + str(time.time() - s1) + " " + str(tot) + " " + str(cgg) + "\n")
    insert_mfg_gpu.set()
    # time.sleep(5)

prof = 0

def util():
    nvmlInit()
    gpu = 0 #GPU 0
    handle = nvmlDeviceGetHandleByIndex(gpu)
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    if (mem_info.free) < (1 << 30) * 4:
        nvmlShutdown()
        return 0  # Not enough memory, clear the event
    else:
        nvmlShutdown()
        return 1  # Enough memory, set the event


def transfer_mfg(f, transfer_queue, gpu_buffer_queue, ggg_launched, done_event, mini_batch,
                 insert_mfg, insert_mfg_gpu, transfer_mfg_gpu):
    # set_affinity(1 << 62)
    stream = torch.cuda.Stream()
    t = t_ = tr = 0
    step = 0
    # ggg_launched.wait()
    global cpu_samples, epoch
    s1 = time.time()
    while not done_event.is_set() or transfer_queue.qsize() > 0:

        if epoch == 0:
            break
        # To prevent transfer_queue.get() throwing a queue empty exception, we sleep the process
        #  until a block has been inserted.
        mfgs = []
        gpu_mfgs = []
        while step != mini_batch:
            s = time.time()
            # insert_mfg.wait()
            blocks = transfer_queue.get(timeout=1)
            mfgs.append(blocks)
            
            e = time.time()
            t += e - s
            step += 1
        
        for blocks in mfgs:
            while not util():
                time.sleep(1)
            
            transfer_mfg_gpu.wait()
            current_stream = torch.cuda.current_stream()
            stream.wait_stream(current_stream)
            s = time.time()
            
            with torch.cuda.stream(stream):
                blocks = recursive_apply(
                                    blocks, lambda x: x.to("cuda", non_blocking=True))
                blocks = recursive_apply(blocks, _record_stream, current_stream)
                # time.sleep(5)
            global stream_event
            stream_event = stream.record_event()
            e = time.time()
            tr += e - s
            gpu_mfgs.append(blocks)
        
        s_ = time.time()
        for blocks in gpu_mfgs:
            while True:
                try:
                    insert_mfg_gpu.wait()
                    gpu_buffer_queue.put(blocks, block=False)
                except Full:
                    continue
                break
        
        e_ = time.time()
        t_ += e_ - s_
        # del blocks 
        # gc.collect()
        
        step = 0
        insert_mfg_gpu.clear()
        f.write("MFG done: " + str(time.time()) + "\n")
        # print("MFG done ", cpu_samples)
        
        if transfer_queue.qsize() == 0:
            insert_mfg.clear()
        
    f.write("MFG thread : " + str(time.time() - s1) +
            " De-queue time : " + str(t) + 
            " Enqueue gpu buffer time : " + str(t_) +
            " GPU transfer time : " + str(tr) 
            + "\n")
        
class SetupNumThreads(object):

    def __init__(self, num_threads):
        self.num_threads = num_threads

    def __call__(self, worker_id):
        torch.set_num_threads(self.num_threads)

def process_func(g, model, train_idx, sampler_, batch, workers, cache_size, epoch_, dataset):
    # numa.bind([0])
    mini_batch = (train_idx.shape[0] + batch - 1) // batch
    print("Mini batch : ", mini_batch)
    f = open("./hybrid_timeline.txt", "a")
    f.write("\n" + dataset + "," + str(batch) + "," + str(sampler_) + "\n")
    ggg_launched = threading.Event()
    done_event = threading.Event()
    samples_ready = threading.Event()
    insert_mfg = threading.Event()
    insert_mfg_gpu = threading.Event()
    transfer_mfg_gpu = threading.Event()
    samples_ready.clear()
    insert_mfg.clear()
    insert_mfg_gpu.set()
    done_event.clear()

    transfer_queue = Queue(mini_batch * (epoch_ / 2)) # Queue size has been kept 4 times the mini_batch size. TODO: Make it dynamic
    gpu_buffer_queue = Queue(mini_batch * epoch_)
    # cpu_samples = torch.multiprocessing.Value('i', epoch_)
    global cpu_samples, epoch
    cpu_samples = epoch = epoch_

    with ThreadPoolExecutor(max_workers=2) as executor:
        mfg_transfer = executor.submit(transfer_mfg, f, transfer_queue, 
                                    gpu_buffer_queue, ggg_launched, done_event, mini_batch, insert_mfg, insert_mfg_gpu, transfer_mfg_gpu)
        hybrid_ggg = executor.submit(ggg, f, g, model,  train_idx, sampler_, batch, workers, cache_size,
                            gpu_buffer_queue, ggg_launched, insert_mfg_gpu, transfer_mfg_gpu)
    
        # torch.set_num_threads(48)
        sampler = ""
        if sampler_ == 0:
            print("fns")
            sampler = NeighborSampler(
                [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
                # prefetch_node_feats=["feat"],
                # prefetch_labels=["label"],
            )
        if sampler_ == 1:
            print("nbr")
            sampler = NeighborSampler(
                [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
                fused = False,
                # prefetch_node_feats=["feat"],
                # prefetch_labels=["label"],
            )
        if sampler_ == 2:
            print("lbr")
            sampler = LaborSampler(
                [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
                # prefetch_node_feats=["feat"],
                # prefetch_labels=["label"],
                # layer_dependency=True,
                # importance_sampling=-1,
            )
        s1 = time.time()
        train_dataloader = DataLoader(
            g,
            train_idx,
            sampler,
            device=torch.device("cpu"),
            batch_size=batch,
            shuffle=True,
            drop_last=False,
            num_workers=workers,
            use_uva=False,
            persistent_workers=True if workers > 0 else False,
            skip_mfg=True,
            mfg_buffer=transfer_queue,
            use_prefetch_thread=True,
            pin_prefetcher=False,
            samples_ready=samples_ready,
            # index_transfer=0,
            # cgg_on_demand=True,
            cgg=True,
            insert_mfg=insert_mfg,
            use_alternate_streams=False,
            # gather_pin_only=True,
            # worker_init_fn=SetupNumThreads(torch.get_num_threads() // workers) if workers > 0 else None,
        )

        # if workers > 0:
        #         for it, (input_nodes, output_nodes, blocks) in enumerate(
        #                 train_dataloader
        #             ):
        #             break

        print("Dataloaded : ", time.time() - s1)
        start_ = time.time()
        print("+++++++++++++++ ", time.time())
        # ggg_launched.wait()
        while cpu_samples > 0:    
            # while transfer_queue.qsize() >= mini_batch and epoch > 0:
            #     time.sleep(1)   
            # if epoch == 0:
            #     break
            start = time.time()
            f.write("Sampling :" + str(time.time()) + "\n")
            iter_ = train_dataloader._begin_sampling()
            samples_ready.wait()
            end = time.time()
            samples_ready.clear()
            # print("Sampling process : ", time.time())
            # print("**************** ", gpu_buffer_queue.qsize(), transfer_queue.qsize())
            cpu_samples -= 1
            
        # file.close()
        end_ = time.time()
        f.write("End-End Sampling Time : " + str(end_ - start_) + " Sampler enqueue : " + str(train_dataloader.index_transfer) + "\n")
        done_event.set()
        mfg_transfer.result()
        hybrid_ggg.result()
        print("#################### ", gpu_buffer_queue.qsize(), transfer_queue.qsize())
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # numa.bind([0])
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )

    parser.add_argument(
      "--batch_size",
      type=int,
        default = 8192)
    
    parser.add_argument(
      "--cache_size",
      type=int,
        default = 10000000)

    parser.add_argument(
      "--workers",
      type=int,
        default = 0)

    parser.add_argument(
      "--sampler",
      type=int,
        default = 0)
    
    parser.add_argument(
        "--epoch",
        type=int,
        default=10
    )

    parser.add_argument(
        "--hid_size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default = "ogbn-arxiv",
        )

    
    args = parser.parse_args()
    
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root="/data/")
                                , save_dir="/data/tmp/")
    g = dataset[0]
    start = time.time()
    g.ndata["label"] = g.ndata["label"].type(torch.LongTensor)
    # train_idx = dataset.train_idx.share_memory_()
    train_idx = dataset.train_idx
    in_size = g.ndata["feat"].shape[1]
    print("Nfeat shape : ", in_size)
    # exit(0)
    out_size = dataset.num_classes
    model = SAGE(in_size, args.hid_size, out_size)
    # print("Sharing memory")
    # g_ = g.shared_memory("g")
    # g_.ndata["label"] = g.ndata["label"].share_memory_()
    # g_.ndata["feat"] = g.ndata["feat"].share_memory_()
    # end = time.time()
    # print("Sharing : ", end - start)
    # if hasattr(g, 'ndata'):
    #     for key in list(g.ndata.keys()):
    #         del g.ndata[key]  # Delete tensors individually
    
    # del g
    gc.collect()
    g.create_formats_()
    g.pin_memory_()
    print("Pre-processing time : ", time.time() - start)
    print("Batch size: ", args.batch_size, "Hidden size: ", args.hid_size)
    process_func(g, model, dataset.train_idx, args.sampler, 
                 args.batch_size, args.workers, args.cache_size, args.epoch, args.dataset)
    