import argparse

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.backend as F_
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
import time
from concurrent.futures import ThreadPoolExecutor
from dgl.utils.internal import recursive_apply
from dgl.createshm import create_shmarray, create_shmoffset, reset_shm, get_shm_ptr, print_offset
import ctypes
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
from dgl.convert import hetero_from_shared_memory
from dgl.utils.pin_memory import pin_memory_inplace
from custom_dl import FriendsterDataset, TwitterDataset, IGB260MDGLDataset
from args import get_args
from load_dataset import fetch_train_graph, fetch_all, fetch_shapes

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

import subprocess, os

def start_perf(batch, pid, e):
    # Command to start perf record
    # command = f"sudo perf record -g -F 999 -o ./stat/perf_{e}_{batch}.data -p {pid} & echo $! > ./perf.pid"
    # command = f"sudo perf stat -o ./stat/stat_{e}_{batch}_read.data -p {pid} & echo $! > ./perf.pid"
    command = f"sudo strace -c -fp {pid} 2>&1 | tee ./stat/strace_{e}_{batch}_.txt & echo $! > ./perf.pid"
    # print(command)
    subprocess.call(command, shell=True)

def stop_perf():
    # Stop perf record by killing the process
    with open('./perf.pid', 'r') as file:
        pid = file.read().strip()
    # os.kill(int(pid), signal.SIGINT)  # SIGKILL
    command = f"sudo kill -2 {pid}"
    subprocess.call(command, shell=True)
    # subprocess.call("perf report -i /tmp/perf.data", shell=True)  # Generate and view the report
import signal

def test_spawn(blocks, ):
    print("Hello")
    for block in blocks:
        print(block[0].srcdata['_ID'])

def training(size, fanout, mfg_read, train_, model, batch_size, mini_batch, epoch,):
    file = open("./results/hybrid_breakdown.txt", "a")
    file.write(f"\nTraining process launched: {time.time()} \n")
    start = time.time()

    offset_gpu_read = create_shmoffset(1024, "offset_gpu_read")
    array_gpu = get_shm_ptr("array_gpu", size, 0)
    b = []
    device = torch.device("cuda")
    train_idx, g = fetch_all()
    file.write(f"Reading from shared memory: {time.time() - start}s\n")

    ggg_train_dataloader = DataLoader(
        g,
        train_idx,
        NeighborSampler(
            [15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        ),
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        use_uva=True,
        gpu_cache={"node": {"feat": 1000000}},
        # transfer_mfg_gpu=transfer_mfg_gpu,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    model.to(device)
    read = 0
    file.write(f"Training process started: {time.time()} \n")
    print("Training process started")
    mfg_read.wait()
    print("Training process started after MFG read")
    # print("Training process: ", os.getpid(), flush=True)
    # while True:
    #     if os.path.exists("/tmp/break_train"):
    #         break
    # print("Training Resuming after SIGCONT")
    global_blocks = []
    deque = 0
    s = time.time()
    for mb in range(mini_batch):
        blocks = []
        s1 = time.time()
        for layer in range(len(fanout)):
            if layer == 0:
                block, input_nodes = dgl.hetero_from_gpu_shared_memory(array_gpu, offset_gpu_read, 0)
            elif layer == len(fanout) - 1:
                block, output_nodes = dgl.hetero_from_gpu_shared_memory(array_gpu, offset_gpu_read, 0)
            else:
                block, _ = dgl.hetero_from_gpu_shared_memory(array_gpu, offset_gpu_read, -1)
            # print(block._graph)
            blocks.append(block)
        blocks[0].srcdata["_ID"] = input_nodes[0]
        blocks[-1].dstdata["_ID"] = output_nodes[0]
        deque += time.time() - s1
        x = ggg_train_dataloader._cgg_on_demand("feat", "_N", blocks[0].srcdata["_ID"])
        y = ggg_train_dataloader._cgg_on_demand("label", "_N", blocks[-1].dstdata["_ID"]) 
        print("Consumer:: ", mb)
        y_hat = model(blocks, x)
        loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("***********",mb)
    end = time.time()
    total_loss = 0
    print("_________________________")
    for it, (input_nodes, output_nodes, blocks) in enumerate(
            ggg_train_dataloader
        ):
            start1 = time.time()
            
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
            
    print("GGG done")
    train_.set()
    file.write(f"GGG time: {time.time() - end}s, CGG time:{end - s}s GPU read time:{deque:.4f}s\n")
    file.close()

def to_gpu_shared_memory(blocks, array_gpu, offset_gpu, fanout,):
    layer = 0
    # gpu_blocks.append(blocks) # TODO: size = max mini_batches that can be hosted on GPU.
    for block in blocks:
        if layer == 0:
            for nframe in block._node_frames:
                if nframe:
                    # gpu_blocks.append(
                        block.shared_memory_gpu(array_gpu, offset_gpu, nframe['_ID'], 0)
                        # )
        elif layer == len(fanout) - 1:
            for nframe in block._node_frames:
                if nframe:
                    # gpu_blocks.append(
                        block.shared_memory_gpu(array_gpu, offset_gpu, nframe['_ID'], 0)
                        # )
        else:
            # gpu_blocks.append(
                block.shared_memory_gpu(array_gpu, offset_gpu, torch.empty(0), -1)
                # )
        layer += 1

def mfg_transfer(mfg, sampling, tail, head, mini_batch, 
                 size, fanout, mfg_read, train_, edge_dir):
    file = open("./results/hybrid_breakdown.txt", "a")
    # start_perf(mini_batch, os.getpid(), e)
    file.write(f"MFG transfer launched: {time.time()} \n")
    array = get_shm_ptr("array", size, 0)
    array_gpu = get_shm_ptr("array_gpu", size, 0)
    read_offset = create_shmoffset(1024, "read_offset")
    offset_gpu = create_shmoffset(1024, "offset_gpu")
    # offset_gpu_read = create_shmoffset(size)
    read_time = transfer_time = reset_time = gpu_enqueue = 0
    mfg.wait()
    # print("Transfer process: ", os.getpid(), flush=True)
    # while True:
    #     if os.path.exists("/tmp/break_transfer"):
    #         break
    start = time.time()
    i = 0
    gpu_blocks = []
    while not sampling.is_set():
        while tail.value < head.value:

            s1 = time.time()
            blocks = []
            for layer in range(len(fanout)):
                if layer == 0:
                    block, output_nodes = dgl.hetero_from_shared_memory_hybrid(array, 0, read_offset, edge_dir)
                elif layer == len(fanout) - 1:
                    block, input_nodes = dgl.hetero_from_shared_memory_hybrid(array, 0, read_offset, edge_dir)
                else:
                    block, _ = dgl.hetero_from_shared_memory_hybrid(array, layer, read_offset, edge_dir)
                
                blocks.insert(0, block)
            blocks[0].srcdata["_ID"] = input_nodes[0]
            blocks[-1].dstdata["_ID"] = output_nodes[0]
            read_time += time.time() - s1

            # Transfer to GPU, TODO: Create separate stream and function for transfer
            s1 = time.time()
            blocks = recursive_apply(
                    blocks, lambda x: x.to("cuda", non_blocking=True))
            transfer_time += time.time() - s1

            # Using cudaIPC to buffer MFG's in GPU.
            s1 = time.time()
            gpu_blocks.append(blocks) 
            to_gpu_shared_memory(blocks, array_gpu, offset_gpu, fanout)
            gpu_enqueue += time.time() - s1

            tail.value = 0 if tail.value == -1 else tail.value
            tail.value += 1 # % mini_batch
            s1 = time.time()
            if (tail.value % mini_batch) == 0:
                # file.write(f"MFG transfer done for epoch {tail.value / mini_batch} : {time.time()}\n")
                reset_shm(read_offset)
                reset_shm(offset_gpu)
            reset_time += time.time() - s1
            # print("Transfer: ", head.value, tail.value, input_nodes[0].shape)
            
        # print(f"Transfer {time.time()} : {time.time() - s: .4f}s")
    end = time.time()
    # stop_perf()
    mfg_read.set()
    train_.wait()
    file.write(f"MFG Transfer E2E: {end - start:.4f}s, CPU Shared read: {read_time}s, Enqueue: {gpu_enqueue:.4f}s, TR: {transfer_time:.4f}s, {time.time()} \n")
    file.close()

def train(file, args, model):
    # create sampler & dataloader
    train_idx, g = fetch_train_graph()
    fanout = [15, 10, 5]
    mfg = torch.multiprocessing.Event()
    sampling = torch.multiprocessing.Event()
    sampling.clear()
    mfg.clear()
    mini_batch = (train_idx.shape[0] + args.batch_size - 1) // args.batch_size
    hybrid_ = True if args.hybrid == 1 else False
    size = args.mfg_size * 1024 * 1024 * 1024

    head = torch.multiprocessing.Value(ctypes.c_long, 0)
    tail = torch.multiprocessing.Value(ctypes.c_long, -1)
    array = []
    array_gpu = []
    offset = []
    read_offset = []
    st = time.time()
    print("Sharing memory: ", mini_batch)
    # start_perf(mini_batch, os.getpid(), args.epoch)
    if hybrid_: 
        array = create_shmarray(size, args.madvise, "array", args.pin_mfg)
        array_gpu = create_shmarray(size, args.madvise, "array_gpu", args.pin_mfg)
        offset = create_shmoffset(1024, "offset")

    print("Memory shared")
    create_time = time.time() - st
    train_ = torch.multiprocessing.Event()
    mfg_read = torch.multiprocessing.Event()   
    train_.clear()
    mfg_read.clear()
    # mini_batch = 1
    sampler = NeighborSampler(
        [15, 10, 5],  
    )
    file.write(f"Launching processes {time.time()} \n")
    mfg_transfer_ = torch.multiprocessing.Process(target=mfg_transfer, args=( mfg, sampling, tail, head, 
                                                    mini_batch, size, fanout, mfg_read, train_, sampler.edge_dir))
    train_pr = torch.multiprocessing.Process(target=training, args=(size, fanout, mfg_read, train_,
                                                                    model, args.batch_size, mini_batch, args.epoch))
    mfg_transfer_.start()
    train_pr.start()
    

    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=torch.device("cpu"),
        skip_mfg=True,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_prefetch_thread= True if args.prefetch_thread == 1 else False,
        cgg=True,
    )

    sampler.hybrid = hybrid_
    # start_perf(mini_batch, os.getpid(), hybrid_)
    start = time.time()
    sampler.array = array
    sampler.offset = offset
    reset_time = 0
    read_time = read_time_ = wait_time = 0
    file.write(f"Starting sampling {time.time()} \n")
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        start_ = time.time()
        
        if hybrid_: 
            reset_shm(offset)

        i = 0
        reset_time += time.time() - start_
        for it, (_, _, b_) in enumerate(
            train_dataloader
        ):
            start1 = time.time()
            while tail.value != -1 and head.value % mini_batch < tail.value % mini_batch:
                continue
            wait_time += time.time() - start1
            head.value += 1 #% mini_batch
            mfg.set()
            
            start1 = time.time()
            while tail.value == -1 and head.value % mini_batch == 0:
                continue
            wait_time += time.time() - start1
                        
        # file.write(f"Epoch {epoch} : {time.time() - start_ - reset_time: .4f}s, {reset_time:.4f}s, {time.time()}\n")
    # mfg.set()
    # stop_perf()
    end = time.time()
    sampling.set()
    print("Sampling done")
    file.write(f"Sampling time: {end - start - wait_time:.4f}s , Creating CPU shared memory: {create_time:.4f}s, Wait time: {wait_time:.4f}s, {time.time()}\n")
    mfg_transfer_.join()
    train_pr.join()
    # stop_perf()
    # file.close()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = get_args()
    print(f"Training in {args.mode} mode.")
    
    file = open("./results/hybrid_breakdown.txt", "a")

    in_size, out_size = fetch_shapes()
    model = SAGE(in_size, 256, out_size)
    # del dataset, g
    # model training
    print("Training...", args.batch_size, args.hybrid)
    print("PID : ", os.getpid())
    # batch_size_ = [1024]
    # for i in batch_size_:
    #     args.batch_size = i
    train(file, args, model)
    file.write(f"Dataset {args.dataset}, Batch size {args.batch_size}, Hybrid {args.hybrid}, Epochs {args.epoch}\n")
    
    file.close()
