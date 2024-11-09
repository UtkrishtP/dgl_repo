from imports import *
# breakdown numbers for basic hybrid + free space logic
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

def pre_epoch():
    '''
        1. nfeat_cache_size.
        2. ggg footprint.
        3. ggg end-end, gpu sampling, ET times.
        4. get_mfg_size - approximate it using gpu and cpu samples , a total of 3 should be a good average.
        5. cpu_sampling times, also best #workers.
        6. 
    '''
    return

def free_space_worker(tail_gpu, mfg_size_array, free_mfg_hbm, train):
    old_ptr = tail_gpu.value

    while not train.is_set():
        if old_ptr != tail_gpu.value:
            old_ptr = 0 if old_ptr == -1 else tail_gpu.value
            free_mfg_hbm.value += mfg_size_array[old_ptr % len(mfg_size_array)]
            # print("Free space worker: ", free_mfg_hbm.value)

def fetch_mfg_gpu_shm(blocks, array_gpu, offset_gpu_read, fanout):
    for layer in range(len(fanout)):
        if layer == 0:
            block, input_nodes = dgl.hetero_from_gpu_shared_memory(array_gpu, offset_gpu_read, 0)
        elif layer == len(fanout) - 1:
            block, output_nodes = dgl.hetero_from_gpu_shared_memory(array_gpu, offset_gpu_read, 0)
        else:
            block, _ = dgl.hetero_from_gpu_shared_memory(array_gpu, offset_gpu_read, -1)
        blocks.append(block)
    blocks[0].srcdata["_ID"] = input_nodes[0]
    blocks[-1].dstdata["_ID"] = output_nodes[0]
    return blocks

def run_ggg(ggg_train_dataloader, model, opt,):
    total_loss = 0
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
    
    return total_loss

def run_gg(ggg_train_dataloader, model, opt, head_gpu, tail_gpu, mini_batch, 
           fanout, array_gpu, offset_gpu_read, epoch, deque, file, file1, g, 
           val_dataloader, num_classes,):
    total_loss = mb = cgg_time = 0
    # Insert logic to prevent returning before consuming the complete mini_batch
    while mb != mini_batch and epoch.value > 0:
        while tail_gpu.value < head_gpu.value and epoch.value > 0:
            start = time.time()
            blocks = []
            s1 = time.time()
            blocks = fetch_mfg_gpu_shm(blocks, array_gpu, offset_gpu_read, fanout)
            deque[0] += time.time() - s1

            x = ggg_train_dataloader._cgg_on_demand("feat", "_N", blocks[0].srcdata["_ID"])
            y = ggg_train_dataloader._cgg_on_demand("label", "_N", blocks[-1].dstdata["_ID"]) 
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            total_loss += loss.item()
            opt.step()
            cgg_time += time.time() - start
            mb += 1
            # print("Consumer GPU: ", tail_gpu.value, tail_gpu.value % mini_batch, tail_gpu.value / mini_batch, time.time())
            tail_gpu.value += 1
            if (tail_gpu.value % mini_batch) == 0:
                print("GG:", epoch.value)
                reset_shm(offset_gpu_read)
                epoch.value -= 1
                file.write(f"GG done for epoch {tail_gpu.value / mini_batch} : {cgg_time:.4f}s\n")
                cgg_time = 0
                mb = 0
                total_loss = 0
                break
        if mb == 0 and tail_gpu.value >= head_gpu.value:
            return

def training_worker(sampler_, size, fanout, train_, model, batch_size, mini_batch, 
                    epoch, head_gpu, tail_gpu, gpu_pinned, cache_size):
    file = open("../results/hybrid_breakdown.txt", "a")
    file1 = open("../results/hybrid_accuracy.txt", "a")
    file.write(f"\nTraining process launched: {time.time()} \n")
    start = time.time()

    in_size, out_size = fetch_shapes()
    offset_gpu_read = create_shmoffset(1024, "offset_gpu_read")
    array_gpu = get_shm_ptr("array_gpu", size, 0)
    device = torch.device("cuda")
    train_idx, val_idx, test_idx, g = fetch_all()
    file.write(f"Reading from shared memory: {time.time() - start}s\n")

    if sampler_ == "nbr":
        sampler = NeighborSampler(
            fanout, 
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"], 
        )
    
    if sampler_ == "lbr":
        sampler = LaborSampler(
            fanout,  
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )

    ggg_train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        use_uva=True,
        gpu_cache={"node": {"feat": cache_size}},
        # transfer_mfg_gpu=transfer_mfg_gpu,
    )
    
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )

    gpu_pinned.set()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    model.to(device)
    file.write(f"Training process started: {time.time()} \n")
    # print("Training process: ", os.getpid(), flush=True)
    # while True:
    #     if os.path.exists("/tmp/break_train"):
    #         break
    # print("Training Resuming after SIGCONT")
    deque = [0]
    s = time.time()
    ggg_time = cgg_time = 0
    while epoch.value > 0:
        total_loss = 0
        start = time.time()
        run_gg(ggg_train_dataloader, model, opt, head_gpu, tail_gpu, 
               mini_batch, fanout, array_gpu, offset_gpu_read, epoch, deque, file,
               file1, g, val_dataloader, out_size,)
        cgg_time += time.time() - start
        
        if epoch.value <= 0:
            break
        start = time.time()
        total_loss = run_ggg(ggg_train_dataloader, model, opt,)
        ggg_time += time.time() - start
        epoch.value -= 1
        file.write(f"GGG {epoch.value} : {ggg_time:.4f}s, {time.time()}\n")
        # print("GGG done")
    train_.set()
    file.write(f"Train: {time.time() - s}s, GGG time: {ggg_time:.4f}s, CGG time:{cgg_time:.4f}s GPU read time:{deque[0]:.4f}s\n")
    print("Training done")
    file.close()
    file1.close()

def free_space(free_space_hbm, mfg_size, free_mfg_hbm):
    nvmlInit()
    gpu = 0 #GPU 0
    handle = nvmlDeviceGetHandleByIndex(gpu)
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    if (mem_info.free + free_space_hbm.value + free_mfg_hbm.value) < mfg_size:
        nvmlShutdown()
        # print("Not enough memory: ", mem_info.free, free_space_hbm.value, free_mfg_hbm.value, mfg_size)
        return 0  # Not enough memory, clear the event
    else:
        nvmlShutdown()
        return 1  # Enough memory, set the event

def to_gpu_shared_memory(blocks, array_gpu, offset_gpu, fanout,):
    layer = 0
    for block in blocks:
        if layer == 0:
            for nframe in block._node_frames:
                if nframe:
                        block.shared_memory_gpu(array_gpu, offset_gpu, nframe['_ID'], 0)
        elif layer == len(fanout) - 1:
            for nframe in block._node_frames:
                if nframe:
                        block.shared_memory_gpu(array_gpu, offset_gpu, nframe['_ID'], 0)
        else:
                block.shared_memory_gpu(array_gpu, offset_gpu, torch.empty(0), -1)
        layer += 1

def fetch_mfg_size(blocks):
    sizes = 0
    for block in blocks:
        coo_nfeat_label_sizes = block.get_mfg_size(torch.device("cpu"))
        sizes += coo_nfeat_label_sizes[0][0] + coo_nfeat_label_sizes[0][1] + coo_nfeat_label_sizes[0][2]
    sizes += blocks[0].srcdata["_ID"].shape[0] * 4 + blocks[-1].dstdata["_ID"].shape[0] * 4
    return sizes

def fetch_mfg_cpu_shm(blocks, array, read_offset, fanout, edge_dir):
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

def transfer_mfg_gpu(stream1, blocks):
    with torch.cuda.stream(stream1):
        blocks = recursive_apply(
                blocks, lambda x: x.to("cuda", non_blocking=True))
    
    return blocks
        
def mfg_transfer_worker(mfg, sampling, tail, head, mini_batch, 
    size, fanout, mfg_read, train_, edge_dir, head_gpu, tail_gpu, gpu_pinned, 
    free_space_hbm, mfg_size_array, free_mfg_hbm):

    file = open("../results/hybrid_breakdown.txt", "a")
    # start_perf(mini_batch, os.getpid(), e)
    file.write(f"MFG transfer launched: {time.time()} \n")

    #Fetch shared memory regions and create offsets
    array = get_shm_ptr("array", size, 0)
    array_gpu = get_shm_ptr("array_gpu", size, 0)
    read_offset = create_shmoffset(1024, "read_offset")
    offset_gpu = create_shmoffset(1024, "offset_gpu")
    read_time = transfer_time = reset_time = gpu_enqueue = 0

    stream1 = torch.cuda.Stream(device=torch.device("cuda"))
    mfg.wait()
    # print("Transfer process: ", os.getpid(), flush=True)
    # while True:
    #     if os.path.exists("/tmp/break_transfer"):
    #         break
    start = time.time()
    gpu_blocks = [None] * mini_batch
    sizes = 0
    mfg_size_time = free_space_wait_time = gpu_consumer_wait_time = 0
    gpu_pinned.wait()
    while not sampling.is_set():
        # time.sleep(2)
        while tail.value < head.value:
            
            # Fetching MFGs from cpu shared memory
            s1 = time.time()
            blocks = []
            fetch_mfg_cpu_shm(blocks, array, read_offset, fanout, edge_dir)
            read_time += time.time() - s1
            # print("Consumer: ", tail.value, tail.value % mini_batch, time.time(), end=" ", flush=True)
            # print_offset(read_offset)
            tail.value += 1 
            # Fetching MFG sizes
            s1 = time.time()
            sizes = fetch_mfg_size(blocks)
            mfg_size_time += time.time() - s1

            '''
                Wait for GPU to finish processing the previous MFG.
                This rate controls the speed of the producer(mfg_transfer) and consumer(training) processes.
                We are maintaining a circular buffer of mini_batch size.
            '''
            # while tail_gpu.value != 0 and tail_gpu.value < head_gpu.value and head_gpu.value % mini_batch == tail_gpu.value % mini_batch:
            s1 = time.time()
            while head_gpu.value - tail_gpu.value >= (mini_batch - 4):
                if sampling.is_set():
                    break
                continue
            gpu_consumer_wait_time = time.time() - s1   
            if sampling.is_set():
                break
            # Check if there is enough space in HBM
            s1 = time.time()
            while not free_space(free_space_hbm, sizes, free_mfg_hbm):
                continue
            free_space_wait_time += time.time() - s1

            # Transfer to GPU
            blocks = transfer_mfg_gpu(stream1, blocks)

            gpu_blocks[head_gpu.value % mini_batch] = blocks
            # Using cudaIPC to buffer MFG's in GPU.
            s1 = time.time()
            to_gpu_shared_memory(blocks, array_gpu, offset_gpu, fanout)
            gpu_enqueue += time.time() - s1
            
            # Updating the head and tail pointers for gpu/cpu signalling
            s1 = time.time()
            # print("Producer GPU: ", head_gpu.value, head_gpu.value % mini_batch, head_gpu.value / mini_batch, time.time())
            
            head_gpu.value += 1
            mfg_size_array[head_gpu.value % mini_batch] = sizes
            # while tail_gpu.value == 0 and head_gpu.value % mini_batch == 0:
            #     continue
            if (tail.value % mini_batch) == 0:
                # file.write(f"MFG transfer done for epoch {tail.value / mini_batch} : {time.time()}\n")
                reset_shm(read_offset)
                reset_shm(offset_gpu)
            reset_time += time.time() - s1
            # print("Transfer: ", head.value, tail.value, input_nodes[0].shape)
        # print(f"Transfer {time.time()} : {time.time() - s: .4f}s")
    print("MFG sizes : ", sizes)
    # print("MFG size time: ", mfg_size_time, mfg_size_time_py)
    end = time.time()
    # stop_perf()
    mfg_read.set()

    # We are mainting global scopes for the GPU shared memory regions, so we need to wait for the taining process
    # to finish before we can terminate the currnet process.
    print("MFG done")
    train_.wait()
    file.write(f"MFG Transfer E2E: {end - start:.4f}s, CPU Shared read: {read_time}s,"
               f"Enqueue: {gpu_enqueue:.4f}s, TR: {transfer_time:.4f}s,"
               f"GPU consumer wait: {gpu_consumer_wait_time:.4f}s, HBM full wait time: {free_space_wait_time:.4f}s,"
               f"{time.time()} \n")
    file.close()

def main_worker(file, args, model, train_idx, val_idx, test_idx, g):
    # create sampler & dataloader
    # train_idx, _, _, g = fetch_train_graph()
    mini_batch = (train_idx.shape[0] + args.batch_size - 1) // args.batch_size
    hybrid_ = True if args.hybrid == 1 else False
    size = args.mfg_size * 1024 * 1024 * 1024
    head = torch.multiprocessing.Value(ctypes.c_long, 0)
    tail = torch.multiprocessing.Value(ctypes.c_long, 0)
    head_gpu = torch.multiprocessing.Value(ctypes.c_long, 0)
    tail_gpu = torch.multiprocessing.Value(ctypes.c_long, 0)
    epoch_ = torch.multiprocessing.Value(ctypes.c_long, args.epoch)
    mfg_size_array = torch.multiprocessing.Array(ctypes.c_long, mini_batch)
    free_mfg_hbm = torch.multiprocessing.Value(ctypes.c_long, 0)
    free_space_hbm = torch.multiprocessing.Value(ctypes.c_long, -(args.ggg_footprint * (1024**3))) # Initialize with -(ggg_footprint + cache_size)
    train_ = torch.multiprocessing.Event()
    mfg_read = torch.multiprocessing.Event()
    gpu_pinned = torch.multiprocessing.Event()
    mfg = torch.multiprocessing.Event()
    sampling = torch.multiprocessing.Event()
    sampling.clear()
    mfg.clear()
    gpu_pinned.clear()   
    train_.clear()
    mfg_read.clear()

    array = []
    array_gpu = []
    offset = []
    st = time.time()
    print("Sharing memory: ", mini_batch)
    # start_perf(mini_batch, os.getpid(), args.epoch)
    if hybrid_: 
        array = create_shmarray(size, args.madvise, "array", args.pin_mfg)
        array_gpu = create_shmarray(mini_batch * len(args.fan_out) * (1024 * 48 + (3 * 64)), args.madvise, "array_gpu", args.pin_mfg)
        offset = create_shmoffset(1024, "offset")

    print("Memory shared")
    create_time = time.time() - st
    if args.sampler == "nbr":
        sampler = NeighborSampler(
            args.fan_out,  
        )
    
    if args.sampler == "lbr":
        sampler = LaborSampler(
            args.fan_out,
        )
    file.write(f"Launching sampler processes {time.time()} \n")
    mfg_transfer_ = torch.multiprocessing.Process(target=mfg_transfer_worker, args=( mfg, sampling, tail, head, 
                                                    mini_batch, size, args.fan_out, mfg_read, train_, sampler.edge_dir,
                                                    head_gpu, tail_gpu, gpu_pinned, free_space_hbm, mfg_size_array,
                                                    free_mfg_hbm))
    train_pr = torch.multiprocessing.Process(target=training_worker, args=(args.sampler, size, args.fan_out, train_,
                                                    model, args.batch_size, mini_batch, epoch_, 
                                                    head_gpu, tail_gpu, gpu_pinned, args.cache_size))
    free_gpu_mem = torch.multiprocessing.Process(target=free_space_worker, args=(tail_gpu, mfg_size_array, 
                                                    free_mfg_hbm, train_))
    mfg_transfer_.start()
    train_pr.start()
    free_gpu_mem.start()

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
        use_alternate_streams=False,
    )

    sampler.hybrid = hybrid_
    # start_perf(mini_batch, os.getpid(), hybrid_)
    
    sampler.array = array
    sampler.offset = offset
    reset_time = 0
    wait_time = 0
    gpu_pinned.wait()
    start = time.time()
    file.write(f"Starting sampling {time.time()} \n")
    while epoch_.value > 0 and args.epoch > 0:
        start_ = time.time()
        if hybrid_: 
            reset_shm(offset)
        reset_time += time.time() - start_
        for it, (_, _, b_) in enumerate(
            train_dataloader
        ):
            '''
                Wait for CPU to finish transferring the previous MFG.
                This rate controls the speed of the producer(sampler) and consumer(mfg_transfer) processes.
                We are maintaining a circular buffer of mini_batch size.
            '''
            start1 = time.time()
            while head.value - tail.value >= (mini_batch - 4):
                continue
            wait_time += time.time() - start1
            # print("Producer: ", head.value, head.value % mini_batch, time.time(), end=" ", flush=True)
            # print_offset(offset)
            head.value += 1
            mfg.set()
        args.epoch -= 1
                        
    # mfg.set()
    # stop_perf()
    end = time.time()
    sampling.set()
    print("Sampling done")
    file.write(f"Sampling time: {end - start - wait_time:.4f}s , Creating CPU shared memory: {create_time:.4f}s, Wait time: {wait_time:.4f}s, {time.time()}\n")
    mfg_transfer_.join()
    train_pr.join()
    free_gpu_mem.join()
    # stop_perf()
    # file.close()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = get_args()
    print(f"Training in {args.mode} mode.")
    
    file = open("../results/hybrid_breakdown.txt", "a")

    in_size, out_size = fetch_shapes()
    train_idx, val_idx, test_idx, g = fetch_all()

    model = SAGE(in_size, 256, out_size, len(args.fan_out))
    print("Training...", args.batch_size, args.hybrid)
    print("PID : ", os.getpid())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file.write(f"{timestamp} Dataset {args.dataset}, Batch size {args.batch_size}, Hybrid {args.hybrid}, Epochs {args.epoch}\n")
    file1 = open("../results/hybrid_accuracy.txt", "a")
    
    main_worker(file, args, model, train_idx, val_idx, test_idx, g)

    file1.write(f"{timestamp} Dataset {args.dataset}, Batch size {args.batch_size}, Hybrid {args.hybrid}, Epochs {args.epoch}\n\n")
    
    file.close()
    file1.close()