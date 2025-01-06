from imports import *
'''
    1. Logic for lbr sampler
'''
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

def run_ggg(ggg_train_dataloader, model, opt):
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
        # print("GGG:", it)
    
    return total_loss

def run_gg(ggg_train_dataloader, model, opt, head_gpu, tail_gpu, mini_batch, 
           fanout, array_gpu, offset_gpu_read, epoch, deque, file, mfg_buffer_size,
           head_cpu, tail_cpu, slack, data, data_mfg, extract_nfeats):
    total_loss = mb = cgg_time = extract_time = train_time = inner_loop_time = outer_loop_time = 0
    # file.write(f"Transferred MFGs:  {head_gpu.value % mini_batch}, Head: {head_cpu.value},"
    #         f" Tail: {tail_cpu.value}, Produced MFGs:{head_cpu.value - tail_cpu.value} Slack:{slack}\n")
    if (head_gpu.value % mini_batch) < slack and (head_cpu.value - tail_cpu.value) < slack:
            return
    _s_ = time.time()
    while mb != mini_batch and epoch.value > 0:
        if mb == 0:
            # file.write(f"Transferred MFGs:  {head_gpu.value % mini_batch}, Head: {head_cpu.value},"
            #            f" Tail: {tail_cpu.value}, Produced MFGs:{head_cpu.value - tail_cpu.value} Slack:{slack}\n")
            data_mfg.append([time.time(),head_gpu.value % mini_batch, head_cpu.value - tail_cpu.value, slack])
            if tail_gpu.value >= head_gpu.value:
                # file.write(f"Total MB consumed: {tail_gpu.value}, {head_gpu.value}\n")
                return
            if (head_gpu.value % mini_batch) < slack and (head_cpu.value - tail_cpu.value) < slack:
                return
        # with util.Timer() as t:
        s_ = time.time()
        while tail_gpu.value < head_gpu.value and epoch.value > 0:
            start = time.time()
            blocks = []
            # with util.Timer() as fetch_mfg_timer:
            s1 = time.time()
            blocks = fetch_mfg_gpu_shm(blocks, array_gpu, offset_gpu_read, fanout)
            # deque[0] += fetch_mfg_timer.elapsed_secs
            deque[0] += time.time() - s1

            extract_nfeats.clear()
            # with util.Timer() as extract_timer:
            x = ggg_train_dataloader._cgg_on_demand("feat", "_N", blocks[0].srcdata["_ID"])
            y = ggg_train_dataloader._cgg_on_demand("label", "_N", blocks[-1].dstdata["_ID"]) 
            extract_nfeats.set()

            # with util.Timer() as train_timer:
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            total_loss += loss.item()
            opt.step()
            # extract_time += extract_timer.elapsed_secs
            # train_time += train_timer.elapsed_secs
            cgg_time += time.time() - start
            mb += 1
            # print("Consumer GPU: ", tail_gpu.value, end=" ", flush=True) #tail_gpu.value % mini_batch, tail_gpu.value / mini_batch, time.time())
            # print_offset(offset_gpu_read)
            tail_gpu.value += 1
            # print("GG done:", tail_gpu.value)
            if (tail_gpu.value % mfg_buffer_size) == 0:
                reset_shm(offset_gpu_read)

            if (tail_gpu.value % mini_batch) == 0:
                print("GG:", epoch.value, cgg_time, flush=True)
                epoch.value -= 1
                data.append([time.time(),"GG",extract_time,train_time,deque[0],cgg_time - deque[0],cgg_time])
                # file.write(f"GG done for epoch {tail_gpu.value / mini_batch} : {extract_time}, {train_time}, {deque[0]} {cgg_time:.4f}, \n")
                cgg_time = mb = total_loss = deque[0] = extract_time = train_time = 0
                break
            # t.elapsed_secs = 0
        inner_loop_time += time.time() - s_
    outer_loop_time += time.time() - _s_
    file.write(f"ET Stall(s): {outer_loop_time - inner_loop_time}\n")
    # file.write(f"Total MB consumed: {tail_gpu.value}, {head_gpu.value}\n")

def free_space(file, free_space_hbm, mfg_size, free_mfg_hbm):
    nvmlInit()
    gpu = 0 #GPU 0
    handle = nvmlDeviceGetHandleByIndex(gpu)
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    # if (mem_info.free + free_space_hbm.value) < mfg_size or free_mfg_hbm.value > mfg_size:
    if (mem_info.free + torch.cuda.memory_stats()['inactive_split_bytes.all.current']) < mfg_size:
        # if free_mfg_hbm.value > 0:
        #     free_mfg_hbm.value -= mfg_size
        # file.write(f"\nFree space: {mem_info.free}, {free_mfg_hbm.value}, MFG_size: {mfg_size},\n")
        # file.write(f"{torch.cuda.memory_stats()['inactive_split_bytes.all.current']}, {torch.cuda.memory_stats()['inactive_split_bytes.large_pool.current']}, {torch.cuda.memory_stats()['inactive_split_bytes.small_pool.current']}")
        # file.write(f"{torch.cuda.memory_stats()['allocated_bytes.all.current']}, {torch.cuda.memory_stats()['allocated_bytes.all.peak']},")
        # file.write(f"{torch.cuda.memory_stats()['reserved_bytes.all.current']}, {torch.cuda.memory_stats()['reserved_bytes.all.peak']},")
        # file.write(f"{torch.cuda.memory_stats()['active_bytes.all.current']}, {torch.cuda.memory_stats()['active_bytes.all.peak']}")
        # file.write(f"{torch.cuda.memory_summary()}\n")
        file.close()
        nvmlShutdown()
        return 0  # Not enough memory, clear the event
    else:
        # file.write(f"\nFree space: {mem_info.free}, {free_mfg_hbm.value}, MFG_size: {mfg_size},\n")
        # file.write(f"{torch.cuda.memory_stats()['inactive_split_bytes.all.current']}, {torch.cuda.memory_stats()['inactive_split_bytes.large_pool.current']}, {torch.cuda.memory_stats()['inactive_split_bytes.small_pool.current']}")
        # file.write(f"{torch.cuda.memory_stats()['allocated_bytes.all.current']}, {torch.cuda.memory_stats()['allocated_bytes.all.peak']},")
        # file.write(f"{torch.cuda.memory_stats()['reserved_bytes.all.current']}, {torch.cuda.memory_stats()['reserved_bytes.all.peak']},")
        # file.write(f"{torch.cuda.memory_stats()['active_bytes.all.current']}, {torch.cuda.memory_stats()['active_bytes.all.peak']}")
        # file.write(f"{torch.cuda.memory_summary()}\n")
        file.close()
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
        coo_row_col_sizes = block.get_mfg_size(torch.device("cpu"))
        sizes += coo_row_col_sizes[0][0] + coo_row_col_sizes[0][1] + coo_row_col_sizes[0][2]
    sizes += blocks[0].srcdata["_ID"].shape[0] * 8 + blocks[-1].dstdata["_ID"].shape[0] * 8
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
    # with torch.cuda.stream(stream1):
    blocks = recursive_apply(
            blocks, lambda x: x.to("cuda", non_blocking=True))
    return blocks

def set_core_affinity(core):
    psutil.Process().cpu_affinity(core)
    set_num_threads(len(core))

def mfg_transfer_worker(mfg, sampling, tail_cpu, head_cpu, mini_batch, 
    size, fanout, mfg_core, enable_affinity, train_, edge_dir, head_gpu, tail_gpu, 
    free_space_hbm, mfg_size_array, free_mfg_hbm, mfg_buffer_size, size_gpu, diff, extract_nfeats):
    
    if enable_affinity:
        set_core_affinity(mfg_core)
    file = open("../results/hybrid/lbr/sampling.txt", "a")
    
    # start_perf(mini_batch, os.getpid(), e)
    # file.write(f"MFG transfer launched: {time.time()} \n")
    launch_time = time.time()
    # cons_file = open("../results/cons.txt", "w+")
    #Fetch shared memory regions and create offsets
    array = get_shm_ptr("array_cpu", size, 0)
    array_gpu = get_shm_ptr("array_gpu", size_gpu, 0)
    # offset_cpu_read = create_shmoffset(size, "offset_cpu_read")
    offset_cpu_write = get_shm_ptr("offset_cpu_write", 16, 0)
    offset_cpu_read = get_shm_ptr("offset_cpu_read", 16, 0) 
    offset_gpu_write = create_shmoffset(size, "offset_gpu_write")
    read_time = transfer_time = reset_time = gpu_enqueue = 0

    # stream1 = torch.cuda.Stream(device=torch.device("cuda"))
    stream1 = torch.cuda.current_stream()
    mfg.wait()
    # print("Transfer process: ", os.getpid(), flush=True)
    # while True:
    #     if os.path.exists("/tmp/break_transfer"):
    #         break
    start = time.time()
    gpu_blocks = [None] * mfg_buffer_size # To maintain scope of MFG's until the consumer finishes processing.
    sizes = total_sizes = 0
    mfg_size_time = free_space_wait_time = gpu_consumer_wait_time = mfg_stall_et = 0
    file1 = open("../results/free_memory.txt", "w+")
    # file2 = open("../results/cpu_consumer_offsets.txt", "a+")
    size_tensor = []
    iterations = 0
    while not sampling.is_set():
        # time.sleep(2)
        while tail_cpu.value < head_cpu.value:
            # Fetching MFGs from cpu shared memory
            s1 = time.time()
            blocks = []
            fetch_mfg_cpu_shm(blocks, array, offset_cpu_read, fanout, edge_dir)
            read_time += time.time() - s1
            # print("Consumer: ", tail.value, tail.value % mini_batch, 
            #     read_offset(offset_cpu_read) - read_offset(offset_cpu_write), head.value % mini_batch, time.time(), end=" ", flush=True) #, tail.value % mini_batch, time.time(), end=" ", flush=True)
            # print_offset(offset_cpu_read)
            # cons_file.write(f"Consumer,{tail.value},{tail.value % mini_batch},{read_offset(offset_cpu_read)},Producer,{head.value}\n") 
            tail_cpu.value += 1 
            # Fetching MFG sizes
            s1 = time.time()
            sizes = fetch_mfg_size(blocks)
            # total_sizes += sizes
            mfg_size_time += time.time() - s1

            '''
                Wait for GPU to finish processing the previous MFG.
                This rate controls the speed of the producer(mfg_transfer) and consumer(training) processes.
                We are maintaining a circular buffer of mini_batch size.
            '''
            # while tail_gpu.value != 0 and tail_gpu.value < head_gpu.value and head_gpu.value % mini_batch == tail_gpu.value % mini_batch:
            s1 = time.time()
            while head_gpu.value - tail_gpu.value >= (mfg_buffer_size - 4):
                if sampling.is_set():
                    break
            
            if sampling.is_set():
                break

            gpu_consumer_wait_time = time.time() - s1   
            
            # Check if there is enough space in HBM
            s1 = time.time()
            
            file1 = open("../results/free_memory.txt", "a+")
            while not free_space(file1, free_space_hbm, sizes, free_mfg_hbm):
                file1 = open("../results/free_memory.txt", "a")
            free_space_wait_time += time.time() - s1

            # Release the block before transfer, verify this via nvml_stats
            gpu_blocks[head_gpu.value % mfg_buffer_size] = None

            # Transfer to GPU
            with util.Timer() as transfer:
                mfg_stall_et -= time.time()
                extract_nfeats.wait()
                mfg_stall_et += time.time()
                blocks = transfer_mfg_gpu(stream1, blocks)
            transfer_time += transfer.elapsed_secs

            gpu_blocks[head_gpu.value % mfg_buffer_size] = blocks
            # Using cudaIPC to buffer MFG's in GPU.
            s1 = time.time()
            to_gpu_shared_memory(blocks, array_gpu, offset_gpu_write, fanout)
            gpu_enqueue += time.time() - s1
            
            # Updating the head and tail pointers for gpu/cpu signalling
            s1 = time.time()
            # print("Producer GPU: ", head_gpu.value, end=" ", flush=True) # head_gpu.value % mini_batch, head_gpu.value / mini_batch, time.time())
            # print_offset(offset_gpu_write)
            head_gpu.value += 1
            # print("MFG trnsferred ", head_gpu.value, sizes)
            mfg_size_array[head_gpu.value % mfg_buffer_size] = sizes
            # while tail_gpu.value == 0 and head_gpu.value % mini_batch == 0:
            #     continue
            if (head_gpu.value % mfg_buffer_size) == 0:
                reset_shm(offset_gpu_write)

            if (tail_cpu.value % mini_batch) == 0:
                # file.write(f"MFG transfer done for epoch {tail.value / mini_batch} : {time.time()}\n")
                # print("MFG size", total_sizes / (1024**3))
                reset_shm(offset_cpu_read)            
                total_sizes = 0
                iterations += 1
            reset_time += time.time() - s1
            # print("Transfer: ", head.value, tail.value, input_nodes[0].shape)
        # print(f"Transfer {time.time()} : {time.time() - s: .4f}s")
    # print("MFG size time: ", mfg_size_time, mfg_size_time_py)
    end = time.time()
    # stop_perf()

    # We are mainting global scopes for the GPU shared memory regions, so we need to wait for the taining process
    # to finish before we can terminate the currnet process.
    # print("MFG done")
    train_.wait()
    data = [["MFG Transfer launch", launch_time],
            ["MFG Transfer E2E", end - start],
            ["CPU Shared read", read_time / iterations],
            ["Enqueue", (gpu_enqueue / head_gpu.value) * mini_batch],
            ["Transfer", (transfer_time / head_gpu.value) * mini_batch],
            ["MFG stalls ET", (mfg_stall_et / head_gpu.value) * mini_batch],
            ["GPU consumer wait", gpu_consumer_wait_time],
            ["HBM full wait time", free_space_wait_time],
            ["MFG size query", (mfg_size_time / head_gpu.value) * mini_batch],
            ["# Epochs transferred", iterations],
            ["# MB's transferred", head_gpu.value],
            ]
    print(tabulate(data, headers=["Metric", "Value"], tablefmt="outline", showindex="always", floatfmt=".4f"))
    file.write(tabulate(data, headers=["Metric", "Value"], tablefmt="outline", showindex="always", floatfmt=".4f"))
    file.close()

def mfg_transfer_worker_lbr(mfg, sampling, mini_batch, size, fanout, mfg_core, 
    enable_affinity, train_, head_gpu, tail_gpu, free_space_hbm, mfg_size_array, free_mfg_hbm, 
    mfg_buffer_size, size_gpu, diff, extract_nfeats, cpu_shared_queue, consumed_mfgs):
    
    if enable_affinity:
        set_core_affinity(mfg_core)
    file = open("../results/hybrid/lbr/sampling.txt", "a")
    
    # start_perf(mini_batch, os.getpid(), e)
    # file.write(f"MFG transfer launched: {time.time()} \n")
    
    # cons_file = open("../results/cons.txt", "w+")
    #Fetch shared memory regions and create offsets
    array_gpu = get_shm_ptr("array_gpu", size_gpu, 0)
    offset_gpu_write = create_shmoffset(size, "offset_gpu_write")
    read_time = transfer_time = reset_time = gpu_enqueue = 0

    # stream1 = torch.cuda.Stream(device=torch.device("cuda"))
    stream1 = torch.cuda.current_stream()
    mfg.wait()
    # print("Transfer process: ", os.getpid(), flush=True)
    # while True:
    #     if os.path.exists("/tmp/break_transfer"):
    #         break
    start = time.time()
    gpu_blocks = [None] * mfg_buffer_size # To maintain scope of MFG's until the consumer finishes processing.
    sizes = total_sizes = 0
    
    mfg_size_time = free_space_wait_time = gpu_consumer_wait_time = mfg_stall_et = 0
    file1 = open("../results/free_memory.txt", "w+")
    # file2 = open("../results/cpu_consumer_offsets.txt", "a+")
    size_tensor = []
    iterations = 0
    launch_time = time.time()
    while not sampling.is_set():
        # time.sleep(2)
        # while tail_cpu.value < head_cpu.value:
        # Fetching MFGs from cpu shared memory
        s1 = time.time()
        idx, data = cpu_shared_queue.get()
        # ResumeIteration flags in the queue
        if data is None:
            continue
        blocks = data[2]
        read_time += time.time() - s1
        consumed_mfgs.value = 0 if consumed_mfgs.value == MIN_VALUE_LONG else consumed_mfgs.value
        consumed_mfgs.value += 1

        # Fetching MFG sizes
        s1 = time.time()
        sizes = fetch_mfg_size(blocks)
        # total_sizes += sizes
        mfg_size_time += time.time() - s1

        '''
            Wait for GPU to finish processing the previous MFG.
            This rate controls the speed of the producer(mfg_transfer) and consumer(training) processes.
            We are maintaining a circular buffer of mini_batch size.
        '''
        s1 = time.time()
        while head_gpu.value - tail_gpu.value >= (mfg_buffer_size - 4):
            if sampling.is_set():
                break
        
        if sampling.is_set():
            break

        gpu_consumer_wait_time = time.time() - s1   
        
        # Check if there is enough space in HBM
        s1 = time.time()
        
        file1 = open("../results/free_memory.txt", "a+")
        while not free_space(file1, free_space_hbm, sizes, free_mfg_hbm):
            file1 = open("../results/free_memory.txt", "a")
        free_space_wait_time += time.time() - s1

        # Release the block before transfer, verify this via nvml_stats
        gpu_blocks[head_gpu.value % mfg_buffer_size] = None

        # Transfer to GPU
        with util.Timer() as transfer:
            mfg_stall_et -= time.time()
            extract_nfeats.wait()
            mfg_stall_et += time.time()
            blocks = transfer_mfg_gpu(stream1, blocks)
        transfer_time += transfer.elapsed_secs

        gpu_blocks[head_gpu.value % mfg_buffer_size] = blocks
        # Using cudaIPC to buffer MFG's in GPU.
        s1 = time.time()
        to_gpu_shared_memory(blocks, array_gpu, offset_gpu_write, fanout)
        gpu_enqueue += time.time() - s1
        
        # Updating the head and tail pointers for gpu/cpu signalling
        s1 = time.time()
        # print("Producer GPU: ", head_gpu.value, end=" ", flush=True) # head_gpu.value % mini_batch, head_gpu.value / mini_batch, time.time())
        # print_offset(offset_gpu_write)
        head_gpu.value += 1
        # print("MFG trnsferred ", head_gpu.value, sizes)
        mfg_size_array[head_gpu.value % mfg_buffer_size] = sizes
        # while tail_gpu.value == 0 and head_gpu.value % mini_batch == 0:
        #     continue
        if (head_gpu.value % mfg_buffer_size) == 0:
            reset_shm(offset_gpu_write)

        if (consumed_mfgs.value % mini_batch) == 0:
            iterations += 1
        reset_time += time.time() - s1
        # print("Transfer: ", head.value, tail.value, input_nodes[0].shape)
        # print(f"Transfer {time.time()} : {time.time() - s: .4f}s")
    # print("MFG size time: ", mfg_size_time, mfg_size_time_py)
    end = time.time()
    # stop_perf()

    # We are mainting global scopes for the GPU shared memory regions, so we need to wait for the taining process
    # to finish before we can terminate the currnet process.
    # print("MFG done")
    train_.wait()
    data = [["MFG Transfer launch", launch_time],
            ["MFG Transfer E2E", end - start],
            ["CPU Shared read", (read_time / consumed_mfgs.value) * mini_batch],
            ["Enqueue", (gpu_enqueue / head_gpu.value) * mini_batch],
            ["Transfer", (transfer_time / head_gpu.value) * mini_batch],
            ["MFG stalls ET", (mfg_stall_et / head_gpu.value) * mini_batch],
            ["GPU consumer wait", gpu_consumer_wait_time],
            ["HBM full wait time", free_space_wait_time],
            ["MFG size query", (mfg_size_time / head_gpu.value) * mini_batch],
            ["# Epochs transferred", iterations],
            ["# MB's transferred", head_gpu.value],
            ["Dequed", consumed_mfgs.value]
            ]
    print(tabulate(data, headers=["Metric", "Value"], tablefmt="outline", showindex="always", floatfmt=".4f"))
    file.write(tabulate(data, headers=["Metric", "Value"], tablefmt="outline", showindex="always", floatfmt=".4f"))
    file.close()

def calculate_slack(args, mini_batch):
    '''
        For cases where CPU sampling falls behind ET, we need to calculate the slack time.
        Slack: What % of MFG's should be prepared by sampler to avoid GPU stalling.
        This will help us decide whether to continue with GGG or ET.
    '''
    slack = 0
    if args.t_sample < args.t_et:
        return -1
    else:
        sample_rate = mini_batch / args.t_sample # Samples per second
        inv_et_rate = args.t_et / mini_batch # Time taken to for 1 MB's ET
        slack = (args.t_sample - (args.t_et * 1.1) + inv_et_rate) * sample_rate
        slack = (int)(math.ceil((args.t_sample - args.t_et + inv_et_rate) * sample_rate))
        return slack

def profile_preSC(args, sampler, train_dataloader):
    '''
        Helper function to measure the following:
        1. MFG size
        2. CPU sampling time
        3. MFG transfer time
        4. ET times.
        5. Access frequency for adj_cache and nfeat_cache
    '''
    cpu_sampling = mfg_size = mfg_timer = 0
    
    #warm-up
    for it, (_, _, b_) in enumerate(
            train_dataloader
        ):
        break

    for epoch in range(2):
        if args.sampler == "nbr" and sampler.hybrid:
            reset_shm(sampler.offset)
            
        start = time.time()
        ovhd = 0
        for it, (_, _, block) in enumerate(
            train_dataloader
        ):
            s = time.time()
            mfg_size += fetch_mfg_size(block)
            ovhd += time.time() - s
            with util.Timer() as t:
                block = recursive_apply(
                    block, lambda x: x.to("cuda", non_blocking=True))
            mfg_timer += t.elapsed_secs
        cpu_sampling += (time.time() - start - ovhd)
    print("Done profiling PreSC")
    args.mfg_size = (int)(mfg_size / 2)
    args.mfg_transfer = mfg_timer / 2
    args.t_sample = (cpu_sampling / 2) - args.mfg_transfer
    return

def ggg_footprint(ggg_footprint_, event):
    nvmlInit()
    gpu = 0 #GPU 0
    handle = nvmlDeviceGetHandleByIndex(gpu)
    used_memory = [0]
    while not event.is_set():
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        used_memory.append(mem_info.used)
        # time.sleep(0.1)
    nvmlShutdown()
    ggg_footprint_.value = max(used_memory)

def profile_ggg(args, ggg_dataloader, model, opt):
    '''
        Helper function to measure the following:
        1. GGG time
        2. ET time
        3. GGG footprint
    '''
    ggg_timer = train_timer = nfeat_fetch = 0
    #warm-up
    for it, (input_nodes, output_nodes, blocks) in enumerate(ggg_dataloader):
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
        y_hat = model(blocks, x)
        loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        break
    
    ggg_dataloader.nfeat_timer = ggg_dataloader.index_transfer = 0
    ggg_footprint_ = torch.multiprocessing.Value(ctypes.c_long, 0)
    footprint_event = torch.multiprocessing.Event()
    footprint_event.clear()
    footprint_thread = torch.multiprocessing.Process(target=ggg_footprint, args=(ggg_footprint_, footprint_event))
    footprint_thread.start()
    start = time.time()

    for it, (input_nodes, output_nodes, blocks) in enumerate(ggg_dataloader):
        tic = time.time()
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
        nfeat_fetch += time.time() - tic
        tic = time.time()
        with util.Timer() as t:
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_timer += t.elapsed_secs
    ggg_timer += time.time() - start
    print("GGG profling done")
    footprint_event.set()
    footprint_thread.join()
    print("GGG:", ggg_timer, train_timer, ggg_dataloader.nfeat_timer, nfeat_fetch, ggg_dataloader.index_transfer)
    args.t_et = train_timer + ggg_dataloader.nfeat_timer + nfeat_fetch
    args.t_ggg = ggg_timer
    args.ggg_footprint = ggg_footprint_.value
    return

def launch_mps(split):
    # user_id = mps_get_user_id()
    # mps_daemon_start()
    # mps_server_start(user_id)
    server_pid = mps_get_server_pid()
    mps_set_active_thread_percentage(server_pid, split)

def sampling_worker(sampler, mfg, sampling, head_cpu, tail_cpu, 
                    epoch_, diff, batch_size, workers, fan_out, hybrid, size, num_threads, hybrid_sampling_timer):
    
    set_num_threads(num_threads)
    file = open("../results/hybrid/lbr/sampling.txt", "a")
    # file.write(f"Sampling process launched: {time.time()} \n")
    launch_time = time.time()
    reset_time = wait_time = 0
    core_list = []
    # for i in range(0, 63):
    #     core_list.append(i)
    # psutil.Process().cpu_affinity(core_list)
    # set_num_threads(64)
    # start_perf(mini_batch, os.getpid(), e)
    # print("Sampling process: ", os.getpid(), flush=True)
    # while True:
    #     if os.path.exists("/tmp/break_sampling"):
    #         break
    train_idx, val_idx, test_idx, g = fetch_all()
    offset_cpu_write = get_shm_ptr("offset_cpu_write", 16, 0)
    array_cpu = get_shm_ptr("array_cpu", size, 0)
    mini_batch = (train_idx.shape[0] + batch_size - 1) // batch_size
    if sampler == "nbr":
        sampler = NeighborSampler(
            fan_out,  
        )
    
    if sampler == "lbr":
        sampler = LaborSampler(
            fan_out,
        )
    
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=torch.device("cpu"),
        skip_mfg=True,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=workers,
        persistent_workers=True if workers > 0 else False,
        # use_prefetch_thread= True if prefetch_thread.value == 1 else False,
        use_alternate_streams=False,
    )
    
    sampled_epochs = epoch_.value
    if hybrid:
        sampler.hybrid = hybrid 
        sampler.array = array_cpu
        sampler.offset = offset_cpu_write
    start = time.time()
    
    while epoch_.value > 0 and sampled_epochs > 0:
        start_ = time.time()
        if hybrid:
            reset_shm(offset_cpu_write)
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
            while head_cpu.value - tail_cpu.value >= diff:
                if epoch_.value <= 0:
                    break
                continue
            
            wait_time += time.time() - start1
            if epoch_.value <= 0:
                    break
            # diff = read_offset(offset_cpu_read) - read_offset(offset_cpu_write)
            # while diff <= 1800000 and diff >= 0:
            #     diff = read_offset(offset_cpu_read) - read_offset(offset_cpu_write)
            #     continue
            
            # prod_file.write(f"Producer,{head.value},{head.value % mini_batch},{read_offset(offset_cpu_write)},Consumer,{tail.value}\n") #(int)(head.value / mini_batch), end=" ", flush=True)
            # print_offset(offset_cpu_write)
            head_cpu.value += 1
            mfg.set()
        sampled_epochs -= 1
        # differences = [size_tensor[i + 1] - size_tensor[i] for i in range(len(size_tensor) - 1)]
        # print("MFG sizes min: ", min(differences), "max: ", max(differences), "avg: ", sum(differences) / len(differences))
    # mfg.set()
    # stop_perf()
    end = time.time()
    sampling.set()
    # print("Sampling done")
    end_ts = time.time()
    sampling_time = end - start - wait_time - reset_time
    epochs = head_cpu.value / mini_batch
    residual = head_cpu.value % mini_batch
    hybrid_sampling_timer.value = (sampling_time / head_cpu.value ) * mini_batch
    # Pair each header with its corresponding value
    table_data = [
        ["Sampler worker launch timestamp", launch_time],
        ["Sampling time", (sampling_time / head_cpu.value ) * mini_batch],
        ["Epochs", epochs],
        ["#Residual MBs", residual],
        ["Wait time", wait_time],
        ["End timestamp", end_ts]
    ]

    # Now you can tabulate:
    table_str = tabulate(
        table_data,
        headers=["Metric", "Value"],     # Column labels
        tablefmt="outline",
        showindex="always",
        floatfmt=".4f"
    )
    file.write(table_str)
    file.close()

def sampling_worker_lbr(sampler, mfg, sampling, epoch_, diff, batch_size, workers,
                     fan_out, hybrid_sampling_timer, cpu_shared_queue, consumed_mfgs):
    
    file = open("../results/hybrid/lbr/sampling.txt", "a")
    # file.write(f"Sampling process launched: {time.time()} \n")
    launch_time = time.time()
    reset_time = wait_time = 0
    core_list = []
    # for i in range(0, 63):
    #     core_list.append(i)
    # psutil.Process().cpu_affinity(core_list)
    # set_num_threads(64)
    # start_perf(mini_batch, os.getpid(), e)
    # print("Sampling process: ", os.getpid(), flush=True)
    # while True:
    #     if os.path.exists("/tmp/break_sampling"):
    #         break
    train_idx, val_idx, test_idx, g = fetch_all()
    mini_batch = (train_idx.shape[0] + batch_size - 1) // batch_size
    
    if sampler == "lbr":
        sampler = LaborSampler(
            fan_out,
            layer_dependency=True,
            importance_sampling=-1,
        )
    
    if sampler == "lbr2":
        sampler = LaborSampler(
            fan_out,
        )
    
    # train_dataloader.graph = g
    # train_dataloader.graph_sampler = sampler
    # train_dataloader.hybrid = True
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=torch.device("cpu"),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=workers,
        use_uva=False,
        persistent_workers=True if workers > 0 else False,
        cpu_shared_queue=cpu_shared_queue,
        pin_prefetcher=False,
        hybrid=True,
        hybrid_wrapper=True,
        # skip_mfg=True,
        # cgg_on_demand=True,
        # gather_pin_only=True,
    )
    
    # file = open("../results/hybrid/lbr/counter.txt", "a")
    sampled_epochs = epoch_.value
    iterations = 0
    start = time.time()
    while epoch_.value > 0 and sampled_epochs > 0:
        # Current prefetch_depth is 2, hence during init (2 * num_workers) mini_batches are enqueued for sampling.
        iterator_obj = train_dataloader.iterate()
        # iterations += 8
        # print(iterator_obj._tasks_outstanding)
        while cpu_shared_queue.qsize() < 1:
            mfg.clear()
            continue
        mfg.set()
        # This condition enqueues all mini_batch to worker's queue
        while iterator_obj._tasks_outstanding < mini_batch: #+ args.workers
            iterator_obj.fetch_next()
            iterations += iterator_obj._tasks_outstanding
        
        # We wait until all the workers have produced a sample.
        while ((consumed_mfgs.value + cpu_shared_queue.qsize()) % mini_batch) != 0:
            # print(f"{consumed_mfgs.value},{cpu_shared_queue.qsize()}\n")
            continue
        # print(iterator_obj._tasks_outstanding)
        print("Queue size:", cpu_shared_queue.qsize())
        
        sampled_epochs -= 1
        
    end = time.time()
    sampling.set()
    # print("Sampling done")
    end_ts = time.time()
    sampling_time = end - start
    epochs = iterations / mini_batch
    residual = iterations % mini_batch
    hybrid_sampling_timer.value = (sampling_time / iterations ) * mini_batch
    # Pair each header with its corresponding value
    table_data = [
        ["Sampler worker launch timestamp", launch_time],
        ["Sampling time", hybrid_sampling_timer.value ],
        ["Epochs", epochs],
        ["Queue size:", cpu_shared_queue.qsize()],
        ["#Residual MBs", residual],
        ["Wait time", wait_time],
        ["End timestamp", end_ts]
    ]

    # Now you can tabulate:
    table_str = tabulate(
        table_data,
        headers=["Metric", "Value"],     # Column labels
        tablefmt="outline",
        showindex="always",
        floatfmt=".4f"
    )
    file.write(table_str)
    file.close()

def check_core_assignment(args):
    '''
        Check if the number of cores assigned to the mfg_transfer and training processes are within the limits
        of total available cores in system.
    '''
    if args.mfg_core[0] > (int)(os.cpu_count() / 2):
        args.mfg_core[0] = (int)(os.cpu_count() / 2)
        args.train_core[0] = (int)(args.mfg_core[0] / 2)

def main_worker(file, args, model, train_idx, val_idx, test_idx, g, timestamp):
    # create sampler & dataloader
    array = array_gpu = offset_cpu_read = offset_cpu_write = None
    dataloader_init = create_shm_time = preSC_timer = ggg_profile = 0
    mini_batch = (train_idx.shape[0] + args.batch_size - 1) // args.batch_size
    hybrid_ = True if args.hybrid == 1 else False
    # array_gpu_size = mini_batch * len(args.fan_out) * (1024 * 48 + (3 * 64)) * 2
    array_gpu_size = 15*1024*1024*1024
    # check_core_assignment(args)
    head_cpu = torch.multiprocessing.Value(ctypes.c_long, 0)
    tail_cpu = torch.multiprocessing.Value(ctypes.c_long, 0)
    head_gpu = torch.multiprocessing.Value(ctypes.c_long, 0)
    tail_gpu = torch.multiprocessing.Value(ctypes.c_long, 0)
    epoch_ = torch.multiprocessing.Value(ctypes.c_long, args.epoch)
    hybrid_sampling_timer = torch.multiprocessing.Value(ctypes.c_float, 0.0)
    mfg_size_array = torch.multiprocessing.Array(ctypes.c_long, mini_batch)
    free_mfg_hbm = torch.multiprocessing.Value(ctypes.c_long, 0)
    consumed_mfgs = torch.multiprocessing.Value(ctypes.c_long, MIN_VALUE_LONG)
    free_space_hbm = torch.multiprocessing.Value(ctypes.c_long, -(args.ggg_footprint * (1024**3))) # Initialize with -(ggg_footprint + cache_size)
    cpu_shared_queue = torch.multiprocessing.Queue()
    train_ = torch.multiprocessing.Event()
    mfg = torch.multiprocessing.Event()
    sampling = torch.multiprocessing.Event()
    extract_nfeats = torch.multiprocessing.Event()
    sampling.clear()
    mfg.clear()
    train_.clear()
    extract_nfeats.set()
    
    start = time.time()
    if args.sampler == "nbr":
        sampler = NeighborSampler(
            args.fan_out,  
        )
        sampler_ = NeighborSampler(
            args.fan_out, 
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"], 
        )
    
    if args.sampler == "lbr2":
        sampler = LaborSampler(
            args.fan_out,
        )
        sampler_ = LaborSampler(
            args.fan_out, 
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"], 
        )
    
    if args.sampler == "lbr":
        sampler = LaborSampler(
            args.fan_out,
            layer_dependency=True,
            importance_sampling=-1,
        )
        sampler_ = LaborSampler(
            args.fan_out, 
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=torch.device("cpu"),
        skip_mfg=True,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.presc_workers,
        persistent_workers=True if args.presc_workers > 0 else False,
        use_prefetch_thread= True if args.prefetch_thread == 1 else False,
        use_alternate_streams=False,
        # hybrid=True,
        hybrid_wrapper=True,
    )

    ggg_dataloader = DataLoader(
        g,
        train_idx,
        sampler_,
        device=torch.device("cuda"),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        use_uva=True,
        gpu_cache={"node": {"feat": args.cache_size}},
        extract_nfeats=extract_nfeats,
        profiler=True,
    )
    dataloader_init += time.time() - start
    start = time.time()
    if hybrid_: 
        array = create_shmarray(args.mfg_size * (1024 ** 3), args.madvise, "array_cpu", args.pin_mfg)
        array_gpu = create_shmarray(array_gpu_size, args.madvise, "array_gpu", args.pin_mfg)
        offset_cpu_write = create_shmoffset(args.mfg_size * 2, "offset_cpu_write")
        offset_cpu_read = create_shmoffset(args.mfg_size * 2, "offset_cpu_read")
        if args.sampler == "nbr":
            sampler.hybrid = hybrid_
            sampler.array = array
            sampler.offset = offset_cpu_write

    offset_gpu_read = create_shmoffset(args.mfg_size * 2, "offset_gpu_read")
    create_shm_time += time.time() - start

    start = time.time()
    set_num_threads(args.num_threads)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    model.to(torch.device("cuda"))
    profile_preSC(args, sampler, train_dataloader)
    preSC_timer += time.time() - start

    start = time.time()
    set_num_threads((int)(os.cpu_count() / 2))
    profile_ggg(args, ggg_dataloader, model, opt)
    ggg_profile += time.time() - start

    args.mfg_per_mb = args.mfg_size / mini_batch
    mfgs_buffer_size = int((torch.cuda.get_device_properties(torch.device("cuda:0")).total_memory - 
                        #(args.cache_size * (args.nfeat_dim / 128) * 0.5 * 1024)  - 
                        (args.ggg_footprint) - (2 * (1024 ** 3))) / args.mfg_per_mb) # Add later mfg_size 
    mfgs_buffer_size = mini_batch if mfgs_buffer_size > mini_batch else mfgs_buffer_size
    print("MFGs buffer size: ", mfgs_buffer_size)
    print("Sharing memory: ", mini_batch)
    diff = (mini_batch - 4) if ((mini_batch * 99) / 100) > (mini_batch - 4) else (int)((mini_batch * 99) / 100)
    slack = calculate_slack(args, mini_batch)
    # start_perf(mini_batch, os.getpid(), args.epoch)
    data = [
        ["Timestamp: ", timestamp],
        ["Dataset", args.dataset],
        ["Batch size", args.batch_size],
        ["Fan-out", args.fan_out],
        ["# epochs", args.epoch],
        ["# workers", args.workers],
        ["# Threads", args.num_threads],
        ["Hidden size", args.hid_size],
        ["Data loader init (s)", dataloader_init],
        ["Create shm (s)", create_shm_time],
        ["PreSC (s)", preSC_timer],
        ["GGG profile (s)", ggg_profile],
        ["t_sample (s)", args.t_sample],
        ["t_et (s)", args.t_et],
        ["mfg_transfer (s)", args.mfg_transfer],
        ["t_ggg (s)", args.t_ggg],
        ["mfg_size (GB)", args.mfg_size / (1024 ** 3)],
        ["ggg_footprint (GB)", (args.ggg_footprint - (args.cache_size * (args.nfeat_dim / 128) * 0.5 * 1024)) / (1024**3)],
        ["cache_size (GB)", (args.cache_size * (args.nfeat_dim / 128) * 0.5) / (1024**2)],
        ["# mini_batch", mini_batch],
        ["mfg_per_mb (MB)", args.mfg_per_mb / (1024 ** 2)],
        ["mfgs_buffer_size (#MBs)", mfgs_buffer_size],
        ["diff  (#MBs)", diff],
        ["slack (#MBs)", slack],
        ["Mps %(MFG transfer)", args.mps_split],
        ["Trainer Worker Timestamp", time.time()],
    ]
    if args.enable_affinity:
        data.append([
        ["MFG pinned core #", args.mfg_core[0]],
        ["Train pinned core #", args.train_core[0]],
        ])

    # Print as a table
    print(tabulate(data, headers=["Metric", "Value"], tablefmt="outline", showindex="always", floatfmt=".4f"))
    file.write(tabulate(data, headers=["Metric", "Value"], tablefmt="outline", showindex="always", floatfmt=".4f"))
    # return
    if args.mps_split != 0:
        launch_mps(args.mps_split)
    if args.sampler == "nbr":
        mfg_transfer_ = torch.multiprocessing.Process(target=mfg_transfer_worker, args=( mfg, sampling, tail_cpu, head_cpu, 
                                                        mini_batch, args.mfg_size * 2, args.fan_out, args.mfg_core, args.enable_affinity,
                                                        train_, sampler.edge_dir, head_gpu, tail_gpu, free_space_hbm, mfg_size_array,
                                                        free_mfg_hbm, mfgs_buffer_size, array_gpu_size, diff, extract_nfeats))
        
        sampling_worker_ = torch.multiprocessing.Process(target=sampling_worker, args=( args.sampler, mfg, sampling, head_cpu, tail_cpu, 
                                                            epoch_, diff, args.batch_size, args.workers, args.fan_out,
                                                            args.hybrid, args.mfg_size * 2, args.num_threads, hybrid_sampling_timer))
    elif args.sampler.startswith("lbr"):
        mfg_transfer_ = torch.multiprocessing.Process(target=mfg_transfer_worker_lbr, args=( mfg, sampling, mini_batch, args.mfg_size * 2, args.fan_out, 
                                                        args.mfg_core, args.enable_affinity, train_,head_gpu, tail_gpu, free_space_hbm,
                                                         mfg_size_array, free_mfg_hbm, mfgs_buffer_size, 
                                                        array_gpu_size, diff, extract_nfeats, cpu_shared_queue, consumed_mfgs))
        
        sampling_worker_ = torch.multiprocessing.Process(target=sampling_worker_lbr, args=(args.sampler, mfg, sampling,
                                                            epoch_, diff, args.batch_size, args.workers, args.fan_out,
                                                            hybrid_sampling_timer, cpu_shared_queue, consumed_mfgs))
    sampling_worker_.start()
    # sampling_worker_.join()
    # return
    mfg_transfer_.start()
    ggg_dataloader.profiler = False
    if args.enable_affinity:
        set_core_affinity(args.train_core)
    file.write(f"\nTraining process started: {time.time()} \n")
    # print("Training process: ", os.getpid(), flush=True)
    # while True:
    #     if os.path.exists("/tmp/break_train"):
    #         break
    # print("Training Resuming after SIGCONT")
    deque = [0]
    s = time.time()
    ggg_time = cgg_time = 0
    data1 = []
    data_mfg = []
    while epoch_.value > 0:
        total_loss = 0
        start = time.time()
        run_gg(ggg_dataloader, model, opt, head_gpu, tail_gpu, 
            mini_batch, args.fan_out, array_gpu, offset_gpu_read, epoch_, deque, file,
            mfgs_buffer_size, head_cpu, tail_cpu, slack, data1, data_mfg, extract_nfeats)
        cgg_time += time.time() - start
        if epoch_.value <= 0:
            break
        # with util.Timer() as ggg_timer:
        start = time.time()
        total_loss = run_ggg(ggg_dataloader, model, opt)
        # time.sleep(100)
        g = time.time() - start
        ggg_time += g
        epoch_.value -= 1
        data1.append([time.time(),"GGG",0,0,0,0,g])
        # file.write(f"GGG {epoch_.value} : {ggg_time:.4f}s, {time.time()}\n") # {ggg_timer.elapsed_secs},
        print("GGG done:", ggg_time, flush=True)
    train_.set()
    file.write(tabulate(data1, headers=["Timestamp", "Variant", "Extract(s)", "Train(s)", "Overhead", "ET(s)", "E2E (s)"], tablefmt="outline",showindex="always", floatfmt=".4f"))
    # file.write(f"Train: {time.time() - s}s, GGG time: {ggg_time:.4f}s, CGG time:{cgg_time:.4f}s GPU read time:{deque[0]:.4f}s\n")
    file.write(tabulate([[time.time() - s, ggg_time, cgg_time, deque[0]]], headers=["End-End(s)", "GGG Times(s)", "GG Times(s)", "GPU deque(s)"], tablefmt="outline", floatfmt=".4f"))
    file.write(tabulate(data_mfg, headers=["Timestamp", "MFG's on GPU", "MFGs on CPU", "Slack"], tablefmt="outline", showindex="always", floatfmt=".4f"))
    print("Training done")
    file.close()
    file1.close()
    sampling_worker_.join()
    mfg_transfer_.join()
    file = open("../results/hybrid/sampler/hybrid_sampler_perf.txt", "a")
    file.write(f"{timestamp}, {args.dataset}, {args.batch_size}, {args.num_threads},"
               f"{args.t_sample}, {hybrid_sampling_timer.value}, {args.t_sample/hybrid_sampling_timer.value},\n")
    file.close()
    # free_gpu_mem.join()
    # stop_perf()
    # file.close()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = get_args()
    print(f"Training in {args.mode} mode.")
    
    file = open("../results/hybrid/lbr/sampling.txt", "a")

    if args.dataset == "friendster":
        args.nfeat_dim = 256
    elif args.dataset == "twitter":
        args.nfeat_dim = 380
    elif args.dataset.startswith("igb"):
        args.fan_out = [15, 10]
        args.dataset_size = args.dataset.split("-")[1]
        if args.dataset_size == "full" or args.dataset_size == "large":
            args.nfeat_dim = 128
        else:
            args.nfeat_dim = 1024
    else:
        args.nfeat_dim = 128

    in_size, out_size = fetch_shapes()
    train_idx, val_idx, test_idx, g = fetch_all()

    model = SAGE(in_size, args.hid_size, out_size, len(args.fan_out))
    print("Training...", args.batch_size, args.hybrid)
    # print("PID : ", os.getpid())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp)
    file.write("\n\n***************************************************************************************\n")
    file.write("***************************************************************************************\n\n")
    # file.write(f"{timestamp} Dataset {args.dataset}, Batch size {args.batch_size}, , Cache Size {args.cache_size}, Hybrid {args.hybrid}, Epochs {args.epoch}\n")
    file1 = open("../results/hybrid_accuracy.txt", "a")
    
    main_worker(file, args, model, train_idx, val_idx, test_idx, g, timestamp)
    os.system("rm /dev/shm/array*")
    os.system("rm /dev/shm/offset_*")
    file.close()
    file1.close()
    if args.mps_split != 0:
        mps_quit()