from imports import *

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
    
class SetupNumThreads(object):

    def __init__(self, num_threads):
        self.num_threads = num_threads

    def __call__(self, worker_id):
        torch.set_num_threads(self.num_threads)


def fetch_mfg_gpu_shm(blocks, array_gpu, offset_gpu_read, fanout, sampler):
    block, blocks = [], []
    idx = None
    if sampler != "shadow":
        for layer in range(len(fanout)):
            if layer == 0:
                block, input_nodes = dgl.hetero_from_gpu_shared_memory(
                    array_gpu, offset_gpu_read, 0
                )
            elif layer == len(fanout) - 1:
                block, output_nodes = dgl.hetero_from_gpu_shared_memory(
                    array_gpu, offset_gpu_read, 0
                )
            else:
                block, _ = dgl.hetero_from_gpu_shared_memory(
                    array_gpu, offset_gpu_read, -1
                )
            blocks.append(block)
        blocks[0].srcdata["_ID"] = input_nodes[0]
        blocks[-1].dstdata["_ID"] = output_nodes[0]
    else:
        blocks, src_nodes, idx = dgl.hetero_from_gpu_shared_memory(
            array_gpu, offset_gpu_read, 0, sampler
        )
        # print(blocks)
        blocks.ndata["_ID"] = src_nodes[0]
        # print(blocks)
        
    return blocks, idx


def to_gpu_shared_memory(
    blocks,
    array_gpu,
    offset_gpu,
    fanout,
    idx=-1,
):
    layer = 0
    if hasattr(blocks, '__len__'):
        for block in blocks:
            if layer == 0:
                block.shared_memory_gpu(
                    array_gpu, offset_gpu, block._node_frames[0]["_ID"], 0
                )
                # for nframe in block._node_frames:
                #     if nframe:
                #             block.shared_memory_gpu(array_gpu, offset_gpu, nframe['_ID'], 0)
            elif layer == len(fanout) - 1:
                block.shared_memory_gpu(
                    array_gpu, offset_gpu, block._node_frames[1]["_ID"], 0
                )
                # for nframe in block._node_frames:
                #     if nframe:
                #             block.shared_memory_gpu(array_gpu, offset_gpu, nframe['_ID'], 0)
            else:
                block.shared_memory_gpu(array_gpu, offset_gpu, torch.empty(0), -1)
            layer += 1
    else:
        blocks.shared_memory_gpu(
            array_gpu, offset_gpu, blocks.ndata["_ID"], 0, idx=idx,
        )


def fetch_mfg_size(blocks):
    sizes = 0
    for block in blocks:
        coo_row_col_sizes = block.get_mfg_size(torch.device("cpu"))
        sizes += (
            coo_row_col_sizes[0][0]
            + coo_row_col_sizes[0][1]
            + coo_row_col_sizes[0][2]
        )
    sizes += (
        blocks[0].srcdata["_ID"].shape[0] * 8
        + blocks[-1].dstdata["_ID"].shape[0] * 8
    )
    return sizes


def fetch_mfg_cpu_shm(blocks, array, read_offset, fanout, edge_dir):
    for layer in range(len(fanout)):
        if layer == 0:
            block, output_nodes = dgl.hetero_from_shared_memory_hybrid(
                array, 0, read_offset, edge_dir
            )
        elif layer == len(fanout) - 1:
            block, input_nodes = dgl.hetero_from_shared_memory_hybrid(
                array, 0, read_offset, edge_dir
            )
        else:
            block, _ = dgl.hetero_from_shared_memory_hybrid(
                array, layer, read_offset, edge_dir
            )

        blocks.insert(0, block)
    blocks[0].srcdata["_ID"] = input_nodes[0]
    blocks[-1].dstdata["_ID"] = output_nodes[0]


def transfer_mfg_gpu(stream1, blocks):
    # with torch.cuda.stream(stream1):
    blocks = recursive_apply(blocks, lambda x: x.to("cuda", non_blocking=True))
    return blocks


def train_cgg(file, args, model, num_classes):
    # create sampler & dataloader
    set_num_threads(args.num_threads)
    train_idx, _, _, g = fetch_all()
    device = torch.device("cpu")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(g.ndata["label"], g.ndata["label"].shape)
    # train_idx = train_idx.to(torch.device("cuda"))

    sampler = ""
    if args.sampler == "nbr":
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    if args.sampler == "lbr":
        # print("nbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # fused = False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    if args.sampler == 2:
        # print("lbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
            # layer_dependency=True,
            # importance_sampling=-1,
        )
    if args.sampler == 3:
        sampler = SAINTSampler(
            mode="node",
            budget=1000,
        )
    train_dataloader = DataLoaderCGG(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
        use_uva=False,
        persistent_workers=True if args.workers > 0 else False,
        gpu_cache={"node": {"feat": args.cache_size}},
        skip_mfg=False,
        cgg_on_demand=True,
        use_prefetch_thread=True,
        use_alternate_streams=False,
        pin_prefetcher=False,
        mfg_transfer=0,
        num_threads=args.num_threads,
        # gather_pin_only=True,
    )

    sampler.hybrid = 0
    print("Dataloaded")
    # if args.workers > 0:
    #     for it, (input_nodes, output_nodes, blocks) in enumerate(
    #             train_dataloader
    #         ):
    #         break

    model.to(torch.device("cuda"))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    file1 = open("../results/cgg_threads.txt", "a")
    file1.write(
        timestamp
        + ","
        + args.dataset
        + ","
        + str(args.batch_size)
        + ","
        + str(args.sampler)
        + ","
        + str(args.cache_size)
        + ","
        + str(args.workers)
        + ","
        + str(args.num_threads)
        + ","
    )
    start_ = time.time()

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        start = time.time()
        profiled_gg_time = extract_time = train_time = mfg_transfer_time = 0
        print("start")
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            s = time.time()
            x = train_dataloader._cgg_on_demand(
                "feat", "_N", blocks[0].srcdata["_ID"]
            )
            y = train_dataloader._cgg_on_demand(
                "label", "_N", blocks[-1].dstdata["_ID"]
            )
            extract_time += time.time() - s
            s = time.time()
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            train_time += time.time() - s

        end = time.time()
        # print(end - start)
        file1.write(f"{end - start},")
        # print("MFG transfer: ", train_dataloader.mfg_transfer)
        train_dataloader.mfg_transfer = 0
    file1.write("\n")
    # file2.write("\n")
    file1.close()
    # file2.close()


def train_c__vanilla(file, args, model, num_classes):
    # create sampler & dataloader
    train_idx, val_idx, test_idx, g = fetch_all()
    mini_batch = (train_idx.shape[0] + args.batch_size - 1) // args.batch_size
    print("Total MB: ", mini_batch)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cpu")
    if args.sampler == "fns":
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    if args.sampler == "nbr":
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            fused=False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    if args.sampler == "lbr":
        # print("lbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )
    if args.sampler == "lbr2":
        # print("lbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
            # layer_dependency=True,
            # importance_sampling=-1,
        )
    if args.sampler == "shadow":
        sampler = ShaDowKHopSampler(
            args.fan_out,
        )

    array_gpu_size = 15 * (1024**3)
    if args.hybrid:
        # array = create_shmarray(args.mfg_size * (1024 ** 3), args.madvise, "array_cpu", args.pin_mfg)
        array_gpu = create_shmarray(
            array_gpu_size, args.madvise, "array_gpu", args.pin_mfg
        )
        # offset_cpu_write = create_shmoffset(args.mfg_size * 2, "offset_cpu_write")
        # offset_cpu_read = create_shmoffset(args.mfg_size * 2, "offset_cpu_read")
        offset_gpu_read = create_shmoffset(array_gpu_size, "offset_gpu_read")
        offset_gpu_write = create_shmoffset(array_gpu_size, "offset_gpu_write")

    # num_threads = torch.get_num_threads() // args.workers
    train_dataloader_ = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
        use_uva=False,
        persistent_workers=True if args.workers > 0 else False,
        pin_prefetcher=False,
        use_prefetch_thread=False,
        # use_prefetch_thread=True,
        # hybrid=True,
        # hybrid_wrapper=True,
        # skip_mfg=True,
        # cgg_on_demand=True,
        # gather_pin_only=True,
        # worker_init_fn=SetupNumThreads(num_threads) if args.setup_threads else None
    )

    # warmup to launch all processes.
    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader_):
        break

    for _ in range(args.epoch):
        start = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader_
        ):
            # print(blocks)
            # if args.sampler == "shadow":
            #     blocks = blocks.to(torch.device("cuda"))
            #     print(blocks.device)
            #     print(blocks.ndata["_ID"].device)
            # else:
            # print(blocks.device)
            # blocks = transfer_mfg_gpu(torch.cuda.current_stream(), blocks)
            # print(blocks.device)
            # print(blocks)
            # to_gpu_shared_memory(
            #     blocks, array_gpu, offset_gpu_write, args.fan_out
            # )
            # break
            continue
        print("Vanilla time", time.time() - start)


def train_gpu_shm(args, fan_out, mb):
    offset_gpu_read = create_shmoffset(30 * (1024 ** 3), "offset_gpu_read")
    array_gpu = get_shm_ptr("array_gpu", 30 * (1024 ** 3), 0)
    in_size, out_size = fetch_shapes()
    train_idx, val_idx, test_idx, g = fetch_all()
    residual = train_idx.shape[0] % args.batch_size
    residual = args.batch_size if residual == 0 else residual
    ggg_train_dataloader = DataLoader(
        g,
        train_idx,
        ShaDowKHopSampler(fan_out),
        device=torch.device("cuda"),
        batch_size=8192,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
        gpu_cache={"node": {"feat": args.cache_size}},
    )

    model = SAGE(in_size, 256, out_size, len(fan_out))
    model.to(torch.device("cuda"))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    it = 0
    while it != mb - 1:
        blocks = []
        blocks, shapes = fetch_mfg_gpu_shm(
            blocks, array_gpu, offset_gpu_read, fan_out, args.sampler
        )
        print(shapes)
        # print(it, blocks.device, blocks.srcdata["_ID"])
        if hasattr(blocks, "__len__"):
            x = ggg_train_dataloader._cgg_on_demand(
                "feat", "_N", blocks[0].srcdata["_ID"]
            )
            y = ggg_train_dataloader._cgg_on_demand(
                "label", "_N", blocks[-1].dstdata["_ID"]
            )
            shapes = blocks[-1].dstdata["_ID"].shape[0]
        else:
            x = ggg_train_dataloader._cgg_on_demand(
                "feat", "_N", blocks.srcdata["_ID"]
            )
            y = ggg_train_dataloader._cgg_on_demand(
                "label", "_N", blocks.dstdata["_ID"]
            )
            
            # shapes = args.batch_size if idx != mb - 1 else residual
            
        y_hat = model(blocks, x)
        loss = F.cross_entropy(y_hat[:shapes], y[:shapes])
        opt.zero_grad()
        loss.backward()
        opt.step()
        # total_loss += loss.item()
        it += 1


def train_c__(file, args, model, num_classes):
    # create sampler & dataloader
    train_idx, val_idx, test_idx, g = fetch_all()
    in_size, out_size = fetch_shapes()
    # print(g.ndata["label"], g.ndata["label"].shape, out_size)
    # print(g.ndata["label"] > out_size)
    # return
    mini_batch = (train_idx.shape[0] + args.batch_size - 1) // args.batch_size
    print("Total MB: ", mini_batch)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cpu")
    if args.sampler == "fns":
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    if args.sampler == "nbr":
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            fused=False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )

    if args.sampler == "lbr2":
        # print("lbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
            # layer_dependency=True,
            # importance_sampling=-1,
        )

    if args.sampler == "lbr":
        # print("lbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )

    if args.sampler == "shadow":
        sampler = ShaDowKHopSampler(
            args.fan_out,
        )
    array_gpu_size = 30 * (1024**3)
    if args.hybrid:
        # array = create_shmarray(args.mfg_size * (1024 ** 3), args.madvise, "array_cpu", args.pin_mfg)
        array_gpu = create_shmarray(
            array_gpu_size, args.madvise, "array_gpu", args.pin_mfg
        )
        # offset_cpu_write = create_shmoffset(args.mfg_size * 2, "offset_cpu_write")
        # offset_cpu_read = create_shmoffset(args.mfg_size * 2, "offset_cpu_read")
        # offset_gpu_read = create_shmoffset(array_gpu_size, "offset_gpu_read")
        offset_gpu_write = create_shmoffset(array_gpu_size, "offset_gpu_write")
    cpu_shared_queue = torch.multiprocessing.Queue(maxsize=mini_batch * args.epoch)
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
        use_uva=False,
        persistent_workers=True if args.workers > 0 else False,
        cpu_shared_queue=cpu_shared_queue,
        pin_prefetcher=False,
        hybrid=True,
        hybrid_wrapper=True,
        # skip_mfg=True,
        # cgg_on_demand=True,
        # gather_pin_only=True,
    )
    # ggg_train_dataloader = DataLoader(
    #     g,
    #     train_idx,
    #     ShaDowKHopSampler(
    #         args.fan_out,
    #         output_device=0,
    #         # prefetch_node_feats=["feat"],
    #     ),
    #     device=torch.device("cuda"),
    #     batch_size=8192,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=0,
    #     use_uva=True,
    #     gpu_cache={"node": {"feat": args.cache_size}},
    # )
    # warmup
    # iterator_obj = train_dataloader.iterate()
    it = 1
    # s = time.time()
    print(args.batch_size if train_idx.shape[0] % args.batch_size == 0 else train_idx.shape[0] % args.batch_size)
    epoch = args.epoch
    while epoch != 0:
        start = time.time()
        iterator_obj = train_dataloader.iterate()

        print(iterator_obj._tasks_outstanding)

        # This will guarantee that all the MB's are fetched, and we avoid the condition for IterableStopIteration class.
        while iterator_obj._tasks_outstanding < mini_batch:  #
            iterator_obj.fetch_next()

        # # iterator_obj.check_index_queue_empty_()
        # time.sleep(15)
        while cpu_shared_queue.qsize() != (mini_batch * it):  # + args.workers):
            # print("Queue size:", cpu_shared_queue.qsize())
            continue
        # print(iterator_obj._tasks_outstanding)
        # print("Queue size:", cpu_shared_queue.qsize())
        epoch -= 1
        it += 1
        print("Epoch time", time.time() - start,)
    # while iterator_obj._tasks_outstanding < mini_batch: #+ args.workers
    #     iterator_obj.fetch_next()
    return
    resume_iter = 0
    deque_ovhd = 0
    count = 0
    print("Queue size:", cpu_shared_queue.qsize())
    block = []
    # model = SAGE(in_size, 256, out_size, len(args.fan_out))
    model.to(torch.device("cuda"))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    while cpu_shared_queue.qsize() > 0:
        s = time.time()
        idx, data = cpu_shared_queue.get()
        # if isinstance(data, torch.utils.data._utils.worker._IterableDatasetStopIteration):
        #     continue
        deque_ovhd += time.time() - s
        # print(data)
        # print(data[2])
        # break
        print(idx, data[0].shape[0], data[1].shape[0])
        blocks = data[2]
        blocks = transfer_mfg_gpu(torch.cuda.current_stream(), blocks)
        # print(count)
        # if hasattr(blocks, "__len__"):
        #     x = blocks[0].srcdata["feat"]
        #     y = blocks[-1].dstdata["label"]
        # else:
        #     x = ggg_train_dataloader._cgg_on_demand(
        #         "feat", "_N", blocks.srcdata["_ID"]
        #     )
        #     y = ggg_train_dataloader._cgg_on_demand(
        #         "label", "_N", blocks.dstdata["_ID"]
        #     )
        # y_hat = model(blocks, x)
        # # residual = args.batch_size if train_idx.shape[0] % args.batch_size == 0 else train_idx.shape[0] % args.batch_size
        # # sh = args.batch_size if idx != mini_batch - 1 else residual
        # sh = args.batch_size
        # loss = F.cross_entropy(
        #     y_hat[:sh], y[:sh]
        # )
        # opt.zero_grad()
        # loss.backward()
        # opt.step()
        # total_loss += loss.item()
        to_gpu_shared_memory(blocks, array_gpu, offset_gpu_write, args.fan_out, data[1].shape[0])
        block.append(blocks)
        count += 1
    #     # break
        if data is None:
            resume_iter += 1
    #     # for block in data[2]:
    #     #     print(block.formats())
    print("Deque ovhds", deque_ovhd)
    print("Per MB deque :", deque_ovhd / mini_batch)
    train_ = torch.multiprocessing.Process(
        target=train_gpu_shm, args=(args, args.fan_out, mini_batch)
    )
    train_.start()
    train_.join()
    # while True:
    #     blocks = []
    #     blocks = fetch_mfg_gpu_shm(
    #         blocks, array_gpu, offset_gpu_read, args.fan_out
    #     )
        # break
        # print(data[1].shape)
        # print(data[0].shape)

    # print("Resume Iter", resume_iter)
    # if isinstance(data, None):
    #     print(idx, data)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = get_args()
    # print(f"Training in {args.mode} mode.")
    train_idx, val_idx, test_idx, g = fetch_all()
    device = torch.device("cuda")
    # val_idx = val_idx.to(device)
    file = open("../results/accuracy.txt", "a")
    # file.write(args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + "\n")
    in_size, out_size = fetch_shapes()
    model = SAGE(in_size, 256, out_size, len(args.fan_out), args.model_type)
    # model training
    print("Training...", args.batch_size, args.hybrid)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp)
    # TODO: Add args for type of variant xxx
    # print("classes", out_size)
    if args.dataset.startswith("igb"):
        args.fanout = [15, 10]

    if args.variant == "cgg":
        train_cgg(file, args, model, out_size)
    elif args.variant == "c__":
        # for workers in [4, 8, 16, 24, 32, 48, 64]:
        for workers in [16, 4]:
            print("Workers", workers)
            args.workers = workers
            train_c__(file, args, model, out_size)
    elif args.variant == "c_vanilla":
        # for workers in [8, 10]: 
        #     for batch in [8192, 1024]:
        #         print("Workers", workers, "Batch", batch)
        #         args.workers = workers
        #         args.batch_size = batch
                train_c__vanilla(file, args, model, out_size)
    # exit(0)

    # Measure CPU Sampling times with different threads.
    # for threads in [64, 32, 16, 8, 4, 2, 1]:
    #     print("Threads", threads)
    #     args.num_threads = threads
    #     train_c__(file, args, model, out_size)

    # acc = layerwise_infer(
    #     torch.device("cpu"), g, test_idx, model, out_size, batch_size=4096
    # )
    # file.write("Test Accuracy {:.4f}\n".format(acc.item()))
