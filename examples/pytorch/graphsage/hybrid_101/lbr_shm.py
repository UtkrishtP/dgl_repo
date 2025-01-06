from imports import *

class SetupNumThreads(object):

    def __init__(self, num_threads):
        self.num_threads = num_threads

    def __call__(self, worker_id):
        torch.set_num_threads(self.num_threads)

def train_cgg(file, args, model, num_classes):
    # create sampler & dataloader
    set_num_threads(args.num_threads)
    train_idx,_,_, g = fetch_all()
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
            budget = 1000,
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
    file1.write(timestamp + "," + args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + str(args.workers) + ","
                + str(args.num_threads) + ",")
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
            x = train_dataloader._cgg_on_demand("feat", "_N", blocks[0].srcdata["_ID"])
            y = train_dataloader._cgg_on_demand("label", "_N", blocks[-1].dstdata["_ID"])
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
    if args.sampler == 'fns':
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    if args.sampler == 'nbr':
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            fused=False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    if args.sampler == 'lbr':
        # print("lbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )
    if args.sampler == 'lbr2':
        # print("lbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
            # layer_dependency=True,
            # importance_sampling=-1,
        )
    
    num_threads = torch.get_num_threads() // args.workers
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
        # hybrid=True,
        # hybrid_wrapper=True,
        # skip_mfg=True,
        # cgg_on_demand=True,
        # gather_pin_only=True,
        worker_init_fn=SetupNumThreads(num_threads) if args.setup_threads else None
    )
    
    # warmup to launch all processes.
    for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader_):
        break

    for _ in range(2):
        start = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
                train_dataloader_
            ):
                continue
        print("Vanilla time", time.time() - start)

def train_c__(file, args, model, num_classes):
    # create sampler & dataloader
    train_idx, val_idx, test_idx, g = fetch_all()
    mini_batch = (train_idx.shape[0] + args.batch_size - 1) // args.batch_size
    print("Total MB: ", mini_batch)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cpu")
    if args.sampler == 'fns':
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
    if args.sampler == 'nbr':
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            fused=False,
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
  
    if args.sampler == 'lbr2':
        # print("lbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
            # layer_dependency=True,
            # importance_sampling=-1,
        )
    
    if args.sampler == 'lbr':
        # print("lbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )
    
    cpu_shared_queue = torch.multiprocessing.Queue()
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
    
    #warmup
    iterator_obj = train_dataloader.iterate()
    it = 1
    while args.epoch != 0:
        start = time.time()
        iterator_obj = train_dataloader.iterate()

        print(iterator_obj._tasks_outstanding)
        
        # This will guarantee that all the MB's are fetched, and we avoid the condition for IterableStopIteration class.
        while iterator_obj._tasks_outstanding < mini_batch: #+ args.workers
            iterator_obj.fetch_next()
        
        # # iterator_obj.check_index_queue_empty_()
        # time.sleep(5)
        while cpu_shared_queue.qsize() != mini_batch * it:
            continue
        print(iterator_obj._tasks_outstanding)
        print("Queue size:", cpu_shared_queue.qsize())  
        args.epoch -= 1
        it += 1
        print("Epoch time", time.time() - start)
    # while iterator_obj._tasks_outstanding < mini_batch: #+ args.workers
    #     iterator_obj.fetch_next()
    
    print("Queue size:", cpu_shared_queue.qsize())
    # while cpu_shared_queue.qsize() > 0:
    #     idx, data = cpu_shared_queue.get()
    #     # print(idx, data)
    #     print(data[1].shape)
    #     print(data[0].shape)

        # if isinstance(data, None):
        #     print(idx, data)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    args = get_args()
    print(f"Training in {args.mode} mode.")
    train_idx, val_idx, test_idx, g = fetch_all()
    device = torch.device("cuda")
    # val_idx = val_idx.to(device)
    file = open("../results/accuracy.txt", "a")
    # file.write(args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + "\n")
    in_size, out_size = fetch_shapes()
    model = SAGE(in_size, 256, out_size, len(args.fan_out))
    # model training
    print("Training...", args.batch_size, args.hybrid)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp)
    # TODO: Add args for type of variant xxx
    print("classes", out_size)
    if args.dataset.startswith("igb"):
        args.fanout = [15, 10]

    if args.variant == "cgg":
        train_cgg(file, args, model, out_size)
    elif args.variant == "c__":
        train_c__(file, args, model, out_size)
    elif args.variant == "c_vanilla":
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
