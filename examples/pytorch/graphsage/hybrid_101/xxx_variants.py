from imports import *

def fetch_mfg_size(blocks):
    sizes = 0
    for block in blocks:
        coo_nfeat_label_sizes = block.get_mfg_size(torch.device("cpu"))
        sizes += coo_nfeat_label_sizes[0][0] + coo_nfeat_label_sizes[0][1] + coo_nfeat_label_sizes[0][2]
    sizes += blocks[0].srcdata["_ID"].shape[0] * 8 + blocks[-1].dstdata["_ID"].shape[0] * 8
    return sizes

#Measured using CGG
def train_gg(file, args, model, num_classes):
    # create sampler & dataloader
    train_idx,_,_, g = fetch_all()
    device = torch.device("cpu")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(g.ndata["label"], g.ndata["label"].shape)
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
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            fused = False,
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
        num_workers=0,
        use_uva=False,
        persistent_workers=True if args.workers > 0 else False,
        gpu_cache={"node": {"feat": args.cache_size}},
        skip_mfg=True,
        cgg_on_demand=True,
        use_prefetch_thread=True,
        use_alternate_streams=True,
        pin_prefetcher=False,
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
    file1 = open("../results/_gg.txt", "a")
    file2 = open("../results/_gg_profiled.txt", "a")
    file1.write(timestamp + "," + args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + str(args.workers) + ",")
    file2.write(timestamp + "," + args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + str(args.workers) + ",")
    start_ = time.time()
    
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        start = time.time()
        step = 0
        gg_time = 0
        profiled_gg_time = extract_time = train_time = mfg_transfer_time = 0
        block = []
        print("start")
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            block.append(blocks)
        
        
        for it, blocks in enumerate(block):
            with util.Timer() as mfg_transfer_timer:
                blocks = recursive_apply(
                    blocks, lambda x: x.to("cuda", non_blocking=False)
                )
            mfg_transfer_time += mfg_transfer_timer.elapsed_secs
            # block[it] = None
            # continue
            start1 = time.time()
            # print("step", it)
            with util.Timer() as gg_timer:
                with util.Timer() as extract_timer:
                    x = train_dataloader._cgg_on_demand("feat", "_N", blocks[0].srcdata["_ID"])
                    y = train_dataloader._cgg_on_demand("label", "_N", blocks[-1].dstdata["_ID"])

                with util.Timer() as train_timer:
                    end1 = time.time()
                    y_hat = model(blocks, x)
                    # print("y_hat", y_hat)
                    loss = F.cross_entropy(y_hat, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                    gg_time += time.time() - start1
                    step = step + 1
                
                extract_time += extract_timer.elapsed_secs
                train_time += train_timer.elapsed_secs
            profiled_gg_time += gg_timer.elapsed_secs
            block[it] = None
            # print("GPU Memory", nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).free / (1024 ** 3))
        
        end = time.time()
        print(end - start, mfg_transfer_time)
        # print("Epoch", epoch, "GG time", gg_time, "Profiled GG time", profiled_gg_time)
        file1.write(str(gg_time) + ",")
        file2.write(f"{extract_time},{train_time},{profiled_gg_time},")
    file1.write("\n")
    file2.write("\n")
    file1.close()
    file2.close()

def train__gg_using_cgg(file, args, model, num_classes):
    # create sampler & dataloader
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
        use_prefetch_thread=False,
        use_alternate_streams=False,
        pin_prefetcher=False,
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
    # file1 = open("../results/_gg_using_cgg.txt", "a")
    file2 = open("../results/_gg/_gg_using_cgg_table.txt", "a")
    # file1.write(timestamp + "," + args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + str(args.workers) + ",")
    file2.write(timestamp + "\n") # + args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + str(args.workers) + ",")
    start_ = time.time()
    data = []
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        start = time.time()
        step = 0
        gg_time = 0
        profiled_gg_time = extract_time = train_time = mfg_transfer_time = 0
        block = []
        print("start")
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            s = time.time()
            with util.Timer() as extract:
                x = train_dataloader._cgg_on_demand("feat", "_N", blocks[0].srcdata["_ID"])
                y = train_dataloader._cgg_on_demand("label", "_N", blocks[-1].dstdata["_ID"])
            extract_time += extract.elapsed_secs
            s = time.time()
            with util.Timer() as train:
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            train_time += train.elapsed_secs       
        
        end = time.time()
        # print(end - start)
        # file1.write(f"{extract_time},{train_time},{extract_time + train_time},")
        data.append([args.dataset,args.batch_size,extract_time,train_time,extract_time + train_time])
    
    file2.write(tabulate(data, headers=["Dataset", "Batch Size", "Extract Time", "Train Time", "Total Time"], tablefmt="outline", floatfmt=".4f", showindex="always"))
    # file1.write("\n")
    file2.write("\n\n")
    # file1.close()
    # file2.close()

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

def mfg_transfer(file, args, model, num_classes):
    # create sampler & dataloader
    train_idx,_,_, g = fetch_all()
    device = torch.device("cpu")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(g.ndata["label"], g.ndata["label"].shape)
    print(train_idx.shape)
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
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            fused = False,
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
        num_workers=0,
        use_uva=False,
        persistent_workers=True if args.workers > 0 else False,
        gpu_cache={"node": {"feat": args.cache_size}},
        skip_mfg=True,
        cgg_on_demand=True,
        use_prefetch_thread=True,
        use_alternate_streams=True,
        pin_prefetcher=False,
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
    # file1 = open("../results/mfg_transfer_async.txt", "a")
    # file1.write(timestamp + "," + args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + str(args.workers) + ",")
    for epoch in range(args.epoch):
        model.train()
        mfg_transfer_time = 0
        mfg_size = 0
        block = []
        print("start")
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            block.append(blocks)
        
        
        for it, blocks in enumerate(block):
            with util.Timer() as mfg_transfer_timer:
                blocks = recursive_apply(
                    blocks, lambda x: x.to("cuda", non_blocking=True)
                )
                mfg_size += fetch_mfg_size(blocks)
            mfg_transfer_time += mfg_transfer_timer.elapsed_secs
            block[it] = None
            continue
        print("MFG size", mfg_size / (1024 ** 3))
    #     file1.write(str(mfg_transfer_time) + ",")
    # file1.write("\n")
    # file1.close()

def train_ggg(file, args, model, num_classes):
    # create sampler & dataloader
    train_idx, val_idx, test_idx, g = fetch_all()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda")
    val_idx = val_idx.to(device)
    train_idx = train_idx.to(device)
    # test_idx = test_idx.to(device)
    if args.sampler == 'nbr':
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )
  
    if args.sampler == 'lbr':
        # print("lbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
            # layer_dependency=True,
            # importance_sampling=-1,
        )
    
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
        gpu_cache={"node": {"feat": args.cache_size}},
        # skip_mfg=True,
        # cgg_on_demand=True,
        # gather_pin_only=True,
    )
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    start_ = time.time()
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )
    # file1 = open("../results/_gg_using_ggg.txt", "a")
    file2 = open("../results/_gg/_gg_using_ggg_table.txt", "a")

    #warm-up
    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            break
    # nvmlInit()
    # print("Before training memory", nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).free / (1024 ** 3))
    # file1.write(timestamp + "," + args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + str(args.workers) + ",")
    file2.write(timestamp + "\n") # + args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + str(args.workers) + ",")
    # with util.Timer() as t:
    data = []
    for epoch in range(args.epoch):
        model.train()
        total_loss = nfeat_fetch = train_time = 0
        start = time.time()
        print("start", epoch)
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            # print(it, time.time())
            s = time.time()
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            nfeat_fetch += time.time() - s
            s = time.time()
            with util.Timer() as t:
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                # train_time += time.time() - s
            train_time += t.elapsed_secs
            # print("Step", it)
            # print("GPU Memory", nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).free / (1024 ** 3))
        # acc = evaluate(model, g, val_dataloader, num_classes)
        # file.write(
        #     "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} \n".format(
        #         epoch, total_loss / (it + 1), acc.item()
        #     )
        # )
        # print(step)
        end = time.time()
        # file1.write(str(end - start) + ",")
        # file1.write(f"{train_dataloader.nfeat_timer + train_time},{train_dataloader.nfeat_timer},{train_time},")
        et = train_dataloader.nfeat_timer + train_time
        data.append([args.dataset, args.batch_size, train_dataloader.nfeat_timer, train_time, et])
        train_dataloader.nfeat_timer = train_dataloader.index_transfer = 0
    
    file2.write(tabulate(data, headers=["Dataset", "Batch Size", "NFeat Fetch Time", "Train Time", "Total Time"], tablefmt="outline", floatfmt=".4f", showindex="always"))
    # file1.write("\n")
    file2.write("\n\n")
    # print(t.elapsed_secs, t.elapsed_secs / args.epoch)    

#Timer inside main loop GGG
def train__gg(file, args, model, num_classes):
    # create sampler & dataloader
    train_idx, val_idx, test_idx, g = fetch_all()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda")
    val_idx = val_idx.to(device)
    train_idx = train_idx.to(device)
    # test_idx = test_idx.to(device)
    if args.sampler == 'nbr':
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
  
    if args.sampler == 'lbr':
        # print("lbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )
    
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
        gpu_cache={"node": {"feat": args.cache_size}},
        # skip_mfg=True,
        # cgg_on_demand=True,
        # gather_pin_only=True,
    )
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    file = open("../results/ggg_nbr_cache.txt", "a")
    
    start_ = time.time()
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )
    file1 = open("../results/_gg.txt", "a")
    nvmlInit()
    # print("Before training memory", nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).free / (1024 ** 3))
    file1.write(timestamp + "," + args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + str(args.workers) + ",")
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        _gg_time = 0
        print("start", epoch)
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            start = time.time()
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            _gg_time += time.time() - start
            
        end = time.time()
        print(_gg_time)
        file1.write(str(_gg_time) + ",")
    file1.write("\n")

def train_g__(file, args, model, num_classes):
    # create sampler & dataloader
    train_idx, val_idx, test_idx, g = fetch_all()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda")
    val_idx = val_idx.to(device)
    train_idx = train_idx.to(device)
    # test_idx = test_idx.to(device)
    if args.sampler == 'nbr':
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
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
    
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
        gpu_cache={"node": {"feat": args.cache_size}},
        # skip_mfg=True,
        # cgg_on_demand=True,
        # gather_pin_only=True,
    )
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    file = open("../results/g__.txt", "a")
    
    start_ = time.time()
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )
    file1 = open("../results/g__.txt", "a")
    nvmlInit()
    print("Before training memory", nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).free / (1024 ** 3))
    file1.write(timestamp + "," + args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + str(args.workers) + ",")
    sizes = 0
    with util.Timer() as t:
        for epoch in range(args.epoch):
            model.train()
            total_loss = 0
            start = time.time()
            for it, (input_nodes, output_nodes, blocks) in enumerate(
                train_dataloader
            ):
                # sizes += fetch_mfg_size(blocks)
                continue
            # torch.cuda.synchronize()
            end = time.time()
            file1.write(str(end - start) + ",")
            print(end - start)
        # print(t.elapsed_secs, )
        # t.elapsed_secs = 0
    file1.write("\n")
    print(t.elapsed_secs, t.elapsed_secs / args.epoch)
    print("MFG size", sizes)
   
def train_c__(file, args, model, num_classes):
    # create sampler & dataloader
    train_idx, val_idx, test_idx, g = fetch_all()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cpu")
    if args.sampler == 'nbr':
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
  
    if args.sampler == 'lbr':
        # print("lbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
            # layer_dependency=True,
            # importance_sampling=-1,
        )
    
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
        gpu_cache={"node": {"feat": args.cache_size}},
        persistent_workers=True if args.workers > 0 else False,
        # skip_mfg=True,
        # cgg_on_demand=True,
        # gather_pin_only=True,
    )
    
    # set_num_threads(args.num_threads)
    if args.hybrid:
        sampler.hybrid = args.hybrid
        sampler.array = create_shmarray(args.mfg_size * (1024 ** 3), args.madvise, "array", args.pin_mfg)
        sampler.offset = create_shmoffset(1024, "offset")
        file1 = open("../results/c__Hybrid.txt", "a")
    else:
        file1 = open("../results/c__lbr.txt", "a")
    
    #warmup
    for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            # sizes.append(fetch_mfg_size(blocks))
            break
    
    # file1.write(timestamp + "," + args.dataset + "," + str(args.batch_size) + "," + str(args.num_threads) + ","
    #             + str(args.sampler) + "," + str(args.cache_size) + "," + str(args.workers) + ",")
    sizes = []
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        
        if args.hybrid:
            reset_shm(sampler.offset)
        # print("Epoch", epoch)
        start = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            # sizes.append(fetch_mfg_size(blocks))
            continue
        # torch.cuda.synchronize()
        end = time.time()
        # file1.write(str(end - start) + ",")
        print("Cpu sampling time", end - start)
        # print("MFG min size:", min(sizes), "MFG max size:", max(sizes), "MFG avg size:", sum(sizes) / len(sizes))
        
    file1.write("\n")
    file1.close()
    # print("MFG size", sizes)

def train_gg_(file, args, model, num_classes):
    # create sampler & dataloader
    train_idx, val_idx, test_idx, g = fetch_all()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda")
    val_idx = val_idx.to(device)
    train_idx = train_idx.to(device)
    # test_idx = test_idx.to(device)
    if args.sampler == 'nbr':
        # print("fns")
        sampler = NeighborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            # prefetch_node_feats=["feat"],
            # prefetch_labels=["label"],
        )
  
    if args.sampler == 'lbr':
        # print("lbr")
        sampler = LaborSampler(
            args.fan_out,  # fanout for [layer-0, layer-1, layer-2]
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
            layer_dependency=True,
            importance_sampling=-1,
        )
    
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
        gpu_cache={"node": {"feat": args.cache_size}},
        # skip_mfg=True,
        # cgg_on_demand=True,
        # gather_pin_only=True,
    )
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    file = open("../results/gg_.txt", "a")
    
    start_ = time.time()
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )
    file1 = open("../results/gg_.txt", "a")
    nvmlInit()
    print("Before training memory", nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).free / (1024 ** 3))
    file1.write(timestamp + "," + args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + str(args.workers) + ",")
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        start = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            continue
        torch.cuda.synchronize()
        end = time.time()
        print(end - start)
        file1.write(str(end - start) + ",")
    file1.write("\n")
    print("Testing...")

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
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

    if args.variant == "ggg": # Currently train_ggg measures each component including GGG, E, T precisely.
        train_ggg(file, args, model, out_size)
    elif args.variant == "cgg":
        train_cgg(file, args, model, out_size)
    elif args.variant == "mfg":
        mfg_transfer(file, args, model, out_size)
    elif args.variant == "_gg":
        train__gg(file, args, model, out_size)
    elif args.variant == "gg_":
        train_gg(file, args, model, out_size)
    elif args.variant == "gg_using_cgg":
        train__gg_using_cgg(file, args, model, out_size)
    elif args.variant == "g__":
        train_g__(file, args, model, out_size)
    elif args.variant == "c__":
        train_c__(file, args, model, out_size)
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
