from imports import *

def train_cgg(file, args, model):
    # create sampler & dataloader
    train_idx,_,_, g = fetch_all()
    device = torch.device("cpu")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        skip_mfg=False,
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
    file1 = open("../results/cggcache.txt", "a")
    file1.write(timestamp + "," + args.dataset + "," + str(args.batch_size) + "," + str(args.sampler) + "," + str(args.cache_size) + "," + str(args.workers) + ",")
    start_ = time.time()
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        start = time.time()
        count = count1 = 0
        step = 0
        print("start")
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            start1 = time.time()
            x = train_dataloader._cgg_on_demand("feat", "_N", blocks[0].srcdata["_ID"])
            y = train_dataloader._cgg_on_demand("label", "_N", blocks[-1].dstdata["_ID"])

            end1 = time.time()
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            end2 = time.time()
            step = step + 1
            
            # count = count + end2 - start1
            # count1 = count1 + end1 - start1
            # continue
        # print(step)
        end = time.time()
        file1.write(str(end - start) + ",")
    file1.write("\n")
        

def train_ggg(file, args, model, num_classes):
    # create sampler & dataloader
    train_idx, val_idx, test_idx, g = fetch_all()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda")
    val_idx = val_idx.to(device)
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
    # file = open("../results/ggg_nbr_cache.txt", "a")
    
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
    file1 = open("../results/ggg_nbr_cache.txt", "a")
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
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader, num_classes)
        # file.write(
        #     "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} \n".format(
        #         epoch, total_loss / (it + 1), acc.item()
        #     )
        # )
        # print(step)
        end = time.time()
        file1.write(str(end - start) + ",")
    file1.write("\n")
    print("Testing...")
    

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = get_args()
    # torch.cuda.empty_cache()
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
    # print("PID : ", os.getpid())
    # batch_size_ = [1024]
    # for i in batch_size_:
    #     args.batch_size = i
    train_ggg(file, args, model, out_size)
    # train_cgg(file, args, model)
    # if args.dataset == "ogbn-products":
    #     args.epoch = 5
    #     train_cgg(file, args, model)

    # acc = layerwise_infer(
    #     torch.device("cpu"), g, test_idx, model, out_size, batch_size=4096
    # )
    # file.write("Test Accuracy {:.4f}\n".format(acc.item()))