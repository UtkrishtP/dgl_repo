import args, time, argparse, threading, json, gc, subprocess, os, torch, dgl
from tqdm import tqdm
from custom_dl import FriendsterDataset, TwitterDataset, IGB260MDGLDataset
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
from dgl.convert import hetero_from_shared_memory
from dgl.utils.pin_memory import pin_memory_inplace
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from args import get_args
from multiprocessing import Manager
from colorama import Fore, Style, init
from mps_utils import *
from datetime import datetime

def deserialize_dtypes(dtypes):
    return [getattr(torch, dtype) for dtype in dtypes]


def fetch_train_graph():
    # global shapes, dtypes
    with open("/dev/shm/shapes_dtypes.json", "r") as f:
        data = json.load(f)
    shapes = data["shapes"]
    dtypes = deserialize_dtypes(data["dtypes"])
    train_idx = get_shared_mem_array("train_idx", shapes[3], dtype=dtypes[3])
    g = hetero_from_shared_memory("graph_formats")
    return train_idx, g


def fetch_all():
    # global shapes, dtypes
    with open("/dev/shm/shapes_dtypes.json", "r") as f:
        data = json.load(f)
    shapes = data["shapes"]
    dtypes = deserialize_dtypes(data["dtypes"])
    train_idx = get_shared_mem_array("train_idx", shapes[3], dtype=dtypes[3])
    g = hetero_from_shared_memory("graph_formats")
    g.ndata["feat"] = get_shared_mem_array("feats", shapes[0], dtype=dtypes[0])
    g.ndata["label"] = get_shared_mem_array(
        "labels", shapes[2], dtype=dtypes[2]
    )
    return train_idx, g


def fetch_shapes():
    # global shapes, num_classes
    with open("/dev/shm/shapes_dtypes.json", "r") as f:
        data = json.load(f)
    shapes = data["shapes"]
    num_classes = data["num_classes"]
    return shapes[0][1], num_classes


def serialize_dtypes(dtypes):
    return [str(dtype).split(".")[-1] for dtype in dtypes]


def run_ncu(
    batch_size,
    file_name,
    sampler, 
    dataset
):
    """Run the node classification task."""
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(
        f"{Fore.GREEN}Running {file_name} with batch size {batch_size} sampler {sampler} {Style.RESET_ALL}"

    )
    total_ = {
        "lbr2": 154,
        "nbr": 84,
        "shadow": 140,    
    }
    log_file = f"./log/ncu_report.log"
    with open(log_file, "a", buffering=1) as f:
        process = subprocess.Popen(
            [
                "sudo", "env", f"PATH={os.environ['PATH']}", "ncu",
                "--target-processes", "all", "--export",
                f"{dataset}_{batch_size}_{sampler}_mem", "-f",
                # "--section", "SpeedOfLight",
                "--section", "MemoryWorkloadAnalysis",
                "--section", "ComputeWorkloadAnalysis",
                # "--section", "SpeedOfLight_RooflineChart",
                "python", file_name,
                "--dataset", dataset,
                "--batch_size", str(batch_size),
                "--epoch", "1",
                "--variant", "g__",
                "--sampler", sampler,
                "--hybrid", "0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # stdout, stderr = process.communicate()
        pbar = tqdm(total=total_[sampler], desc=f"Running {file_name}", dynamic_ncols=True,leave=False, position=0)

        # Read and log stdout and stderr in real-time
        for line in process.stdout:
            # print(line, end="", flush=True)  # Print live to stdout
            f.write(str(line) + "\n")  # Write to log file immediately
            f.flush()  # Ensure immediate file write
            pbar.update(1)  # Simulated progress update

        for line in process.stderr:
            # if line.__contains__("dlopen"):
            #     break
            print(line, end="", flush=True)  # Print stderr live
            f.write("\n--- STDERR ---\n")
            f.write(str(line) + "\n")
            f.flush()

        pbar.close()

    if process.returncode == 0:
        print(
            f"{Fore.LIGHTGREEN_EX}{file_name} completed successfully with batch size {batch_size}.{Style.RESET_ALL}"
        )
    # else:
    #     print(f"Error during {file_name}: {stderr.decode()}")


def run_nsys(
    batch_size,
    file_name,
    sampler, 
    dataset,
    model,
    variant,
):
    """Run the node classification task."""
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(
        f"{Fore.GREEN}Running {file_name} with batch size {batch_size} sampler {sampler} variant {variant} model {model} {Style.RESET_ALL}"

    )
    # total_ = {
    #     "lbr2": 154,
    #     "nbr": 84,
    #     "shadow": 140,    
    # }
    log_file = f"./log/nsys_report.log"
    with open(log_file, "a", buffering=1) as f:
        process = subprocess.Popen(
            [
                "/opt/nvidia/nsight-systems/2024.2.1/bin/nsys",
                "profile", "-t", "cuda", "-o",
                f"{dataset}_{batch_size}_{variant}_{sampler}_{model}", "-f","true",
                "python", file_name,
                "--dataset", dataset,
                "--batch_size", str(batch_size),
                "--epoch", "1",
                "--variant", variant,
                "--sampler", sampler,
                "--hybrid", "0",
                "--workers", "32",
                "--model_type", model,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        # pbar = tqdm(total=total_[sampler], desc=f"Running {file_name}", dynamic_ncols=True,leave=False, position=0)

        f.write(stdout.decode())  # Write stdout to the log file
        if stderr:
            f.write("\n--- STDERR ---\n")
            f.write(stderr.decode())
        # Read and log stdout and stderr in real-time
        # for line in process.stdout:
        #     # print(line, end="", flush=True)  # Print live to stdout
        #     f.write(str(line) + "\n")  # Write to log file immediately
        #     f.flush()  # Ensure immediate file write
            # pbar.update(1)  # Simulated progress update

        # for line in process.stderr:
        #     # if line.__contains__("dlopen"):
        #     #     break
        #     print(line, end="", flush=True)  # Print stderr live
        #     f.write("\n--- STDERR ---\n")
        #     f.write(str(line) + "\n")
        #     f.flush()

        # pbar.close()

    if process.returncode == 0:
        print(
            f"{Fore.LIGHTGREEN_EX}{file_name} completed successfully with batch size {batch_size}.{Style.RESET_ALL}"
        )
    # else:
    #     print(f"Error during {file_name}: {stderr.decode()}")



def run_node_classification(
    batch_size,
    file_name,
    epoch,
    mfg_size,
    dataset,
    cache,
    sampler,
    workers,
    num_threads,
    variant,
    hid_size,
    diff=0,
    fanout="",
    slack_test=False,
    nfeat_dim=128,
    hybrid=1,
    model="sage",
    mfg_buffer_size=0,
    opt2=0,
    ablation=0,
    # diff=0,
):
    """Run the node classification task."""
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(
        f"{Fore.GREEN}Running {file_name} with batch size {batch_size} "
        f"epochs {epoch} sampler {sampler} workers {workers} variant {variant} model {model} {Style.RESET_ALL}"

    )
    log_file = f"./log/{file_name}.log"
    with open(log_file, "a") as f:
        # Build the command first, so we can show it to the user
        cmd = [
            "python",
            file_name,
            "--epoch", str(epoch),
            "--batch_size", str(batch_size),
            "--mfg_size", str(mfg_size),
            "--dataset", dataset,
            "--cache_size", str(cache),
            "--sampler", sampler,
            "--workers", str(workers),
            "--num_threads", str(num_threads),
            "--variant", variant,
            "--hid_size", str(hid_size),
            "--diff", str(diff),
            "--fan_out", fanout,
            "--slack_test", str(slack_test),
            "--nfeat_dim", str(nfeat_dim),
            "--hybrid", str(hybrid),
            "--model_type", model,
            "--mfg_buffer_size", str(mfg_buffer_size),
            "--opt", str(opt2),
            "--ablation", str(ablation),
        ]
        
        # Print the full command for debugging/record‐keeping
        print("Executing command:", " ".join(cmd), flush=True)
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,      # merge stderr into stdout
            bufsize=1,
            universal_newlines=True,       # so we get text lines, not bytes
        )
        # stream‐read line by line
        for line in process.stdout:
            f.write(line)               # each line already has its newline
            f.flush()
        process.stdout.close()
        returncode = process.wait()    # wait for the process to exit
        
        if returncode != 0:
            f.write(f"\nProcess exited with return code {returncode}\n")
            f.flush()
        else:
            print(
            f"{Fore.LIGHTGREEN_EX}{file_name} completed successfully with batch size {batch_size}.{Style.RESET_ALL}"
        )


def release_dataset():
    """Release shared memory and clean up."""
    print("Releasing dataset from shared memory.")
    gc.collect()


def clear_shared_memory():
    """Clear all files from /dev/shm using os.system."""
    # TODO: Add cudaHostUnregister to unpin the memory
    command = "rm -rf /dev/shm/*"
    print(f"{Fore.RED}Executing command: {command} {Style.RESET_ALL}")
    os.system(command)  # Execute the command directly


def main():
    args = get_args()
    datasets = ["igb-large"]
    # datasets = ["friendster", "ogbn-papers100M", "twitter", "igb-large"] #
    # datasets = ["friendster", "ogbn-papers100M"]
    batch_sizes = [8192, 1024]
    # batch_sizes = [1024]
    sampler_xxx = ["shadow"]
    # sampler_xxx = ["fns", "lbr2", "shadow"]
    # sampler_hybrid = ["fns", "lbr2", "shadow"]
    sampler_hybrid = ["shadow"]
    model_type = ["sage", "gcn"]
    # model_type = ["gcn"]
    hid_size_ = [256]
    
    # file_name_ = ["xxx_variants.py"] #, "hybrid_shadow.py"] #"hybrid_ablation.py",
    file_name_ = ["hybrid_shadow.py"]
    # threads_ = [1, 4, 16, 64]
    threads_ = [64]
    workers_xxx = [32]
    # workers_hybrid = [16, 4, 1]
    workers_hybrid = [64]
    # variants = ["c_vanilla",] 
    variants = ["ccg", "ggg"] # Run only for 2 datasets, FR/PA
    # varaints = ["c__", "g__", ]
    cache_size = 12000000
    # cache_size = 0
    fanout_ = ["15,10,5"]
    opt2 = [0]

    for _dataset in datasets:
        # workers_hybrid = [64] if _dataset == "igb-large" else workers_hybrid
        # sampler_hybrid = ["fns"] if _dataset == "igb-large" else sampler_hybrid
        clear_shared_memory()
        if _dataset == "friendster":
            args.nfeat_dim = 256
            dataset = FriendsterDataset(args)
            num_classes = 64
            train_idx = torch.nonzero(
                dataset[0].ndata["train_mask"], as_tuple=True
            )[0]
            val_idx = torch.nonzero(
                dataset[0].ndata["val_mask"], as_tuple=True
            )[0]
            test_idx = torch.nonzero(
                dataset[0].ndata["test_mask"], as_tuple=True
            )[0]
            args.mfg_size = 15

        elif _dataset == "twitter":
            args.nfeat_dim = 380
            dataset = TwitterDataset(args)
            num_classes = 64
            train_idx = torch.nonzero(
                dataset[0].ndata["train_mask"], as_tuple=True
            )[0]
            val_idx = torch.nonzero(
                dataset[0].ndata["val_mask"], as_tuple=True
            )[0]
            test_idx = torch.nonzero(
                dataset[0].ndata["test_mask"], as_tuple=True
            )[0]
            args.mfg_size = 15

        elif _dataset.startswith("igb"):
            args.dataset_size = _dataset.split("-")[1]
            args.num_classes = num_classes = 19
            args.fan_out = "15,10"
            dataset = IGB260MDGLDataset(args)
            train_idx = torch.nonzero(
                dataset[0].ndata["train_mask"], as_tuple=True
            )[0]
            val_idx = torch.nonzero(
                dataset[0].ndata["val_mask"], as_tuple=True
            )[0]
            test_idx = torch.nonzero(
                dataset[0].ndata["test_mask"], as_tuple=True
            )[0]
            args.mfg_size = 120
            # args.epoch = 3
            cache_size = 80449000
        else:
            dataset = AsNodePredDataset(
                DglNodePropPredDataset(_dataset, root="/data/"),
                save_dir="/data/tmp/",
            )
            num_classes = dataset.num_classes
            train_idx = dataset.train_idx
            val_idx = dataset.val_idx
            test_idx = dataset.test_idx
            args.mfg_size = 15
        g = dataset[0]
        g.ndata["label"] = g.ndata["label"].type(torch.long)
        
        if args.dataset.startswith("ogb"):
            g = dgl.add_self_loop(g)
        # Creating shared + pinned memory regions for graph formats {coo, csr, csc}, nfeat, labels, train_idx
        shapes = [
            g.ndata["feat"].shape,
            (2, g.edges()[0].shape[0]),
            g.ndata["label"].shape,
            train_idx.shape,
            val_idx.shape,
            test_idx.shape,
        ]
        dtypes = [
            g.ndata["feat"].dtype,
            g.edges()[0].dtype,
            g.ndata["label"].dtype,
            train_idx.dtype,
            val_idx.dtype,
            test_idx.dtype,
        ]
        start = time.time()
        feat = create_shared_mem_array("feats", shapes[0], dtype=dtypes[0])
        label = create_shared_mem_array("labels", shapes[2], dtype=dtypes[2])
        feat[:] = g.ndata["feat"]
        label[:] = g.ndata["label"]
        # feat_ = pin_memory_inplace(feat)
        # label_ = pin_memory_inplace(label)
        train_idx_ = create_shared_mem_array(
            "train_idx", shapes[3], dtype=dtypes[3]
        )
        val_idx_ = create_shared_mem_array(
            "val_idx", shapes[4], dtype=dtypes[4]
        )
        test_idx_ = create_shared_mem_array(
            "test_idx", shapes[5], dtype=dtypes[5]
        )
        shared_graph_formats = g.shared_memory("graph_formats")
        # shared_graph_formats.pin_memory_()
        train_idx_[:] = train_idx
        val_idx_[:] = val_idx
        test_idx_[:] = test_idx
        del dataset
        gc.collect()
        # Save to a shared memory file
        with open("/dev/shm/shapes_dtypes.json", "w") as f:
            json.dump(
                {
                    "shapes": shapes,
                    "dtypes": serialize_dtypes(dtypes),
                    "num_classes": num_classes,
                },
                f,
            )
        # print ("Loading data finished")

        print(f"Dataset {_dataset} loaded and ready.")
        model_types = model_type
        # Run node classification for each batch size sequentially
        for file_name in file_name_:
            # sampler_hybrid = ["lbr", "fns"] if _dataset == "igb-large" else sampler_hybrid
            for batch_size in batch_sizes:

                if args.mps_split:
                    user_id = mps_get_user_id()
                    mps_daemon_start()
                    mps_server_start(user_id)

                if args.ncu:
                    for s in sampler_xxx:
                            run_ncu(
                                batch_size,
                                file_name,
                                s,
                                _dataset,
                            )
                    continue

                if args.nsys:
                    for s in sampler_xxx:
                        for m in model_type:
                            for v in variants:
                                run_nsys(
                                    batch_size,
                                    file_name,
                                    s,
                                    _dataset,
                                    m,
                                    v,
                                )
                    continue

                if file_name.startswith("xxx") or file_name.startswith("lbr"):
                    for hid_size in hid_size_:
                        for variant in variants:
                            epochs = 3
                            sampler_xxx_ = sampler_xxx
                            
                            for s in sampler_xxx_:
                                if s == "shadow" and variant == "ggg":
                                    continue
                                model_types_ = model_types #if s == "lbr2" else ["gcn"]
                                if s == "fns":
                                    
                                    workers_xxx_ = [0]
                                elif s == "lbr2":
                                    workers_xxx_ = [8]
                                else:
                                    workers_xxx_ = [32]
                                threads__ = [64]
                                # workers_xxx_ = [8] if s == "lbr2" else [32]
                                for model in model_types_:
                                    for threads in threads__:
                                        # if s == "shadow":
                                        #     worker = 16
                                        # elif s == "lbr2":
                                        #     worker = 8
                                        # else:
                                        #     worker = 0
                                        
                                        for worker in workers_xxx_:
                                            for fanout in fanout_:
                                                run_node_classification(
                                                    batch_size,
                                                    file_name,
                                                    epochs,
                                                    args.mfg_size,
                                                    _dataset,
                                                    cache_size,
                                                    s,
                                                    worker,
                                                    threads,
                                                    variant,
                                                    hid_size,
                                                    fanout=fanout,
                                                    slack_test=args.slack_test,
                                                    nfeat_dim=args.nfeat_dim,
                                                    hybrid=0,
                                                    model=model,
                                                )
                else:
                    # if _dataset != "igb-large":
                    #     continue
                    for hid_size in hid_size_:
                        for s in sampler_hybrid:
                            args.mfg_size = 0 if s != "fns" else args.mfg_size
                            
                            # model_types = ["sage"] if _dataset == "ogbn-papers100M" else model_type
                            # threads__ = [64] if s == "lbr2" else threads_
                            # workers_hybrid_ = [0] if s == "fns" else workers_hybrid
                            if file_name.startswith("hybrid_shadow"):
                                # opt2 = [1] if s == "lbr2" and ((_dataset == "ogbn-papers100M" and batch_size == 1024) or _dataset.startswith("igb")) else [0]
                                opt2 = [0]
                                workers_hybrid = [64]
                                ablation = 0
                            else:
                                workers_hybrid = [32]
                                opt2 = [0]
                                ablation = 1

                            epochs = 20 # if _dataset.startswith("igb") else 30
                            for model in model_types:
                                for threads in threads_:
                                    for fanout in fanout_:
                                        for opt2_ in opt2:
                                            for w in workers_hybrid:
                                                run_node_classification(
                                                    batch_size,
                                                    file_name,
                                                    epochs,
                                                    args.mfg_size,
                                                    _dataset,
                                                    cache_size,
                                                    s,
                                                    # 0 if s == "fns" else 64,
                                                    w if s != "fns" else 0,
                                                    threads,
                                                    "hybrid",
                                                    hid_size,
                                                    fanout=fanout,
                                                    slack_test=args.slack_test,
                                                    nfeat_dim=args.nfeat_dim,
                                                    model=model,
                                                    mfg_buffer_size=args.mfg_buffer_size,
                                                    opt2=opt2_,
                                                    ablation=ablation,
                                                )
                if args.mps_split:
                    mps_quit()
        del feat, label, train_idx_, val_idx_, test_idx_, shared_graph_formats
        # del dataset  # Ensure the dataset is fully cleared from memory

    print("All datasets processed successfully.")


if __name__ == "__main__":
    main()
