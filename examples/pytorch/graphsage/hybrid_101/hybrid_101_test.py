import args, time, argparse, threading, json, gc, subprocess, os, torch, dgl
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
    g.ndata["label"] = get_shared_mem_array("labels", shapes[2], dtype=dtypes[2])
    return train_idx, g

def fetch_shapes():
    # global shapes, num_classes
    with open("/dev/shm/shapes_dtypes.json", "r") as f:
        data = json.load(f)
    shapes = data["shapes"]
    num_classes = data["num_classes"]
    return shapes[0][1], num_classes

def serialize_dtypes(dtypes):
    return [str(dtype).split('.')[-1] for dtype in dtypes]

def run_node_classification(batch_size, file_name, epoch, mfg_size, dataset, cache, sampler, workers, 
                            num_threads, variant, hid_size, diff=0, fanout=""):
    """Run the node classification task."""
    print(f"{Fore.GREEN}Running {file_name} with batch size {batch_size}{Style.RESET_ALL}")
    log_file = f"./log/{file_name}_test.log"
    with open(log_file, "a") as f:
        process = subprocess.Popen(
            ["python", file_name, "--epoch", str(epoch), "--batch_size", str(batch_size), "--mfg_size", str(mfg_size)
            ,"--dataset", dataset, "--cache_size", str(cache), "--sampler", sampler, "--workers", str(workers),
            "--num_threads", str(num_threads), "--variant", variant, "--hid_size", str(hid_size), "--diff", str(diff),
            "--fan_out", fanout],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Capture output and write it to the log file
        stdout, stderr = process.communicate()
        f.write(stdout.decode())  # Write stdout to the log file
        if stderr:
            f.write("\n--- STDERR ---\n")
            f.write(stderr.decode())  # Write stderr to the log file (if any)

    if process.returncode == 0:
        print(f"{Fore.LIGHTGREEN_EX}{file_name} completed successfully with batch size {batch_size}.{Style.RESET_ALL}")
    else:
        print(f"Error during {file_name}: {stderr.decode()}")

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
    # datasets = ["ogbn-products"]
    datasets = ["friendster" , "ogbn-papers100M", "twitter", "igb-large"] 
    # datasets = ["igb-large"]
    # datasets = ["ogbn-papers100M"]
    batch_sizes = [8192, 1024]
    hid_size = args.hid_size
    # batch_sizes = [8192,]
    file_name_ = ["hybrid_lbr.py"]
    # file_name_ = ["hybrid_104.py", "xxx_variants.py"]
    # threads_ = [1, 4, 16, 64]
    threads_ = [64]
    # diff = [1, 2, 4, 6, 10]
    diff = [1]
    variants = ["cgg"] #"ggg", "gg_using_cgg"]
    cache_size = 12000000
    fanout_ = ["15,10,5"]

    for _dataset in datasets:
        clear_shared_memory()
        if _dataset == "friendster":
            dataset = FriendsterDataset(args)
            num_classes = 64
            train_idx = torch.nonzero(dataset[0].ndata['train_mask'], as_tuple=True)[0]
            val_idx = torch.nonzero(dataset[0].ndata['val_mask'], as_tuple=True)[0]
            test_idx = torch.nonzero(dataset[0].ndata['test_mask'], as_tuple=True)[0]
            args.mfg_size = 15
    
        elif _dataset == "twitter":
            dataset = TwitterDataset(args)
            num_classes = 64
            train_idx = torch.nonzero(dataset[0].ndata['train_mask'], as_tuple=True)[0]
            val_idx = torch.nonzero(dataset[0].ndata['val_mask'], as_tuple=True)[0]
            test_idx = torch.nonzero(dataset[0].ndata['test_mask'], as_tuple=True)[0]
            args.mfg_size = 15

        elif _dataset.startswith("igb"):
            args.dataset_size = _dataset.split("-")[1]
            args.num_classes = num_classes = 19
            args.fan_out = "15,10"
            dataset = IGB260MDGLDataset(args)
            train_idx = torch.nonzero(dataset[0].ndata['train_mask'], as_tuple=True)[0]
            val_idx = torch.nonzero(dataset[0].ndata['val_mask'], as_tuple=True)[0]
            test_idx = torch.nonzero(dataset[0].ndata['test_mask'], as_tuple=True)[0]
            args.mfg_size = 120
            # args.epoch = 3
            cache_size = 80449000
        else:
            dataset = AsNodePredDataset(
                DglNodePropPredDataset(
                    _dataset, root="/data/"
                    ), save_dir="/data/tmp/"
            )
            num_classes = dataset.num_classes
            train_idx = dataset.train_idx
            val_idx = dataset.val_idx
            test_idx = dataset.test_idx
            args.mfg_size = 15
        g = dataset[0]
        g.ndata["label"] = g.ndata["label"].type(torch.long)

        # Creating shared + pinned memory regions for graph formats {coo, csr, csc}, nfeat, labels, train_idx
        shapes = [g.ndata["feat"].shape, (2, g.edges()[0].shape[0]), g.ndata["label"].shape, train_idx.shape, val_idx.shape, test_idx.shape]
        dtypes = [g.ndata["feat"].dtype, g.edges()[0].dtype, g.ndata["label"].dtype, train_idx.dtype, val_idx.dtype, test_idx.dtype]
        start = time.time()
        feat = create_shared_mem_array("feats", shapes[0], dtype=dtypes[0])
        label = create_shared_mem_array("labels", shapes[2], dtype=dtypes[2])
        feat[:] = g.ndata['feat']
        label[:] = g.ndata['label']
        # feat_ = pin_memory_inplace(feat)
        # label_ = pin_memory_inplace(label)
        train_idx_ = create_shared_mem_array("train_idx", shapes[3], dtype=dtypes[3])
        val_idx_ = create_shared_mem_array("val_idx", shapes[4], dtype=dtypes[4])
        test_idx_ = create_shared_mem_array("test_idx", shapes[5], dtype=dtypes[5])
        shared_graph_formats = g.shared_memory("graph_formats")
        # shared_graph_formats.pin_memory_()
        train_idx_[:] = train_idx
        val_idx_[:] = val_idx
        test_idx_[:] = test_idx
        del dataset
        gc.collect()
        # Save to a shared memory file
        with open("/dev/shm/shapes_dtypes.json", "w") as f:
            json.dump({"shapes": shapes, "dtypes": serialize_dtypes(dtypes), "num_classes": num_classes}, f)
        # print ("Loading data finished")

        print(f"Dataset {_dataset} loaded and ready.")

        # Run node classification for each batch size sequentially
        for file_name in file_name_:
            for batch_size in batch_sizes:
                
                if args.mps_split:
                    user_id = mps_get_user_id()
                    mps_daemon_start()
                    mps_server_start(user_id)

                if file_name.startswith("xxx"):
                    for variant in variants:
                        for threads in threads_:
                            for fanout in fanout_:
                                run_node_classification(batch_size, file_name, 5 if file_name.startswith("xxx") else 10,
                                                        args.mfg_size, _dataset, cache_size, args.sampler, args.workers,
                                                        threads, variant, hid_size, fanout)
                else:
                    for threads in threads_:
                        for fanout in fanout_:
                            for d in diff:
                                run_node_classification(batch_size, file_name, 5 if file_name.startswith("xxx") else 10,
                                                            args.mfg_size, _dataset, cache_size, args.sampler, args.workers,
                                                            threads, "xcc", hid_size, d, fanout)
                if args.mps_split:
                    mps_quit()
        del feat, label, train_idx_, val_idx_, test_idx_, shared_graph_formats
        # del dataset  # Ensure the dataset is fully cleared from memory

    print("All datasets processed successfully.")

if __name__ == "__main__":
    main()