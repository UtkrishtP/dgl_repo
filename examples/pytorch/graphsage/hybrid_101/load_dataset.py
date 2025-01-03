from custom_dl import FriendsterDataset, TwitterDataset, IGB260MDGLDataset
import torch
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
from dgl.convert import hetero_from_shared_memory
from dgl.utils.pin_memory import pin_memory_inplace
import dgl
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
import args, time, argparse, threading, json, gc, os
from args import get_args
from multiprocessing import Manager

def deserialize_dtypes(dtypes):
    return [getattr(torch, dtype) for dtype in dtypes]

def fetch_train_graph():
    # global shapes, dtypes
    with open("/dev/shm/shapes_dtypes.json", "r") as f:
        data = json.load(f)
    shapes = data["shapes"]
    dtypes = deserialize_dtypes(data["dtypes"])
    train_idx = get_shared_mem_array("train_idx", shapes[3], dtype=dtypes[3])
    val_idx = get_shared_mem_array("val_idx", shapes[4], dtype=dtypes[4])
    test_idx = get_shared_mem_array("test_idx", shapes[5], dtype=dtypes[5])
    g = hetero_from_shared_memory("graph_formats")
    return train_idx, val_idx, test_idx, g

def fetch_all():
    # global shapes, dtypes
    with open("/dev/shm/shapes_dtypes.json", "r") as f:
        data = json.load(f)
    shapes = data["shapes"]
    dtypes = deserialize_dtypes(data["dtypes"])
    train_idx = get_shared_mem_array("train_idx", shapes[3], dtype=dtypes[3])
    val_idx = get_shared_mem_array("val_idx", shapes[4], dtype=dtypes[4])
    test_idx = get_shared_mem_array("test_idx", shapes[5], dtype=dtypes[5])
    g = hetero_from_shared_memory("graph_formats")
    g.ndata["feat"] = get_shared_mem_array("feats", shapes[0], dtype=dtypes[0])
    g.ndata["label"] = get_shared_mem_array("labels", shapes[2], dtype=dtypes[2])
    return train_idx, val_idx ,test_idx, g

def fetch_shapes():
    # global shapes, num_classes
    with open("/dev/shm/shapes_dtypes.json", "r") as f:
        data = json.load(f)
    shapes = data["shapes"]
    num_classes = data["num_classes"]
    return shapes[0][1], num_classes

def serialize_dtypes(dtypes):
    return [str(dtype).split('.')[-1] for dtype in dtypes]

def clear_shared_memory():
    """Clear all files from /dev/shm using os.system."""
    command = "rm -rf /dev/shm/*"
    print(f"Executing command: {command}")
    os.system(command)  # Execute the command directly

def host_datas(e):
    # global shapes, dtypes, num_classes
    if args.dataset == "friendster":
        dataset = FriendsterDataset(args)
        num_classes = 64
        train_idx = torch.nonzero(dataset[0].ndata['train_mask'], as_tuple=True)[0]
        val_idx = torch.nonzero(dataset[0].ndata['val_mask'], as_tuple=True)[0]
        test_idx = torch.nonzero(dataset[0].ndata['test_mask'], as_tuple=True)[0]
    
    elif args.dataset == "twitter":
        dataset = TwitterDataset(args)
        num_classes = 64
        train_idx = torch.nonzero(dataset[0].ndata['train_mask'], as_tuple=True)[0]
        val_idx = torch.nonzero(dataset[0].ndata['val_mask'], as_tuple=True)[0]
        test_idx = torch.nonzero(dataset[0].ndata['test_mask'], as_tuple=True)[0]

    elif args.dataset.startswith("igb"):
        args.dataset_size = args.dataset.split("-")[1]
        args.num_classes = num_classes = 19
        dataset = IGB260MDGLDataset(args)
        train_idx = torch.nonzero(dataset[0].ndata['train_mask'], as_tuple=True)[0]
        val_idx = torch.nonzero(dataset[0].ndata['val_mask'], as_tuple=True)[0]
        test_idx = torch.nonzero(dataset[0].ndata['test_mask'], as_tuple=True)[0]
        # print("Label: ", dataset[0].ndata['label'])
        # print(torch.all(dataset[0].ndata['label'] < num_classes).item())
    else:
        dataset = AsNodePredDataset(
            DglNodePropPredDataset(
                args.dataset, root="/data/"
                ), save_dir="/data/tmp/"
        )
        num_classes = dataset.num_classes
        train_idx = dataset.train_idx
        val_idx = dataset.val_idx
        test_idx = dataset.test_idx
    g = dataset[0]
    g.ndata["label"] = g.ndata["label"].type(torch.long)

    # Creating shared + pinned memory regions for graph formats {coo, csr, csc}, nfeat, labels, {train/val/test}_idx
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
    print ("Loading data finished")
    print(" Press Ctrl+D to release shared memory")
    breakpoint()

if __name__ == "__main__":
    args = get_args()
    event = threading.Event()
    threading.Thread(target=host_datas,
                     args=[event],
                     daemon=True).start()
    try:
        event.wait()
    except KeyboardInterrupt:
        clear_shared_memory()