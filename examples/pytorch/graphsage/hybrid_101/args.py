import argparse


# Function to parse fanout input
def parse_fan_out(fan_out_str):
    """Converts comma-separated string into a list of integers."""
    return [int(f) for f in fan_out_str.split(",")]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    parser.add_argument("--cgg", type=bool, default=True, choices=[True, False])
    parser.add_argument(
        "--set",
        type=str,
        default="ggg",
        choices=["ggg", "g__", "cc_", "c__"],
        help="Set of operations (S,E,T) to be performed & where",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default='nbr',
        choices=['nbr', 'lbr', 'lbr2'],
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--pin_mfg",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--madvise",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--mfg_size",
        type=int,
        default=5,
        help="Specify the size of shared memory region in GBs"
    )
    parser.add_argument(
        "--hybrid",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=10000000,
    )

    parser.add_argument(
        "--ggg_footprint",
        type=int,
        default=5,
    )
    
    parser.add_argument(
        "--hid_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--prefetch_thread",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--path",
        type=str,
        default="/data/",
        help="path containing the datasets",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv",
    )
    parser.add_argument(
        "--dataset_size",
        type=str,
        default="medium",
        choices=["tiny", "small", "medium", "large", "full"],
        help="size of the datasets",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=64,
        choices=[19, 2983],
        help="number of classes",
    )
    parser.add_argument(
        "--in_memory",
        type=int,
        default=1,
        choices=[0, 1],
        help="0:read only mmap_mode=r, 1:load into memory",
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=1,
        choices=[0, 1],
        help="0:nlp-node embeddings, 1:random",
    )
    parser.add_argument(
        "--nfeat_dim",
        type=int,
        default=380,
    )

    # Model
    parser.add_argument(
        "--model_type", type=str, default="sage", choices=["gat", "sage", "gcn"]
    )
    parser.add_argument("--modelpath", type=str, default="deletethis.pt")
    parser.add_argument("--model_save", type=int, default=0)

    # Model parameters
    parser.add_argument(
        "--fan_out",
        type=str,
        default="15,10,5",
        help="Comma-separated fanout values for each layer (e.g.,\
            '15,10,5' for fanout for [layer-0, layer-1, layer-2]).",
    )
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--learning_rate", type=int, default=0.01)
    parser.add_argument("--decay", type=int, default=0.001)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)

    parser.add_argument(
        "--log_to_file",
        type=bool,
        default=False,
        help="Enable logging of the run times to file"
    )
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    # Parse the fanout string into a list of integers
    args.fan_out = parse_fan_out(args.fan_out)

    return args

