## Argument list and explanation

- Script usage to run igb_large dataset:
```
python hybrid_103.py --dataset igb-large --batch_size 1024 --cache_size 80449000 --mfg_size 100 --fan_out 15,10 --epoch 10 --ggg_footprint 0 --mfg_per_mb 1.5*1024*1024  2>&1 | tee log.txt
```

List of all **essential** arguments used:
- `dataset`
- `batch_size`
- `cache_size`
- `mfg_size` : Size of shared memory region created on CPU where sampler and MFG_transfer will operate.
- `epoch`
- `fan_out`
- `ggg_footprint` : Static entry, will be configured dynamically later.
- `mfg_per_mb` : Size of an MFG per mini_batch, used to determine the #MB's that can be fit on HBM. Currently static, will be configured dynamically later.

## Automated script

`hybrid_101_test.py`

The above file when run:
- Loads specified datasets into shared memory.
- Can run all specified files.
- Specify Batch sizes/cache_size/etc.
```
    datasets = ["friendster" ,"twitter", "ogbn-papers100M", "igb-large"]
    batch_sizes = [8192, 1024]
    file_name_ = ["xxx_variants.py", "hybrid_103.py"]
```