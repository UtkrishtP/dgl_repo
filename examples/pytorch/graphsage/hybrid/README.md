This directory contains all the test files and the implementation of hybrid sampling

# Details

 ## Experiment Details

```

Dataset : ogbn-papers100M
Fanout : [10, 10, 10]
Model : GraphSage

```
 ## Overview

 Hybrid sampling has been implemented on top of DGL 1.1.2 for the following three algos:
 - Fused Neighbor Sampling
 - Neighbor Sampling
 - Labor Sampling
 - SAINT sampling (Will be added)

 For each of the sampling currently we have implemented the following prefetching to GPU scenarios:
 - MFG + nfeat
 - MFG
 - nfeat
 - None

 Since hybrid sampling consist of both CPU and GPU based dataloaders working simultaneously, for CPU based dataloaders we have tuned the best combinations for # workers.

 ## Implementation

- We launch a thread from the main process, responsible for producing CPU based samples continuously. 
- We keep a shared array either on CPU or GPU based on the variant used, to store the MFG's produced by the CPU sampler.
- Initially the pipeline is started with a complete GPU based training since initially no CPU samples will be available.
- After end of every epoch, we check the flag ``` cpu_samples ``` to see if any cpu sample is ready or not.
- If yes, we directly start the training on GPU by using the produced samples for the particular epoch.
- Once we have exhausted all the CPU produced samples, we revert back to a pure GPU pipeline.
- The above process keeps on repeating to and fro until total epochs required for training.

## To-do
