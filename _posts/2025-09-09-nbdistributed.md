---
layout: post
title: "Nbdistributed – Distributed Training Without Losing Your Sanity"
date: 2025-09-09 10:00:00 +0200
categories: [distributed, pytorch, llms]
tags: [distributed-training, pytorch, collective-ops, nccl, tutorial]
series: "Distributed Training in PyTorch"
---

> 📚 **Part 1 of the Distributed Training series** — Next up: [Distributed Data Parallelism, a gentle intro](/posts/ddp/)

# Nbdistributed - Introduction and Basic Operations

## Why Distributed Training Matters

Training deep learning models used to fit on a single GPU. Then came transformers… and then Large Language Models (LLMs).

These beasts don’t just *prefer* more GPUs, they **demand** them. Why?

- **Memory limits:** A single A100 GPU has 40–80 GB of memory. Sounds big… until you try to fit a 65B parameter model that requires terabytes.
- **Speed:** Training on one GPU could take years. Distributing across many GPUs lets us split the work (and get results before we retire).
- **Scalability:** Want to train across multiple nodes in a cluster? You’ll need to coordinate GPUs as if they were one big “virtual GPU.”

But scaling comes with its own headaches:

- How do GPUs talk to each other?
- What happens if one process crashes?
- How do we avoid wasting time in communication bottlenecks?

👉 That’s exactly what these sessions are about. We’ll explore the **building blocks of distributed training**, understand their quirks, and learn how to wield them without blowing up your notebook kernel.

## Why `nbdistributed`?

In this course we rely on **`torch.distributed`**, which provides the low-level primitives to communicate between multiple GPUs.

Running distributed code inside a **notebook**  used to be painful. Previously, one had to rely on the `accelerate` notebook launcher, which came with several drawbacks:

- You couldn’t interact with CUDA until the notebook was fully launched.
- Errors would only appear at runtime, often forcing a full restart.

`nbdistributed` was built ( thanks to zach mueller, check his blogs at :  https://muellerzr.github.io/)to fix this workflow:

- Distributed initialization works directly from within a notebook cell.
- Errors remain isolated to the process group thread → you can simply re-run the faulty cell without restarting the entire notebook.

Nb : In this first episode I will be using Modal’s Notebook ( as they are sponsor for the course and their Notebook tool works pretty well with nbdistributed ), for the remaining episodes I may switch between Modal and Lambda.

## Initializing and Shutting Down

To start a distributed session:

First Make sure to install nbdistributed using : 

```python
# After initializing your environment 
!uv pip install nbdistributed
#Then 
%load_ext nbdistributed
```

Then : 

![image.png](/assets/img/nbdistributed/image.png)

- This launches two processes, mapped to GPU 3 and GPU 4.
- GPU indexing is resolved by **hardware generation order** (you can check via `nvidia-smi`).

Once finished, you can shut down the distributed environment : 

```python
%dist_shutdown
```

After this, the notebook goes back to single-GPU (or CPU) mode.

⚠️ If you want to re-initialize with different GPUs, you must first shutdown the distributed environment.

## Handling Errors Gracefully

With `nbdistributed`, each rank runs in isolation. If rank 0 throws an error, other ranks remain unaffected.

This allows you to re-define variables only where needed:

.

```python
%%rank [0]
t = torch.tensor([1, 2, 3]).to(device)
print(t)  

# At this point, only rank 0 knows about `t`.

%%rank [1]
t = torch.tensor([1, 2, 3]).to(device)
print(t)

```

No need to restart the entire notebook 🎉.

****

## **Core Operations**

### 1. Broadcast (One → All)

Broadcast sends a tensor from one rank (the *source*) to all others.

Each receiving rank must allocate memory before the operation:

![image.png](/assets/img/nbdistributed/image%201.png)

Rank 1’s tensor received Rank 0’s tensor. 

### 2. Point-to-Point Communication (A → B)

- **Synchronous send/receive**: the sender and receiver must both be ready.
    
    The program stalls until the transfer completes.
    
    ![image.png](/assets/img/nbdistributed/image%202.png)
    
- **Asynchronous send/receive**: operations return immediately, allowing overlap with other computation.
    
    You can later synchronize to ensure completion.
    
    Useful for large tensors (such as model weights).
    

### 3. Scatter (One → Many)

Splits a tensor into chunks and distributes each chunk to a different rank.

![image.png](/assets/img/nbdistributed/image%203.png)

By the way, data type or tensor shapes mismatches between ranks will cause a system crash.

### 4. Reduction Operations

Reduction aggregates tensors across ranks into one result.

- **All-reduce (All → All):** each rank ends up with the reduced result.
- **Reduce (All → One):** only the destination rank receives the result.

Example (sum reduction):

![image.png](/assets/img/nbdistributed/image%204.png)

There are other ops for the reduce mapping including MAX, MIN and PRODUCT. 

### 5. Gather / All-gather

This is one of the most important operation, why you may ask ? 

Here’s why:

1. **Model Weights Synchronization**
    
    In data-parallel training, every GPU trains on a different mini-batch. After a step, gradients must be *shared*.
    
    - The usual approach is an **all-reduce** (sum/avg gradients).
    - But to **rebuild parameters** (in sharded or checkpointed setups), we often need **all-gather** to reassemble pieces of tensors stored across GPUs.
        
        → Without all-gather, you’d never be able to reconstruct the full weight matrix on a single rank.
        
2. **Embedding Layers and Token Tables (LLMs)**
    
    For language models, embeddings can be *huge* (tens or hundreds of GB).
    
    - We often shard the embedding matrix across GPUs to fit memory.
    - When tokens need to be looked up, **all-gather** is used to collect partial results so each rank can continue computation.
        
        → Example: each GPU stores 1/8 of the vocabulary embeddings. When you feed tokens into the model, the results must be gathered so the next layer sees the complete batch.
        
3. **Checkpointing and Model Saving**
    
    When saving a distributed model, each rank might only hold a *shard* of the weights.
    
    - Before writing to disk, we typically need a **gather** to assemble them into one coherent state dict.
    - Otherwise, you’d just end up with random tensor slices per GPU.
4. **Inference and Output Assembly**
    
    In multi-GPU inference, results may be produced in parallel.
    
    - To return predictions (such as generated text), you need to **gather outputs** to a single process (often rank 0).
    - Without this, you’d get fragmented results scattered across devices.

In summary, Gather is like the “group project” moment in distributed training: everyone does their piece locally, but eventually you need to bring all the parts together in one place to continue.

Alright, after this thorough explanation, let’s get back to business.

There are two types of gather operations : 

- **Gather:** collects tensors from all ranks into one rank.
- **All-gather:** collects tensors from all ranks into **every** rank.

![image.png](/assets/img/nbdistributed/image%205.png)

## Performance Notes

Distributed training is only as fast as its slowest link.

- Example: transferring **100 GB** of weights across a 12.5 GB/s PCIe bus takes ~8 seconds.
- That’s 8 seconds of idle GPUs.

Asynchronous operations help overlap communication with computation, so GPUs aren’t just waiting around.

Rule of thumb:

- Use communication efficiently.
- Minimize unnecessary sync points.
- Remember: PCIe usually beats Ethernet, unless you’ve got serious HPC gear (100 Gbps Infiniband).

👉 That was all for this session, next sessions will dive deeper into how these primitives combine into **data parallelism** and **model parallelism**, which are the backbone of scaling LLM training.