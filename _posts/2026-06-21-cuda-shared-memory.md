---
layout: post
title: "CUDA shared memory: the block-local scratchpad"
date: 2026-06-21 12:00:00 +0200
categories: [deep-dive]
tags: [cuda, gpu-architecture, shared-memory, kernels, performance]
series: "GPU Kernel Notes"
lede: "A detailed, end-to-end explanation of CUDA shared memory through a block reduction kernel: why it exists, where __syncthreads() matters, how bank conflicts happen, and how to test the kernel from host code."
math: true
comments: true
published: true
---

CUDA shared memory is not a cache you hope the hardware uses. It is a small
piece of memory that you explicitly allocate per thread block, explicitly fill,
explicitly synchronize around, and explicitly reuse.

That is the useful mental model:

```text
global memory: large, high latency, visible to every block
shared memory: small, low latency, visible only inside one block
registers: private to one thread
```

Shared memory becomes valuable when threads in the same block need to cooperate.
They can load data from global memory once, stage it in a block-local scratchpad,
then perform many reads and writes without going back to DRAM every time.

The trade-off is that CUDA stops protecting you from yourself. If one thread
reads a shared-memory location before another thread has written it, the bug is
yours. If several threads hit the same shared-memory bank in an unfriendly
pattern, the access can serialize. If you allocate too much shared memory per
block, occupancy can drop.

This post builds the concept from the memory model up to a complete kernel.

## The block is the boundary

Every CUDA kernel launch creates a grid of thread blocks. Threads inside the
same block can cooperate through shared memory. Threads in different blocks
cannot directly communicate while the kernel is running.

That boundary is important. If you allocate:

```cpp
extern __shared__ float sdata[];
```

you do not get one global array. You get one `sdata` array per block. Block 0 has
its own `sdata`, block 1 has another, and so on.

For a block reduction, that is exactly what we want. Each block reduces a chunk
of the input and writes one partial sum. A second pass can reduce the partials.

{% include interactives/cuda-shared-memory.html %}

## Why not just use global memory?

Consider summing a large array. The naive approach is:

```cpp
// Not a complete reduction. This only shows the bad idea.
global_tmp[threadIdx.x] = x[i];
__syncthreads(); // This does NOT synchronize global memory across blocks.
global_tmp[threadIdx.x] += global_tmp[threadIdx.x + stride];
```

There are two problems.

First, global memory is much higher latency than shared memory. Repeatedly using
global memory as a scratchpad is expensive.

Second, `__syncthreads()` only synchronizes threads inside one block. It does not
make a global array safe as a communication mechanism between arbitrary blocks.
If the algorithm needs cooperation within a block, shared memory is the natural
tool. If it needs cooperation across blocks, you usually need another kernel
launch, atomics, cooperative groups, or a different algorithm.

## Static and dynamic shared memory

CUDA gives you two common ways to allocate shared memory.

Static shared memory has a compile-time size:

```cpp
__global__ void kernel(...) {
    __shared__ float tile[256];
}
```

Dynamic shared memory is sized at launch:

```cpp
__global__ void kernel(...) {
    extern __shared__ float tile[];
}

kernel<<<grid, block, block.x * sizeof(float)>>>(...);
```

For a teaching reduction, dynamic shared memory is useful because the shared
array size follows the block size. If the block has 256 threads, allocate 256
floats. If it has 512 threads, allocate 512 floats.

## A complete block reduction kernel

The kernel below reduces two input elements per thread. That first local sum is
kept in a register, then written once to shared memory. The block then performs a
tree reduction inside `sdata`.

This is a clarity-first kernel. It is correct and easy to inspect. It is not the
last word in reduction performance.

For this version, use a power-of-two block size such as 128, 256, or 512. That
keeps the reduction loop simple. Production reductions usually handle arbitrary
sizes or delegate the whole problem to a tuned primitive.

```cpp
// reduce_sum.cu
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err__));      \
            std::exit(1);                                                     \
        }                                                                    \
    } while (0)

__global__ void reduce_sum_kernel(const float* __restrict__ x,
                                  float* __restrict__ partial,
                                  int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int block_start = blockIdx.x * blockDim.x * 2;
    unsigned int i = block_start + tid;

    float sum = 0.0f;
    if (i < n) {
        sum += x[i];
    }
    if (i + blockDim.x < n) {
        sum += x[i + blockDim.x];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int n = 1 << 20;
    const int block_size = 256;
    const int elements_per_block = block_size * 2;
    const int num_blocks = (n + elements_per_block - 1) / elements_per_block;

    std::vector<float> h_x(n);
    for (int i = 0; i < n; ++i) {
        h_x[i] = 1.0f / (1.0f + static_cast<float>(i % 17));
    }

    float* d_x = nullptr;
    float* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    size_t shared_bytes = block_size * sizeof(float);
    reduce_sum_kernel<<<num_blocks, block_size, shared_bytes>>>(d_x, d_partial, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_partial(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial,
                          num_blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));

    double gpu_sum = 0.0;
    for (float v : h_partial) {
        gpu_sum += v;
    }

    double cpu_sum = 0.0;
    for (float v : h_x) {
        cpu_sum += v;
    }

    std::printf("gpu partial sum: %.8f\n", gpu_sum);
    std::printf("cpu reference:   %.8f\n", cpu_sum);
    std::printf("absolute error:  %.8f\n", std::abs(gpu_sum - cpu_sum));

    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_x));
    return 0;
}
```

Build it with:

```bash
nvcc -O3 -arch=sm_80 reduce_sum.cu -o reduce_sum
./reduce_sum
```

Change `sm_80` to match your GPU architecture. For example, Ampere A100 is
`sm_80`, Ada Lovelace RTX 4090 is `sm_89`, and Hopper H100 is `sm_90`.

## Walking the kernel

The global index is arranged so each block covers `2 * blockDim.x` elements:

```cpp
unsigned int block_start = blockIdx.x * blockDim.x * 2;
unsigned int i = block_start + threadIdx.x;
```

Thread `t` reads:

```text
x[block_start + t]
x[block_start + t + blockDim.x]
```

Then it adds those two values in a register:

```cpp
float sum = 0.0f;
if (i < n) sum += x[i];
if (i + blockDim.x < n) sum += x[i + blockDim.x];
```

This is a small but important detail. A register is private to the thread and
very fast. There is no reason to write both values to shared memory and then add
them there. First do the cheap private work, then store one value per thread:

```cpp
sdata[tid] = sum;
__syncthreads();
```

The barrier matters. Without it, thread 0 might start reading `sdata[128]`
before thread 128 has stored its value.

After the barrier, the reduction halves the active thread count every iteration:

```cpp
for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
}
```

For `blockDim.x = 256`, the strides are:

```text
128, 64, 32, 16, 8, 4, 2, 1
```

At stride 128, thread 0 adds slot 128 into slot 0, thread 1 adds slot 129 into
slot 1, and so on. At stride 64, only threads 0 to 63 remain active. Eventually
thread 0 holds the block sum.

## Why the final sum is still on the CPU

The kernel writes one partial sum per block:

```cpp
partial[blockIdx.x] = sdata[0];
```

The host code copies those partials back and sums them on the CPU. That is fine
for an explanation, but not always what you want in production.

For large workloads, you would usually launch the same reduction kernel again on
the partial array until only one value remains, or you would use a library such
as CUB. The point here is not to beat CUB. The point is to understand what shared
memory is doing.

## Shared memory banks

Shared memory is split into banks. On modern NVIDIA GPUs, the useful beginner
model is 32 banks, where consecutive 32-bit words map across consecutive banks:

```text
bank = word_index % 32
```

If a warp reads:

```cpp
sdata[threadIdx.x]
```

then lane 0 hits bank 0, lane 1 hits bank 1, and so on. That is the happy path.

If a warp reads:

```cpp
sdata[threadIdx.x * 2]
```

then lanes collide on fewer banks. Bank 0 receives lane 0 and lane 16, bank 2
receives lane 1 and lane 17, and so on. The hardware may need to serialize the
access into multiple transactions.

That is a bank conflict. It is not the same thing as an uncoalesced global-memory
load. Global coalescing concerns DRAM transactions. Bank conflicts concern the
shared-memory crossbar inside an SM.

One exception is broadcast. If all lanes read the same shared-memory address, the
hardware can broadcast the value instead of treating it as a normal conflict.

## When shared memory helps

Shared memory is useful when at least one of these is true:

- Threads in a block reuse the same data multiple times.
- Threads need to exchange intermediate values.
- You want to transform a bad global-memory access pattern into a better one.
- You need a block-local staging buffer before a collective operation.

Classic examples:

- tiled matrix multiplication;
- reductions and scans;
- histogram bins per block;
- stencil computations;
- tiled transpose kernels;
- staging data before tensor-core fragments are loaded.

Shared memory is not automatically faster. If each value is used once, and the
global access is already coalesced, staging through shared memory can just add
instructions and barriers.

## The cost model

Shared memory has three major costs.

First, the bytes are limited. The exact limit depends on the GPU and the selected
shared-memory/L1 configuration. If a block uses too much shared memory, fewer
blocks can reside on the same SM.

Second, barriers are not free. `__syncthreads()` waits until every non-exited
thread in the block reaches the barrier. A reduction with many barriers is easy
to reason about, but not necessarily optimal.

Third, bank conflicts can serialize memory operations. A shared-memory access
pattern can look local and still be slow.

This is why production kernels often mix shared memory with warp-level
primitives. A common optimized reduction uses shared memory across warps, then
uses warp shuffles inside the final warp. The teaching kernel above keeps the
barriers because they make the synchronization boundary explicit.

## Checklist for writing a shared-memory kernel

When writing your own kernel, check these points in order:

1. What data is reused by threads in the same block?
2. How many bytes of shared memory does each block need?
3. Which threads write each shared-memory location?
4. Which threads read each location?
5. Is there a required `__syncthreads()` between the writes and reads?
6. What is the bank mapping for the hot shared-memory access?
7. Does the shared-memory allocation reduce occupancy too much?

If you cannot answer these questions, the kernel may still work, but you do not
yet understand why.

## Takeaway

Shared memory is CUDA's explicit cooperation mechanism inside a block. It is
fast because it lives near the SM. It is dangerous because it is manual. The
kernel writer decides what gets staged, when threads synchronize, and whether
the access pattern is friendly to banks.

The reduction kernel is small, but it contains the essential pattern:

```text
global load -> register work -> shared store -> block barrier
-> shared-memory collective -> one global write
```

Once that pattern is clear, tiled matmul, transpose kernels, scans, and more
advanced GPU kernels become much easier to read.
