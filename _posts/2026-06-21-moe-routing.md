---
layout: post
title: "Mixture of Experts: routing tokens without melting the experts"
date: 2026-06-21 10:00:00 +0200
categories: [deep-dive]
tags: [mixture-of-experts, transformers, routing, llms, systems]
series: "Model Scaling Notes"
lede: "A visual explanation of how sparse MoE layers route tokens, why expert capacity matters, and why load balancing is not a cosmetic loss."
math: true
comments: true
published: true
---

Dense transformer layers spend the same feed-forward compute on every token. A
Mixture of Experts layer changes that bargain: keep many feed-forward networks
available, but activate only a small subset for each token.

That is the main appeal. The model can carry more parameters without paying the
full compute cost on every forward pass. The price is that the model now needs a
traffic controller.

## The sparse layer

An MoE block usually replaces the dense feed-forward network inside a transformer
block:

```text
token hidden state -> router -> selected experts -> weighted combine
```

The router is a small learned projection. For each token representation `x`, it
produces one score per expert:

```text
scores = x W_router
```

Then the layer keeps the top-k experts. In a top-1 router, a token goes to one
expert. In top-2 routing, it goes to two experts and the outputs are mixed back
together using the router probabilities.

This is where the system stops being just a model architecture question. Tokens
are not routed one at a time in isolation. They are routed as a batch, and every
expert has finite capacity.

{% include interactives/moe-router.html %}

## Capacity is a systems constraint

If a batch has `T` tokens, `E` experts, and each token is sent to `k` experts,
then the average load per expert is:

$$
\frac{T \cdot k}{E}
$$

But average load is not the operational problem. The problem is the tail. If the
router prefers one expert too often, that expert receives more tokens than it can
process in the fixed dispatch buffer.

Most MoE implementations choose a capacity per expert:

$$
\text{capacity} =
\left\lfloor
\frac{T \cdot k}{E} \cdot \text{capacity factor}
\right\rfloor
$$

Tokens that arrive after the expert is full are usually dropped, padded, or sent
through a fallback path depending on the implementation. That sounds like an
implementation detail until you remember that a dropped token does not get the
expert computation the router wanted.

## Why load balancing exists

The router is trained to minimize the model loss. Without extra pressure, it can
discover shortcuts: route too many tokens to a few experts that happen to be
useful early in training. That creates two problems at once:

- Hot experts become communication and compute bottlenecks.
- Cold experts receive too few examples to specialize.

So MoE models add an auxiliary load-balancing objective. The exact formula
varies, but the intent is stable: encourage the probability mass and actual token
counts to spread across experts.

This does not mean "make every expert identical." It means "do not let the router
collapse before the experts have a chance to learn useful specializations."

## The important mental model

MoE is not free parameters. It is a routing problem wrapped around a model
scaling trick.

For a clean mental model, track four quantities:

1. `top_k`: how many experts each token activates.
2. `num_experts`: how much sparse capacity the layer has.
3. `capacity_factor`: how much overflow room each expert gets.
4. `load_balance`: how strongly training discourages expert collapse.

The first two decide the shape of the layer. The second two decide whether the
layer is usable at scale.

## What to watch for in real systems

Production MoE layers are dominated by dispatch and combine operations. Tokens
must be grouped by expert, sent to the right device, processed, then scattered
back to their original order. On a single GPU this is mostly indexing and
packing. Across GPUs, it becomes an all-to-all communication problem.

That is why MoE papers often report both model quality and routing statistics:
tokens per expert, dropped-token rate, capacity factor, and communication cost.
If those numbers are bad, the architecture may look elegant while the system is
quietly losing the benefit.

The router is small. The consequences of the router are not.
