---
layout: post
title: "MoE routing visual prototype: PixiJS edition"
date: 2026-06-21 11:00:00 +0200
categories: [prototype]
tags: [mixture-of-experts, pixijs, routing, visualization]
series: "Model Scaling Notes"
lede: "A more polished PixiJS prototype for visualizing MoE routing, capacity, router collapse, and dispatch overflow."
math: false
comments: false
published: true
---

This page is a visual prototype for the MoE article format. It is intentionally
short: the goal is to test the feel of a high-quality interactive explanation
inside the Jekyll post layout.

The scene treats the MoE layer as a live routing table:

- tokens arrive as a batch;
- a small gate scores the experts;
- the top-k experts receive token copies;
- each expert has finite slots;
- overflow appears when the router sends too much traffic to the same expert.

{% include interactives/moe-router-pixi.html %}

For a real deep dive, this would be one section in a longer article. The next
sections would slow down and explain router logits, capacity factor, auxiliary
load-balancing loss, and why distributed MoE turns into an all-to-all dispatch
problem.
