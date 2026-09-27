# kheims.github.io

Personal research portfolio for Djamel Mesbah. The site is a small Jekyll app:
posts live in `_posts`, projects in `_data/projects.yml`, publications in
`_data/publications.yml`, reading-list entries in `_data/reading.yml`, and
top-level pages in `_tabs`.

## Local workflow

```sh
bundle install
bash tools/run.sh
```

Production-style validation:

```sh
bash tools/test.sh
```

The test script validates the content data, builds the site, and runs
html-proofer with external links disabled.

## Add a post

Use the scaffold command:

```sh
ruby tools/new-post "How DDP Actually Synchronizes Gradients" \
  --tags pytorch,distributed-training,ddp \
  --series "Distributed Training in PyTorch"
```

This creates a hidden post in `_posts` with `published: false`. Preview it with:

```sh
bundle exec jekyll serve --unpublished
```

When the article is ready, change `published: true`.

Recommended front matter:

```yaml
---
title: "Post title"
date: 2026-06-16 10:00:00 +0200
categories: [deep-dive]
tags: [pytorch, distributed-training]
series: "Distributed Training in PyTorch"
lede: "One sentence that says what the reader will learn."
math: false
comments: true
repo: https://github.com/Kheims/example
paper: https://doi.org/example
published: true
---
```

Use the generated sections as the article recipe: TL;DR, Context, Setup, Core
Idea, Walkthrough, Results, and References.

## Add a project

Edit `_data/projects.yml`.

```yaml
- name: Project name
  url: https://github.com/Kheims/project
  repo: https://github.com/Kheims/project
  status: active
  tag: systems
  featured: true
  blurb: >-
    Short concrete description of the problem, the approach, and why it matters.
  stack: [Python, PyTorch, CUDA]
```

Set `featured: true` only for items that should appear on the homepage. For
private consulting or collaboration work, omit `url` and `repo`, then add
`visibility: private` or `visibility: collaboration`.

## Add a publication

Edit `_data/publications.yml`.

```yaml
- year: 2026
  title: "Paper title"
  authors: "Djamel Mesbah, Coauthor Name"
  venue: "Conference or journal name."
  note: "Best Paper"
  links:
    - { label: DOI, url: "https://doi.org/..." }
    - { label: Code, url: "https://github.com/Kheims/..." }
    - { label: HAL, url: "https://hal.science/..." }
```

## Add a reading-list item

Edit `_data/reading.yml`. The list accepts papers, blogs, X threads, talks,
repositories, or any other reference you want to keep public.

```yaml
- title: "Reference title"
  url: "https://example.com/reference"
  kind: paper
  source: arXiv
  authors: "Author One, Author Two"
  category: ML systems
  tags: [distributed-training, pytorch, scaling]
  saved_on: 2026-06-16
  note: >-
    One short personal note about why this is worth reading or how it connects
    to your work.
```

Useful `kind` values: `paper`, `blog`, `thread`, `talk`, `repo`, `docs`,
`book`, `course`.

Run `bash tools/test.sh` before pushing.
