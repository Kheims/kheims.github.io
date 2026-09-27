# kheims.github.io

Personal site and blog, built with [Astro](https://astro.build) and deployed to GitHub Pages on every push to `main`.

## Setup

```sh
npm install
npm run dev        # http://localhost:4321, reloads on save
npm run build      # production build into dist/
```

## Writing a post

```sh
npm run new "Tiling a matmul kernel"          # Markdown post
npm run new "Tiling a matmul kernel" -- --mdx # MDX post (can use components)
```

This creates `src/content/posts/tiling-a-matmul-kernel/index.md` with `draft: true`.
Drafts show up in `npm run dev` only. Set `draft: false` and push to publish.

Front matter:

```yaml
title: "Tiling a matmul kernel"
description: "One line shown in the post list and in link previews."
date: 2026-09-27
updated: 2026-10-02   # optional
tags: [cuda, gpu]
draft: false
```

`src/content/posts/writing-guide/index.mdx` is a draft showing every feature. Open it in `npm run dev` for a live reference.

### Sidenotes

Use normal Markdown footnotes. They are rendered in the right margin on wide screens and open on tap on phones.

```md
Shared memory is banked.[^banks]

[^banks]: 32 banks of 4 bytes on recent NVIDIA GPUs.
```

In MDX posts, `<MarginNote>...</MarginNote>` adds an unnumbered note.

### Math

`$inline$` and `$$display$$`, rendered with KaTeX at build time.

### Code

Fenced blocks with a language (` ```python `, ` ```cuda `, ` ```cpp `...), highlighted with Shiki.

### Figures

Keep images in the post folder. An image alone on its line becomes a figure, and its title becomes the caption:

```md
![Tiled matmul](./tiling.svg "Each block loads one tile of A and B into shared memory.")
```

SVG line drawings are inverted in dark mode, so draw them black on a transparent background.
In MDX posts, `<Figure src={img} alt="..." caption="..." wide />` extends a figure into the margin (import the image first: `import img from './tiling.svg'`).

Drawing tools that export clean SVG: Excalidraw (also as an Obsidian plugin), draw.io, Figma. For plots, matplotlib with `plt.savefig("fig.svg", transparent=True)`.

## Other pages

| Page | Edit |
| --- | --- |
| Home intro | `src/pages/index.astro` |
| About | `src/pages/about.md` |
| Projects | `src/data/projects.yml` |
| Publications | `src/data/publications.yml` |
| Reading | `src/data/reading.yml` |
| Nav, links | `src/site.ts` |
| Colors, fonts, layout | `src/styles/global.css` (tokens at the top) |

The previous Jekyll site, including unpublished drafts, is kept on the `archive/jekyll` branch.
