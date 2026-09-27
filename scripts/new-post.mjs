// Usage: npm run new "Post title" [-- --mdx]
import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const args = process.argv.slice(2);
const mdx = args.includes('--mdx');
const title = args.filter((a) => !a.startsWith('--')).join(' ').trim();

if (!title) {
  console.error('Usage: npm run new "Post title"   (add -- --mdx for an MDX post)');
  process.exit(1);
}

const slug = title
  .toLowerCase()
  .normalize('NFD')
  .replace(/[̀-ͯ]/g, '')
  .replace(/[^a-z0-9]+/g, '-')
  .replace(/^-|-$/g, '');
const date = new Date().toISOString().slice(0, 10);
const dir = join('src/content/posts', slug);
const file = join(dir, mdx ? 'index.mdx' : 'index.md');

if (existsSync(dir)) {
  console.error(`${dir} already exists`);
  process.exit(1);
}

mkdirSync(dir, { recursive: true });
writeFileSync(
  file,
  `---
title: "${title.replace(/"/g, '\\"')}"
description: ""
date: ${date}
tags: []
draft: true
---

Write here. Put images next to this file and link them as ![alt](./figure.svg "Caption").
`,
);
console.log(`Created ${file}\nSet draft: false when it's ready to publish.`);
