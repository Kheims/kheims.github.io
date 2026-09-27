import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSidenotes from './src/plugins/rehype-sidenotes.mjs';
import rehypeFigures from './src/plugins/rehype-figures.mjs';

export default defineConfig({
  site: 'https://kheims.github.io',
  integrations: [mdx(), sitemap()],
  markdown: {
    // Sidenotes, figures and math need remark/rehype plugins, so use the unified pipeline.
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex, rehypeFigures, rehypeSidenotes],
    }),
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark-dimmed' },
      defaultColor: false,
      langAlias: { cuda: 'cpp' },
    },
  },
});
