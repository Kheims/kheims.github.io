import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { glob, file } from 'astro/loaders';
import { load } from 'js-yaml';

// YAML lists keep their order: each item gets a zero-padded index as id.
const yamlList = (text: string) =>
  Object.fromEntries((load(text) as object[]).map((item, i) => [String(i).padStart(3, '0'), item]));

const posts = defineCollection({
  loader: glob({ pattern: '**/index.{md,mdx}', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});

const link = z.object({ label: z.string(), url: z.string() });

const projects = defineCollection({
  loader: file('src/data/projects.yml', { parser: yamlList }),
  schema: z.object({
    name: z.string(),
    url: z.string(),
    repo: z.string().optional(),
    status: z.string().optional(),
    tag: z.string().optional(),
    featured: z.boolean().default(false),
    blurb: z.string(),
    stack: z.array(z.string()).default([]),
  }),
});

const publications = defineCollection({
  loader: file('src/data/publications.yml', { parser: yamlList }),
  schema: z.object({
    year: z.number(),
    title: z.string(),
    authors: z.string(),
    venue: z.string(),
    note: z.string().optional(),
    links: z.array(link).default([]),
  }),
});

const reading = defineCollection({
  loader: file('src/data/reading.yml', { parser: yamlList }),
  schema: z.object({
    title: z.string(),
    url: z.string(),
    kind: z.string(),
    source: z.string().optional(),
    authors: z.string().optional(),
    category: z.string().default('Other'),
    tags: z.array(z.string()).default([]),
    saved_on: z.coerce.date(),
    note: z.string().optional(),
  }),
});

export const collections = { posts, projects, publications, reading };
