// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://combine-lab.github.io',
  base: '/caps-sa',
  integrations: [
    starlight({
      title: 'caps-sa',
      logo: {
        src: './src/assets/capssa-icon.svg',
      },
      favicon: '/favicon.svg',
      head: [
        { tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.googleapis.com' } },
        { tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' } },
        {
          tag: 'link',
          attrs: {
            rel: 'stylesheet',
            href: 'https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap',
          },
        },
      ],
      components: {
        Header: './src/components/Header.astro',
      },
      customCss: ['./src/styles/custom.css'],
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/COMBINE-lab/caps-sa',
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/COMBINE-lab/caps-sa/edit/main/docs/',
      },
      sidebar: [
        {
          label: 'Getting started',
          items: [
            { label: 'Introduction', slug: 'getting-started/introduction' },
            { label: 'Installation', slug: 'getting-started/installation' },
            { label: 'Quick start', slug: 'getting-started/quick-start' },
          ],
        },
        {
          label: 'Concepts',
          items: [
            { label: 'The algorithm', slug: 'concepts/algorithm' },
            { label: 'Geometric LCP memoization', slug: 'concepts/geometric-memoization' },
            { label: 'The library & the CLI', slug: 'concepts/crates' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'CLI parameters', slug: 'reference/cli' },
            { label: 'Library API', slug: 'reference/api' },
            { label: 'Performance', slug: 'reference/performance' },
          ],
        },
      ],
    }),
  ],
});
