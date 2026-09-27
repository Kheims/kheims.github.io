// Wraps a paragraph that holds only an image into <figure>.
// The Markdown image title becomes the caption:
//   ![alt text](./diagram.svg "Caption shown under the figure")
import { visit } from 'unist-util-visit';
import { h } from 'hastscript';

const isBlank = (node) => node.type === 'text' && !node.value.trim();

export default function rehypeFigures() {
  return (tree) => {
    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'p' || !parent) return;
      const content = node.children.filter((c) => !isBlank(c));
      if (content.length !== 1 || content[0].type !== 'element' || content[0].tagName !== 'img') return;

      const img = content[0];
      const caption = img.properties?.title;
      if (caption) delete img.properties.title;
      parent.children[index] = h('figure', [img, caption ? h('figcaption', caption) : null].filter(Boolean));
    });
  };
}
