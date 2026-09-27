// Turns GFM footnotes ([^1]) into margin sidenotes.
// Each reference becomes: <label> number + hidden checkbox + <span class="sidenote">.
// The checkbox lets narrow screens toggle the note inline without JavaScript.
import { visit, SKIP } from 'unist-util-visit';
import { h } from 'hastscript';

const isElement = (node, tag) => node?.type === 'element' && (!tag || node.tagName === tag);

function hasClass(node, name) {
  const cls = node.properties?.className;
  return Array.isArray(cls) ? cls.includes(name) : cls === name;
}

// Footnote bodies are block content, but a sidenote sits inside a <p>,
// so paragraphs are rewritten as block-styled spans.
function toInline(children) {
  const out = [];
  for (const child of children) {
    if (isElement(child) && child.properties?.dataFootnoteBackref !== undefined) continue;
    if (isElement(child, 'p')) {
      out.push(h('span.sidenote-par', toInline(child.children)));
    } else if (isElement(child)) {
      out.push({ ...child, children: toInline(child.children) });
    } else if (child.type === 'text' && !child.value.trim() && out.length === 0) {
      continue;
    } else {
      out.push(child);
    }
  }
  return out;
}

export default function rehypeSidenotes() {
  return (tree) => {
    const notes = new Map();

    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'section') return;
      if (node.properties?.dataFootnotes === undefined && !hasClass(node, 'footnotes')) return;
      visit(node, 'element', (li) => {
        if (li.tagName === 'li' && li.properties?.id) notes.set(String(li.properties.id), li.children);
      });
      parent.children.splice(index, 1);
      return [SKIP, index];
    });

    if (notes.size === 0) return;
    let counter = 0;

    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'sup' || !parent) return;
      const link = node.children.find((c) => isElement(c, 'a') && c.properties?.dataFootnoteRef !== undefined);
      if (!link) return;
      const target = String(link.properties.href || '').replace(/^#/, '');
      const body = notes.get(target);
      if (!body) return;

      counter += 1;
      const id = `sn-${counter}`;
      const number = link.children.map((c) => c.value ?? '').join('') || String(counter);

      parent.children.splice(
        index,
        1,
        h('label.sidenote-number', { for: id, 'aria-label': `Note ${number}` }, number),
        h('input.margin-toggle', { type: 'checkbox', id }),
        h('span.sidenote', { 'data-number': number }, toInline(body)),
      );
      return [SKIP, index + 3];
    });
  };
}
