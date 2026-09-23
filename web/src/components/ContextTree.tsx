/**
 * What the agent was shown, laid over what exists.
 *
 * The whole repository tree, with the files that actually reached the prompt
 * highlighted and weighted by their token cost. Everything else is dimmed —
 * and the dimmed part is the point: this is the view that makes "the agent
 * invented code from a filename list" visible in one glance instead of after
 * an afternoon of reading logs.
 */

import { useMemo } from "react";
import type { ContextManifest } from "../api";

interface Node {
  name: string;
  path: string;
  children: Map<string, Node>;
  tokens: number | null;
}

function build(tree: string[], files: Record<string, number>): Node {
  const root: Node = { name: "", path: "", children: new Map(), tokens: null };
  // Files that were injected but are absent from the (possibly truncated)
  // tree still belong on screen: omitting them would hide the very thing the
  // view exists to show.
  const paths = new Set([...tree, ...Object.keys(files)]);
  for (const path of paths) {
    let node = root;
    const parts = path.split("/").filter(Boolean);
    parts.forEach((part, index) => {
      let child = node.children.get(part);
      if (!child) {
        child = {
          name: part,
          path: parts.slice(0, index + 1).join("/"),
          children: new Map(),
          tokens: null,
        };
        node.children.set(part, child);
      }
      node = child;
    });
    node.tokens = files[path] ?? null;
  }
  return root;
}

function Row({ node, depth, max }: { node: Node; depth: number; max: number }) {
  const children = [...node.children.values()].sort((a, b) => {
    const folder = Number(b.children.size > 0) - Number(a.children.size > 0);
    return folder !== 0 ? folder : a.name.localeCompare(b.name);
  });
  const isFolder = children.length > 0;
  const seen = node.tokens !== null;
  const width = seen && max > 0 ? Math.max(4, Math.round((node.tokens! / max) * 100)) : 0;

  return (
    <>
      <div className={`tree-row ${seen ? "seen" : "unseen"}`} style={{ paddingLeft: depth * 14 }}>
        <span className="tree-name">
          {isFolder ? "▾ " : ""}
          {node.name}
          {isFolder ? "/" : ""}
        </span>
        {seen && (
          <span className="tree-cost" title={`${node.tokens} tokens in the prompt`}>
            <span className="tree-bar" style={{ width: `${width}%` }} />
            <span className="tree-tokens">{node.tokens!.toLocaleString()} tk</span>
          </span>
        )}
      </div>
      {children.map((child) => (
        <Row key={child.path} node={child} depth={depth + 1} max={max} />
      ))}
    </>
  );
}

export function ContextTree({ manifest }: { manifest: ContextManifest }) {
  const root = useMemo(
    () => build(manifest.tree, manifest.files),
    [manifest.tree, manifest.files],
  );
  const max = useMemo(
    () => Math.max(0, ...Object.values(manifest.files)),
    [manifest.files],
  );
  const shown = Object.keys(manifest.files).length;
  const used = manifest.estimated_tokens;
  const budget = manifest.budget_tokens;
  const fill = budget > 0 ? Math.min(100, Math.round((used / budget) * 100)) : 0;

  return (
    <section className="panel context">
      <header className="panel-head">
        <h3>
          What the {manifest.role.toLowerCase()} was shown
        </h3>
        <span className={`badge ${shown === 0 ? "bad" : "ok"}`}>
          {shown === 0 ? "no code at all" : `${shown} file${shown > 1 ? "s" : ""}`}
        </span>
      </header>

      <div className="budget">
        <div className="budget-bar">
          <span style={{ width: `${fill}%` }} />
        </div>
        <span className="budget-text">
          {used.toLocaleString()} / {budget.toLocaleString()} tokens
        </span>
      </div>

      {shown === 0 && (
        <p className="warn">
          The prompt carried a file listing and no source. Anything the agent
          produced was invented rather than derived.
        </p>
      )}

      <div className="tree">
        {[...root.children.values()]
          .sort((a, b) => {
            const folder = Number(b.children.size > 0) - Number(a.children.size > 0);
            return folder !== 0 ? folder : a.name.localeCompare(b.name);
          })
          .map((child) => (
            <Row key={child.path} node={child} depth={0} max={max} />
          ))}
      </div>

      {manifest.notes.length > 0 && (
        <ul className="notes">
          {manifest.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
