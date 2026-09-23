/**
 * Where a project comes from.
 *
 * A project is its files, uploaded here, and nothing else: there is no field
 * for a path on the API host, because a path is a claim about a machine the
 * person at this screen cannot see, and the one time projects were created
 * that way they all named the same mount and the agents ran against whatever
 * happened to be there. The upload is the whole tree, so what the agents see
 * is what was chosen.
 *
 * One list of files feeds the form, filled from either of two pickers: plain
 * files, which arrive with only a basename and suit a flat project or a single
 * zip, or a folder, which arrives with each file's path under it.
 */

import { useState, type ChangeEvent, type FormEvent } from "react";

export interface Upload {
  name: string;
  files: { path: string; file: File }[];
  build_command?: string;
  test_command?: string;
}

/** What the server would only reject or bloat on: never source, always large. */
const NOT_SOURCE = new Set([".git", "node_modules", "__pycache__", ".venv"]);

/**
 * Turn a picker's selection into paths relative to the project root.
 *
 * A folder pick reports every path from the chosen folder down, including the
 * folder's own name as the first segment. That segment is dropped: the person
 * chose that folder *as* the project, so its contents are the root, and
 * keeping it would put every build command one directory above the code. A
 * plain pick has no directory to report and the basename is the path.
 */
function relativePaths(list: FileList, fromFolder: boolean) {
  const kept: Upload["files"] = [];
  let skipped = 0;
  for (const file of Array.from(list)) {
    const nested = fromFolder && file.webkitRelativePath ? file.webkitRelativePath : "";
    const path = nested ? nested.split("/").slice(1).join("/") : file.name;
    if (path.split("/").some((segment) => NOT_SOURCE.has(segment))) {
      skipped += 1;
      continue;
    }
    kept.push({ path, file });
  }
  const folder = fromFolder && list[0]?.webkitRelativePath ? list[0].webkitRelativePath.split("/")[0] : "";
  return { kept, skipped, folder };
}

/** Sizes in the coarsest unit that still shows a digit before the point. */
function size(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const isArchive = (files: Upload["files"]) =>
  files.length === 1 && files[0]?.path.toLowerCase().endsWith(".zip") === true;

export function NewProject({
  onCreate,
  busy,
}: {
  onCreate: (upload: Upload) => Promise<boolean>;
  busy: boolean;
}) {
  const [name, setName] = useState("");
  const [files, setFiles] = useState<Upload["files"]>([]);
  const [skipped, setSkipped] = useState(0);
  const [build, setBuild] = useState("");
  const [test, setTest] = useState("");
  // Why the form will not submit, or null. It is only ever set on a submit
  // attempt, so a blank form is not shouted at before anything was tried.
  const [refusal, setRefusal] = useState<string | null>(null);

  const total = files.reduce((sum, f) => sum + f.file.size, 0);

  const pick = (fromFolder: boolean) => (event: ChangeEvent<HTMLInputElement>) => {
    const list = event.target.files;
    if (!list || list.length === 0) return;
    const { kept, skipped: dropped, folder } = relativePaths(list, fromFolder);
    setFiles(kept);
    setSkipped(dropped);
    setRefusal(null);
    // The folder's name is the likeliest project name, and offering it costs
    // nothing; a name already typed is left alone.
    if (folder && !name.trim()) setName(folder);
    // The same folder or file chosen twice must fire change twice.
    event.target.value = "";
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return setRefusal("A name is required.");
    if (files.length === 0) return setRefusal("Choose a zip, some files, or a folder first.");
    setRefusal(null);
    const created = await onCreate({
      name: trimmed,
      files,
      build_command: build.trim() || undefined,
      test_command: test.trim() || undefined,
    });
    if (created) {
      setName("");
      setFiles([]);
      setSkipped(0);
      setBuild("");
      setTest("");
    }
  };

  return (
    <form className="new-project" onSubmit={(e) => void submit(e)}>
      <input
        value={name}
        maxLength={200}
        onChange={(e) => setName(e.target.value)}
        placeholder="Project name"
        aria-label="Project name"
      />

      <div className="pickers">
        <label className="pick">
          <input type="file" multiple onChange={pick(false)} />
          <span>files or .zip…</span>
        </label>
        <label className="pick">
          <input
            type="file"
            multiple
            onChange={pick(true)}
            ref={(el) => {
              // React's attribute types predate this one, but the DOM property
              // has been there since every browser that renders this viewer.
              if (el) el.webkitdirectory = true;
            }}
          />
          <span>folder…</span>
        </label>
      </div>
      {/* What the next request will carry, and what it will not: a skipped
          count is shown rather than dropped, so a folder that arrives lighter
          than it left is not a mystery. */}
      <span
        className={files.length === 0 ? "manifest empty" : "manifest"}
        title={skipped > 0 ? `skipped: ${Array.from(NOT_SOURCE).join(", ")}` : undefined}
      >
        {files.length === 0
          ? "nothing chosen"
          : isArchive(files)
            ? `1 archive · ${size(total)}`
            : `${files.length} file${files.length === 1 ? "" : "s"} · ${size(total)}`}
        {skipped > 0 && ` · ${skipped} skipped`}
      </span>

      {/* These two fields were read as "what should the agents do": someone
          typed a sentence into the test command, it was accepted, and a run
          was spent before validation reported that no program by that name
          exists. The heading, the monospace placeholders and the sentence
          about the objective are all there to make the mistake hard. */}
      <p className="field-head">
        Shell commands the project runs on <em>itself</em> to check a change.
        Detected from the files when left empty. What the agents should do is
        asked when you start a run, not here.
      </p>
      <input
        className="command"
        value={build}
        onChange={(e) => setBuild(e.target.value)}
        placeholder="build, e.g. make"
        aria-label="Build command, a shell command"
        spellCheck={false}
      />
      <input
        className="command"
        value={test}
        onChange={(e) => setTest(e.target.value)}
        placeholder="test, e.g. pytest -q"
        aria-label="Test command, a shell command"
        spellCheck={false}
      />

      {refusal && <small className="warn-inline">{refusal}</small>}

      <div className="row">
        <button type="submit" disabled={busy}>
          {busy ? "uploading…" : "Create project"}
        </button>
      </div>
    </form>
  );
}
