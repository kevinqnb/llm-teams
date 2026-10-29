---
name: experimentlog
description: End of an experiment session — write devlog/experiment/<id>.md. Maps to /experimentlog. Run once a submitted job has finished.
argument-hint: "[id]"
disable-model-invocation: true
---

# /experimentlog: record an experiment session

Argument: an id (`YYYY-MM-DD-slug-NN`), or blank. Run at the end of an experiment
session, or once a submitted job has finished.

This writes exactly one public, committed file — `devlog/experiment/<id>.md` — and,
if needed, commits the experiment config. It has no dependency on any private notes
system: run it in any clone of this repo. It does not record results or an
assessment of them — only that the run happened and why. Pulling in finished results
is a separate step this skill does not cover.

## 1. Resolve the id

If no id was given, propose one: today's date + a short lowercase slug, `NN=01`
unless `devlog/experiment/` already has a file with that date+slug. The id must
match the committed config's `id` field
(`configs/<id>.yaml`, per the harness contract). Confirm the id with the user before writing anything.

## 2. Commit the config, if it isn't already

A run isn't reproducible until its config is committed. Check whether
`configs/<id>.yaml` (and any runner code
this session touched) is committed. If not, show the diff and a proposed commit
message, and only commit on explicit go-ahead. Push (will prompt). Take the
resulting short sha(s).

## 3. Write or append `devlog/experiment/<id>.md`

Create the file if it doesn't exist, or append a new `## Session <date>` block if it
does — never rewrite an earlier session's block.

```markdown
---
id: <id>
config: configs/<id>.yaml
---

## Session <today's date>

### Prompts
<!-- the human's experiment requests this session, verbatim or lightly trimmed.
     Drop typo-fixes and bare approvals. -->

### Hypothesis
<!-- the human's general-form hypothesis for this run, kept verbatim — do not
     reword, sharpen, or formalize it. -->

### What was run
<!-- the config(s) used, entry point invoked (scripts/run_experiment.py),
     and the run directory, repo-relative if it lives in the repo -->

### Commits
<!-- one line per commit from step 2: `<short-sha> <subject>` -->
```

Public file, same bar as a commit message: no secrets, no absolute cluster/machine
paths (repo-relative only), no session identifiers from any local tooling, no result
specifics (this skill runs before or independent of results being in). If a prompt
contained something that shouldn't be public, paraphrase it and say so plainly
rather than pasting it verbatim.

## 4. Commit the explog file on its own

Stage `devlog/experiment/<id>.md` alone:

```bash
git commit -m "explog: <id> session <date>"
```

with this project's standard attribution trailers, then push. This is a trailing
commit — every hash it lists already exists and it does not list itself.

Report the explog commit hash.

## Scope note

This skill only ever touches this repo, and it never records results, an
assessment, or a session id. It does not read or write any external notes system.
If the person running it also keeps a personal, private experiment log elsewhere,
that's a separate step they take on their own — this skill's output is complete
without it.
