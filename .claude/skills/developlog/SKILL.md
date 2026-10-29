---
name: developlog
description: End of a development session — write devlog/develop/<id>.md and commit the session's code. Maps to /developlog.
argument-hint: "[id]"
disable-model-invocation: true
---

# /developlog: record a development session

Argument: an id (`YYYY-MM-DD-slug-NN`), or blank.

This writes exactly one public, committed file — `devlog/develop/<id>.md` — and
commits the session's code. It has no dependency on any private notes system: run it
in any clone of this repo.

## 1. Resolve the id

If no id was given, propose one: today's date + a short lowercase slug describing
what was built, `NN=01` unless `devlog/develop/` already has a file with that
date+slug (then increment). Confirm the id with the user before writing anything.

## 2. Show the diff, propose the commit, get a go-ahead

Check the working tree for uncommitted implementation changes from this session.
Show the diff summary and a proposed commit message, and only commit on explicit
go-ahead — this skill commits code, which is a visible, hard-to-reverse action.
Then push (this will prompt for confirmation; that's expected). Take the resulting
short sha(s).

If there's nothing uncommitted (the session's code was already committed earlier),
skip straight to step 3 using the sha(s) from that earlier commit.

## 3. Write or append `devlog/develop/<id>.md`

Create the file if it doesn't exist, or append a new `## Session <date>` block if it
does — never rewrite an earlier session's block.

```markdown
---
id: <id>
config:            # configs/<id>.yaml if this build has one, else omit the value
---

## Session <today's date>

### Prompts
<!-- the human's requests this session, verbatim or lightly trimmed. Keep the
     substantive ones — what to build, changes of direction, what was ruled out.
     Drop typo-fixes and bare approvals. -->

### Implemented
<!-- 3-5 sentences: what changed, which entry points, which configs, which
     tests/checks passed. Pointer-length, not a transcript. -->

### Commits
<!-- one line per commit from step 2: `<short-sha> <subject>` -->
```

Public file, same bar as a commit message: no secrets, no absolute cluster/machine
paths (repo-relative only), no session identifiers from any local tooling, no
unpublished-result specifics. If a prompt contained something that shouldn't be
public, paraphrase it and say so plainly rather than pasting it verbatim.

## 4. Commit the devlog file on its own

Stage `devlog/develop/<id>.md` alone:

```bash
git commit -m "devlog: <id> session <date>"
```

with this project's standard attribution trailers, then push. This is a trailing
commit — every hash it lists already exists and it does not list itself.

Report the devlog commit hash.

## Scope note

This skill only ever touches this repo. It does not read or write any external
notes system, and it never records a session id. If the person running it also
keeps a personal, private build log elsewhere, that's a separate step they take on
their own — this skill's output is complete without it.
