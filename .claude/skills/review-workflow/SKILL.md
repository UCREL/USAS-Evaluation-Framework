---
name: review-workflow
description: Review, audit or change a GitHub Actions workflow in .github/workflows/ for correctness and security, using actionlint, zizmor and a security and project checklist, then report findings and explain every change made. Use when asked to review, audit, lint, harden or edit a workflow file.
---

# Review a GitHub workflow

Tool background (why the tools run through `uvx`, token safety, upstream actionlint vs the kjanat fork) lives in the README section "Reviewing GitHub workflows" ([README.md](../../../README.md)). Read that section first. The README lists only the checks the tools can't do. The full checklist Claude uses is below.

## Checklist

[.github/workflows/docs.yaml](../../../.github/workflows/docs.yaml) follows every item, so use it as a reference. When reporting a finding, explain the risk using the "why" given here.

Covered by zizmor, but check them anyway so findings can be explained and false positives judged:

- **Dangerous triggers** (zizmor `dangerous-triggers`): avoid `pull_request_target` and `workflow_run`. They run with write access and secrets, so checking out pull request code in them lets a fork run code with those privileges.
- **Permissions** (`excessive-permissions`): a top-level `permissions: {}`, with each job granting only what it needs. A compromised step can only do what the token allows.
- **Action pinning** (`unpinned-uses`; `ref-version-mismatch` and `impostor-commit` in online mode): every `uses:` is pinned to a full commit SHA with a `# vX.Y.Z` comment that matches it. A tag can be moved to point at malicious code, but a SHA can't. Check without a token: `git ls-remote --tags https://github.com/OWNER/REPO vX.Y.Z`; for annotated tags, take the line ending in `^{}`.
- **Credentials** (`artipacked`, `secrets-inherit`): `actions/checkout` uses `persist-credentials: false`, so the token isn't left in `.git/config` for later steps or uploaded artifacts. Secrets are only available through an `environment` on the job that needs them.
- **Injection** (`template-injection`): no attacker-controllable values, such as `${{ github.event.pull_request.title }}` or `${{ github.head_ref }}`, written directly into `run:`, because they are pasted into the script before the shell runs it. Pass them through `env:` and use the shell variable.
- **Caching** (`cache-poisoning`): only `main` saves caches, so a pull request can't plant a poisoned cache that `main` later restores.

Not checked by any tool (these are also in the README):

- **Path filters**: `paths` lists every file the job reads, including the workflow file itself. A missing path means a change that breaks the build doesn't trigger a run.
- **Deploy gating**: deploy jobs only run on `main` and never for `pull_request`, and `concurrency` never cancels a deploy part way, which could leave Pages half deployed.
- **Version sync**: tool versions are pinned and match, for example uv in the workflow and in [.devcontainer/Dockerfile](../../../.devcontainer/Dockerfile), and dependencies are installed with `uv sync --locked`, so CI uses exactly the locked versions.
- **Local parity**: steps call `make` targets, so CI builds the same way as a developer does locally and flags live in one place.
- **Limits**: every job has `timeout-minutes`, so a hung job doesn't run for the 6-hour default. Uploads have a short `retention-days` and never include `.git` or secrets.

## Steps

1. **Run the automated checks** with `make lint-workflows`. Run it without a token, so actionlint works offline and zizmor runs only its offline audits. Only use zizmor's online mode if the user asks for it and sets `GH_TOKEN` themselves. Never run `gh auth token`, and never print, log or write a token anywhere.
2. **Walk the checklist above** for each workflow in scope, and cite every finding as `file:line`.
3. **Check the tools are current.** Compare the `*_VERSION` variables in the [makefile](../../../makefile) with the latest releases of zizmor, actionlint-py and shellcheck-py. Apply the README's rule on upstream actionlint vs the kjanat fork: if upstream rhysd/actionlint has had no release for 12 months, or looks unmaintained, tell the user and suggest re-evaluating the fork. Don't switch on your own.
4. **Report findings before changing anything.** Group them by severity (high, medium, low, informational). Keep tool findings, with their rule name (for example zizmor `artipacked` or actionlint `expression`), separate from manual checklist findings. For each, say what the risk is in plain words and what the fix would be. Then ask the user which findings to fix.
5. **Fix only what the user agreed to**, following these rules:
   - Never write or change a pinned commit SHA without looking it up. Use `git ls-remote --tags https://github.com/OWNER/REPO vX.Y.Z`; for annotated tags, take the SHA from the line ending in `^{}`. Keep the `# vX.Y.Z` comment in step with the SHA.
   - Don't apply zizmor's `--fix` output blindly. Read each suggested change and apply it by hand.
   - Only add a `# zizmor: ignore[<rule>]` comment when the user agrees the finding is a deliberate choice, and add a comment explaining why.
   - Keep the project's conventions, such as top-level `permissions: {}`, steps that call `make` targets, and the uv version matching [.devcontainer/Dockerfile](../../../.devcontainer/Dockerfile).
6. **Re-run `make lint-workflows`** after the fixes and confirm it's clean, or say what remains.
7. **Tell the user what changed and why.** End with a change summary, with one entry per change:
   - **Where**: `file:line`, and the job and step name.
   - **What**: before → after, for example "`actions/checkout` SHA bumped from v7.0.0 to v7.0.1".
   - **Why**: the finding it fixes (the tool rule or checklist item) and the risk it removes, in plain words.
   - **How verified**: for example "SHA checked with `git ls-remote`", "`make lint-workflows` now clean".

   Also list the findings deliberately left unfixed (suppressed or deferred) and why, so nothing is changed or skipped silently.
