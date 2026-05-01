# CLAUDE.md

## Documentation

Read `docs/` for project setup, architecture, and components before making changes. Start with `docs/source/introduction.md`.

## Git

- **NEVER** use `git checkout --`, `git restore`, `reset --hard`, or `clean -f` to discard uncommitted changes. Leave files unstaged instead — do not destroy in-progress work.
- Do not use destructive git operations without explicit user approval.
- **NEVER** add `Co-Authored-By: Claude` to commit messages.
- When merging commits in a plan, clean up: renumber all commits, update dependency references, and remove stale descriptions.
- Keep commit messages **compact**: short title + minimal body. No per-file bullet points — summarize the intent, not every change.

## Quality Checks

Always run before committing:
- `uv run ruff check .` — lint (fix with `--fix` when possible)
- `uv run mypy -p p2pfl` — type checking

## Code Review Workflow

1. **Build a commit plan first.** Organize changes into logical commits. If changes need to be split across commits, identify that upfront. If a file has changes for multiple commits, consider merging those commits or editing the file to separate concerns.
2. **Big-picture review first.** Before going file by file, read ALL files in the commit. Evaluate design, code quality, and cross-file issues from three perspectives:
   - **Programmer** — code quality, correctness, edge cases, error handling.
   - **P2PFL user** — usability, API clarity, configuration defaults, documentation.
   - **Architect** — design coherence, encapsulation, scalability, cross-component impact.
   Present a compact numbered list of what's wrong — no pattern names, no verbose explanations.
3. **Review files one by one.** For each file show the full path (clickable), a short description, and the diff. Wait for user feedback before moving to the next file.
4. **Thorough review.** Double-check everything: code quality, code smells, stale references to deleted code, unused imports, dead code paths, and dangling docstrings. When removing code, trace all references (imports, settings, docstrings, comments) and clean them up.
5. **After reviewing all files in a commit**, provide the staging and commit commands (as two separate commands). Never include `Co-Authored-By`.
6. **After all commits**, verify no file appears in multiple commits and scopes don't cross.
