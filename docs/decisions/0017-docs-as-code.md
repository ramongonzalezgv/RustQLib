# 0017. Documentation as part of done
Status: Accepted (2026-10-17, PR #53)

## Context
Design reasoning, benchmark numbers and accepted deviations lived only in PR descriptions; PR #30 noted "there is no changelog file in the repo". Model maths and limits had no home outside rustdoc and issues.

## Decision
`AGENTS.md` gains a Documentation section; docs ship in the same PR as the code:
- Decision records in `docs/decisions/NNNN-slug.md` for every design choice. Accepted records are never edited; a new record supersedes them (`Superseded by NNNN`).
- Model and engine pages (`docs/models/*.md`, `docs/engines/*.md`) updated with any change to maths, discretisation, defaults or validation.
- `CHANGELOG.md` (Keep a Changelog): one line per user-visible change under `Unreleased`, with the issue/PR number.
- Public items get a one-line rustdoc; formulas live in the model page. CI runs `cargo doc --workspace --no-deps` with `RUSTDOCFLAGS=-D warnings`.
- Records 0001–0016 are backfilled from merged PRs #22–#30, #47.

## Alternatives considered
- PR descriptions as the record (status quo): not versioned with the code.

## Consequences
- Reviewers treat missing or stale docs as a finding.
- Broken intra-doc links fail CI.
