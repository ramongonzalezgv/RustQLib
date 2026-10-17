# 0019. Tiered agent pipeline with blind spec tests
Status: Accepted (2026-10-17)

## Context
[0015](0015-agents-process.md) requires separate write, review and test agents for non-trivial issues, but not what each sees. Bugs still reached later sessions: tests written with the code checked what it does, not what it should do. Running the full process on every issue costs too many tokens.

## Decision
`/ship-issue N` (`.claude/skills/ship-issue`) classifies each issue:
- Trivial: main session only, plus `scripts/gates.sh`.
- Normal: `implementer` then `reviewer`, both Sonnet.
- Complex (new engine, model, product, calibration or maths): a committed spec in `.claude/specs/`, then `spec-tester` writes failing tests from the spec without seeing the implementation, using independent oracles; `implementer` may not edit them; `reviewer` (Opus) reads the diff cold. At most 2 fix rounds.

Agents share only the spec file and reply in fixed short formats. Gates run as a script that prints failures only.

## Alternatives considered
- Full pipeline for every issue: most issues are local fixes; cost several times higher for no gain.
- Tests written after the code by a separate agent: the tester anchors on the implementation's output.

## Consequences
- Specs are versioned and feed decision records and model pages.
- A wrong spec fails the pipeline loudly after 2 rounds instead of shipping.
