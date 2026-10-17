---
name: ship-issue
description: Implement a GitHub issue end to end with the tiered write / test / review pipeline. Use for "/ship-issue N", "implement issue N", "fix #N".
---

# Ship issue $ARGUMENTS

Read the issue (`gh issue view N`) and `AGENTS.md`. Branch off `main` (`fix/N-slug` or `feat/N-slug`).

## 1. Pick the tier

Choose the cheapest tier that fits. The full pipeline costs several times the others; use it only for complex issues.

- **Trivial**: typos, renames, docs-only, dead-code removal, a default value, a one-line bug with an obvious fix. No subagents: do it yourself, run `scripts/gates.sh`.
- **Normal**: a localised bug fix or small feature in existing code, no new maths. `implementer` then `reviewer`, both on Sonnet. No spec file; pass the issue body as the spec. The implementer adds the regression test itself.
- **Complex**: a new engine, model, product or calibration; any change to pricing maths, discretisation or numerical scheme; anything spanning several crates. Full pipeline below.

State the tier and a one-line reason before starting. If a normal issue turns out to need new maths, stop and escalate it to complex.

## 2. Spec (complex only)

Write `.claude/specs/N-slug.md` and commit it (`docs: spec for #N`). It is the only context the subagents share, so it must stand alone:

```
# #N <title>
## Maths
Formulas with paper and equation number. Conventions: vol or variance, rates continuous, day count.
## API
Public signatures added or changed.
## Validation
Inputs rejected at construction, and the error returned.
## Acceptance criteria
A1. <observable behaviour> — oracle: <closed form | parity | limit | reference value | MC within k·se>
## Docs
Decision record / model page / CHANGELOG lines required.
```

## 3. Run the pipeline

Pass agents the spec path, never its contents or your reasoning.

Complex:
1. `spec-tester` with the spec path. Note the test commit sha.
2. `implementer` with the spec path and test sha. Override `model: opus` only when the issue introduces new maths (a new model or scheme); otherwise keep its Sonnet default.
3. If the implementer disputes a test, you decide against the spec: fix the spec and re-run `spec-tester`, or tell the implementer the test stands.
4. `reviewer` with the spec path, base `main` and test sha.
5. Blockers or majors: send the findings to the same implementer via SendMessage (it keeps its context), then review again. At most 2 rounds; after that, stop and report to the user, since the spec is likely wrong.

Normal: steps 2, 4, 5 with `model: sonnet` on the reviewer and no test sha.

## 4. Finish

Run `scripts/gates.sh` once yourself. Open a PR linking the issue, listing the tier, the acceptance criteria and which test covers each. Report to the user in three lines: tier, PR link, findings fixed or disputed.
