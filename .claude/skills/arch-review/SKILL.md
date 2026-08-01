# Arch Review

> Verifies `ARCHITECTURE.md`'s claims against the actual source — the semantic-accuracy half of
> architecture documentation that `just ci-arch` can't check. `ci-arch` only confirms a section
> *exists* for every component; it says nothing about whether that section is still *true*.
> Invoke with `/arch-review`.

## What this skill does

Reads each component section in `ARCHITECTURE.md`, re-derives the same claims from the current
source, and reports every place they've drifted apart — stale function/constant names, wrong
dependency lists, described behavior the code no longer does, diagrams missing a component that
now exists. Reports findings, then asks before editing. It never rewrites `ARCHITECTURE.md`
silently.

## Steps

1. **Parse `ARCHITECTURE.md`.** For each `##`-level component section, extract every checkable
   claim: named functions/classes/constants, dependency lists, described data/control flow,
   numeric claims (line counts, limits, timeouts), and what the ASCII diagram shows connected to
   what.

2. **Re-derive each claim from source, not memory.** For a component at
   `src/{{ python_package_name }}/<name>/`:
   - Named symbols: `grep -n` the claimed name in that path — flag if absent or renamed.
   - Dependencies: check actual imports (`grep -rn "^import\|^from" <path>`) against the
     section's stated dependency list — flag both directions (claimed-but-absent,
     present-but-unclaimed).
   - Numeric claims: recompute directly (`wc -l`, a grep count) rather than trusting the
     doc's number.
   - Diagram: confirm every box in the diagram still corresponds to something real, and that
     nothing load-bearing added since is missing from it.

3. **Classify each finding.** Split into:
   - **Stale** — the doc describes something that changed (rename, removed, behavior differs).
   - **Missing** — real behavior/dependency/symbol with no mention anywhere in the section.
   - **Diagram drift** — the ASCII diagram itself is now wrong or incomplete.

4. **Report, then ask.** Present findings grouped by component, each with the specific claim,
   what the source actually shows, and the file:line evidence. Ask before editing
   `ARCHITECTURE.md`. If confirmed, make the correction and say what changed — don't touch
   sections with no findings.

5. **If a component from `src/{{ python_package_name }}/` has no section at all**, note it but
   don't fix it here — that's `just ci-arch`'s job (a missing section is a coverage gap, not a
   drift finding this skill should silently patch over).

## When to use

- Periodically, as a standing check independent of any single PR.
- After a PR that changed a documented component's internals, interface, or dependencies —
  before merge, not after.
- Before a release, as a final accuracy pass.

## When NOT to use

- As a substitute for `just ci-arch` — that gate catches *missing* sections; this skill catches
  *wrong* ones. Both are needed; neither replaces the other.
- On a component that hasn't changed since its last review — nothing to re-verify.
