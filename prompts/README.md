# prompts/

> **Bucket A seam (12-factor-agents F2 — "own your prompts").** Prompts are versioned artifacts,
> not inline string literals. Enforced by the `no-inline-prompt-literals` opengrep rule
> (`.opengrep/rules/prompts-as-code.yaml`).

## Convention

- One file per prompt (or prompt family): `prompts/<name>.md` or `prompts/<name>.j2` if templated.
- Code loads prompts from this directory at runtime — never embeds them as inline string literals
  in application code.
- Prompt changes are diffable, reviewable, and versioned the same way code is — this is what
  makes prompts auditable and lets the ADLC agent-change gate (`.github/workflows/adlc-gate.yaml`)
  trigger eval + security-review when they change.

## Why

Inline prompt literals hide in diffs, can't be independently reviewed by a domain expert, and
make it hard to answer "what prompt produced this behavior" during an audit. Treating prompts as
code (own them, version them, diff them) is one of the build/rent calls 12-factor-agents makes
explicit (F2/F3/F5/F8/F12 = BUILD, never delegated to the rented runtime).
