# Security Policy

## Supported versions

Only the latest commit on `main` (and the latest released tag, when tags exist)
is supported. Fixes ship forward; there are no backported patch branches for
older `0.x` cuts.

## Reporting a vulnerability

Please report suspected vulnerabilities **privately**, not in a public issue
or pull request.

- Preferred: open a [GitHub private security advisory](https://github.com/km2411/qortia/security/advisories/new)
  ("Report a vulnerability").
- Alternatively, email the maintainer at the address on the git commit history
  (GitHub noreply).

Include enough to reproduce: version/commit, whether multi-tenant RLS is
enabled, and impact. We'll acknowledge, work a fix and coordinated disclosure,
and credit you unless you prefer anonymity.

## Scope

In scope for Qortia:

- Cross-tenant or cross-agent memory leakage
- RLS / auth bypass on the HTTP admin or memory APIs
- Injection via recall/remember payloads that escalates privilege
- Secret or API-key leakage through logs or error responses

Out of scope: vulnerabilities in Postgres, LiteLLM, or embedding providers —
report those upstream; tell us if Qortia's defaults make the exposure worse.
