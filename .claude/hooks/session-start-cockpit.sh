#!/usr/bin/env bash
# SessionStart hook — refreshes the AI-usage cockpit and reminds the agent it exists.
# Whatever this script prints to stdout is added directly to the agent's context at
# session start (before the first prompt) — see chassis's docs/explanation/
# ai-usage-cockpit.md and design.md ("Coaching" / "Context Engineering" layers).
#
# Best-effort only: SessionStart hooks cannot block the session, so every step here
# is guarded to fail silently rather than surface a startup error for a bucket-C
# (on-demand, never-blocking) tool that may not even be installed in this project.
set -uo pipefail

if command -v graphify >/dev/null 2>&1; then
  graphify update . >/dev/null 2>&1 || true
fi

if command -v ctx >/dev/null 2>&1; then
  ctx setup >/dev/null 2>&1 || true
fi

cat <<'MSG'
AI-usage cockpit active for this project (see AGENTS.md "AI-usage cockpit &
coaching" for the full writeup):
- graphify: graphify-out/GRAPH_REPORT.md was just refreshed — check its God Nodes /
  communities before exploring unfamiliar code, instead of re-grepping from scratch.
- ctx: run `ctx search "<topic>"` before starting non-trivial work — it searches full
  past session transcripts, not just memory files, for prior decisions or corrections.
- /dev-coach: run periodically, or right after a corrected mistake, to turn friction
  into a durable AGENTS.md rule instead of relearning it next session.
MSG

exit 0
