#!/usr/bin/env bash
# post-create.sh — runs once after the devcontainer is created.
# Installs the full toolchain defined in the chassis spec (Layer 1).
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

# ── Fix root-owned named-volume mounts ────────────────────────────────────────
# Docker creates a brand-new named volume's mount point as root:root before this
# script's remoteUser (vscode) ever runs, which breaks every cache-writing step
# below (uv sync, pip install --user, pre-commit install) with PermissionError.
# ~/.cache is purely container-local (not one of the host bind mounts in
# devcontainer.json), so a full chown is safe.
sudo chown -R vscode:vscode "$HOME/.cache" 2>/dev/null || true

# ── Docker-outside-of-docker socket permission fix (no-op unless enabled) ─────
# If the project was stamped with enable_docker_outside_of_docker (default: true),
# devcontainer.json carries the docker-outside-of-docker feature + docker.sock
# mount, and the socket's host-side ownership/GID frequently does
# not match this container's pre-baked `docker` group (root:root inside a Docker
# Desktop/OrbStack VM, or behind a socket proxy, is common) — the feature adds
# vscode to a `docker` group at a GID that may not be the one the runtime-mounted
# socket actually has, since the mount only attaches after the feature installs.
# Group-matching across hosts is fragile; chmod is not. Guarded on the socket
# existing, so this is a safe no-op for every project that hasn't opted in.
if [ -S /var/run/docker.sock ]; then
  sudo chmod 666 /var/run/docker.sock 2>/dev/null \
    || echo "[warn] could not chmod /var/run/docker.sock — docker-outside-of-docker may not work this session"
fi

# ── uv (Python package/venv manager) ─────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  echo "→ Installing uv …"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ── just (task runner) ────────────────────────────────────────────────────────
if ! command -v just >/dev/null 2>&1; then
  echo "→ Installing just …"
  mkdir -p "$HOME/.local/bin"
  curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
    | bash -s -- --to "$HOME/.local/bin"
fi

# ── pre-commit ────────────────────────────────────────────────────────────────
if ! command -v pre-commit >/dev/null 2>&1; then
  echo "→ Installing pre-commit …"
  pip install pre-commit --quiet --user
fi

# ── gitleaks (secret scanning) ────────────────────────────────────────────────
if ! command -v gitleaks >/dev/null 2>&1; then
  echo "→ Installing gitleaks …"
  # renovate: datasource=github-releases depName=gitleaks/gitleaks extractVersion=^v(?<version>.*)$
  GITLEAKS_VERSION="8.21.2"
  ARCH=$(uname -m)
  case "$ARCH" in arm64|aarch64) GA="arm64" ;; *) GA="x64" ;; esac
  TMP=$(mktemp -d)
  curl -LsSf \
    "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_${GA}.tar.gz" \
    | tar -xz -C "$TMP" gitleaks
  mv "$TMP/gitleaks" "$HOME/.local/bin/gitleaks"
  rm -rf "$TMP"
fi

# ── opengrep (semantic lint / self-weakening rules) ───────────────────────────
if ! command -v opengrep >/dev/null 2>&1; then
  echo "→ Installing opengrep …"
  # renovate: datasource=github-releases depName=opengrep/opengrep extractVersion=^v(?<version>.*)$
  OPENGREP_VERSION="1.26.0"
  ARCH=$(uname -m)
  case "$ARCH" in arm64|aarch64) OG_ASSET="opengrep_manylinux_aarch64" ;; *) OG_ASSET="opengrep_manylinux_x86" ;; esac
  curl -LsSf \
    "https://github.com/opengrep/opengrep/releases/download/v${OPENGREP_VERSION}/${OG_ASSET}" \
    -o "$HOME/.local/bin/opengrep"
  chmod +x "$HOME/.local/bin/opengrep"
fi

# ── Claude Code CLI ────────────────────────────────────────────────────────────
# The ~/.claude and ~/.claude.json bind mounts above bring the host's config,
# memory, and account state, but not the binary itself — install it explicitly.
# The cockpit tools below (codeburn) instrument/extend this CLI.
if ! command -v claude >/dev/null 2>&1; then
  echo "→ Installing Claude Code CLI …"
  npm install -g @anthropic-ai/claude-code --silent 2>/dev/null \
    || echo "[warn] claude-code install failed (needs npm) — install manually: npm install -g @anthropic-ai/claude-code"
fi

# ── codeburn (AI usage cost/burn cockpit — bucket C) ──────────────────────────
# Local-first: reads Claude Code's own session JSONL, no OTEL required.
# Pinned — a template installing unattended in other people's containers doesn't run "latest".
# renovate: datasource=npm depName=codeburn
CODEBURN_VERSION="0.9.15"
if ! command -v codeburn >/dev/null 2>&1; then
  echo "→ Installing codeburn v${CODEBURN_VERSION} …"
  npm install -g "codeburn@${CODEBURN_VERSION}" --silent 2>/dev/null \
    || echo "[warn] codeburn install failed (needs npm) — install manually: npm install -g codeburn@${CODEBURN_VERSION}"
fi

# ── abtop (live session monitor — bucket C) ───────────────────────────────────
# Pinned to a specific release's own installer (not releases/latest/) — the installer
# fetches that version's prebuilt binary and verifies its checksum. abtop only reads
# local files and process metadata; the one exception is its optional session summaries,
# which shell out to `claude --print` (a real API call).
# renovate: datasource=github-releases depName=graykode/abtop extractVersion=^v(?<version>.*)$
ABTOP_VERSION="0.5.3"
if ! command -v abtop >/dev/null 2>&1; then
  echo "→ Installing abtop v${ABTOP_VERSION} …"
  curl --proto '=https' --tlsv1.2 -LsSf \
    "https://github.com/graykode/abtop/releases/download/v${ABTOP_VERSION}/abtop-installer.sh" \
    2>/dev/null | sh 2>/dev/null \
    || echo "[warn] abtop install failed — install manually from https://github.com/graykode/abtop/releases"
fi

# ── AI Engineer Coach (VS Code dashboard — anti-patterns, practice score) ─────
# https://github.com/microsoft/ai-engineering-coach (MIT). Opt-in (install_ai_coach in
# copier.yml, default false) and off by default for two reasons: it publishes no release,
# so this is a git clone + `npm ci` + build of upstream source (a supply-chain surface a
# template shouldn't run unattended in strangers' containers), and its harness docs cover
# GitHub Copilot — Claude Code session-log support is unconfirmed. When enabled, the clone
# is pinned to a specific commit. Chassis's own repo has no .copier-answers.yml, so the
# grep is false here and the block stays off for chassis too.
AEC_COMMIT="81d8eb2d76ef7f538f9c23ef0d950c1985e3270e"
if grep -q '^install_ai_coach: true' .copier-answers.yml 2>/dev/null \
   && command -v code >/dev/null 2>&1 \
   && ! code --list-extensions 2>/dev/null | grep -qi "ai-engineer-coach"; then
  echo "→ Building AI Engineer Coach @ ${AEC_COMMIT:0:12} (clone + npm ci + package — this takes a minute) …"
  AEC_DIR="$HOME/.local/share/ai-engineering-coach"
  if [ ! -d "$AEC_DIR" ]; then
    git clone --filter=blob:none https://github.com/microsoft/ai-engineering-coach.git "$AEC_DIR" 2>/dev/null \
      && git -C "$AEC_DIR" checkout "$AEC_COMMIT" 2>/dev/null
  fi
  if [ -d "$AEC_DIR" ]; then
    (
      cd "$AEC_DIR"
      npm ci --silent && npm run package --silent
      VSIX=$(find . -maxdepth 1 -name "*.vsix" | head -n1)
      if [ -n "$VSIX" ]; then
        code --install-extension "$VSIX"
        echo "✓ AI Engineer Coach installed — open via Cmd/Ctrl+Shift+P → 'AI Engineer Coach: Open Dashboard'"
      else
        echo "[warn] AI Engineer Coach build produced no .vsix — install manually, see https://github.com/microsoft/ai-engineering-coach"
      fi
    ) || echo "[warn] AI Engineer Coach build failed — install manually, see https://github.com/microsoft/ai-engineering-coach"
  fi
fi

# ── graphify (AI-powered knowledge graph — bucket C) ──────────────────────────
# https://github.com/Graphify-Labs/graphify. Local-first tree-sitter parsing + assistant-
# driven semantic extraction into graph.json/graph.html/GRAPH_REPORT.md. Installs as a uv
# tool + multi-assistant skill (`/graphify` in Claude Code and 15+ other assistants).
# The PyPI distribution is `graphifyy` (double-y — the project's own name); the CLI it
# installs is `graphify`. Pinned.
# renovate: datasource=pypi depName=graphifyy
GRAPHIFY_VERSION="0.9.14"
if command -v uv >/dev/null 2>&1 && ! command -v graphify >/dev/null 2>&1; then
  echo "→ Installing graphify v${GRAPHIFY_VERSION} …"
  uv tool install "graphifyy==${GRAPHIFY_VERSION}" --quiet 2>/dev/null \
    && graphify install >/dev/null 2>&1 \
    || echo "[warn] graphify install failed — install manually: uv tool install graphifyy==${GRAPHIFY_VERSION} && graphify install"
fi

# ── ctx (cross-session agent history search — bucket C) ───────────────────────
# https://github.com/ctxrs/ctx. Indexes local coding-agent session history into SQLite;
# `ctx search "…"` retrieves prior decisions/failed attempts across sessions instead of
# repeating work. Pinned to a specific release's prebuilt binary + SHA-256 verification,
# rather than piping the unpinned https://ctx.rs/install script through sh — this template
# runs unattended in other people's containers, so it fetches a known artifact and checks
# it. ctx is young and releases fast (multiple a week); bump CTX_VERSION deliberately.
# renovate: datasource=github-releases depName=ctxrs/ctx extractVersion=^v(?<version>.*)$
CTX_VERSION="0.24.0"
if ! command -v ctx >/dev/null 2>&1; then
  echo "→ Installing ctx v${CTX_VERSION} …"
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)             CTX_ASSET="ctx-linux-x64" ;;
    Linux-aarch64|Linux-arm64) CTX_ASSET="ctx-linux-aarch64" ;;
    Darwin-arm64)             CTX_ASSET="ctx-macos-arm64" ;;
    Darwin-x86_64)            CTX_ASSET="ctx-macos-x64" ;;
    *)                        CTX_ASSET="" ;;
  esac
  if [ -n "$CTX_ASSET" ]; then
    CTX_TMP=$(mktemp -d)
    CTX_BASE="https://github.com/ctxrs/ctx/releases/download/v${CTX_VERSION}"
    CTX_OK=""
    if curl --proto '=https' --tlsv1.2 -fsSL "$CTX_BASE/$CTX_ASSET" -o "$CTX_TMP/ctx" \
       && curl --proto '=https' --tlsv1.2 -fsSL "$CTX_BASE/SHA256SUMS" -o "$CTX_TMP/SHA256SUMS"; then
      CTX_WANT=$(grep " ${CTX_ASSET}\$" "$CTX_TMP/SHA256SUMS" | awk '{print $1}')
      CTX_GOT=$(sha256sum "$CTX_TMP/ctx" | awk '{print $1}')
      [ -n "$CTX_WANT" ] && [ "$CTX_WANT" = "$CTX_GOT" ] && CTX_OK=1
    fi
    if [ -n "$CTX_OK" ]; then
      chmod +x "$CTX_TMP/ctx" && mv "$CTX_TMP/ctx" "$HOME/.local/bin/ctx"
    else
      echo "[warn] ctx download/checksum failed — install manually from https://github.com/ctxrs/ctx/releases"
    fi
    rm -rf "$CTX_TMP"
  else
    echo "[warn] no prebuilt ctx asset for $(uname -s)-$(uname -m) — install manually from https://github.com/ctxrs/ctx/releases"
  fi
fi
if command -v ctx >/dev/null 2>&1; then
  ctx setup >/dev/null 2>&1 || true
fi

# ── Agent memory durability ───────────────────────────────────────────────────
# Canonical memory is git-tracked in-repo (.agents/memory). The conventional Claude
# path lives on the ephemeral home overlay (the ~/.claude bind mount only persists on
# LOCAL devcontainers, not remote/cloud ones), so recreate it as a symlink each build.
# Mirrors the .claude/skills → .agents/skills pattern. Wrapped in a subshell: a
# permission failure on the host-mounted ~/.claude (UID mismatch, restrictive bind
# mount) must degrade to a warning, not abort the rest of this script (uv sync,
# pre-commit install haven't run yet at this point).
(
  MEM_CANON="$PWD/.agents/memory"
  MEM_LINK="$HOME/.claude/projects/${PWD//\//-}/memory"
  mkdir -p "$MEM_CANON"
  if [ -e "$MEM_LINK" ] && [ ! -L "$MEM_LINK" ]; then
    cp -rn "$MEM_LINK"/. "$MEM_CANON"/ 2>/dev/null || true   # rescue files written before relink
    rm -rf "$MEM_LINK"
  fi
  mkdir -p "$(dirname "$MEM_LINK")"
  ln -sfn "$MEM_CANON" "$MEM_LINK"
  echo "→ agent memory linked: $MEM_LINK → $MEM_CANON"
) || echo "[warn] agent memory symlink setup failed (often a ~/.claude bind-mount permission/UID mismatch) — .agents/memory/ is still there and git-tracked, just not auto-linked into ~/.claude/projects/.../memory this build. Safe to ignore or fix the mount permissions manually."

# ── Project setup ─────────────────────────────────────────────────────────────
if [ -f pyproject.toml ]; then
  echo "→ uv sync …"
  uv sync
fi

if [ -f .pre-commit-config.yaml ]; then
  echo "→ pre-commit install …"
  pre-commit install --install-hooks 2>/dev/null || pre-commit install
fi

echo ""
echo "✓ Devcontainer ready. Run 'just ci' to verify."
