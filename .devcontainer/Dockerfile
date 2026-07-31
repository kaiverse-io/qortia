FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# The base image bakes in Node/Yarn (independent of the devcontainer "node" feature)
# with an apt source at /etc/apt/sources.list.d/yarn.list. That repo's signature no
# longer verifies against any key in Yarn's current published keyring — the base
# image's own keyring is frozen as of its last publish (2025-07-10) and predates
# Yarn's 2026-01 key rotation. Because this is baked into the image, it's already
# present before *any* devcontainer feature's RUN layer executes, so the very first
# feature that runs `apt-get update` (e.g. docker-outside-of-docker, if a project
# opts in) fails with "NO_PUBKEY 62D54FD4003F6525" — regardless of feature order or
# any feature's own options. Corepack/npm install Yarn fine without apt, and no
# project's post-create.sh shells out to the apt-installed yarn binary, so dropping
# the broken source outright has no downside. See .agents/memory/
# devcontainer-yarn-apt-key-drift.md.
RUN rm -f /etc/apt/sources.list.d/yarn.list
