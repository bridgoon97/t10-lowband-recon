#!/usr/bin/env bash
# ping-reviewer.sh — send a markdown report to the remote reviewer via hapi.
#
# No secrets in this file (the repo is PUBLIC).  Hub + token live OUT of the
# repo in ~/.config/t10-lowband-recon/reviewer.env (chmod 600), created once.
#
# Usage:
#   bash scripts/ping-reviewer.sh /tmp/report.md
#   REVIEWER_PEER=<short-id> bash scripts/ping-reviewer.sh /tmp/report.md
#
# To set up the env file once:
#   mkdir -p ~/.config/t10-lowband-recon
#   printf 'HAPI_API_URL=http://bridgoon.nat100.top\nCLI_API_TOKEN=<token>\n' \
#     > ~/.config/t10-lowband-recon/reviewer.env && chmod 600 ~/.config/.../reviewer.env
set -euo pipefail

MSG="${1:?usage: ping-reviewer.sh <message-file>}"
[ -f "$MSG" ] || { echo "not a file: $MSG" >&2; exit 1; }

ENV_FILE="${REVIEWER_ENV:-$HOME/.config/t10-lowband-recon/reviewer.env}"
[ -f "$ENV_FILE" ] || {
  echo "missing $ENV_FILE — create it with HAPI_API_URL + CLI_API_TOKEN (chmod 600)" >&2
  exit 1
}
# export vars from the env file WITHOUT clobbering the agent's own hapi session:
# only these two are set, scoped to this subshell + the one hapi call below.
set -a; . "$ENV_FILE"; set +a

PEER="${REVIEWER_PEER:-f9f50934}"   # short id resolves to full UUID
exec hapi ping-peer "$PEER" --message-file "$MSG"
