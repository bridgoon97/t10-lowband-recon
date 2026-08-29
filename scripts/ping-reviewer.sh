#!/usr/bin/env bash
# ping-reviewer.sh — send a markdown report to the remote reviewer via hapi.
#
# No secrets in this file (the repo is PUBLIC).  Hub + token + peer id all
# live OUT of the repo in ~/.config/t10-lowband-recon/reviewer.env (chmod 600),
# sourced here.  The reviewer's session id CHANGES — update REVIEWER_PEER in
# that env file (no git commit needed).
#
# Usage:
#   bash scripts/ping-reviewer.sh /tmp/report.md            # pre-check + send
#   PING_REVIEWER_SKIP_CHECK=1 bash scripts/ping-reviewer.sh /tmp/report.md   # skip pre-check (self-test)
#
# PRE-CHECK (default, on every official report): run `hapi ping-peer --list`
# and verify the target peer is still active=true.  If it's gone or
# active=false, ABORT (exit 2) and print "channel anomaly" — do NOT send.
#
# To set up the env file once:
#   mkdir -p ~/.config/t10-lowband-recon
#   printf 'HAPI_API_URL=http://bridgoon.nat100.top\nCLI_API_TOKEN=<token>\nREVIEWER_PEER=<prefix>\n' \
#     > ~/.config/t10-lowband-recon/reviewer.env && chmod 600 ~/.config/.../reviewer.env
set -euo pipefail

MSG="${1:?usage: ping-reviewer.sh <message-file>}"
[ -f "$MSG" ] || { echo "not a file: $MSG" >&2; exit 1; }

ENV_FILE="${REVIEWER_ENV:-$HOME/.config/t10-lowband-recon/reviewer.env}"
[ -f "$ENV_FILE" ] || {
  echo "missing $ENV_FILE — create it with HAPI_API_URL + CLI_API_TOKEN + REVIEWER_PEER (chmod 600)" >&2
  exit 1
}
# export these three from the env file, scoped to this subshell + the hapi call
# (does NOT clobber the agent's own ~/.hapi or ~/.bashrc).
set -a; . "$ENV_FILE"; set +a

: "${HAPI_API_URL:?HAPI_API_URL not set in $ENV_FILE}"
: "${CLI_API_TOKEN:?CLI_API_TOKEN not set in $ENV_FILE}"
PEER="${REVIEWER_PEER:?REVIEWER_PEER not set in $ENV_FILE (session id changes — update it there)}"

# --- pre-check: is the target still active=true on the hub? ---
if [ "${PING_REVIEWER_SKIP_CHECK:-0}" != "1" ]; then
  list_out=$(hapi ping-peer --list 2>&1) || {
    echo "channel anomaly: 'hapi ping-peer --list' failed — hub $HAPI_API_URL reachable?" >&2
    printf '%s\n' "$list_out" >&2
    exit 2
  }
  peer_line=$(printf '%s\n' "$list_out" | grep -F "$PEER" | head -1) || peer_line=""
  if [ -z "$peer_line" ]; then
    echo "channel anomaly: target peer '$PEER' NOT FOUND in --list (session id changed? update REVIEWER_PEER in $ENV_FILE) — paused, nothing sent." >&2
    exit 2
  fi
  case "$peer_line" in
    *active=true*) : ;;   # OK, proceed
    *)
      echo "channel anomaly: target peer '$PEER' is NOT active=true — paused, nothing sent:" >&2
      printf '  %s\n' "$peer_line" >&2
      exit 2 ;;
  esac
fi

# --- send (always via --message-file; never stdin/heredoc) ---
exec hapi ping-peer "$PEER" --message-file "$MSG"
