# Agent ↔ Reviewer Channel (hapi ping-peer)

This project is developed in a no-GPU isolation environment.  The coding agent
communicates with the remote reviewer through the **hapi** CLI's `ping-peer`
subcommand.  This file documents the method so it is reusable.

## The command

**Preferred (no inline secrets — the repo is public):**

```bash
bash scripts/ping-reviewer.sh /tmp/<report>.md
```

The hub + token + peer id ALL live OUT of the repo in
`~/.config/t10-lowband-recon/reviewer.env` (chmod 600), sourced by the
wrapper.  The reviewer's session id CHANGES — so REVIEWER_PEER is in the env
file (update there, no git commit).  Set it up once:
```bash
mkdir -p ~/.config/t10-lowband-recon
printf 'HAPI_API_URL=http://bridgoon.nat100.top\nCLI_API_TOKEN=<token>\nREVIEWER_PEER=<prefix>\n' \
  > ~/.config/t10-lowband-recon/reviewer.env && chmod 600 ~/.config/t10-lowband-recon/reviewer.env
```

**Equivalent raw form (if the wrapper is unavailable):**
```bash
HAPI_API_URL='http://bridgoon.nat100.top' \
CLI_API_TOKEN='<token>' \
hapi ping-peer <peer-prefix> --message-file /tmp/<report>.md
```

- `HAPI_API_URL` — the hapi hub the reviewer's peer is registered on.
- `CLI_API_TOKEN` — auth token for that hub (DO NOT commit to the public repo).
- `REVIEWER_PEER` / `<peer-prefix>` — the reviewer's CURRENT session id (8-char
  prefix OK; hapi resolves it to the full UUID).  Changes over time — verify
  with `--list` before each report (see pre-check below).
- `--message-file <path>` — **required**: write the message to a file and pass
  the path.  Do NOT pipe via stdin/heredoc — it does not work reliably.

⚠️ The agent's OWN hapi (`hapi auth status` → localhost) uses different
credentials — do NOT clobber `~/.hapi/settings.json` or set these env vars
globally in `~/.bashrc`; that would break the agent's own hapi sessions.
The wrapper scopes them to the one `hapi ping-peer` call.

**Pre-check protocol (run before EVERY official report):** the wrapper (by
default) runs `hapi ping-peer --list`, finds the target peer prefix, and
verifies `active=true`.  If the peer is gone or `active=false`, the wrapper
ABORTS (exit 2) printing `channel anomaly …` and sends NOTHING — then report
the anomaly in-session (do NOT retry/send blind).  Session ids die and rotate;
this check is what keeps pings from vanishing.  (Skip for self-tests:
`PING_REVIEWER_SKIP_CHECK=1 bash scripts/ping-reviewer.sh …`.)

A successful run prints (and the message is delivered):
```
hapi ping-peer: resolved <full-uuid>  active=true  name="<reviewer's current design notes>"
hapi ping-peer: sending message (N chars)...
hapi ping-peer: OK - delivered to <full-uuid>
```

## Workflow

1. **Do the work first**, then report.  Each report corresponds to a concrete
   milestone (a spec section, a rework item, a review point).  Don't ping
   half-finished work.
2. **Write the report to `/tmp/<name>.md`** — a self-contained markdown summary
   (the reviewer reads pings; they do NOT see the agent's in-session text, so
   the ping must carry the substance: what was done, the numbers, the
   caveats, the open questions).  Raw command output (git push deltas,
   measured metrics) pasted verbatim is good — it is the hard evidence.
3. **Ping it** with the command above.
4. **Commit + push to git BEFORE pinging** — the git remote is the ground
   truth the reviewer checks against.  `git ls-remote origin master` is the
   proof a push landed; include the SHA + the push delta line in the report.
5. **Report after each item, don't batch** (per the reviewer's standing
   instruction).  Exception: small loose-end cleanups can ride with the next
   report.

## Delivery is ASYNC (important)

The channel delivers every message in both directions, but with **queue lag**:
- A message the reviewer sends may land AFTER the agent has started the next
  task — the agent reads it when the current task finishes.
- A report from the agent arrives at the reviewer in order, but it responds to
  the reviewer's EARLIER message.  So a reviewer complaint ("you didn't push")
  may have been true when written but stale by the time the agent reads it.

Consequences (agreed protocol with this reviewer):
- **Do NOT resend reports** — both directions deliver; a missing ping is a
  lag artifact, not a delivery failure.  Re-sending creates noise.
- **Do NOT investigate the channel** — `OK - delivered` means delivered.
- **Judge by CODE STATE (git), not by a message's wording.**  If a reviewer
  message complains about something the code already does (stale), skip it
  with a one-liner ("this is stale — already done in `<SHA>`") and move on.
  Don't defend line-by-line.
- The agent processes roughly one reviewer message per turn; the reviewer
  batches messages to the gaps between the agent's reports to reduce lag.

## What to put in a report (self-contained, since the reviewer reads pings only)

- **The milestone** done (which spec section / review point).
- **The key numbers / raw output** (test counts, measured metrics, git push
  deltas, `git ls-remote` SHA) — verbatim where possible; these are the
  evidence the reviewer verifies.
- **Caveats / boundaries** (sample size, what the measurement does NOT cover,
  falsified predictions reported as such — honesty over looking good).
- **Open questions** for the reviewer (decisions that need them, e.g. "should
  Arm A use oracle F0 for males?").
- **The remote SHA** so the reviewer can `git pull` and verify.

## Gotchas learned this project

- `--message-file` is mandatory; heredoc/stdin does not work.
- The peer id short form (8 chars) resolves to the full UUID — use either; the
  prefix is in REVIEWER_PEER (env file, out of repo).
- The reviewer's `name` field carries their current design notes (a snippet);
  it is NOT the message content.
- `OK - delivered` is the only success signal; there is no read-receipt, so
  "did they read it?" cannot be checked — assume delivered + lag.
- When stuck or a decision is ambiguous, **ask immediately** in the ping
  rather than guessing and burning a turn.
