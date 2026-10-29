# rhobear-neo: notes for AI agents

## Every agent signs its work (owner rule, 2026-10-29) — read this first

Every agent in this estate pushes, comments and opens PRs through the **owner's GitHub account**, so GitHub cannot tell
who did what. You are welcome to fix things from wherever you run (ChatGPT, Codex, Kimi, Claude, anything else), but you
must say who you are:

1. **Commits:** end every commit message with a trailer line `Agent: <who> (<model>, <where it ran>)`, for example
   `Agent: ChatGPT agent (Astra, chatgpt.com)`, `Agent: Codex (gpt-6.1-sol, build box)`, `Agent: Kimi Work (K3.1, owner PC)`.
2. **PRs and comments:** the first line says `Agent: <who>`; the last line of a comment is `— <who>`.
3. **Outside agents** (anything not dispatched from the sprint board, e.g. a ChatGPT run the owner started): work on
   your own branch `agent/<who>/<topic>` and open your own PR. **Never push to someone else's open PR branch**: a lane
   worker may be mid-round on it and your push collides with theirs. If you think an open PR needs your fix, open a
   separate PR against its branch and say so on the original PR.
4. Never merge, never force-push, never rewrite history, never post `@rhobear-reviews`. Say what you ran and what you
   did not run (tests, probes); never claim a pass you did not see.


<!-- lane-start -->
## Start here (RHOBEAR lanes, 2026-10-24)

This repo belongs to lane **L1 Reviews**. Before you change anything, read in `rhobear-north-star` (`docs/north-star/lanes/`): `README.md`, `GATE.md`, then `L1-reviews/BRIEF.md` and `L1-reviews/SOURCES.md`.

- What wins when sources disagree: the owner's newest words, then the final iteration (the newest board and the newest
  code), then the docs, this file included.
- No worker starts here until the lane architect's gate is signed and a worker brief names your files.
- Old UX never comes back: a screen, style, font, colour or mark that is not on the current board is deleted, not
  restyled. Fonts: `docs/north-star/FONTS.md` in `rhobear-north-star` (Nacelle is banned).
- Workers run only on Core, Peak, Summit or Guest. No keys in prompts, files or commits.
<!-- /lane-start -->

