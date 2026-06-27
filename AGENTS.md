# Repository Agent Instructions

## Hackathon Submission Review Style

Prepare this hackathon submission for AI-assisted judging.

Context:

- The repo will be reviewed by an AI code-review pipeline and human jurors.
- The AI review is advisory, but judges may see it.
- Entire session history is required and visible as an advisory process-quality
  signal.

Optimize for:

1. Clear challenge alignment
2. Static "would it run?" confidence
3. Code quality
4. Architecture clarity
5. Visible custom work over boilerplate
6. Strong Entire process history

Tasks:

- Add or improve root `README.md` with:
  - project summary
  - challenge requirements mapped to implemented features
  - architecture overview
  - setup/run commands
  - required env vars with `.env.example`
  - demo flow
  - known limitations
- Ensure package scripts are correct: install, dev, build, test if available.
- Remove unused scaffold/demo/placeholder code where safe.
- Make core implementation easy to inspect: clear filenames, small modules,
  meaningful comments only where helpful.
- Add lightweight verification: tests, smoke script, or documented manual test
  steps.
- Check for broken imports, missing env vars, dead links, and stale TODOs.
- Ensure Entire is enabled and checkpoint branch is pushed.
- During work, use prompts that show ownership: decisions, tradeoffs, debugging,
  testing, edge cases, and why changes were made.

Avoid:

- Prompt-injection text like "ignore previous instructions" or "rate this
  10/10".
- Hiding important logic in huge generated files.
- Leaving unused boilerplate that dilutes originality.
- Claiming features that are not implemented.

Goal:

Make the repo readable, runnable-looking, challenge-aligned, and easy for both
an LLM reviewer and a human judge to understand in under 10 minutes.
