---
id: 2026-11-05-structured-messages-01
kind: build
---

## Session 2026-11-05

### Prompts
- Asked to restrict message-board posts to the form "I raised / I lowered" as one setting, with free messaging as the other value, and to plan it first. Chose forward-looking intent ("I will raise / lower / hold") with a `hold` token.
- "commit this and write a devlog note. Then let's submit the smoke to test it out."

### Implemented
Added `comms.message_format` (`free` default, `structured`). In structured mode agents reply `MESSAGE: raise|lower|hold`, the board only ever holds "I will raise/lower/hold", and off-vocabulary replies are dropped and counted as `invalid_messages`. The default is left out of the condition hash so existing runs keep their hashes. Prompt template branches on the setting; new smoke config `experiments/experiment-configs/sum_clique_structured_smoke.yaml`; tests in `tests/test_message_format.py`. The analysis metric for stated intent vs. actual action is not yet written.

### Commits
10cc171 feat: comms.message_format=structured restricts board posts to raise/lower/hold
