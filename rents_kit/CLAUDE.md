# Rents kit - instructions for Claude Code

Microburbs Hackathon Task 4 ("What will this flat rent for?"), Sat 7 Nov 2026. Windows, PowerShell, Python.
Specs live in spec/ (PRD, requirements, design, tasks); requirements.md wins on conflicts. README.md has the runbook.

Hard rules
- Nothing dated after 2026-07-31 may reach the model. Every source goes through guard.py.
- Never read files whose name contains "frozen".
- Never read, edit, print, upload or commit ..\data, ..\cache, .env or the API key.
- Never call the API's listings/for-rent or undated endpoints; api_client.py blocks them - do not loosen the block list.
- Change model settings in config.py rather than code where possible.
- After ANY code change run `python run.py test --data ..\data_synth` and show the summary line; commit only if it says all passed.
- No code changes after 1:45 pm; only `run.py final` and `run.py report`. Submissions close 3:00 pm.
- When suggesting changes, cite the requirement id (e.g. T5.1).
