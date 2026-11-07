# Rents kit: "What will this flat rent for?" (Microburbs Hackathon, 7 Nov 2026)

Estimates the weekly asking rent, with and without a car space, for all 65,777 City of Sydney flats.
Specs: `PRD.md`, `requirements.md` (test ids T1.1 ...), `design.md`, `tasks.md` in the project.
Tested on synthetic data only: accuracy on synthetic data proves plumbing and leak-safety, not the real score.

## Setup on Windows (PowerShell, once, tonight)

```powershell
cd rents_kit
python --version                      # needs 3.10 or newer
python -m venv .venv
.venv\Scripts\Activate.ps1            # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python make_synthetic.py --out ..\data_synth          # ~30 s
python run.py test --data ..\data_synth                # expect "39/39 passed"
```

Folders sit BESIDE the repo so private data never enters git:

```
hackathon\
  rents_kit\        <- this repo (public)
  data\             <- the three supplied files at 11:00 (never commit)
  cache\api\        <- API pulls (never commit)
  data_synth\       <- rehearsal data
```

## Rehearsal (tonight, ~10 minutes)

```powershell
python api_pull.py leases --mock ..\data_synth\mock_api --cache ..\cache_synth
python run.py inspect   --data ..\data_synth
python run.py calibrate --data ..\data_synth
python run.py eval      --data ..\data_synth                                   # supplied only (~1 min)
python run.py eval      --data ..\data_synth --sources merged --cache ..\cache_synth   # ~3 min
python run.py final     --data ..\data_synth --sources merged --cache ..\cache_synth --team "Our Team"
python run.py report    --data ..\data_synth --sources merged --cache ..\cache_synth
python lookup.py "102/55 dunkerley pl" --asking 950 --flats ..\data_synth\rents-input-flats-city-of-sydney.csv
python score_private.py out\rents_our-team.csv ..\data_synth\rents-input-flats-city-of-sydney.csv ..\data_synth\private\synthetic_frozen.csv
```

## Hackathon day runbook

| Time | Command | What to look at / decide |
| --- | --- | --- |
| 11:00 | copy `.env.example` to `.env` and paste the key after `MICROBURBS_API_KEY=`; save files to `..\data` | never commit `.env` |
| 11:00 | `python api_pull.py smoke` | key works? rows show gnaf_pid, listing_date, weekly_rent? |
| 11:02 | `python api_pull.py leases` (second terminal, leave running) | "suburbs: N adverts"; 403 = outside area; stops 12:30 |
| 11:00 | `python run.py inspect --data ..\data` | columns found? date range? match methods? If a column is not found: add `--col rent="Their Name"` to every command |
| 11:20 | `python run.py calibrate --data ..\data` | PASS/GAP against the sample report ($100, 56.8%, R² 0.645, 3,954). A GAP: check date format, parking field, dedupe - note it, continue |
| 11:35 | `python api_pull.py coverage --flats ..\data\rents-input-flats-city-of-sydney.csv --leases ..\data\rents-past-leases-city-of-sydney.csv` | API fill: NOT POSSIBLE / NOT NEEDED / OPEN |
| 11:40 | `python run.py eval --data ..\data` | model vs baselines; blend weight; bias |
| 12:00 | `python run.py final --data ..\data --team "Our Team"` | **valid CSV by 12:15** (insurance) |
| 12:15 | `python run.py eval --data ..\data --sources merged` | does API history beat supplied-only? |
| 12:30 | `... --sources merged --late-api drop` and `--index-end trend` and `--no-views` | keep each only if typical miss AND one-per-listing both improve |
| 1:30 | `python run.py eval ... --confirm` (once) | confirmation window, same settings |
| **1:45** | `python run.py final --data ..\data --team "Our Team" [--sources merged] [--index-end trend] [--blend W]` | freeze |
| 1:50 | `python run.py report --data ..\data [same flags]` | read report.html end to end |
| 2:30 | copy `out\rents_<team>.csv` and `out\report.html` to the repo root; README; videos | |
| 2:45 | `git status` - no data, cache, .env | upload by 3:00 |

Settings you may change are all in `config.py` (top of each section). The `--blend W` value comes from the eval line
"blend weight": use it only if it is below 1.0.

## What we built (for the submission README - edit on the day)

- **What:** two weekly rents per City of Sydney flat (with and without a car space), plus a report and a lookup demo.
- **Who it helps:** tenants checking an asking rent, landlords setting one, investors comparing flats.
- **Clever step:** we carry each flat's own past rent forward with a City of Sydney rent index built from repeat
  adverts, then measure the parking premium within the same building.
- **Data:** Microburbs supplied files and Microburbs Hackathon API (rent history up to 31 July 2026 only).
- **How to run:** see the runbook above; data files are not included (supplied for the day only).

### Credits

Data © Microburbs (microburbs.com.au); estimates are automated and indicative only. Addresses use Open G-NAF, © Geoscape Australia, licensed by the Commonwealth
with an attribution and mailing restriction. 2021 block counts use ABS Census data, © Commonwealth of Australia,
CC BY 4.0. Add every other source you use.

## Rules the code enforces

- Nothing dated after 2026-07-31 reaches the model: one guard (`guard.py`) for every source; API requests ask for
  `to_date=2026-07-31` and are filtered again; live listings and undated API snapshots are blocked in `api_client.py`.
- Files named "frozen" are refused. The `leased` field is dropped.
- The CSV is checked against T1.1 to T1.6 before it is written (`validate_submission.py`).
