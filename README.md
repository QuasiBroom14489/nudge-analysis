# What makes a nudge useful?

[Vesper](https://zanejohnson.vercel.app/projects/vesper), my personal AI assistant, runs a local model every ten minutes that looks at my deadlines and inbox and raises "nudges". I mark each one **useful**, **wrong** or **noise** with one tap. This project asks what separates the useful ones, and then tests one change to the engine based on the answer.

**Baseline (Oct 15, 2026):** 1,725 nudges raised and 343 judged, of which 35% were useful. See [findings.md](findings.md).

- **Repetition is the problem.** A rule's 2nd or 3rd firing is useful 58% of the time, but its 11th or later firing only 15% of the time. In week 4, 101 of the 140 judged nudges were 11th-or-later repeats.
- **The model's confidence barely helps.** On held-out data, its confidence alone ranks useful nudges at AUC 0.57. A simple logistic model on timing, type and repeat count reaches 0.69.
- **Planned change: a per-rule repeat cap.** Replayed on the judged history, a cap of 10 firings per rule cuts nudge volume by 54%, keeps 82% of the useful nudges and lifts precision from 35% to 48%. That replay is in-sample hindsight. The real test is shipping the cap and comparing two weeks of new judgments.

## Privacy

The raw data is my own event log, so only derived features leave it. `scripts/export_features.py` reads the database read-only and writes one row per nudge to `data/private/` (gitignored). Each row has the day, hour and weekday, rule category, urgency, confidence, text length, evidence count, prior firings of the same rule, the verdict and the minutes until the verdict.

No nudge text, evidence, verdict notes or rule keys are exported. Everything committed here is aggregated, and any group with fewer than 5 judged nudges is dropped.

## Run it

```bash
uv run python scripts/export_features.py   # needs the Vesper database; writes data/private/
uv run python scripts/analyze.py           # writes data/public/, charts/ and findings.md
```

The analysis uses DuckDB for the group-bys, scikit-learn for the logistic model (a time-ordered 70/30 split, compared against base-rate baselines) and matplotlib for the SVG charts.

## Caveats

- Judged nudges are the ones I chose to answer, not a random sample.
- Usefulness drifted down over time, so the model ranks better than chance but doesn't beat a constant that already knows the test period's rate.
- One person, six weeks: these are findings about my assistant, not general results.
