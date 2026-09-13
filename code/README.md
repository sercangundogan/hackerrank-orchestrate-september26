# Buy or Wait? — runnable package

Python 3.11+ standard library only. No `pip install` is required for predictions.

## Layout the evaluator should use

Place the provided `dataset/` directory next to this extracted package:

```text
<workspace>/
  dataset/          # organizer input files
  main.py           # this package (contents of code.zip)
  config.py
  ...
```

or the original repository layout:

```text
<workspace>/
  dataset/
  code/
    main.py
```

`config.py` locates `dataset/` in either case.

## Generate predictions

```bash
python3 main.py --generate-output
```

Writes `<workspace>/output.csv` (250 evaluation rows + header).
Does not modify `dataset/output.csv`.

The token-usage report for the submitted run is `evaluation/usage_report.md`.

## Image amounts and API credentials

Sixteen financial events have a blank `amount` that must be read from the
linked PNG in `dataset/media/images/`. Those amounts are never treated as zero.

The submitted `output.csv` was produced with a local validated evidence cache
(`evaluation/cache/evidence.json` in the development checkout). That cache is
not part of `code.zip`.

A cold run that does not already have that cache must set `OPENAI_API_KEY` or
`BUYORWAIT_API_KEY` so the vision model can extract the image-linked amounts.

Without those credentials, `--generate-output` fails with
`UnresolvedForecastError` rather than inventing or zeroing missing amounts.
