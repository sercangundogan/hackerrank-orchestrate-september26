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
