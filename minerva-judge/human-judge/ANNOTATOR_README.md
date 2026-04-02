# Human Judge

This bundle is for human annotation only. It does not include hidden GPT labels
or model identities.

## Requirements

- Python 3.10 or newer

## Run

On macOS or Linux:

```bash
python3 annotator_app.py
```

Or:

```bash
./start_annotation.sh
```

On Windows:

```bat
python annotator_app.py
```

Or double-click `start_annotation.bat`.

## Use

1. Open `http://127.0.0.1:8787/` in a browser.
2. Choose `pointwise_pilot_50_correct_only`.
3. Enter your annotator ID.
4. For each item:
   - read the prompt
   - read the blind model response
   - score all 3 rubric criteria from `1` to `4`
   - click `Save Scores And Next`

The app saves progress automatically. To resume, run the app again and use the
same annotator ID and subset.

## Return File

When finished, send back the saved annotation file:

```text
annotations/<subset>/<annotator>.jsonl
```
