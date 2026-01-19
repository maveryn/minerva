# Detection Dataset Builders

Scripts in this folder build technique-ID prediction datasets from multiple detection-rule sources.
Outputs are JSONL files with Minerva-style fields (`task`, `input`, `ground_truth`, `reward_fn`, etc.).

## Sources
- Atomic Red Team (ART): `redcanaryco/atomic-red-team`
- Microsoft Sentinel analytics rules: `Azure/Azure-Sentinel`
- Splunk Security Content: `splunk/security_content`
- Elastic detection rules: `elastic/detection-rules`

## Quickstart
```
python detection/build_all.py \
  --output-dir dataset/detection \
  --cache-dir dataset/detection_cache
```

## Optional hard-negative MCQ
Enable candidate lists with distractors (selected by token overlap in technique names):
```
python detection/build_all.py \
  --output-dir dataset/detection \
  --cache-dir dataset/detection_cache \
  --with-distractors \
  --include-names \
  --distractor-count 6
```

## Individual builders
```
python detection/build_art.py --output-dir dataset/detection --cache-dir dataset/detection_cache
python detection/build_sentinel.py --output-dir dataset/detection --cache-dir dataset/detection_cache
python detection/build_splunk.py --output-dir dataset/detection --cache-dir dataset/detection_cache
python detection/build_elastic.py --output-dir dataset/detection --cache-dir dataset/detection_cache
```

## Notes
- Technique IDs and technique names from the ground-truth list are stripped from the snippets to reduce extraction.
- Builders currently emit single-label technique rows; multi-technique rules are skipped.
