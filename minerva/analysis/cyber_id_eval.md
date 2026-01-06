# Cybersecurity Identifier Association Evaluation

This section describes the construction of an evaluation dataset that tests whether a model can map natural‑language cues to canonical cybersecurity identifiers. The task spans five identifier families: MITRE ATT&CK techniques and sub‑techniques, MITRE ATT&CK mitigations, CAPEC attack patterns, and CWE weaknesses. For each family, we create two prompt variants: (i) description‑based and (ii) title‑based, enabling separate evaluation of semantic understanding versus name‑level recall.

## Data sources

We rely on the same public sources used in the Minerva pipeline:

- MITRE ATT&CK Enterprise STIX bundle (techniques, sub‑techniques, and mitigations).
- CAPEC STIX bundle.
- CWE XML catalog.

Each record yields a canonical ID, a title (name), and an official description. For ATT&CK, we treat `attack‑pattern` objects as techniques or sub‑techniques (based on the sub‑technique flag or a dotted ID), and `course‑of‑action` objects as mitigations. For CAPEC, we use `attack‑pattern` objects. For CWE, we use the `Weakness` entries and their `Description` tags. All inputs are sourced from official bundles without additional external enrichment.

## Prompt construction

For every eligible record, we generate up to two questions:

1. **Description‑based prompt**: The model is given a sanitized official description and asked to return exactly one ID in the correct format.
2. **Title‑based prompt**: The model is given only the title (name) and asked to return exactly one ID in the correct format.

The templates are explicit about the expected ID type and format (e.g., `T####`, `T####.###`, `M####`, `CAPEC-<number>`, `CWE-<number>`) and require the output to contain only the ID with no extra text. Description‑based prompts include a "Description:" field, while title‑based prompts include a "Title:" field.

## Sanitization and ambiguity filtering

We apply strict sanitization to prevent leakage of the target label:

- Remove any identifier strings that match ATT&CK, CAPEC, or CWE ID patterns.
- Remove URLs.
- Collapse whitespace.

We also drop records with empty content after sanitization or missing required fields. To avoid ambiguous supervision, we remove any prompt whose sanitized text (title or description) is shared by more than one canonical ID within the same variant set. Ambiguity filtering is applied independently for title‑based and description‑based prompts, which can lead to different counts across the two sets.

## Output format

Each example is written as a single JSONL row with:

- `uid`: unique identifier for the example.
- `id_type`: one of {`ATTACK_TECHNIQUE`, `ATTACK_SUBTECHNIQUE`, `ATTACK_MITIGATION`, `CAPEC`, `CWE`}.
- `category`: a human‑readable label that distinguishes title vs. description (e.g., "CWE ID (Title)").
- `prompt`: the task prompt.
- `answer`: the gold ID.
- `evidence`: the sanitized description or title used to construct the prompt.
- `source_url`: provenance string (bundle URL or file path).
- `notes`: optional.

This structure supports both standard classification accuracy and retrieval‑style evaluation, while preserving strict label hygiene in the prompt and evidence fields.

## Rationale

The description‑based setting targets semantic understanding of cybersecurity concepts, while the title‑based setting isolates canonical name recall. By enforcing strict output‑only constraints and filtering ambiguous text, the dataset provides clean, automatable evaluation of identifier associations that are common in CTI workflows and knowledge‑base linking tasks.
