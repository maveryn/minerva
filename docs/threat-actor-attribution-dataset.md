# Threat Actor Attribution Dataset (High-Level)

This note summarizes how the threat actor attribution subset of Minerva was created.

## Source data
- Start from MITRE ATT&CK Enterprise intrusion-set entries (actors).
- For each actor, collect its "Techniques Used" relationships to ATT&CK techniques.
- From those techniques, extract the associated procedure descriptions that describe how the actor operates.

## Sampling and balancing
- Build a per-actor pool of procedure descriptions.
- Sample the number of examples per actor in proportion to the amount of available procedure text so actors with
  more coverage contribute more samples.
- Apply lightweight balancing (binning by technique count and minimum coverage per actor) to avoid a few large
  actors dominating the dataset.

## Anonymization
- Before constructing prompts, replace the actor name with a generic placeholder ("A threat actor").
- Generalize other entity mentions by type (campaign -> "A campaign", malware/tool -> "A malware") while preserving
  leading articles.
- Keep the true actor name only as the label; aliases are recorded in `threat_actor_lookup.json` for reward scoring.

## Output
- Input: list of anonymized procedures.
- Output: threat actor name (canonical or alias).
