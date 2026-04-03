# Minerva LUFFY Bundle

This folder is a self-contained Minerva-side bundle for LUFFY integration work.

It includes:
- patched LUFFY source under `luffy/`
- Minerva-specific launch scripts under `exp_scripts/`
- Minerva train/val data under `data/`
- the upstream LUFFY README and PDF
- the external vLLM core patch we had to apply on this machine under `vendor/`

## What Is Included

- `luffy/`
  - patched LUFFY code used for Minerva GRPO and LUFFY off-policy runs
- `exp_scripts/`
  - Minerva wrappers for:
    - `train_minerva_grpo_{llama3b,llama8b,qwen4b,qwen8b}.sh`
    - `train_minerva_luffy_{llama3b,llama8b,qwen4b,qwen8b}.sh`
- `data/`
  - `minerva_base_train.parquet`
  - `minerva_base_dev.parquet`
  - `minerva_train_dart32k_offpolicy.parquet`
  - Athena validation parquets
  - `seceval_mini.parquet`
  - `threat_actor_lookup.json`
- `vendor/vllm_v1_engine_core.py`
  - patched vLLM file from the working runtime on this machine
- `apply_vllm_core_patch.py`
  - helper script to install that patch into a target Python environment

## Important Limitation

This bundle contains the code and datasets needed to run Minerva LUFFY training, but it does not include:
- Hugging Face model weights
- a complete Conda or pip environment snapshot

The working runtime details and setup steps are documented in [SETUP.md](/home/jovyan/work/minerva/luffy/SETUP.md).

## Key Run Commands

From the copied bundle root on a new machine:

```bash
cd /path/to/minerva/luffy
bash exp_scripts/train_minerva_luffy_llama3b.sh
```

Other model wrappers:

```bash
bash exp_scripts/train_minerva_luffy_llama8b.sh
bash exp_scripts/train_minerva_luffy_qwen4b.sh
bash exp_scripts/train_minerva_luffy_qwen8b.sh
```

GRPO baselines:

```bash
bash exp_scripts/train_minerva_grpo_llama3b.sh
bash exp_scripts/train_minerva_grpo_llama8b.sh
bash exp_scripts/train_minerva_grpo_qwen4b.sh
bash exp_scripts/train_minerva_grpo_qwen8b.sh
```

## Notes

- The Minerva data in `data/` is already local, so the train scripts do not depend on `/home/jovyan/work/minerva/...` at run time.
- The one remaining external runtime dependency is the installed `vllm` package, which must be patched with `vendor/vllm_v1_engine_core.py` on a fresh machine.
- The copied LUFFY source already includes the local Ray and modern vLLM compatibility fixes that were required for this Blackwell machine.
