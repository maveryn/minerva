# Setup Notes For Another Machine

These are the exact practical steps another Codex should follow to run the Minerva LUFFY bundle on a new machine.

## 1. Copy The Bundle

Copy this whole directory:

```bash
/home/jovyan/work/minerva/luffy
```

to the target machine.

## 2. Install A Compatible Python Runtime

The successful runtime on this machine used:

```text
torch==2.9.0
vllm==0.13.0
ray==2.54.1
transformers==4.57.5
datasets==4.8.4
tensordict==0.9.1
flash_attn==2.8.3
wandb==0.25.1
omegaconf==2.3.0
```

Upstream LUFFY requirement files are included in:

- [luffy/requirements.txt](/home/jovyan/work/minerva/luffy/luffy/requirements.txt)
- [luffy/requirements.v2.txt](/home/jovyan/work/minerva/luffy/luffy/requirements.v2.txt)

Those are useful as a baseline, but the working Minerva LUFFY run on this machine used the newer stack above, not the original older LUFFY stack.

## 3. Apply The vLLM Core Patch

The working runtime also required a patch to the installed `vllm` package.

From the bundle root:

```bash
python apply_vllm_core_patch.py
```

This replaces:

```text
<site-packages>/vllm/v1/engine/core.py
```

with:

```text
vendor/vllm_v1_engine_core.py
```

## 4. Hugging Face Access

You still need model access for the launch scripts, for example:

- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen3-4B-Base`
- `Qwen/Qwen3-8B-Base`

The scripts download model weights through the local Hugging Face environment.

## 5. Launch

LUFFY off-policy:

```bash
cd /path/to/minerva/luffy
bash exp_scripts/train_minerva_luffy_llama3b.sh
```

GRPO baseline:

```bash
cd /path/to/minerva/luffy
bash exp_scripts/train_minerva_grpo_llama3b.sh
```

## 6. Minerva Data Already Included

The copied `data/` directory already contains:

- Minerva RL train/dev
- the 32k DART-aligned LUFFY off-policy parquet
- Athena validation parquets
- Seceval mini
- threat actor lookup JSON

So no extra Minerva repo checkout is required to run the training scripts.

## 7. Known Runtime Notes

- The LUFFY train scripts auto-pick a Python executable with `sm_120` support when available.
- The current copied code includes local Ray startup limits that were needed to keep Ray stable on a 120-core Blackwell machine.
- The working LUFFY validation/checkpoint naming matches the Minerva convention:
  - `val-core/minerva-dev/reward/mean`
  - `val-core/athena-bench/reward/mean`
  - `val-core/seceval/reward/mean`
  - `val-core/global-val/reward/mean`
