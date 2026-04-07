# LiveSQLBench Local Setup

This directory contains a repo-local LiveSQLBench setup for the `sql-r1` branch.

It is designed to evaluate four model slots on `LiveSQLBench-Base-Full v1`:

- `Qwen/Qwen2.5-Coder-3B-Instruct`
- `MPX0222forHF/SQL-R1-3B`
- local GRPO checkpoint
- local MinervaRL checkpoint

The benchmark assets needed by these scripts are already included in `sql-benchmark/`:

- public dataset clone:
  - `sql-benchmark/livesqlbench-base-full-v1/`
- emailed GT/test-case supplement:
  - `sql-benchmark/livesqlbench_base_full_v1_gt_kg_testcases_0904.jsonl`
- PostgreSQL dump zip:
  - `sql-benchmark/bird-interact-full-dumps.zip`

The dump zip is tracked with Git LFS. After cloning on another machine, run:

```bash
git lfs install
git lfs pull
```

If you copy the repo directory directly instead of cloning, no extra download is needed.

## What These Scripts Do

1. `setup_assets.py`
   - merges the public dataset with the emailed GT/test-case file
   - normalizes field names for the official evaluator
   - downloads the official LiveSQLBench baseline/evaluation helper files
   - unpacks the PostgreSQL dump bundle

2. `start_local_postgres.sh`
   - starts a local PostgreSQL server without Docker
   - imports the Full-v1 databases from the provided dump bundle
   - creates the template and real DBs expected by the evaluator

3. `generate_sql_predictions.py`
   - builds prompts that mirror the official LiveSQLBench baseline prompt
   - runs local generation with `vllm`
   - extracts SQL from model responses
   - writes an evaluation-ready JSONL

4. `evaluate_predictions.sh`
   - runs the official LiveSQLBench evaluator against a local PostgreSQL host

5. `run_*.sh`
   - convenience wrappers for the four model slots

## Runtime Requirements

Python packages in the generation/eval environment:

- `vllm`
- `transformers`
- `torch`
- `psycopg2-binary`
- `sqlglot`
- `datasets`
- `tqdm`

PostgreSQL binaries on `PATH`:

- `initdb`
- `pg_ctl`
- `psql`
- `createdb`
- `dropdb`

If PostgreSQL is missing, one simple option is:

```bash
conda install -y -c conda-forge postgresql
pip install -r sql-benchmark/livesqlbench/requirements-local.txt
```

## One-Time Setup

```bash
cd /home/jovyan/work/minerva

python sql-benchmark/livesqlbench/setup_assets.py
bash sql-benchmark/livesqlbench/start_local_postgres.sh
```

`setup_assets.py` does three things:

- writes the merged benchmark JSONL
- ensures the official helper files exist under `sql-benchmark/livesqlbench/official/`
- unpacks the PostgreSQL dump zip under `sql-benchmark/livesqlbench/artifacts/postgre_table_dumps_full/`

After setup, the local PostgreSQL server defaults to:

- host: `127.0.0.1`
- port: `15432`
- user: `root`
- password: `123123`

Tracked setup artifacts:

- merged benchmark JSONL:
  - `sql-benchmark/livesqlbench/artifacts/livesqlbench_base_full_v1_full.jsonl`
- official helper files:
  - `sql-benchmark/livesqlbench/official/...`

## Run The Public Models

Base Qwen:

```bash
cd /home/jovyan/work/minerva
bash sql-benchmark/livesqlbench/run_qwen25coder3b_base.sh
```

Released SQL-R1:

```bash
cd /home/jovyan/work/minerva
bash sql-benchmark/livesqlbench/run_sqlr1_3b_released.sh
```

## Run The Local Trained Models

GRPO:

```bash
cd /home/jovyan/work/minerva
SQLR1_GRPO_MODEL_PATH=/abs/path/to/grpo_model \
bash sql-benchmark/livesqlbench/run_grpo_qwen25coder3b.sh
```

MinervaRL:

```bash
cd /home/jovyan/work/minerva
SQLR1_MINERVARL_MODEL_PATH=/abs/path/to/minervarl_model \
bash sql-benchmark/livesqlbench/run_minervarl_qwen25coder3b.sh
```

Note: I do not currently see surviving local merged checkpoints for the SQL GRPO or SQL MinervaRL runs in this workspace, so those wrappers intentionally take explicit model paths.

## Common Knobs

These wrappers forward the same optional environment overrides:

- `LIVESQLBENCH_LIMIT`
- `LIVESQLBENCH_N`
- `LIVESQLBENCH_TEMPERATURE`
- `LIVESQLBENCH_TOP_P`
- `LIVESQLBENCH_TP`
- `LIVESQLBENCH_MAX_TOKENS`
- `LIVESQLBENCH_MAX_MODEL_LEN`
- `LIVESQLBENCH_BACKEND`
- `CUDA_VISIBLE_DEVICES`

The default backend is `vllm`, with:

- `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- `TRANSFORMERS_NO_TF=1`

Example smoke run:

```bash
cd /home/jovyan/work/minerva
LIVESQLBENCH_LIMIT=5 \
LIVESQLBENCH_N=2 \
CUDA_VISIBLE_DEVICES=0 \
bash sql-benchmark/livesqlbench/run_qwen25coder3b_base.sh
```

## Outputs

Each run is written under:

- `sql-benchmark/livesqlbench/runs/<run_name>/`

Key files:

- `predictions.jsonl`
- `generation_summary.json`
- `evaluation_report.txt`
- `evaluation_status.jsonl`
- per-instance logs if evaluator logging is enabled

## Portable Copy Checklist

To run this elsewhere, the copied repo should include:

- `sql-benchmark/livesqlbench/`
- `sql-benchmark/livesqlbench-base-full-v1/`
- `sql-benchmark/livesqlbench_base_full_v1_gt_kg_testcases_0904.jsonl`
- `sql-benchmark/bird-interact-full-dumps.zip`

The only missing piece is the model weights for the local `GRPO` and `MinervaRL` slots; those are passed in at runtime with:

- `SQLR1_GRPO_MODEL_PATH`
- `SQLR1_MINERVARL_MODEL_PATH`
