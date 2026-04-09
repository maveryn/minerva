# SQL-E1 Paper Notes

This folder contains the paper-facing documentation for our SQL-E1 experiments.

In these notes, `SQL-E1` refers to our Minerva/VeRL adaptation of the SQL-R1 setup:

- SQL-R1 public training and validation data
- SQL-R1 executable reward
- `Qwen2.5-Coder-3B-Instruct` as the base model
- two training variants:
  - vanilla `GRPO`
  - Minerva Noctua-style `MinervaRL`

The main files are:

- [training_setup.md](/home/jovyan/work/minerva/paper/training_setup.md): training data, hyperparameters, checkpoints, and the differences between GRPO and MinervaRL
- [evaluation_setup.md](/home/jovyan/work/minerva/paper/evaluation_setup.md): benchmark definitions, metrics, self-consistency settings, and LiveSQLBench evaluation details
- [results.md](/home/jovyan/work/minerva/paper/results.md): consolidated result tables for all evaluated models and datasets

Implementation paths remain under the repo's existing `sql-r1` integration, but this folder uses the paper-facing `SQL-E1` name consistently.
