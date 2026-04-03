import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path("/home/jovyan/work/LUFFY")
DATA_DIR = REPO_ROOT / "data"

DEFAULT_FILES = {
    "minerva_base_train.parquet": Path(
        "/home/jovyan/work/minerva/rlvr/mydata/minerva_base/minerva_base_train.parquet"
    ),
    "minerva_base_dev.parquet": Path(
        "/home/jovyan/work/minerva/rlvr/mydata/minerva_base/minerva_base_dev.parquet"
    ),
    "threat_actor_lookup.json": Path(
        "/home/jovyan/work/minerva/rlvr/mydata/minerva_base/threat_actor_lookup.json"
    ),
    "athena_cti_ate.parquet": Path(
        "/home/jovyan/work/minerva/rlvr/mydata/athena/athena_cti_ate.parquet"
    ),
    "athena_cti_ckt.parquet": Path(
        "/home/jovyan/work/minerva/rlvr/mydata/athena/athena_cti_ckt.parquet"
    ),
    "athena_cti_rcm.parquet": Path(
        "/home/jovyan/work/minerva/rlvr/mydata/athena/athena_cti_rcm.parquet"
    ),
    "athena_cti_rms.parquet": Path(
        "/home/jovyan/work/minerva/rlvr/mydata/athena/athena_cti_rms.parquet"
    ),
    "athena_cti_taa.parquet": Path(
        "/home/jovyan/work/minerva/rlvr/mydata/athena/athena_cti_taa.parquet"
    ),
    "athena_cti_vsp.parquet": Path(
        "/home/jovyan/work/minerva/rlvr/mydata/athena/athena_cti_vsp.parquet"
    ),
    "seceval_mini.parquet": Path(
        "/home/jovyan/work/minerva/rlvr/mydata/seceval/seceval_mini.parquet"
    ),
}


def copy_file(src: Path, dst: Path, overwrite: bool) -> dict:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    if dst.exists() and not overwrite:
        action = "kept_existing"
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        action = "copied"
    return {
        "source": str(src),
        "destination": str(dst),
        "size_bytes": dst.stat().st_size,
        "action": action,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize Minerva CTI train/dev/validation datasets inside LUFFY/data."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--summary",
        default=str(DATA_DIR / "minerva_data_manifest.json"),
        help="Where to write the manifest JSON.",
    )
    args = parser.parse_args()

    records = {}
    for name, src in DEFAULT_FILES.items():
        dst = DATA_DIR / name
        records[name] = copy_file(src, dst, overwrite=args.overwrite)

    offpolicy_path = DATA_DIR / "minerva_train_dart32k_offpolicy.parquet"
    if offpolicy_path.exists():
        records["minerva_train_dart32k_offpolicy.parquet"] = {
            "source": str(offpolicy_path),
            "destination": str(offpolicy_path),
            "size_bytes": offpolicy_path.stat().st_size,
            "action": "present",
        }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
