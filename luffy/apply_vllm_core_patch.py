from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    source = here / "vendor" / "vllm_v1_engine_core.py"
    if not source.exists():
        raise FileNotFoundError(f"Missing source patch file: {source}")

    import importlib.metadata as md
    import vllm  # noqa: F401

    try:
        import vllm.v1.engine.core as core_mod
    except ModuleNotFoundError as exc:
        if exc.name == "vllm.v1" or exc.name.startswith("vllm.v1."):
            print(f"Skipping vLLM core patch for vllm=={md.version('vllm')} (no v1 engine module)")
            return
        raise

    target = Path(core_mod.__file__).resolve()
    backup = target.with_suffix(target.suffix + ".bak")

    if not backup.exists():
        shutil.copy2(target, backup)

    shutil.copy2(source, target)
    print(f"Patched {target}")
    print(f"Backup at {backup}")


if __name__ == "__main__":
    main()
