from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


def main() -> None:
    spec = importlib.util.find_spec("deepspeed")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("deepspeed is not installed in the active environment")

    target = Path(spec.submodule_search_locations[0]) / "git_version_info.py"
    backup = target.with_suffix(target.suffix + ".bak")
    content = target.read_text()

    if "except Exception" in content:
        print(f"DeepSpeed CUDA_HOME patch already present in {target}")
        return

    original = """from .ops.op_builder.all_ops import ALL_OPS

compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    op_compatible = builder.is_compatible()
    compatible_ops[op_name] = op_compatible
    compatible_ops["deepspeed_not_implemented"] = False
"""
    patched = """from .ops.op_builder.all_ops import ALL_OPS

compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    try:
        op_compatible = builder.is_compatible()
    except Exception:
        op_compatible = False
    compatible_ops[op_name] = op_compatible
    compatible_ops["deepspeed_not_implemented"] = False
"""

    if original in content:
        updated = content.replace(original, patched)
    elif "except MissingCUDAException" in content:
        updated = content.replace("except MissingCUDAException:", "except Exception:")
        updated = updated.replace("from .ops.op_builder.builder import MissingCUDAException\n", "")
    else:
        raise RuntimeError(f"Could not find expected patch target in {target}")

    if not backup.exists():
        shutil.copy2(target, backup)

    target.write_text(updated)
    print(f"Patched {target}")
    print(f"Backup at {backup}")


if __name__ == "__main__":
    main()
