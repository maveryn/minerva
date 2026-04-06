from __future__ import annotations

from pathlib import Path


OLD = """        self.apply_rotary_emb_flash_attn = None\n        if find_spec(\"flash_attn\") is not None:\n            from flash_attn.ops.triton.rotary import apply_rotary\n\n            self.apply_rotary_emb_flash_attn = apply_rotary\n"""

NEW = """        self.apply_rotary_emb_flash_attn = None\n        if find_spec(\"flash_attn\") is not None:\n            try:\n                from flash_attn.ops.triton.rotary import apply_rotary\n            except ImportError:\n                apply_rotary = None\n\n            self.apply_rotary_emb_flash_attn = apply_rotary\n"""


def main() -> None:
    import vllm.model_executor.layers.rotary_embedding.common as common_mod

    target = Path(common_mod.__file__).resolve()
    text = target.read_text()
    if NEW in text:
        print(f"Already patched {target}")
        return
    if OLD not in text:
        raise RuntimeError(f"Expected flash-attn import block not found in {target}")
    target.write_text(text.replace(OLD, NEW))
    print(f"Patched {target}")


if __name__ == "__main__":
    main()
