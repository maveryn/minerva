# verl/extensions/bridging/main_ppo_bridged.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from verl.extensions.bridging.use_bridged_dataset import make_bridged_trainloader
from verl.trainer.main_ppo import main as base_main


def main():
    # Reuse VeRL hydra config composition
    base_cfg = hydra.compose(config_name="ppo_trainer", overrides=[])
    cfg = OmegaConf.merge(base_cfg, OmegaConf.from_cli())
    if cfg.get("bridging", {}).get("enable", False):
        # monkey-patch the dataloader builder used in the base main
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        orig_init = RayPPOTrainer.__init__

        def wrapped_init(self, *a, **kw):
            orig_init(self, *a, **kw)
            # swap train dataset/dataloader
            self.train_dataloader = make_bridged_trainloader(cfg, self.tokenizer)

        RayPPOTrainer.__init__ = wrapped_init
    base_main()


if __name__ == "__main__":
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra.initialize_config_module("verl.trainer.config")
    main()
