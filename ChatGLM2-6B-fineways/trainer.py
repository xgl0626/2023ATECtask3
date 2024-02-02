# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
import os
from typing import Dict, List, Optional
from transformers import Trainer

import torch
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.utils import logging

logger = logging.get_logger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"
VALUE_HEAD_FILE_NAME = "value_head.bin"

def get_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]: # get state dict containing trainable parameters
    state_dict = model.state_dict()
    filtered_state_dict = {}

    for k, v in model.named_parameters():
        if v.requires_grad:
            filtered_state_dict[k] = state_dict[k].cpu().clone().detach()

    return filtered_state_dict
        
class PrefixTrainer(Trainer):
    def __init__(self, *args, save_changed=False, **kwargs):
        self.save_changed = save_changed
        super().__init__(*args, **kwargs)
        
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        r"""
        Saves trainable parameters as model checkpoint.

        This function will only be executed at the process zero.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        model = unwrap_model(self.model)

        if hasattr(model, "pretrained_model"): # for models with valuehead
            backbone_model = getattr(model, "pretrained_model")
        else:
            backbone_model = model

        if hasattr(backbone_model, "peft_config"): # peft methods
            backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model)) # save lora weights
        else:
            torch.save(get_state_dict(backbone_model), os.path.join(output_dir, WEIGHTS_NAME)) # save trainable weights

        if hasattr(model, "v_head"): # save valuehead weights
            torch.save(get_state_dict(getattr(model, "v_head")), os.path.join(output_dir, VALUE_HEAD_FILE_NAME))

        with open(os.path.join(output_dir, TRAINING_ARGS_NAME), "w", encoding="utf-8") as f:
            f.write(self.args.to_json_string() + "\n")