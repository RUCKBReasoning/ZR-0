from transformers.feature_extraction_utils import BatchFeature
from .qwen_vl_backbone import QwenVLBackbone
from torch import nn
from typing import Tuple
from .flow_matching_action_head import FlowmatchingActionHead, FlowmatchingActionHeadConfig

import torch
import json
import os
from safetensors.torch import save_file, load_file

class ZR0Model(nn.Module):
    def __init__(
            self,
            vlm_name_or_path: str,
            action_expert_name_or_path: str,
            action_expert_config: FlowmatchingActionHeadConfig,
            tune_vlm=True,
            tune_action_expert=True,
            detach_vlm_outputs_for_action_expert=False,
            lora_args=None,
        ):
        super().__init__()

        self.action_expert_config = action_expert_config

        self.backbone = QwenVLBackbone(vlm_name_or_path, tune_vlm, lora_args)
        print("the size of VLM's last layer hidden state:", self.backbone.model.config.text_config.hidden_size)
        self.action_expert_config.vlm_output_embedding_dim = self.backbone.model.config.text_config.hidden_size
        # self.action_expert_config.vlm_output_embedding_dim = self.backbone.model.config.hidden_size
        self.action_expert = FlowmatchingActionHead(self.action_expert_config, tune_action_expert)
        self.detach_vlm_outputs_for_action_expert = detach_vlm_outputs_for_action_expert
        
        if action_expert_name_or_path:
            print(f"load pre-trained weights from {action_expert_name_or_path} to initialize the action expert")
            self.action_expert.load_state_dict(
                load_file(os.path.join(action_expert_name_or_path, "action_expert.safetensors"))
            )

    def prepare_vla_input(self, batch_inputs) -> Tuple[BatchFeature, BatchFeature]:
        backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
        action_expert_inputs = self.action_expert.prepare_inputs(batch_inputs)
        # print(f"prepare a batched samples spends {time.time()-start_time}s")
        return backbone_inputs, action_expert_inputs

    @torch.inference_mode()
    def get_action_direct(self, batch_inputs: BatchFeature, num_denoised_steps: int = 5) -> BatchFeature:
        with torch.no_grad():
            backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
            backbone_outputs = self.backbone(backbone_inputs)

            action_expert_inputs = self.action_expert.prepare_inputs(batch_inputs)
            action_expert_outputs = self.action_expert.get_action(
                backbone_outputs, action_expert_inputs, num_denoised_steps
            )
        # return action_pred
        return BatchFeature(data={"action_pred": action_expert_outputs["action_pred"]})
    
    @torch.inference_mode()
    def get_n_actions_direct(self, batch_inputs: BatchFeature, num_denoised_steps: int = 5, n: int = 8) -> BatchFeature:
        # sampling `n` different action chunks
        with torch.no_grad():
            backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
            backbone_outputs = self.backbone(backbone_inputs)

            action_expert_inputs = self.action_expert.prepare_inputs(batch_inputs)
            action_preds_list = []
            for i in range(n):
                action_expert_outputs = self.action_expert.get_action(
                    backbone_outputs, action_expert_inputs, num_denoised_steps
                )
                action_preds_list.append(action_expert_outputs["action_pred"][0])
        n_action_preds = torch.stack(action_preds_list, dim=0)

        # return action_pred
        return BatchFeature({"n_action_preds": n_action_preds})
  
    @torch.inference_mode()
    def get_action_subtask(self, batch_inputs: BatchFeature, num_denoised_steps: int = 5) -> BatchFeature:
        with torch.no_grad():
            prompt_len = len(batch_inputs["input_ids"][0])
            
            # 1. generate subtask
            new_input_ids = self.backbone.model.generate(
                input_ids = batch_inputs["input_ids"],
                attention_mask = batch_inputs["attention_mask"],
                pixel_values = batch_inputs["pixel_values"],
                image_grid_thw = batch_inputs["image_grid_thw"],
                do_sample=False,
                eos_token_id=[153718]
            )
            new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.long)

            print("output subtask:", self.backbone.processor.tokenizer.decode(new_input_ids[0][prompt_len:]))

            # 2. perpare new VLM inputs (prompt + subtask)
            backbone_inputs = {
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
                "pixel_values": batch_inputs["pixel_values"],
                "image_grid_thw": batch_inputs["image_grid_thw"],
                "sub_task_flag": batch_inputs["sub_task_flag"]
            }
            # 3. VLM forward
            backbone_outputs = self.backbone(backbone_inputs)
            
            # 4. action expert diffusion
            action_expert_inputs = self.action_expert.prepare_inputs(batch_inputs)
            action_expert_outputs = self.action_expert.get_action(
                backbone_outputs, action_expert_inputs, num_denoised_steps
            )
        # return action_pred
        return BatchFeature(data={"action_pred": action_expert_outputs["action_pred"]})

    def forward(self, batch_inputs, training_progress: float, dynamic_loss_type = "vlm_and_action", 
                vlm_loss_weight = 1.0, action_expert_loss_weight = 1.0) -> BatchFeature:
        # backbone_inputs, action_expert_inputs = self.prepare_vla_input(batch_inputs)
        backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        vlm_loss = backbone_outputs["vlm_loss"]

        if dynamic_loss_type == "vlm":
            return BatchFeature(data={"loss": vlm_loss, "vlm_loss": vlm_loss})

        if self.detach_vlm_outputs_for_action_expert:
            backbone_outputs["backbone_embeddings"] = backbone_outputs["backbone_embeddings"].detach()
        
        action_expert_inputs = self.action_expert.prepare_inputs(batch_inputs)
        action_expert_outputs = self.action_expert(backbone_outputs, action_expert_inputs, training_progress)
        action_expert_loss = action_expert_outputs["action_expert_loss"]

        if dynamic_loss_type == "action":
            return BatchFeature(data={"loss": action_expert_loss, "action_expert_loss": action_expert_loss})
        elif dynamic_loss_type == "vlm_and_action":
            return BatchFeature(
                data={
                      "loss": vlm_loss_weight * vlm_loss + action_expert_loss_weight * action_expert_loss,
                      "vlm_loss": vlm_loss,
                      "action_expert_loss": action_expert_loss,
                      "weighted_vlm_loss": vlm_loss_weight * vlm_loss,
                      "weighted_action_expert_loss": action_expert_loss_weight * action_expert_loss,
                      "vlm_loss weight": vlm_loss_weight,
                      "action_expert_loss weight": action_expert_loss_weight
                    }
            )
        else:
            raise ValueError(f"Unrecognized dynamic_loss_type: {dynamic_loss_type}. It should be [vlm, action, vlm_and_action]")
    
    @property
    def device(self):
        return next(iter(self.parameters())).device

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # save backbone model parameter and model config (Qwen-VL)
        self.backbone.model.save_pretrained(save_directory)
        # save backnone's processor
        self.backbone.processor.save_pretrained(save_directory)
        # save action expert's model parameter
        save_file(self.action_expert.state_dict(), os.path.join(save_directory, "action_expert.safetensors"))
        # save action expert's config
        with open(os.path.join(save_directory, "action_expert_config.json"), "w") as json_file:
            json.dump(self.action_expert_config.to_dict(), json_file, indent=4)
    
    @classmethod
    def from_pretrained(cls, save_directory, tune_vlm=False, tune_action_expert=False, detach_vlm_outputs_for_action_expert=False):
        # `save_directory` should contain complete VLA weights (a VLM and an action expert)
        action_expert_config_json = json.load(open(os.path.join(save_directory, "action_expert_config.json")))
        action_expert_config = FlowmatchingActionHeadConfig(**action_expert_config_json)

        return cls(
            vlm_name_or_path = save_directory,
            action_expert_name_or_path = save_directory,
            action_expert_config = action_expert_config,
            tune_vlm = tune_vlm,
            tune_action_expert = tune_action_expert,
            detach_vlm_outputs_for_action_expert = detach_vlm_outputs_for_action_expert
        )
