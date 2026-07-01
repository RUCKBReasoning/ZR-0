import torch
import time
import torch.distributed as dist

from torch import nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import is_flash_attn_2_available

class QwenVLBackbone(nn.Module):
    def __init__(
        self, model_name: str, tune_vlm: bool, lora_args: dict
    ):
        super().__init__()

        self.tune_vlm = tune_vlm
        if is_flash_attn_2_available():
            attn_implementation = "flash_attention_2"
        else:
            print(
                "[Warning] Flash Attention 2 is not available. "
                "It is highly recommended to install flash-attn for faster training speed "
                "and lower GPU memory usage."
            )
            attn_implementation = "eager"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation
        )
        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model_name, trust_remote_code=True,
        #     attn_implementation=attn_implementation
        # )

        if lora_args and not self.tune_vlm:
            raise ValueError("you set tune_vlm=False, which means that the parameters of the VLM do not change in any way, \
                             therefore you can not set use_lora=True.")

        if lora_args and lora_args["use_lora"]:
            target_modules = [target_module.strip() for target_module in lora_args["target_modules"].split(',')]
            print("Lora target_modules:", target_modules)
            peft_config = LoraConfig(
                task_type = TaskType.CAUSAL_LM, 
                target_modules = target_modules, 
                r = lora_args["r"], 
                lora_alpha = lora_args["lora_alpha"], 
                lora_dropout = lora_args["lora_dropout"]
            )
            self.model = get_peft_model(self.model, peft_config)

        if self.tune_vlm:
            self.model.gradient_checkpointing_enable()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.set_trainable_parameters()

    def set_trainable_parameters(self):
        if not self.tune_vlm:
            self.model.requires_grad_(False)
        
    def set_frozen_modules_to_eval_mode(self):
        if self.training and not self.tune_vlm:
            self.model.eval()

    def prepare_inputs(self, batch: dict):
        vl_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"],
            "image_grid_thw": batch["image_grid_thw"]
        }

        if "labels" in batch:
            vl_inputs["labels"] = batch["labels"]
        if "sub_task_flag" in batch:
            vl_inputs["sub_task_flag"] = batch["sub_task_flag"]
        
        return BatchFeature(data=vl_inputs)

    # def generate(self, vl_inputs: BatchFeature) -> BatchFeature:
    #     # TODO: we only support batch size = 1 during the inference
    #     if vl_inputs["input_ids"].shape[0] > 1:
    #         raise ValueError("we only support batch size = 1 during the inference, current batch size is: ", vl_inputs["input_ids"].shape[0])

    #     # obtain the VLM's last layer hidden states
    #     vlm_outputs = self.forward(vl_inputs)

    #     return vlm_outputs

    def extract_action_expert_cross_attn_mask_only_prompt(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: LongTensor, shape (bs, L)
        Returns: BoolTensor, shape (bs, L)

        Rule: For each row, set True for positions before and including the last occurrence (guaranteed unique) of the token ids [151644, 77091, 198], 
                and False for positions after.
        """
        # 151644 is '<|im_start|>', 77091 is 'assistant'; 198 is '\n'. 
        # They are adjacent tokens in the generation prompt
        a, b, c = 151644, 77091, 198
        bs, L = input_ids.shape
        if L < 3:
            return torch.ones((bs, L), dtype=torch.bool, device=input_ids.device)

        # Ternary neighbor matching yields a Boolean tensor of shape (bs, L-2), 
        # with True indicating that the position is the start of the ternary sequence
        matches3 = (input_ids[:, :-2] == a) & (input_ids[:, 1:-1] == b) & (input_ids[:, 2:] == c)
        
        # check which batch actually found the sub-sequence
        has_match = matches3.any(dim=1)  # (bs,) bool
        
        # Each sequence appears exactly once, use argmax to take the starting position
        start_idx = matches3.to(torch.int64).argmax(dim=1)  # (bs,)
        end_idx = start_idx + 2  # position of the last token of the ternary sequence
        
        # if not found, set end_idx for that row to L-1, so that all mask be set to True
        end_idx = end_idx.masked_fill(~has_match, L-1)
        
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)
        mask = pos <= end_idx.unsqueeze(1)  # (bs, L), bool
        return mask

    def extract_action_expert_cross_attn_mask_prompt_plus_subtask(self, input_ids):
        """
        input_ids: LongTensor, shape (bs, L)
        Returns: BoolTensor, shape (bs, L)

        Rule:
            For each row, set True for positions before and including the unique occurrence
            of the token id 153718, and False for positions after.

        Note:
            Token id 153718 is guaranteed to appear exactly once per sequence.
        """
        target_token = 153718 # here, 153718 is the token id of '</SUB_TASK>'
        bs, L = input_ids.shape

        # check which batch actually found the target token
        token_mask_sum = (input_ids == target_token).sum(dim=1)
        
        # Find index of the unique target token per batch (shape: bs,)
        idx = (input_ids == target_token).to(torch.int64).argmax(dim=1)
        
        # if not found, set end_idx for that row to L-1, so that all mask be set to True
        idx = idx.masked_fill(token_mask_sum == 0, L-1)

        # Generate boolean mask: True for all positions <= idx
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)
        mask = pos <= idx.unsqueeze(1)  # (bs, L), bool

        return mask

    def forward(self, vl_inputs: BatchFeature) -> BatchFeature:
        # set frozen module to eval
        self.set_frozen_modules_to_eval_mode()

        use_cache = False if self.training else True
        model_outputs = self.model(**vl_inputs, return_dict=True, 
                                   output_hidden_states=True, use_cache=use_cache)

        # return the embeddings of all tokens in the middle or the last layer (bs, seq_len, hidden_size)
        # layer_idx = self.model.config.text_config.num_hidden_layers//2
        layer_idx = -1
        embeddings = model_outputs["hidden_states"][layer_idx] # e.g., (4, 4096, 2048)

        # let the action expert only attend to the input (i.e., "image+text prompt" part) of the VLM,
        # including the generation prompt '<|im_start|>assistant\n'. Thus, during inference, we should set `add_generation_prompt` to True.
        # action_expert_cross_attn_mask = self.extract_action_expert_cross_attn_mask(vl_inputs["input_ids"])
        
        # Compute both masks
        mask_mode1 = self.extract_action_expert_cross_attn_mask_only_prompt(vl_inputs["input_ids"])
        mask_mode2 = self.extract_action_expert_cross_attn_mask_prompt_plus_subtask(vl_inputs["input_ids"])

        # Flag for mode selection: 0 -> mode1, 1 -> mode2
        sub_task_flag = vl_inputs["sub_task_flag"].unsqueeze(1)  # (bs, 1)

        # Combine masks by mode
        action_expert_cross_attn_mask = torch.where(
            sub_task_flag == 1,
            mask_mode2,
            mask_mode1
        )
        
        return BatchFeature(
            data={
                "backbone_embeddings": embeddings,
                "action_expert_cross_attn_mask": action_expert_cross_attn_mask,
                "vlm_loss": model_outputs["loss"] if "labels" in vl_inputs else None
            }
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device
