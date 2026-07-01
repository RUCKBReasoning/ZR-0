import torch
import numpy as np
import json
import time
import os
import math

from typing import Dict
from torchvision.transforms import ToTensor
from transformers import AutoProcessor
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from model import ZR0Model
from utils.load_training_dataset import prepare_qwen_vl_inputs_cpu, prepare_action_expert_inputs_cpu, custom_collate_fn
from utils.obs_buffer import ObservationBuffer
from utils.normalization import min_max_denorm
from utils.constants import DATASET2FEATURE
from policies.base_policy import BasePolicy

class ZR0Policy(BasePolicy):
    def __init__(
        self,
        dataset_entry,
        ckpt_dir,
        inference_mode,
        window_size,
        num_denoised_steps=5,
        max_pad_state_and_action_length=64,
        device="cuda:0"
    ):
        super().__init__()
        # env params
        self.dataset_entry = dataset_entry
        self.ckpt_dir = ckpt_dir
        self.window_size = window_size
        self.num_denoised_steps = num_denoised_steps
        self.max_pad_state_and_action_length = max_pad_state_and_action_length
        self.device = device
        self.inference_mode = inference_mode
        self.prompt_suffix = "" if inference_mode == "direct_action" else "Sub task:"

        # load relevant configurations from the constant pool
        self.dataset_path = DATASET2FEATURE[dataset_entry]["dataset_path"]
        self.use_quantile = DATASET2FEATURE[dataset_entry]["use_quantile"]
        self.action_dim = json.load(open(os.path.join(self.dataset_path, "meta", "info.json")))["features"]["action"]["shape"][0]

        # load VLM's processor
        self.processor = AutoProcessor.from_pretrained(ckpt_dir)
        # load dataset's metadata
        self.dataset_meta = LeRobotDatasetMetadata(
            repo_id = self.dataset_path.split("/")[-1],
            root = self.dataset_path
        )
        # load model
        self.model = ZR0Model.from_pretrained(ckpt_dir).to(device).to(torch.bfloat16)
        self.model.eval()

        # use `torch.compile` to speed up inference
        self.model = torch.compile(self.model, mode="default") # or mode="reduce-overhead"
        
        # initialize buffer
        self.ob_buffer = ObservationBuffer(max_recent_observations = window_size)
        self.to_tensor = ToTensor()
        self.global_inference_steps = 0
        
    def infer(self, obs: Dict) -> Dict:
        task = obs.get('task')
        state = obs.get('observation.state')
        n_action_steps = obs.get('n_action_steps')
        print("task:", task)
        if isinstance(state, np.ndarray):
            print("state:", state.tolist())
        else:
            print("state:", state)
        print("n_action_steps:", n_action_steps)

        # convert image from numpy to tensor
        multi_view_images = dict()
        for cam_key in self.dataset_meta.camera_keys:
            multi_view_images[cam_key] = self.to_tensor(np.array(obs.get(cam_key), dtype=np.uint8))

        # manage historical observation buffer
        self.ob_buffer.add_observation(multi_view_images)
        
        # get the latest observations from the buffer
        observations = self.ob_buffer.get_inference_time_observations(self.dataset_meta.camera_keys, visualize = True)
        
        data_sample = {
            "task": task, "observation.state": state
        } # "embodiment_id": self.embodiment_id, 
        for key, value in observations.items():
            data_sample[key] = value
        
        action_expert_inputs = prepare_action_expert_inputs_cpu(
            data_sample, self.dataset_meta.stats, self.max_pad_state_and_action_length, self.use_quantile,
            self.model.action_expert_config.action_horizon, self.action_dim
        )
        vl_inputs = prepare_qwen_vl_inputs_cpu(
            data=data_sample,
            camera_keys=self.dataset_meta.camera_keys,
            grounding_camera_keys=self.dataset_meta.grounding_camera_keys,
            processor=self.processor,
            process_mode="eval",
            prompt_suffix=self.prompt_suffix,
            fast_tokenizer=None, # FAST tokenizer is not needed during inference
        )
        sub_task_flag = 0 if self.inference_mode == "direct_action" else 1

        # prepare batch input
        vla_input = custom_collate_fn([{**action_expert_inputs, **vl_inputs, "sub_task_flag": torch.tensor(sub_task_flag)}])
        for key, value in vla_input.items():
            vla_input[key] = value.to(self.device)

        # model inference
        with torch.no_grad():
            st = time.time()
            if self.inference_mode == "direct_action":
                vla_outputs = self.model.get_action_direct(vla_input, self.num_denoised_steps)
            elif self.inference_mode == "subtask_then_action":
                vla_outputs = self.model.get_action_subtask(vla_input, self.num_denoised_steps)
            print(f"vla forward takes {time.time()-st} seconds.")

        '''
        1. bs=1
        2. only retain the next `n_action_steps` steps in the chunk
        3. slice out the first few `action_dim` (the remaining dim are padded with 0 during training)
        '''
        action_chunk = vla_outputs["action_pred"][0][:n_action_steps, :self.action_dim]
        action_chunk = min_max_denorm(action_chunk, self.dataset_meta.stats["action"], self.use_quantile)
        action_chunk = action_chunk.tolist()

        print("action_chunk:")
        for action in action_chunk:
            print(action)

        self.global_inference_steps += 1
        print("-"*30)

        return {"actions": action_chunk}