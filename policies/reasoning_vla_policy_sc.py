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

import torch
import torch.nn.functional as F

def consensus_median_then_pick(n_action_preds, pick_candidate=True):
    Xn = n_action_preds  # (K, N, M)

    # consensus actions
    consensus = Xn.median(dim=0).values  # (N, M)

    if not pick_candidate:
        return consensus

    # pick up the trajectory the most close to the consensus
    dists = (Xn - consensus).abs().mean(dim=(1, 2))  # (K,)
    best_idx = dists.argmin().item()
    return best_idx


def trimmed_l1_medoid_then_pick(n_action_preds, trim_ratio=0.2):
    X = n_action_preds

    # Pairwise distance matrix D[i,j], using L1 norm
    diff = X.unsqueeze(1) - X.unsqueeze(0)    # (K, K, N, M)
    diff = diff.abs()
    D = diff.mean(dim=(2, 3))                 # (K, K)

    K = X.size(0)
    keep = max(1, int(math.ceil((1 - trim_ratio) * (K - 1))))
    scores = torch.empty(K, device=D.device)

    # For each candidate i, take the minimum keep of its distance (remove itself) and average it.
    for i in range(K):
        # remove self
        vals = torch.cat([D[i, :i], D[i, i+1:]], dim=0)  # (K-1,)
        topk_vals, _ = torch.topk(vals, k=keep, largest=False)
        scores[i] = topk_vals.mean()

    best_idx = scores.argmin().item()
    return best_idx

def mse_then_pick(n_action_preds):
    K = n_action_preds.size(0)
    mse_scores = torch.zeros(K)

    # calculate MSE among candidates
    for i in range(K):
        diff_sum = 0.0
        for j in range(K):
            if i != j:
                diff = n_action_preds[i] - n_action_preds[j]
                diff_sum += torch.mean(diff ** 2).item()
        mse_scores[i] = diff_sum / (K - 1)

    print("mse_scores:", mse_scores)
    # select the best candidate
    best_idx = torch.argmin(mse_scores).item()

    return best_idx

class ZR0Policy(BasePolicy):
    def __init__(
        self,
        dataset_entry,
        ckpt_dir,
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
        observations = self.ob_buffer.get_inference_time_observations(self.dataset_meta.camera_keys, visualize = False)
        
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
            fast_tokenizer=None, # FAST tokenizer is not needed during inference
        )

        # prepare batch input
        vla_input = custom_collate_fn([{**action_expert_inputs, **vl_inputs}])
        for key, value in vla_input.items():
            vla_input[key] = value.to(self.device)

        # # model inference
        # with torch.no_grad():
        #     st = time.time()
        #     vla_outputs = self.model.get_action(vla_input, self.num_denoised_steps)
        #     print(f"vla forward takes {time.time()-st} seconds.")
        # '''
        # 1. bs=1
        # 2. only retain the next `n_action_steps` steps in the chunk
        # 3. slice out the first few `action_dim` (the remaining dim are padded with 0 during training)
        # '''
        # action_chunk = vla_outputs["action_pred"][0][:n_action_steps, :self.action_dim]
        # action_chunk = min_max_denorm(action_chunk, self.dataset_meta.stats["action"], self.use_quantile)
        # action_chunk = action_chunk.tolist()

        # model inference
        with torch.no_grad():
            st = time.time()
            # action_pred = self.model.get_action(vla_input, self.num_denoised_steps)["action_pred"][0]
            n_action_preds = self.model.get_n_actions_direct(vla_input, self.num_denoised_steps, 8)["n_action_preds"] # [8, 32, 64]
            print(f"vla forward takes {time.time()-st} seconds.")

        # self-consistency
        n_action_preds = n_action_preds[:, :n_action_steps, :self.action_dim]

        best_idx_mse = mse_then_pick(n_action_preds)
        print("best_idx (MSE):", best_idx_mse)

        best_idx_consensus_median = consensus_median_then_pick(n_action_preds)
        print("best_idx_consensus_median:", best_idx_consensus_median)
        
        best_idx_trimmed_l1_medoid = trimmed_l1_medoid_then_pick(n_action_preds)
        print("best_idx_trimmed_l1_medoid:", best_idx_trimmed_l1_medoid)

        picks = torch.tensor([best_idx_mse, best_idx_consensus_median, best_idx_trimmed_l1_medoid])
        most_common_idx = torch.mode(picks).values.item() # common value
        print(most_common_idx)

        action_chunk = n_action_preds[most_common_idx]  # shape: (N, M)
        action_chunk = min_max_denorm(action_chunk, self.dataset_meta.stats["action"], self.use_quantile)
        action_chunk = action_chunk.tolist()

        # single inference
        # action_chunk = action_pred[:n_action_steps, :self.action_dim]
        # action_chunk = min_max_denorm(action_chunk, self.dataset_meta.stats["action"], self.use_quantile)
        # action_chunk = action_chunk.tolist()

        print("action_chunk:")
        for action in action_chunk:
            print(action)

        self.global_inference_steps += 1
        print("-"*30)

        return {"actions": action_chunk}