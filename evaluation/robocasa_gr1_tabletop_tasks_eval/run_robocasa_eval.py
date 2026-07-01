'''
Modified from https://github.com/NVIDIA/Isaac-GR00T/blob/d483f00b1c13116bda020bead9d16dca497b2f6d/gr00t/eval/rollout_policy.py
'''

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import time
from typing import Any
import uuid
import imageio
from PIL import Image

from evaluation.multistep_wrapper import MultiStepWrapper
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from utils.websocket_client_policy import WebsocketClientPolicy
import os

import robocasa  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
import robosuite  # noqa: F401

os.environ["MUJOCO_GL"] = "egl"

@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        video_delta_indices: Indices of video observations to stack
        state_delta_indices: Indices of state observations to stack
        n_action_steps: Number of action steps to execute
        max_episode_steps: Maximum number of steps per episode
    """

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 720
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    """Container for various environment wrapper configurations.

    Attributes:
        video: Configuration for video recording
        multistep: Configuration for multi-step processing
    """

    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


def get_robocasa_env_fn(
    env_name: str,
):
    def env_fn():
        return gym.make(env_name, enable_render=True)

    return env_fn

def get_gym_env(env_name: str, env_idx: int, total_n_envs: int):
    """Create Ray environment factory function without wrappers."""

    env_fn = get_robocasa_env_fn(env_name)

    return env_fn()


def create_eval_env(
    env_name: str, env_idx: int, total_n_envs: int, wrapper_configs: WrapperConfigs
) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        idx: Environment index (used to determine video recording)
        wrapper_configs: Configuration for environment wrappers
    Returns:
        Wrapped gymnasium environment
    """

    env = get_gym_env(env_name, env_idx, total_n_envs)

    env = MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: WebsocketClientPolicy,
    wrapper_configs: WrapperConfigs,
    n_episodes: int = 10,
    n_envs: int = 1,
    n_action_steps: int = 8
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy_fn: Function that creates a policy instance
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        wrapper_configs: Configuration for environment wrappers
        ray_env: Whether to use ray gym env to create each env.
    Returns:
        Collection results from running the episodes
    """
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_name,
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    # Storage for results
    episode_lengths = []
    current_rewards = [0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)

    # Initial reset
    observations, _ = env.reset()
    i = 0
    frames = []

    pbar = tqdm(total=n_episodes, desc="Episodes")
    while completed_episodes < n_episodes:
        request_data = {
            "observation.images.ego_view": observations["video.ego_view_bg_crop_pad_res256_freq20"][0][0], # the client supports packing numpy array
            "observation.state": np.concatenate([
                observations["state.left_arm"][0][0], # 0-7
                observations["state.left_hand"][0][0], # 7-13
                [0.0]*6, # 13-19
                [0.0]*3, # 19-22
                observations["state.right_arm"][0][0], # 22-29
                observations["state.right_hand"][0][0], # 29-35
                [0.0]*6, # 35-41
                observations["state.waist"][0][0], # 41-44
            ]),
            "task": observations["annotation.human.coarse_action"][0],
            "n_action_steps": n_action_steps
        }
        frames.append(request_data["observation.images.ego_view"])

        # actions, _ = policy.get_action(observations)
        actions = policy.infer(request_data)["actions"]
        actions = np.array(actions)

        action_dict = {
            "action.left_arm": actions[None, :, 0:7], # 0-7
            "action.left_hand": actions[None, :, 7:13], # 7-13
            "action.left_leg": [0.0]*6, # 13-19, actions[None, :, 13:19]
            "action.neck": [0.0]*3, # 19-22, actions[None, :, 19:22]
            "action.right_arm": actions[None, :, 22:29], # 22-29
            "action.right_hand": actions[None, :, 29:35], # 29-35
            "action.right_leg": [0.0]*6, # 35-41, actions[None, :, 35:41]
            "action.waist": actions[None, :, 41:44], # 41-44
        }

        next_obs, rewards, terminations, truncations, env_infos = env.step(action_dict)
        # NOTE (FY): Currently we don't properly handle policy reset. For now, our policy are stateless,
        # but in the future if we need policy to be stateful, we need to detect env reset and call policy.reset()
        i += 1
        # Update episode tracking
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, list):
                    env_success = np.any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            else:
                current_successes[env_idx] = False

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                env_success = env_infos["final_info"][env_idx]["success"]
                if isinstance(env_success, list):
                    env_success = any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1

            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                if "final_info" in env_infos:
                    current_successes[env_idx] |= any(env_infos["final_info"][env_idx]["success"])
                if "task_progress" in env_infos:
                    episode_infos["task_progress"].append(env_infos["task_progress"][env_idx][-1])
                if "q_score" in env_infos:
                    episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                if "valid" in env_infos:
                    episode_infos["valid"].append(all(env_infos["valid"][env_idx]))
                # Accumulate results
                episode_lengths.append(current_lengths[env_idx])
                episode_successes.append(current_successes[env_idx])
                print("(in evaluation progress...) episode successes: ", episode_successes)
                os.makedirs(os.path.join("data/robocasa_gr1_tabletop_tasks", env_name), exist_ok=True)
                imageio.mimsave(os.path.join("data/robocasa_gr1_tabletop_tasks", env_name, f"episode-{completed_episodes}.mp4"), np.stack(frames), fps=10)
                frames = []

                # Reset trackers for this environment.
                current_successes[env_idx] = False
                # only update completed_episodes if valid
                if "valid" in episode_infos:
                    if episode_infos["valid"][-1]:
                        completed_episodes += 1
                        pbar.update(1)
                else:
                    # envs don't return valid
                    completed_episodes += 1
                    pbar.update(1)
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0
        observations = next_obs
    pbar.close()

    env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    return env_name, episode_successes, episode_infos


def create_gr00t_sim_policy(
    policy_host: str = "",
    policy_port: int | None = None,
) -> WebsocketClientPolicy:
    policy = WebsocketClientPolicy(policy_host, policy_port)

    return policy


def run_gr00t_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    policy_host: str = "",
    policy_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
):
    wrapper_configs = WrapperConfigs(
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )

    policy = create_gr00t_sim_policy(
        policy_host, policy_port
    )

    results = run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=n_episodes,
        n_envs=n_envs,
        n_action_steps=n_action_steps,
    )
    return results

'''
For example:
python -m evaluation.robocasa_gr1_tabletop_tasks_eval.run_robocasa_eval --max_episode_steps 504 --n_episodes 1 --policy_host "127.0.0.1" --policy_port 8001 --env_name "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env" --n_envs 1 --n_action_steps 8
'''

'''
All available env_name:
[
  "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
  "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
]
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episode_steps", type=int, default=504)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--policy_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=None)
    parser.add_argument(
        "--env_name",
        type=str,
        default="gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    )
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--n_action_steps", type=int, default=8)

    args = parser.parse_args()

    # Validate n_envs
    if args.n_envs > 1:
        print("[Warning] n_envs > 1 is not currently supported. Using n_envs=1.")
        args.n_envs = 1

    results = run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
    )
    print("results: ", results)
    print("success rate: ", np.mean(results[1]))