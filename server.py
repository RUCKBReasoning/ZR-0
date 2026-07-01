import argparse
from policies.reasoning_vla_policy import ZR0Policy
from utils.websocket_server_policy import WebsocketPolicyServer
import random
import numpy as np
import torch
import json

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_entry', type=str, default="libero_v21",
                        help="the pre-registration dataset entry in dataset2feature.yaml")
    parser.add_argument('--ckpt_dir', type=str, help="checkpoint directory")
    parser.add_argument('--inference_mode', type=str, help="specify the inference mode, in `direct_action` or `subtask_then_action`")
    parser.add_argument('--window_size', type=int, default=1, help="window size")
    parser.add_argument('--num_denoised_steps', type=int, default=5, help="number of denoised steps")
    parser.add_argument('--max_pad_state_and_action_length', type=int, default=64, help="max padding length")
    parser.add_argument('--port', type=int, default=8000, help="server port")
    
    args = parser.parse_args()
    return args

def deploy():
    opt = parse_option()
    set_all_seeds(42)

    assert opt.inference_mode in ["direct_action", "subtask_then_action"]
    # settings
    kwargs = {
        "dataset_entry": opt.dataset_entry,
        "ckpt_dir": opt.ckpt_dir,
        "inference_mode": opt.inference_mode,
        "window_size": opt.window_size,
        "num_denoised_steps": opt.num_denoised_steps,
        "max_pad_state_and_action_length": opt.max_pad_state_and_action_length,
        "device": "cuda:0"
    }
    print(json.dumps(kwargs, indent=2, ensure_ascii=False))
    policy = ZR0Policy(**kwargs)
    
    print("Start serving.")
    host = "0.0.0.0"
    port = opt.port
    server = WebsocketPolicyServer(policy, host, port)
    server.serve_forever()

if __name__ == "__main__":
    deploy()