from transformers import Qwen3VLForConditionalGeneration
from peft import PeftModel
import torch
import argparse

def parse_option():
    parser = argparse.ArgumentParser()

    # args for the path of VLM, action expert, and FAST tokenizer
    parser.add_argument('--base_model_path', type = str, default="/root/hf_models/Qwen/Qwen3-VL-2B-Instruct-Add-Action-Tokens")
    parser.add_argument('--peft_model_path', type = str)
    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_option()
    base_model_path = opt.base_model_path
    peft_model_path = opt.peft_model_path

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path, torch_dtype = torch.bfloat16
    )
    print(model.dtype)

    model = PeftModel.from_pretrained(model, peft_model_path)

    print("before merging:")
    for name, param in model.named_parameters():
        print(name)

    model = model.merge_and_unload(progressbar = True)
    print(model.dtype)

    print("after merging:")
    for name, param in model.named_parameters():
        print(name)

    model.save_pretrained(
        peft_model_path, 
        max_shard_size = "100GB"
    )