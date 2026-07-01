# Qwen Special Token Addition Script

Easily add new special tokens to [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) (or compatible models), and save the resulting model and tokenizer for local loading.

## Usage

```bash
source_model_id=/your_model_path/Qwen/Qwen3-VL-2B-Instruct
target_model_id=/your_model_path/Qwen/Qwen3-VL-2B-Instruct-Add-Action-Tokens
tokens_file=new_tokens.txt

python add_special_tokens_to_qwen.py \
  --model-id "${source_model_id}" \
  --tokens-file "${tokens_file}" \
  --save-dir "${target_model_id}" \
  --init-strategy normal
```

`new_tokens.txt` should contain the special tokens, one per line, that you wish to add to the tokenizer.

## Arguments

- `--model-id`: Hugging Face model ID or path to a local model directory.
- `--save-dir`: Directory to save the updated model and tokenizer.
- `--tokens-file`: Path to your text file containing tokens to add.
- `--init-strategy`: Strategy to initialize embeddings for new tokens. Options: `avg`, `normal`, `zero`.
- `--as-special` / `--no-as-special`: Add as special tokens (default) or regular tokens.
- `--padding-side`: Padding direction, either `left` or `right`.
- `--device`: Device to run the script on: `cpu`, `cuda`, `mps`, or `auto`.

## Results

The script will:
1. Add the specified special tokens to the tokenizer.
2. Extend the model’s `nn.Embedding` layer to accommodate the new tokens.

**You can verify the effect as follows:**

```python
from transformers import AutoProcessor

action_seq = "<robot_action_0><robot_action_1><robot_action_2>"

# Load processor without new tokens:
processor_no_new = AutoProcessor.from_pretrained("/your_model_path/Qwen/Qwen3-VL-2B-Instruct")
token_ids = processor_no_new.tokenizer(action_seq)["input_ids"]
print("Tokenizer without new tokens:")
for token_id in token_ids:
    print(f"{token_id} -> {processor_no_new.tokenizer.decode(token_id)}")

# Load processor with new tokens:
processor_new = AutoProcessor.from_pretrained("/your_model_path/Qwen/Qwen3-VL-2B-Instruct-Add-Action-Tokens")
token_ids = processor_new.tokenizer(action_seq)["input_ids"]
print("Tokenizer with new tokens:")
for token_id in token_ids:
    print(f"{token_id} -> {processor_new.tokenizer.decode(token_id)}")
```

**Sample Output:**
```
Tokenizer without new tokens:
27 -> '<'
18247 -> 'robot'
7931 -> '_action'
62 -> '_'
15 -> '0'
1784 -> '><'
...        (truncated)
Tokenizer with new tokens:
151669 -> '<robot_action_0>'
151670 -> '<robot_action_1>'
151671 -> '<robot_action_2>'
```

## Acknowledgements

The core script is adapted from [starVLA](https://github.com/starVLA/starVLA/tree/starVLA/starVLA/model/modules/vlm/tools/add_qwen_special_tokens) — thanks to their impressive contribution!