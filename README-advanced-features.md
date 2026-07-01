# Advanced Features
This document provides an overview of several advanced features supported by this repository, such as training with multiple nodes, training model with a mixture of datasets, VQA data co-training, and more.

## Table of Contents
- [Command-Line Arguments of the Training Script](#command-Line-arguments-of-the-training-script)
- [Multiple nodes training](#multiple-nodes-training)
- [Training with a Mixture of Datasets](#training-with-a-mixture-of-datasets)
- [Training with VLA and VQA Datasets](#training-with-vla-and-vqa-datasets)
- [Extending VLA Datasets with Embodied CoT Annotations](#extending-vla-datasets-with-embodied-cot-annotations)
- [LoRA Training](#lora-training)

## Command-Line Arguments of the Training Script
Before diving into specific features, here is a comprehensive explanation of the command-line arguments accepted by the `train_vla.py` script:

### Model Paths

- **`--vlm_name_or_path`**  
  *Type:* `str`  
  Path to the pretrained Vision-Language Model (VLM) checkpoint.

- **`--action_expert_name_or_path`**  
  *Type:* `str`  
  Path to the pretrained action expert. If unset, a randomly initialized action expert will be used.

- **`--FAST_tokenizer_path`**  
  *Type:* `str`  
  Path to the pretrained FAST action tokenizer.

---

### Training Hyperparameters

- **`--per_device_train_batch_size`**  
  *Type:* `int`, *Default:* `8`  
  Batch size per GPU during training.

- **`--seed`**  
  *Type:* `int`, *Default:* `42`  
  Random seed for reproducibility.

- **`--epochs`**  
  *Type:* `int`, *Default:* `16`  
  Total number of training epochs.

- **`--save_ckpt_interval`**  
  *Type:* `int`, *Default:* `1`  
  Interval (in epochs) to save model checkpoints.

- **`--save_step_interval`**  
  *Type:* `int`, *Default:* `20000`  
  Interval (in steps) to save model checkpoints.

- **`--peak_learning_rate`**  
  *Type:* `float`, *Default:* `1e-5`  
  The peak learning rate during training when using a scheduler.

- **`--min_lr_rate`**  
  *Type:* `float`, *Default:* `0.1`  
  The minimum learning rate at the end of training, expressed as a fraction of the peak learning rate. For example, `0.1` means the final LR will be `peak_learning_rate * 0.1`.

- **`--tensorboard_log_dir`**  
  *Type:* `str`, *Default:* `"./outputs/train_logs/ZR-0"`  
  Directory where TensorBoard logs are saved.

- **`--output_ckpt_dir`**  
  *Type:* `str`, *Default:* `"./outputs/ckpts/ZR-0"`  
  Directory where model checkpoints are saved.

---

### Fine-tuning and Model Control

- **`--tune_vlm`**  
  *Action:* `store_true`  
  Whether to fine-tune the VLM.

- **`--tune_action_expert`**  
  *Action:* `store_true`  
  Whether to fine-tune the action expert (including projectors and DiT).

- **`--detach_vlm_outputs_for_action_expert`**  
  *Action:* `store_true`  
  If set, gradients from the action expert will not be back-propagated into the VLM. (Breaks the gradient flow from action expert to VLM.)

- **`--loss_type`**  
  *Type:* `str`, *Default:* `"vlm_and_action"`  
  Specifies which loss type(s) to optimize. Choices:
    - `vlm_and_action`: Jointly train both VLM and action expert losses.
    - `vlm`: Train only the VLM loss.
    - `action`: Train only the action expert loss.

- **`--vlm_loss_weight`**  
  *Type:* `float`, *Default:* `1.0`  
  Weight for the VLM loss when using joint loss.

- **`--action_expert_loss_weight`**  
  *Type:* `float`, *Default:* `1.0`  
  Weight for the action expert loss when using joint loss.

- **`--lr_scheduler`**  
  *Type:* `str`, *Default:* `"cosine"`  
  Learning rate scheduler type. Choices: `cosine`, `constant`.

- **`--resume_training`**  
  *Action:* `store_true`  
  Whether to resume training from a previous checkpoint. Restores model, optimizer, and LR scheduler states.

- **`--save_optimizer_and_lr_states`**  
  *Action:* `store_true`  
  Whether to save the optimizer and LR scheduler states alongside model checkpoints (needed for training resumption).

---

### LoRA (Low-Rank Adapters) Options

- **`--use_lora`**  
  *Action:* `store_true`  
  Whether to enable LoRA fine-tuning for VLM.

- **`--target_modules`**  
  *Type:* `str`, *Default:* `"gate_proj, up_proj, down_proj"`  
  Comma-separated names of modules for applying LoRA adapters.

- **`--r`**  
  *Type:* `int`, *Default:* `16`  
  LoRA attention dimension (the `rank` of the adapter).

- **`--lora_alpha`**  
  *Type:* `int`, *Default:* `32`  
  Scaling parameter for LoRA (typically double the value of `r`).

- **`--lora_dropout`**  
  *Type:* `float`, *Default:* `0.0`  
  Dropout probability to use within the LoRA layers.

---

### Dataset & DataLoader Options

- **`--dataset_entries`**  
  *Type:* `str`, *nargs='+', Required*  
  A list of dataset entries (space-separated) to use for training, e.g., `"bridge_orig_lerobot fractal20220817_data_lerobot libero_v21"`.

- **`--window_size`**  
  _Type:_ `int` &nbsp; *Default:* `1`  
  Specifies the sliding window size for historical image observations.  
  A value of `1` means the VLA model uses only the current frame as input;  
  values greater than `1` include the most recent `window_size` frames (including the current frame) as input, allowing the model to leverage temporal context from previous observations.

- **`--action_horizon`**  
  *Type:* `int`, *Default:* `32`  
  The length of the action chunk.

- **`--max_pad_state_and_action_length`**  
  *Type:* `int`, *Default:* `64`  
  Maximum dimension for padded state/action sequences.


## Multiple nodes training
We provide training scripts that support distributed multi-node training with DeepSpeed. For example, the following command demonstrates how to launch training across 2 nodes, each with 8 GPUs (i.e., 16 GPUs in total).

```sh
sh scripts/run_finetune_on_multiple_nodes.sh \
   --machine_rank $ROLE_INDEX \
   --main_process_ip $WORKER_HOST \
   --main_process_port $WORKER_PORT \
   --num_machines 2 \
   --num_gpu 16 \
   --base_model_path /your/path/to/ZR-0 \
   --fast_tokenizer_path /your/path/to/fast/tokenizer \
   --per_device_bs 16 \
   --epochs 32 \
   --save_ckpt_interval 8 \
   --peak_lr 1e-5 \
   --min_lr_rate 0.1 \
   --tensorboard_log_dir "./outputs/train_logs/ZR-0-multiple-nodes-example" \
   --output_ckpt_dir "./outputs/ckpts/ZR-0-multiple-nodes-example" \
   --dataset_entries "dataset_entry"
```

**Multi-Node Environment Variables**  
In this multi-node training example, the environment variables `ROLE_INDEX`, `WORKER_HOST`, and `WORKER_PORT` are **automatically injected by our internal cluster scheduler**.

If you are using a different cluster or launch system (e.g., Slurm, OpenMPI, torchrun, or other schedulers), these environment variable names may differ. In such cases, please modify the script accordingly to match your cluster’s distributed launch configuration.

## Training with a Mixture of Datasets

Our training scripts support mixing multiple datasets to train a single VLA model, enabling the development of generalist models capable of controlling diverse robot embodiments and performing a variety of tasks, like ZR-0 does.

To use multiple datasets, simply specify all desired entries in the `--dataset_entries` argument, separated by spaces:

```sh
sh scripts/run_finetune_on_single_node.sh \
   --base_model_path /your/path/to/ZR-0 \
   --fast_tokenizer_path /your/path/to/fast/tokenizer \
   --num_gpu 8 \
   --per_device_bs 16 \
   --epochs 32 \
   --save_ckpt_interval 8 \
   --peak_lr 1e-5 \
   --min_lr_rate 0.1 \
   --tensorboard_log_dir "./outputs/train_logs/ZR-0-multiple-datasets" \
   --output_ckpt_dir "./outputs/ckpts/ZR-0-multiple-datasets" \
   --dataset_entries "dataset_entry1 dataset_entry2 dataset_entry3 dataset_entry4"
```

> **Note:** Ensure that `dataset_entry1`, `dataset_entry2`, `dataset_entry3`, and `dataset_entry4` are all registered in your `dataset2feature.yaml` configuration file prior to launching training. This feature allows for scalable and flexible model training across heterogeneous data sources with minimal configuration overhead.

## Training with VLA and VQA Datasets

Our training pipeline supports mixing both Vision-Language-Action (VLA) and Vision Question Answering (VQA) datasets.  
- For VLA datasets, the model computes next-token prediction losses in the VLM component and flow-matching losses in the action expert component.
- For VQA datasets, only the next-token prediction loss is calculated in the VLM component.

### VQA Dataset Format Requirements

To work with our dataloader, VQA datasets should be provided in parquet format with the following structure:

- One column named `"json"` (with data type: JSON-format string), which contains the messages between the user and the assistant.
- Example for the `"json"` column:

  ```json
  [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "resized_height": 448,
          "resized_width": 448,
          "image_index": 0
        },
        {
          "type": "text",
          "text": "How many clubs are pictured on this card?"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "The card shown is a five of clubs. There are a total of ...."
    }
  ]
  ```

- **Image Handling:**  
  Unlike some official formats, we use `"image_index"` as a placeholder for images within the conversation. The actual image data is stored in separate columns named `"imageX"`, where `X` matches the `"image_index"`. Each image column contains the raw image bytes (e.g., PIL-encoded).  
  Our dataloader will automatically locate the corresponding images using the `"image_index"` and insert them into the appropriate positions within the JSON messages.

We provide an example VQA dataset in `demo_data/vqa_data`, which contains 50 data samples.

After preparing your VQA dataset, be sure to register it in the `dataset2feature.yaml` configuration file.  
For example:

```yaml
demo_data.vqa_data:
  dataset_path: demo_data/vqa_data
  dataset_type: vlm
  sample_ratio: 1.0
```

> **Note:** For VQA datasets, the `dataset_type` should be set to `vlm`, whereas VLA datasets use `vla`.

To use the VQA dataset for model training, simply include the corresponding entry name in the `--dataset_entries` argument.

## Extending VLA Datasets with Embodied CoT Annotations

We have extended the `LeRobotDataset` class to support storing Embodied Chain-of-Thought (ECoT) information. Specifically, three new columns are introduced in the parquet files:

- `future_sub_tasks`: (*string*, JSON-encoded list)
- `bbox`: (*string*, JSON-encoded dictionary)
- `cot`: (*string*, free-form text)

**Examples of these columns:**

- **`future_sub_tasks`:**
  ```json
  ["pick up the green apple", "place the green apple on the green plate"]
  ```

- **`bbox`:**
  ```json
  {
    "observation.images.cam_wrist": [
      {"bbox_2d": [0.588, 0.486, 0.76, 0.731], "label": "green apple"}
    ],
    "observation.images.cam_front": [
      {"bbox_2d": [0.329, 0.433, 0.381, 0.525], "label": "green apple"}
    ]
  }
  ```

> In many cases, multiple camera views are provided as input; however, grounding is not required for all views.  
To inform the model which views should be used for grounding, list the corresponding camera keys in the `meta/info.json` file. For example:
> ```json
> "grounding_camera_keys": [
>   "observation.images.cam_wrist",
>   "observation.images.cam_front"
> ]
> ```
> The `grounding_camera_keys` should match exactly the camera keys present in `bbox`.

- **`cot`:**
  ```
  Image shows a green apple and a banana on the table, with three plates available: blue, green, and pink. This task is not finished because neither the green apple nor the banana is on the green plate. The robot should pick up the green apple and place it on the green plate, then pick up the banana and place it on the green plate as well.
  ```

Manually annotating ECoT for every frame in a VLA dataset is impractical. To address this, we have developed a new framework—powered by VLMs—for *automatically* annotating ECoT in any VLA dataset (LeRobot v2 format). For more details, please refer to [ProcVLM](https://github.com/ProcVLM/ProcVLM).

**Note:**  
Once ECoT annotations are added to your VLA datasets, the training pipeline will automatically detect and use them.

**Serving with ECoT (subtask reasoning mode):**
```sh
python server.py \
    --dataset_entry xxx \
    --ckpt_dir /path/to/checkpoint \
    --inference_mode subtask_then_action \
    --port 8000
```

## LoRA Training (not recommended)

When training resources are constrained, full model fine-tuning may be impractical. To address this, our training scripts support parameter-efficient LoRA fine-tuning.

**Example:**
```sh
sh scripts/run_finetune_on_single_node_lora.sh \
   --base_model_path /your/path/to/ZR-0 \
   --fast_tokenizer_path /your/path/to/fast/tokenizer \
   --num_gpu 8 \
   --per_device_bs 16 \
   --epochs 32 \
   --save_ckpt_interval 8 \
   --peak_lr 1e-5 \
   --min_lr_rate 0.1 \
   --tensorboard_log_dir "./outputs/train_logs/ZR-0-lora-finetuning" \
   --output_ckpt_dir "./outputs/ckpts/ZR-0-lora-finetuning" \
   --dataset_entries "dataset_entry1 dataset_entry2 dataset_entry3" ...
```

You can configure LoRA-related hyperparameters including `--target_modules`, `--r`, `--lora_alpha`, and `--lora_dropout` in the `scripts/run_finetune_on_single_node_lora.sh` script to fit your requirements. In practice, however, this is not recommended, as LoRA fine-tuning may severely degrade performance. Full-parameter fine-tuning is strongly preferred instead.

> **Note:** In our scripts, only the VLM component is trained with LoRA, while the action expert module is still fully fine-tuned. Since the action expert contains only 5 million parameters, it does not significantly impact GPU memory usage and can be quickly adapted to new settings. This approach achieves the best balance between computational efficiency, GPU memory usage, and fine-tuning performance.

Before evaluation, make sure to merge the LoRA adapters into the base model by running:

```sh
python merge_lora_adapter.py --base_model_path /your/path/to/ZR-0 --peft_model_path ./outputs/ckpts/ZR-0-lora-finetuning/step-xxxx
```
