# LIBERO Evaluation Benchmark  – Evaluation Guide

**LIBERO** is a widely used benchmark for studying knowledge transfer and lifelong learning in robotic manipulation.  
It consists of multiple task suites—Spatial, Object, Goal, and Long-Horizon (10).

For more details about the benchmark, please refer to the [official LIBERO website](https://libero-project.github.io/main.html).

---

## Evaluation Results

To reproduce the following evaluation results, please use the **[ZR-0-LIBERO](xxx)** checkpoint, which is based on ZR-0 and was fine-tuned on the [LIBERO](https://huggingface.co/datasets/HuggingFaceVLA/libero) dataset.

| Model                     | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|---------------------------|:--------------:|:-------------:|:-----------:|:---------:|:-------:|
| π0.5 @ 30k (finetuned)    |     **98.8%**       |    98.2%       |   **98.0%**      |   92.4%    |  96.85%  |
| **ZR-0-LIBERO**   |     97.4%           |      **99.4%**     |   **98.0%**    |   **96.4%**    |   **97.8%**  |

---

## How to Evaluate ZR-0-LIBERO

### 1. Set up the evaluation environment

Please follow the instructions from the official [LIBERO GitHub repository](https://github.com/Lifelong-Robot-Learning/LIBERO) to install the simulation environment:

```sh
apt update
apt-get install libgl1 libosmesa6 libosmesa6-dev

conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install websockets msgpack tyro
# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
cd ..
```

> **Note:** The commands above are adapted from the official LIBERO repository.

---

### 2. Launch the server and evaluation client

Open two terminals:

#### Terminal 1 – Server

```sh
conda activate ZR-0
python server.py \
    --env_type demo_data.libero_v21 \
    --ckpt_dir /your/path/to/ZR-0-LIBERO \
    --port 8000
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

#### Terminal 2 – Client

```sh
conda activate libero
python -m evaluation.libero_eval.run_libero_eval \
    --args.replan_steps 10 \
    --args.task-suite-name libero_spatial \
    --args.port 8000
```

To evaluate different LIBERO task suites, set `--args.task-suite-name` to one of:

```text
libero_spatial | libero_object | libero_goal | libero_10
```

---

## Fine-Tuning ZR-0 on LIBERO

We provide ready-to-use scripts for fine-tuning **ZR-0** on the LIBERO dataset.

### 1. Download the training dataset
```sh
# download dataset
hf download HuggingFaceVLA/libero --repo-type=dataset --revision v2.1 --local-dir /your/path/to/libero
# calculate statistics
python calculate_global_stats.py --dataset_path /your/path/to/libero
```

Then register the downloaded dataset in `dataset2feature.yaml`.  
For example:

```yaml
libero_finetuning:
  dataset_path: /your/path/to/libero
  dataset_type: vla
  sample_ratio: 1.0
  use_quantile: true
```

---

### 2. Run the fine-tuning scripts
Fine-tune ZR-0 on your registered dataset using the provided script:

```sh
sh scripts/run_finetune_on_single_node.sh \
   --base_model_path /your/path/to/ZR-0 \
   --fast_tokenizer_path /your/path/to/fast/tokenizer \
   --num_gpu 8 \
   --per_device_bs 16 \
   --epochs 32 \
   --save_ckpt_interval 8 \
   --peak_lr 3e-5 \
   --min_lr 3e-6 \
   --tensorboard_log_dir "./outputs/train_logs/ZR-0-LIBERO-finetuning" \
   --output_ckpt_dir "./outputs/ckpts/ZR-0-LIBERO-finetuning" \
   --dataset_entries "libero_finetuning"
```

Adjust `--num_gpu` and `--per_device_bs` to match your available hardware resources. 