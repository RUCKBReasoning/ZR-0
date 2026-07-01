# ZR-0 RoboTwin Evaluation

This document describes how to evaluate **ZR-0** on the **RoboTwin** benchmark.

---

# 1. Environment Setup

## 1.1 Install System Dependencies

```bash
sudo apt update
sudo apt install -y libvulkan1 mesa-vulkan-drivers vulkan-tools
```

---

## 1.2 Create Conda Environment

```bash
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
```

Install RoboTwin:

```bash
bash script/_install.sh
```

---

## 1.3 Manual Installation (Optional)

If the installation script fails, install the dependencies manually.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch3D:

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

Install CuRobo:

```bash
cd envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
cd ../..
```

---

## 1.4 Download Assets

Download RoboTwin assets, texture libraries and robot embodiments.

```bash
bash script/_download_assets.sh
```

---

# 2. Configure Evaluation

All evaluation parameters are specified in

```text
evaluation/RoboTwin-main/policy/ZR0/deploy_policy.yml
```

Example:

```yaml
policy_name: ZR0

task_name: adjust_bottle
task_config: demo_randomized
ckpt_setting: demo_clean

seed: 0
instruction_type: unseen

host: "127.0.0.1"
port: 9000

max_episode_steps: 717
n_episodes: 100
n_action_steps: 16
```

The client loads **all evaluation settings** from this configuration file.

| Parameter         | Description                   |
| ----------------- | ----------------------------- |
| task_name         | RoboTwin task name            |
| task_config       | RoboTwin task configuration   |
| ckpt_setting      | Evaluation checkpoint setting |
| instruction_type  | Instruction type              |
| host              | ZR-0 server address           |
| port              | ZR-0 server port              |
| max_episode_steps | Maximum rollout length        |
| n_episodes        | Number of evaluation episodes |
| n_action_steps    | Action chunk size             |

> **Important:** `port` must be identical to the server port.

---

# 3. Launch ZR-0 Server

Open a new terminal.

```bash
conda activate ZR-0

python server.py \
    --dataset_entry demo_data.robotwin2.0-aloha-agilex \
    --ckpt_dir /path/to/ZR-0-RoboTwin \
    --port 9000 \
    --inference_mode direct_action
```


---

# 4. Launch RoboTwin Evaluation Client

Open another terminal.

```bash
conda activate RoboTwin

cd evaluation/RoboTwin-main

python script/eval_policy_client_zr0.py \
    --config policy/ZR0/deploy_policy.yml
```

The client will

1. Create the RoboTwin environment.
2. Read all evaluation parameters from `deploy_policy.yml`.
3. Send observations to the ZR-0 server.
4. Receive an action chunk.
5. Execute the actions inside RoboTwin.
6. Repeat until the episode terminates.

---

# 5. Evaluating Other Tasks

To evaluate another RoboTwin task, only modify
`policy/ZR0/deploy_policy.yml`.

For example,

```yaml
task_name: beat_block_hammer
task_config: demo_randomized

host: "127.0.0.1"
port: 9001

max_episode_steps: 400
n_episodes: 100
n_action_steps: 16
```

Then launch the server using the same port:

```bash
python server.py \
    --dataset_entry demo_data.robotwin2.0-aloha-agilex \
    --ckpt_dir /path/to/ZR-0-RoboTwin \
    --port 9001 \
    --inference_mode direct_action
```

Run the evaluation client:

```bash
python script/eval_policy_client_zr0.py \
    --config policy/ZR0/deploy_policy.yml
```

No additional command-line arguments are required.


# 6. Notes

* The server is responsible for loading the ZR-0 model.
* The evaluation client does **not** load any model checkpoint.
* Images are transmitted at their original RoboTwin camera resolution without resizing, cropping, padding, or rotation.
* `n_action_steps` should match the action horizon used during training.
* `max_episode_steps` determines the maximum rollout horizon for each evaluation episode.
* `n_episodes` specifies the number of evaluation episodes.
* If any required field is missing from `deploy_policy.yml`, the client will terminate with an error.
