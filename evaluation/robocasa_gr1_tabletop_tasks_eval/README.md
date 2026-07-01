# RoboCasa GR1 Tabletop Tasks – Evaluation Guide

RoboCasa GR1 Tabletop Tasks is a comprehensive simulation benchmark for generalist robotic manipulation in tabletop scenarios.
Built on the modular [RoboCasa](https://github.com/robocasa/robocasa) framework, it focuses on the GR-1 humanoid robot and includes diverse household manipulation environments, assets, and tooling for policy evaluation in real-world-inspired tasks.

For full benchmark details, see the [official repository](https://github.com/robocasa/robocasa-gr1-tabletop-tasks).

## Evaluation Results

To reproduce the results, use the checkpoint [ZR-0-Robocasa-GR1](), fine-tuned on the [PhysicalAI-Robotics-GR00T-Teleop-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim) dataset.

| task (env_name) | ZR-0-Robocasa-GR1 | dataset_entry | max_episode_steps |
| ----------------------- | -------------------------------- | ------------- | ----------------- |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env | 85% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA | 717 |
| gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env | 40% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PnPWineToCabinetClose | 1940 |
| gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 81% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA | 1002 |
| gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 81% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA | 705 |
| gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env | 74% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA | 787 |
| gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env | 45% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PnPMilkToMicrowaveClose | 1483 |
| gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env | 89% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA | 2380 |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env | 82% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA | 743 |
| gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 85% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA | 1057 |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env | 92% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA | 667 |
| gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env | 78% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA | 681 |
| gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 81% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA | 804 |
| gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env | 82% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA | 518 |
| gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env | 59% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PnPPotatoToMicrowaveClose | 1721 |
| gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env | 88% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA | 796 |
| gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env | 20% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PnPCupToDrawerClose | 1236 |
| gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env | 50% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA | 1237 |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env | 79% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA | 534 |
| gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env | 74% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA | 716 |
| gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env | 39% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PnPBottleToCabinetClose | 1141 |
| gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env | 80% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA | 921 |
| gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env | 47% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PnPCanToDrawerClose | 973 |
| gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env | 46% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA | 1402 |
| gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env | 87% | demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA | 678 |


---

## How to Evaluate ZR-0-Robocasa-GR1

### 1. Environment Setup

Follow the instructions below to prepare the RoboCasa GR-1 Tabletop simulation environment.

```sh
apt update
apt-get install libegl1-mesa-dev libglu1-mesa

# 1. Set up conda environment
conda create -c conda-forge -n robocasa python=3.10
conda activate robocasa
pip install websockets

# 2. Clone and install robosuite
wget https://github.com/ARISE-Initiative/robosuite/archive/refs/tags/v1.5.1.zip -O robosuite-1.5.1.zip
unzip robosuite-1.5.1.zip
mv robosuite-1.5.1 robosuite
pip install -e robosuite

# 3. Clone and install robocasa-gr1-tabletop-tasks
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git
pip install -e robocasa-gr1-tabletop-tasks
pip install numpy==1.26.4 imageio[ffmpeg]

# 4. Download assets
cd robocasa-gr1-tabletop-tasks
python robocasa/scripts/download_tabletop_assets.py -y
```

> **Note:** Commands are adapted from the official repository instructions. 

---

### Launch Server & Client for Evaluation

Take `PosttrainPnPNovelFromCuttingboardToPotSplitA` as an example.

#### Terminal 1 – Server

```sh
conda activate ZR-0
python server.py --dataset_entry demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA \
    --ckpt_dir /your/path/to/ZR-0-Robocasa-GR1 \
    --port 9000 \
    --use_ecot
```

This starts the policy server on port 9000.

#### Terminal 2 – Client

```sh
conda activate robocasa
python -m evaluation.robocasa_gr1_tabletop_tasks_eval.run_robocasa_eval \
    --max_episode_steps 717 \
    --n_episodes 100 \
    --policy_host "127.0.0.1" \
    --policy_port 9000 \
    --env_name "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env" \
    --n_envs 1 \
    --n_action_steps 8
```

🔄 Evaluating Other Tasks
To run evaluations for different tasks:
- Change `--dataset_entry` in the server command.
- Change `--max_episode_steps` and `--env_name` in the client command.
- Use the values from the evaluation results table above.
