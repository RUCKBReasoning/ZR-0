# Demo Data

Note: These datasets are **incomplete** and intended **only for demonstration purposes**.

When launching the server, you need to initialize the `LeRobotDatasetMetadata` class, which requires four key metadata files: `episodes.jsonl`, `info.json`, `stats.json`, and `tasks.jsonl`.

To enable out-of-the-box evaluation of ZR-0 without the need to download the full datasets, we have extracted these metadata files from the original training sets of LIBERO, DROID, Robotwin 2.0 (aloha agilex randomized), Bridge, Fractal, RoboChallenge Table30, and robocasa GR1 tabletop tasks.

All demo entries are **pre-registered** in the `dataset2feature.yaml` configuration file, with their names starting with `demo_data.`.

## Supported Demo Datasets

**LIBERO**
```
demo_data.libero_v21
```
**DROID**
```
demo_data.droid_1.0.1_lerobot
```
**Robotwin 2.0 (aloha agilex)**
```
demo_data.robotwin2.0-aloha-agilex
```
**Robocasa GR1 Tabletop Tasks**
```
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PnPWineToCabinetClose
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PnPMilkToMicrowaveClose
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PnPPotatoToMicrowaveClose
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PnPCupToDrawerClose
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PnPBottleToCabinetClose
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PnPCanToDrawerClose
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA
demo_data.robocasa_gr1_tabletop_tasks.gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA
```

## Example Usage

For example, to launch a ZR-0 server for DROID using demo data (no need to download full real data), simply run:

```sh
conda activate ZR-0
python server.py \
    --env_type demo_data.droid_1.0.1_lerobot \
    --ckpt_dir /your/path/to/ZR-0-Preview \
    --port 8000 \
    --use_ecot
```
