# constants.py
import yaml
import os

yaml_path = os.path.join(os.path.dirname(__file__), "../dataset2feature.yaml")
with open(yaml_path, "r") as f:
    DATASET2FEATURE = yaml.safe_load(f)