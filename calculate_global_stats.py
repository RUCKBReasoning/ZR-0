import pandas as pd
import glob
from tqdm import tqdm
import numpy as np
import json
import argparse
from pathlib import Path
from lerobot.common.datasets.utils import calculate_global_stats

"""
This script computes global statistical metrics for the 'action' and 'observation.state' fields
across a LeRobot-formatted dataset. For each field, it aggregates samples from all available
parquet files and calculates mean, standard deviation, minimum, maximum, and the 1st/99th percentiles.
The resulting statistics are saved as a JSON file under '<dataset_path>/meta/stats.json'.
"""

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help="LeRobot dataset path")
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_option()
    calculate_global_stats(opt.dataset_path)