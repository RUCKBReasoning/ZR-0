import torch
import hashlib
import time
import io
import json
import random
import orjson

from PIL import Image
from datasets import load_dataset
from torchvision.transforms import ToPILImage
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.normalization import min_max_norm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from torch.utils.data import ConcatDataset, DataLoader
from utils.constants import DATASET2FEATURE

def pad_2d_to_max_length(tensor_2d, max_pad_length=64, pad_value=0.0):
    """
    Pads each row of a 2D tensor to max_length with pad_value and returns the corresponding mask.
    
    Args:
        tensor_2d (torch.Tensor): shape (N, L), L <= max_length
        max_length (int): pad length
        pad_value (int or float): value for padding
    
    Returns:
        padded_tensor (torch.Tensor): shape (N, max_length)
        mask (torch.Tensor): shape (N, max_length), 1=real, 0=pad
    """
    N, L = tensor_2d.shape
    if L >= max_pad_length:
        return tensor_2d[:, :max_pad_length], torch.ones((N, max_pad_length), dtype=torch.long, device=tensor_2d.device)
    pad_len = max_pad_length - L
    pad_tensor = torch.full((N, pad_len), pad_value, dtype=tensor_2d.dtype, device=tensor_2d.device)
    padded_tensor = torch.cat([tensor_2d, pad_tensor], dim=1)
    mask = torch.cat([
        torch.ones((N, L), dtype=torch.long, device=tensor_2d.device),
        torch.zeros((N, pad_len), dtype=torch.long, device=tensor_2d.device)
    ], dim=1)
    return padded_tensor, mask

def prepare_action_expert_inputs_cpu(data, stats, max_pad_length, use_quantile, action_horizon=None, action_dim=None):
    action_expert_inputs = {}
    # a scalar (shape is [])
    # if not isinstance(data["embodiment_id"], torch.Tensor):
    #     data["embodiment_id"] = torch.tensor(data["embodiment_id"])
    # action_expert_inputs["embodiment_id"] = data["embodiment_id"].to(torch.long)

    # 1. per-dim min-max norm; 2. dim-size padding + return mask
    data["norm_state_wo_pad"] = min_max_norm(data["observation.state"], stats["observation.state"], use_quantile) # (1, state_dim)
    padded_states, state_masks = pad_2d_to_max_length(data["norm_state_wo_pad"], max_pad_length)
    action_expert_inputs["observation.state"] = padded_states  # (1, max_pad_length)
    action_expert_inputs["state_mask"] = state_masks           # (1, max_pad_length)

    if "action" in data:
        data["norm_action_wo_pad"] = min_max_norm(data["action"], stats["action"], use_quantile) # (action_horizon, action_dim)
        padded_actions, action_masks = pad_2d_to_max_length(data["norm_action_wo_pad"], max_pad_length)
        action_expert_inputs["action"] = padded_actions    # (action_horizon, max_pad_length)
        action_expert_inputs["action_mask"] = action_masks # (action_horizon, max_pad_length)
    else:
        infer_action_mask = torch.zeros((action_horizon, max_pad_length), dtype=torch.long)
        infer_action_mask[:, :action_dim] = 1
        action_expert_inputs["infer_action_mask"] = infer_action_mask # (action_horizon, max_pad_length)

    return action_expert_inputs

def find_subtensor_index(tensor, sub_tensor):
    """
    Find the first occurrence of the 1D sub-string sub_tensor in the 1D tensor tensor, 
        and return -1 if it is not found
    """
    n = sub_tensor.numel()
    if n == 0 or n > tensor.numel():
        return -1
    # create sliding windows of length n on the original tensor
    windows = tensor.unfold(0, n, 1)          # [L-n+1, n]
    # compare each window with sub_tensor
    matches = (windows == sub_tensor).all(dim=1)
    # find indices where a full match occurs
    idxs = torch.where(matches)[0]
    return idxs[0].item() if idxs.numel() > 0 else -1

def mask_prompt_loss_1d(labels_1d: torch.Tensor, assistant_start_ids, pad_token_id):
    """
    Mask a single sample (1D) labels:
    - Set all pad_token_ids to -100.
    - Set the content before and including the assistant starting tokens to -100.
    """
    # pad -> -100
    if pad_token_id is not None:
        labels_1d[labels_1d == pad_token_id] = -100
    # mask prompt（assistant starting token sequence: [151644, 77091, 198]）
    ast = torch.tensor(list(assistant_start_ids), dtype=labels_1d.dtype, device=labels_1d.device)
    start_idx = find_subtensor_index(labels_1d, ast)
    if start_idx != -1:
        labels_1d[:start_idx + len(ast)] = -100
    return labels_1d

def convert_fast_tokens_to_vlm_action_seq(fast_tokens: list[int]) -> str:
    """
    convert fast action tokens to VLM's (special) action tokens.
    E.g., action tokens [0, 343, 745] are converted to the string "<robot_action_0><robot_action_343><robot_action_745>"
    """
    return ''.join([f"<robot_action_{token}>" for token in fast_tokens])

def tokenize_vision_language_inputs(msg, process_mode, processor, max_length=1200):
    # for inference, the generation prompt should be added.
    add_generation_prompt = (process_mode != "train")
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=add_generation_prompt)

    # vision input
    image_inputs, video_inputs = process_vision_info(msg, image_patch_size=16) # image_patch_size, 14 for Qwen2.5-VL and 16 for Qwen3-VL

    st = time.time()
    if process_mode == "train":
        processor.tokenizer.padding_side = "right"
        vl_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            do_resize=False,
            return_tensors="pt",
            padding_side="right"
        )
    else:
        # don't perform padding during evaluation
        vl_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=True,
            do_resize=False,
            return_tensors="pt"
        )
    
    # remove the batch dim of input_ids and attention_mask
    vl_inputs["input_ids"] = vl_inputs["input_ids"][0] # (seq_len,)
    vl_inputs["attention_mask"] = vl_inputs["attention_mask"][0] # (seq_len,)
    # pixel_values and image_grid_thw do not contain the batch dim
    vl_inputs["pixel_values"] = vl_inputs["pixel_values"] # (pixel_values_width, pixel_values_height)
    vl_inputs["image_grid_thw"] = vl_inputs["image_grid_thw"] # (image_grid_thw_width, image_grid_thw_height)

    if process_mode == "train":
        vl_inputs["labels"] = vl_inputs["input_ids"].clone() # (seq_len,)

        # mask prompt loss. 151643 is the default pad id of Qwen2.5
        pad_id = getattr(processor.tokenizer, "pad_token_id", 151643)
        masked_labels = mask_prompt_loss_1d(
            vl_inputs["labels"],
            assistant_start_ids=(151644, 77091, 198), # generation prompt: <|im_start|>assistant\n
            pad_token_id=pad_id
        )
        vl_inputs["labels"] = masked_labels
    return vl_inputs

def prepare_qwen_vl_inputs_cpu(
    data: dict,
    camera_keys: list[str],
    grounding_camera_keys: list[str],
    processor: AutoProcessor,
    process_mode: str,
    prompt_suffix: str,
    fast_tokenizer: AutoProcessor
):
    to_pil = ToPILImage()
    resized_height = 224
    resized_width = 224

    # store contextual images (history + current)
    context_images = []
    if len(data[camera_keys[0]].shape) == 4:
        # for each camera, the lerobot dataset gives several historical observations and one current observations
        # For example, torch.Size([4, 3, 480, 640]) means that it contains 3 historical obs and 1 current obs
        obs_num = data[camera_keys[0]].shape[0]
        for obs_idx in range(obs_num):
            for camera_key in camera_keys:
                cam_key = camera_key if obs_idx == obs_num - 1 else f"historical-observation-{obs_idx}.{camera_key}"
                context_images.append([cam_key, to_pil(data[camera_key][obs_idx])])
    elif len(data[camera_keys[0]].shape) == 3:
        # for each camera, the lerobot dataset only gives one current observation
        for camera_key in camera_keys:
            context_images.append([camera_key, to_pil(data[camera_key])])
    else:
        raise ValueError(
            f"Expected image shape with 3 or 4 dimensions for data['{camera_keys[0]}'], "
            f"but got shape with {len(data[camera_keys[0]].shape)} dimensions: {data[camera_keys[0]].shape}."
        )

    prompt = {"role": "user", "content": []}
    for cam_key, img in context_images:
        prompt["content"].append({"type": "text", "text": cam_key})
        prompt["content"].append({"type": "image", "image": img, "resized_height": resized_height, "resized_width": resized_width})

    prompt["content"].append({"type": "text", "text": "<TASK> " + data["task"].strip() + " <\TASK>\n" + prompt_suffix})
    msg = [prompt]

    if process_mode == "train":
        discrete_action_tokens = fast_tokenizer(data["norm_action_wo_pad"])
        discrete_action_tokens_seq = convert_fast_tokens_to_vlm_action_seq(discrete_action_tokens[0])

        ecot_str = data["embodied_cot"] # load ECoT JSON string
        ecot_json = json.loads(ecot_str)
        if prompt_suffix == "":
            # add discrete action tokens
            ecot_json["Discrete Action Tokens"] = discrete_action_tokens_seq
            # output seq is a complete embodied chain-of-thought reasoning path
            output_seq = json.dumps(ecot_json, indent=2, ensure_ascii=False)
            # output_seq = "<place holder>"
        elif prompt_suffix == "Sub task:":
            # output seq is a sub task
            if "To-do Actions" in ecot_json:
                output_seq = "<SUB_TASK>" + ecot_json["To-do Actions"][0] + "</SUB_TASK>"
            else:
                output_seq = "<SUB_TASK>done</SUB_TASK>"
        
        msg.append({"role": "assistant", "content": output_seq})
    
    vl_inputs = tokenize_vision_language_inputs(msg, process_mode, processor)
    
    return vl_inputs

def deterministic_test_time_n_action_steps(action_horizon: int, ds_id: int, local_idx: int, epoch: int) -> int:
    # Avoiding the random salt effects of Python's default hash with stable hashing
    key = f"{ds_id}-{local_idx}-{epoch}".encode("utf-8")
    h = hashlib.sha1(key).hexdigest()
    # Take the first 8 bytes as an integer
    val = int(h[:8], 16)
    return 1 + (val % action_horizon)

class StreamingLeRobotSampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        lerobot_dataset: LeRobotDataset, # a LeRobot Dataset instance
        use_quantile: bool,
        sample_ratio: float,
        processor: AutoProcessor,
        fast_tokenizer: AutoProcessor,
        window_size: int,
        action_horizon: int,
        process_mode: str,
        dataset_id: int,
        max_pad_state_and_action_length: int,
        dataset_entry: str
    ):
        self.ds = lerobot_dataset
        self.use_quantile = use_quantile
        self.sample_ratio = sample_ratio
        self.processor = processor
        self.fast_tokenizer = fast_tokenizer
        self.window_size = max(1, window_size)
        self.action_horizon = action_horizon
        self.process_mode = process_mode
        self.dataset_id = dataset_id
        self.max_pad_state_and_action_length = max_pad_state_and_action_length
        self.dataset_entry = dataset_entry

        self.fps = self.ds.meta.fps
        self.camera_keys = self.ds.meta.camera_keys
        self.stats = self.ds.meta.stats
        self.grounding_camera_keys = self.ds.meta.grounding_camera_keys

        self._epoch = 0  # used for deterministic randomness
        self._init_sampled_indices()

        self.ecot_supported = self.is_ecot_enhanced(self.ds.hf_dataset)
        if self.ecot_supported:
            print(f"{dataset_entry} is ECoT-enhanced.")
        else:
            print(f"{dataset_entry} is not ECoT-enhanced.")

    def is_ecot_enhanced(self, hf_dataset) -> bool:
        """
        Check if a Lerobot dataset is ECoT-enhanced 
        (contains required annotation columns).
        
        Args:
            hf_dataset: HuggingFace Dataset object or its .features mapping.
        
        Returns:
            bool: True if dataset contains all required ECoT columns, False otherwise.
        """
        # Required columns for ECoT-enhanced datasets
        ecot_required_columns = {"future_sub_tasks", "bbox", "cot"}
        
        # Extract dataset features mapping
        features = getattr(hf_dataset, "features", hf_dataset)
        
        # Check if the dataset contains all required columns
        dataset_columns = set(features.keys())
        return ecot_required_columns.issubset(dataset_columns)

    def _init_sampled_indices(self):
        total = len(self.ds)
        n_samples = max(1, int(total * self.sample_ratio))
        rng = random.Random(42 + self._epoch)
        self.subset_indices = rng.sample(range(total), n_samples)

    def __len__(self):
        return len(self.subset_indices)

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        self._init_sampled_indices()

    def _obtain_delta_timestamps(self, test_time_n_action_steps: int):
        delta_timestamps = dict()

        # During inference, we set `test_time_n_action_steps` to an integer between [1, action_horizon]. 
        # Thus, during training, the observation images in the prompt should come from the delta indices: [-(N-1)*test_time_n_action_steps, ..., -1*test_time_n_action_steps, 0],
        # where N is `window_size` and `test_time_n_action_steps` is a random integer between 1 and action_horizon
        slide_window_observation_delta_indices = list(range(self.window_size))[::-1]
        slide_window_observation_delta_indices = [-idx * test_time_n_action_steps for idx in slide_window_observation_delta_indices]

        if self.window_size > 1:
            # set camera's delta timestamps
            for camera_key in self.camera_keys:
                delta_timestamps[camera_key] = [index / self.fps for index in slide_window_observation_delta_indices]

        # set action's delta timestamps
        action_delta_indices = list(range(self.action_horizon))
        delta_timestamps["action"] = [idx / self.fps for idx in action_delta_indices]
        return delta_timestamps

    def __getitem__(self, idx: int):
        orig_idx = self.subset_indices[idx]
        tries = 0
        max_tries = 30
        while tries < max_tries:
            try:
                idx = orig_idx if tries == 0 else random.choice(self.subset_indices)
                # sample a `n_action_steps` value for current data (epoch stable)
                test_time_n_action_steps = deterministic_test_time_n_action_steps(
                    self.action_horizon, ds_id=self.dataset_id, local_idx=idx, epoch=self._epoch
                )
                delta_timestamps = self._obtain_delta_timestamps(test_time_n_action_steps)
                # fetch raw data from Lerobot Dataset
                data_sample = self.ds.getitem_with_delta_timestamps(idx, delta_timestamps)
    
                # online pre-processing
                action_expert_inputs = prepare_action_expert_inputs_cpu(
                    data_sample, self.stats, self.max_pad_state_and_action_length, self.use_quantile
                )
                prompt_suffix = ""
                sub_task_flag = 0
                if self.ecot_supported and random.random() < 0.1:
                    # If the dataset is ECoT-enhanced, randomly select 10% of the samples
                    # to follow the Two-Stage inference strategy:
                    # 1. VLM first generates sub-task description from the prompt.
                    # 2. Action expert then conditions on (prompt + sub-task) to predict continuous actions.
                    prompt_suffix = "Sub task:"
                    sub_task_flag = 1

                vl_inputs = prepare_qwen_vl_inputs_cpu(
                    data=data_sample,
                    camera_keys=self.camera_keys,
                    grounding_camera_keys=self.grounding_camera_keys,
                    processor=self.processor,
                    process_mode=self.process_mode,
                    prompt_suffix=prompt_suffix,
                    fast_tokenizer=self.fast_tokenizer
                )
                preprocessed_data_sample = {**action_expert_inputs, **vl_inputs, "sub_task_flag": torch.tensor(sub_task_flag)}
                return preprocessed_data_sample
            except Exception as e:
                tries += 1
                # print(f"retry times: {tries}")
        return None

class HFDatasetWrapper:
    def __init__(self, path, split="train"):
        self.ds = load_dataset(
            "parquet",
            data_dir=path,
            split=split,
            keep_in_memory=False,
        )

    def get(self, idx):
        return self.ds[idx]

    def __len__(self):
        return len(self.ds)

class VQADataset(torch.utils.data.Dataset):
    """
    Parquet + image-bytes VQA Dataset.
    Safe for:
      - DataLoader(num_workers > 0)
      - DDP / multi-node
    """

    def __init__(
        self,
        dataset_path: str,
        sample_ratio: float,
        processor: AutoProcessor,
        process_mode: str = "train",
        max_pad_state_and_action_length: int = 64,
        action_horizon: int = 32,
    ):
        self.processor = processor
        self.process_mode = process_mode
        self.max_pad_state_and_action_length = max_pad_state_and_action_length
        self.action_horizon = action_horizon
        self.sample_ratio = sample_ratio

        # parquet dataset
        self.hf_ds = HFDatasetWrapper(f"{dataset_path}/data")

        self._epoch = 0
        self._init_sampled_indices()

    def _init_sampled_indices(self):
        total = len(self.hf_ds)
        n_samples = max(1, int(total * self.sample_ratio))
        rng = random.Random(42 + self._epoch)
        self.subset_indices = rng.sample(range(total), n_samples)

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        self._init_sampled_indices()

    def __len__(self):
        return len(self.subset_indices)

    def generate_dummy_action_expert_inputs(self):
        return {
            "observation.state": torch.randn(
                1, self.max_pad_state_and_action_length
            ),
            "state_mask": torch.ones(
                1, self.max_pad_state_and_action_length
            ),
            "action": torch.randn(
                self.action_horizon, self.max_pad_state_and_action_length
            ),
            "action_mask": torch.zeros(
                self.action_horizon, self.max_pad_state_and_action_length
            ),
        }

    def _decode_image(self, img_bytes):
        # PNG image bytes -> PIL.Image
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def _inject_images(self, msg, image_bytes_dict):
        """
        msg: parsed JSON
        image_bytes_dict: {0: bytes, 1: bytes, 2: bytes}
        """
        for turn in msg:
            content = turn.get("content")
            if not isinstance(content, list):
                continue

            new_content = []
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "image"
                    and "image_index" in item
                ):
                    idx = item["image_index"]
                    img_bytes = image_bytes_dict[idx]
                    img = self._decode_image(img_bytes)

                    new_item = dict(item)
                    new_item.pop("image_index", None)
                    new_item["image"] = img
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            turn["content"] = new_content
        return msg

    def __getitem__(self, idx: int):
        try:
            real_idx = self.subset_indices[idx]
            sample = self.hf_ds.get(real_idx)

            image_bytes_dict = {
                int(key[5:]): val
                for key, val in sample.items()
                if key.startswith("image") and val is not None
            }

            # 1. parse JSON
            msg = orjson.loads(sample["json"])
            # 2. inject images
            msg = self._inject_images(msg, image_bytes_dict)
            # 3. tokenize / process
            vl_inputs = tokenize_vision_language_inputs(msg, self.process_mode, self.processor)
            # 4. dummy expert inputs
            action_expert_inputs = self.generate_dummy_action_expert_inputs()

            return {**action_expert_inputs, **vl_inputs, "sub_task_flag": torch.tensor(0)}

        except Exception as e:
            print("a VQA data point raises an error:", str(e))
            return None

def build_concat_streaming_dataset(
    dataset_entries: list[str],
    model_name_or_path: str,
    fast_tokenizer_path: str,
    window_size: int,
    action_horizon: int,
    accelerator,
    process_mode: str = "train",
    max_pad_state_and_action_length: int = 64
):
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)

    datasets = []
    for dataset_id, dataset_entry in enumerate(dataset_entries):
        print(f"loading {dataset_entry}")
        dataset_path = DATASET2FEATURE[dataset_entry]["dataset_path"]
        sample_ratio = DATASET2FEATURE[dataset_entry]["sample_ratio"]
        dataset_type = DATASET2FEATURE[dataset_entry]["dataset_type"]

        if dataset_type == "vla":
            lerobot_dataset = LeRobotDataset(repo_id=dataset_path.split("/")[-1], root=dataset_path)
            use_quantile = DATASET2FEATURE[dataset_entry]["use_quantile"]
            ds = StreamingLeRobotSampleDataset(
                lerobot_dataset=lerobot_dataset,
                use_quantile=use_quantile,
                sample_ratio=sample_ratio,
                processor=processor,
                fast_tokenizer=fast_tokenizer,
                window_size=window_size,
                action_horizon=action_horizon,
                process_mode=process_mode,
                dataset_id=dataset_id,
                max_pad_state_and_action_length=max_pad_state_and_action_length,
                dataset_entry=dataset_entry
            )
        elif dataset_type == "vlm":
            ds = VQADataset(
                dataset_path=dataset_path,
                sample_ratio=sample_ratio,
                processor=processor,
                process_mode=process_mode,
                max_pad_state_and_action_length=max_pad_state_and_action_length,
                action_horizon=action_horizon,
            )
        else:
            raise ValueError(
                f"Dataset '{dataset_path}' is not configured. "
                "To use this dataset, please add its configuration to `utils/constants.py`. "
                "Refer to the existing dataset entries for the required format."
            )
        if accelerator is not None:
            accelerator.wait_for_everyone()
        datasets.append(ds)

    concat = ConcatDataset(datasets)
    return concat

def custom_collate_fn(batch):
    """
    Generic collate_fn that stacks all keys except for certain keys,
    for which it uses cat along the first dimension.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    cat_keys = ['pixel_values', 'image_grid_thw']
    keys = batch[0].keys()
    result = {}
    for key in keys:
        items = [item[key] for item in batch]
        if key in cat_keys:
            result[key] = torch.cat(items, dim=0)
        else:
            result[key] = torch.stack(items, dim=0)

    if 'input_ids' in result and 'attention_mask' in result:
        attention_mask = result['attention_mask']  # [B, L]
        valid_lens = attention_mask.sum(dim=1).long()  # [B]
        max_len = int(valid_lens.max().item()) + 1
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in result:
                result[key] = result[key][:, :max_len].contiguous()

    return result

def create_dataloader_for_concat(
    concat_dataset,
    batch_size_per_device: int,
    num_workers: int = 8,
    prefetch_factor: int = 2
):
    dataloader = DataLoader(
        concat_dataset, batch_size=batch_size_per_device, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=False,
        prefetch_factor=prefetch_factor, collate_fn=custom_collate_fn
    )

    return dataloader
