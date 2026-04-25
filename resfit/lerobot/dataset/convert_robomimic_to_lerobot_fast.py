# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

#!/usr/bin/env python3
"""
Script to convert Robomimic HDF5 trajectory data into LeRobot dataset format.

Usage:
  python convert_robomimic_to_lerobot.py --dataset /path/to/robomimic_dataset.hdf5 \\
    --output_dir /path/to/lerobot_dataset --repo_id [your-hf-account]/dataset-name

~/personal_abhi/Thesis-Docs/Reward_Func/Robomimic/robomimic/datasets/lift/mh/robomimic-mh-lift-image_v15_dense.hdf5
Dataset name: poolvarine/robomimic-mh-lift-image-dense

python convert_robomimic_to_lerobot.py \
    --dataset ~/personal_abhi/Thesis-Docs/Reward_Func/Robomimic/robomimic/datasets/lift/mh/robomimic-mh-lift-image_v15_dense.hdf5 \
    --output_dir ~/personal_abhi/.cache/huggingface/lerobot/poolvarine/robomimic-mh-lift-image-dense \
    --repo_id poolvarine/robomimic-mh-lift-image-dense

python convert_robomimic_to_lerobot_fast.py \
  --dataset ~/personal_abhi/Thesis-Docs/Reward_Func/Robomimic/robomimic/datasets/square/mh/robomimic-mh-square-image_v15_dense.hdf5 \
  --output_dir ~/.cache/huggingface/lerobot/poolvarine/robomimic-mh-square-image-dense \
  --repo_id poolvarine/robomimic-mh-square-image-dense

"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_info
from tqdm import tqdm
from huggingface_hub import DatasetCard
import math
import tempfile
import multiprocessing as mp
from datasets import load_dataset, concatenate_datasets, disable_caching

# Disable HuggingFace caching so we don't blow up your drive with temp files
disable_caching()

# Import our LeRobot upload utilities
from lerobot_upload_utils import upload_with_retry

import shutil


def get_env_metadata_from_dataset(dataset_path: str) -> dict[str, Any]:
    """
    Retrieves env metadata from robomimic dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:
            'env_name': name of environment
            'type': type of environment
            'env_kwargs': dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    return env_meta  # noqa: RET504


def get_dataset_trajectories(
    dataset_path: str,
    filter_key: str | None = None,
    max_episodes: int | None = None,
    exclude_episodes: list[int] | None = None,
) -> list[str]:
    """
    Get list of trajectory keys from the dataset.

    Args:
        dataset_path (str): path to dataset
        filter_key (str, optional): filter key to select subset of trajectories
        max_episodes (int, optional): maximum number of episodes to include
        exclude_episodes (list[int], optional): list of 1-indexed episode numbers to exclude

    Returns:
        List[str]: sorted list of trajectory keys (e.g., ['demo_0', 'demo_1', ...])
    """
    with h5py.File(dataset_path, "r") as f:
        if filter_key is not None:
            demos = [
                elem.decode("utf-8") if isinstance(elem, bytes) else elem for elem in np.array(f[f"mask/{filter_key}"])
            ]
        else:
            demos = list(f["data"].keys())

    # Sort in increasing number order
    inds = np.argsort([int(elem[5:]) for elem in demos])  # demo_N -> N
    demos = [demos[i] for i in inds]

    # Filter out excluded episodes (convert from 1-indexed to 0-indexed)
    if exclude_episodes is not None:
        # Convert 1-indexed episode numbers to 0-indexed for filtering
        demos = [demo for i, demo in enumerate(demos, start=1) if i not in exclude_episodes]

    if max_episodes is not None and max_episodes > 0:
        demos = demos[:max_episodes]

    return demos


def analyze_dataset_structure(dataset_path: str, demo_keys: list[str]) -> tuple[dict[str, Any], list[str], int]:
    """
    Analyze the dataset structure to determine features and image keys.

    Args:
        dataset_path (str): path to dataset
        demo_keys (List[str]): list of demonstration keys

    Returns:
        Tuple containing:
        - features dict for LeRobot dataset
        - list of image observation keys
        - fps (frames per second)
    """
    with h5py.File(dataset_path, "r") as f:
        # Get first demo for structure analysis
        first_demo = demo_keys[0]
        demo_grp = f[f"data/{first_demo}"]

        # Get basic trajectory info (some keys might not exist)
        actions = demo_grp["actions"][()]

        # Analyze observation structure
        obs_grp = demo_grp["obs"]
        obs_keys = list(obs_grp.keys())

        # Get environment metadata to determine expected image keys
        env_meta = json.loads(f["data"].attrs["env_args"])
        env_name = env_meta.get("env_name", "")

        # Define expected image keys based on environment name
        expected_image_keys = get_expected_image_keys(env_name)

        # Define expected low_dim_keys based on environment name
        expected_low_dim_keys = get_expected_low_dim_keys(env_name)

        # Separate image and non-image observations
        image_keys = []
        state_keys = []

        for key in obs_keys:
            obs_data = obs_grp[key]
            if len(obs_data.shape) == 4:  # (T, H, W, C) - image data
                image_keys.append(key)
            else:  # non-image observations
                state_keys.append(key)

        # Filter image keys to only include expected ones
        if expected_image_keys:
            image_keys = [key for key in image_keys if key in expected_image_keys]

        # Filter state keys to only include expected low_dim_keys
        if expected_low_dim_keys:
            state_keys = [key for key in state_keys if key in expected_low_dim_keys]

        # Build features dict
        features = {}

        # Action feature
        features["action"] = {
            "dtype": "float32",
            "shape": actions.shape[1:],
            "names": get_action_names(env_name, actions.shape[1]),
        }

        # Done feature - create from episode length if dones doesn't exist
        features["next.done"] = {
            "dtype": "bool",
            "shape": (1,),
            "names": ["done"],
        }

        # Reward feature (optional — present when HDF5 has rewards)
        if "rewards" in demo_grp:
            print("Dataset contains rewards, including 'next.reward' feature.")
            features["next.reward"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward"],
            }

        # State observations (concatenate only expected low_dim_keys)
        if state_keys:
            state_components = []
            state_names = []
            for key in state_keys:
                if key in obs_grp:  # Make sure the key exists in the dataset
                    obs_data = obs_grp[key][0]  # First timestep
                    if obs_data.ndim == 0:  # scalar
                        obs_data = np.array([obs_data])
                    state_components.append(obs_data)
                    if obs_data.shape == ():
                        state_names.append(key)
                    else:
                        state_names.extend([f"{key}_{i}" for i in range(len(obs_data))])
                else:
                    print(f"Warning: Expected low_dim_key '{key}' not found in dataset observations")

            if state_components:
                concatenated_state = np.concatenate(state_components)
                features["observation.state"] = {
                    "dtype": "float32",
                    "shape": concatenated_state.shape,
                    "names": state_names,
                }

        # Image features
        for img_key in image_keys:
            img_data = obs_grp[img_key]
            # Remove '_image' suffix if present for cleaner naming
            clean_key = img_key.replace("_image", "")
            features[f"observation.images.{clean_key}"] = {
                "dtype": "video",
                "shape": img_data.shape[1:],  # (H, W, C)
                "names": ["height", "width", "channel"],
            }

    # Default fps - robomimic datasets typically don't store fps info
    fps = 20

    return features, image_keys, fps


def get_expected_image_keys(env_name: str) -> list[str]:
    """
    Get expected image keys based on environment name from generate_training_config.py

    Args:
        env_name (str): Environment name

    Returns:
        List[str]: List of expected image observation keys
    """
    # Mapping based on generate_training_config.py
    panda_image_keys = [
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot1_eye_in_hand_image",
    ]

    panda_transport_image_keys = [
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot1_eye_in_hand_image",
        "shouldercamera0_image",
        "shouldercamera1_image",
    ]

    humanoid_image_keys = [
        "agentview_image",
        "robot0_eye_in_left_hand_image",
        "robot0_eye_in_right_hand_image",
    ]

    humanoid_can_sort_image_keys = [
        "frontview_image",
        "robot0_eye_in_left_hand_image",
        "robot0_eye_in_right_hand_image",
    ]

    # Map environment names to image keys
    env_lower = env_name.lower()
    if "transport" in env_lower:
        return panda_transport_image_keys
    if "cansort" in env_lower or "can_sort" in env_lower:  # Check both variants
        return humanoid_can_sort_image_keys
    if any(task in env_lower for task in ["pouring", "coffee"]):
        return humanoid_image_keys

    # Default to panda keys for other tasks
    return panda_image_keys


def get_expected_low_dim_keys(env_name: str) -> list[str]:
    """
    Get expected low_dim_keys based on environment name from generate_training_config.py

    Args:
        env_name (str): Environment name

    Returns:
        List[str]: List of expected low-dimensional observation keys
    """
    # Mapping based on generate_training_config.py
    panda_low_dim_keys = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot1_eef_pos",
        "robot1_eef_quat",
        "robot1_gripper_qpos",
    ]

    humanoid_low_dim_keys = [
        "robot0_right_eef_pos",
        "robot0_right_eef_quat",
        "robot0_right_gripper_qpos",
        "robot0_left_eef_pos",
        "robot0_left_eef_quat",
        "robot0_left_gripper_qpos",
    ]

    # Map environment names to low_dim_keys
    env_lower = env_name.lower()
    if any(task in env_lower for task in ["pouring", "coffee"]) or "cansort" in env_lower or "can_sort" in env_lower:
        return humanoid_low_dim_keys
    # Default to panda keys for other tasks
    return panda_low_dim_keys


def get_action_names(env_name: str, action_dim: int) -> list[str]:
    """
    Get meaningful action names based on environment name and action dimension.

    Based on the action annotations from the DexMimicGen wrapper:
    - 0-2: Right wrist Δpos Cartesian x/y/z offset
    - 3-5: Right wrist Δrot Axis-angle components rx,ry,rz
    - 6-11: Right Inspire-hand joints
    - 12-14: Left wrist Δpos Cartesian x/y/z offset
    - 15-17: Left wrist Δrot Axis-angle components for left EE
    - 18-23: Left Inspire-hand joints

    Args:
        env_name (str): Environment name
        action_dim (int): Total action dimension

    Returns:
        List[str]: List of meaningful action names
    """
    env_lower = env_name.lower()

    # Check if this is a dexterous hand environment (PandaDex or similar)
    is_dexterous = any(task in env_lower for task in ["lifttray", "boxcleanup", "drawercleanup"])

    # Check if this is a humanoid environment
    is_humanoid = any(task in env_lower for task in ["pouring", "coffee", "cansort", "can_sort"])

    # Check if this is a single-arm environment
    # Include both underscored and compact variants to handle values like "ToolHang".
    single_arm_tasks = [
        "lift",
        "can",
        "pickplacecan",
        "square",
        "nutassemblysquare",
        "threading",
        "tool_hang",
        "toolhang",
    ]
    is_single_arm = any(task in env_lower for task in single_arm_tasks)

    if is_dexterous and action_dim == 24:
        # 24-DOF dexterous bimanual actions
        return [
            "right_wrist_delta_pos_x",
            "right_wrist_delta_pos_y",
            "right_wrist_delta_pos_z",  # 0-2: Right wrist position
            "right_wrist_delta_rot_rx",
            "right_wrist_delta_rot_ry",
            "right_wrist_delta_rot_rz",  # 3-5: Right wrist rotation
            "right_thumb_flexion",
            "right_thumb_opposition",
            "right_index_flexion",  # 6-8: Right thumb and index
            "right_middle_flexion",
            "right_ring_flexion",
            "right_pinky_flexion",  # 9-11: Right middle, ring, pinky
            "left_wrist_delta_pos_x",
            "left_wrist_delta_pos_y",
            "left_wrist_delta_pos_z",  # 12-14: Left wrist position
            "left_wrist_delta_rot_rx",
            "left_wrist_delta_rot_ry",
            "left_wrist_delta_rot_rz",  # 15-17: Left wrist rotation
            "left_thumb_flexion",
            "left_thumb_opposition",
            "left_index_flexion",  # 18-20: Left thumb and index
            "left_middle_flexion",
            "left_ring_flexion",
            "left_pinky_flexion",  # 21-23: Left middle, ring, pinky
        ]
    if is_humanoid and action_dim == 24:
        # 24-DOF humanoid bimanual actions (similar structure but for humanoid)
        return [
            "right_eef_delta_pos_x",
            "right_eef_delta_pos_y",
            "right_eef_delta_pos_z",  # 0-2: Right end-effector position
            "right_eef_delta_rot_rx",
            "right_eef_delta_rot_ry",
            "right_eef_delta_rot_rz",  # 3-5: Right end-effector rotation
            "right_thumb_flexion",
            "right_thumb_opposition",
            "right_index_flexion",  # 6-8: Right thumb and index
            "right_middle_flexion",
            "right_ring_flexion",
            "right_pinky_flexion",  # 9-11: Right middle, ring, pinky
            "left_eef_delta_pos_x",
            "left_eef_delta_pos_y",
            "left_eef_delta_pos_z",  # 12-14: Left end-effector position
            "left_eef_delta_rot_rx",
            "left_eef_delta_rot_ry",
            "left_eef_delta_rot_rz",  # 15-17: Left end-effector rotation
            "left_thumb_flexion",
            "left_thumb_opposition",
            "left_index_flexion",  # 18-20: Left thumb and index
            "left_middle_flexion",
            "left_ring_flexion",
            "left_pinky_flexion",  # 21-23: Left middle, ring, pinky
        ]
    if action_dim == 14:
        # Standard bimanual Panda actions (7 DOF per arm)
        return [
            "robot0_eef_delta_pos_x",
            "robot0_eef_delta_pos_y",
            "robot0_eef_delta_pos_z",  # 0-2: Robot0 end-effector position
            "robot0_eef_delta_rot_rx",
            "robot0_eef_delta_rot_ry",
            "robot0_eef_delta_rot_rz",  # 3-5: Robot0 end-effector rotation
            "robot0_gripper_action",  # 6: Robot0 gripper
            "robot1_eef_delta_pos_x",
            "robot1_eef_delta_pos_y",
            "robot1_eef_delta_pos_z",  # 7-9: Robot1 end-effector position
            "robot1_eef_delta_rot_rx",
            "robot1_eef_delta_rot_ry",
            "robot1_eef_delta_rot_rz",  # 10-12: Robot1 end-effector rotation
            "robot1_gripper_action",  # 13: Robot1 gripper
        ]
    if is_single_arm and action_dim == 7:
        # Single-arm Panda actions (7 DOF) - for Lift, Can, Square, Threading tasks
        return [
            "eef_delta_pos_x",
            "eef_delta_pos_y",
            "eef_delta_pos_z",  # 0-2: End-effector position
            "eef_delta_rot_rx",
            "eef_delta_rot_ry",
            "eef_delta_rot_rz",  # 3-5: End-effector rotation (axis-angle)
            "gripper_action",  # 6: Gripper open/close
        ]
    if action_dim == 20:
        # Potential 20-DOF configuration (could be dual-arm with different setup)
        # This might be for environments with different gripper configurations
        return [
            "robot0_eef_delta_pos_x",
            "robot0_eef_delta_pos_y",
            "robot0_eef_delta_pos_z",  # 0-2: Robot0 position
            "robot0_eef_delta_rot_rx",
            "robot0_eef_delta_rot_ry",
            "robot0_eef_delta_rot_rz",  # 3-5: Robot0 rotation
            "robot0_finger_0",
            "robot0_finger_1",
            "robot0_finger_2",
            "robot0_finger_3",  # 6-9: Robot0 multi-finger gripper
            "robot1_eef_delta_pos_x",
            "robot1_eef_delta_pos_y",
            "robot1_eef_delta_pos_z",  # 10-12: Robot1 position
            "robot1_eef_delta_rot_rx",
            "robot1_eef_delta_rot_ry",
            "robot1_eef_delta_rot_rz",  # 13-15: Robot1 rotation
            "robot1_finger_0",
            "robot1_finger_1",
            "robot1_finger_2",
            "robot1_finger_3",  # 16-19: Robot1 multi-finger gripper
        ]
    if action_dim == 12:
        # 12-DOF configuration (could be dual-arm with 6 DOF each)
        return [
            "robot0_eef_delta_pos_x",
            "robot0_eef_delta_pos_y",
            "robot0_eef_delta_pos_z",  # 0-2: Robot0 position
            "robot0_eef_delta_rot_rx",
            "robot0_eef_delta_rot_ry",
            "robot0_eef_delta_rot_rz",  # 3-5: Robot0 rotation
            "robot1_eef_delta_pos_x",
            "robot1_eef_delta_pos_y",
            "robot1_eef_delta_pos_z",  # 6-8: Robot1 position
            "robot1_eef_delta_rot_rx",
            "robot1_eef_delta_rot_ry",
            "robot1_eef_delta_rot_rz",  # 9-11: Robot1 rotation
        ]
    if action_dim == 6:
        # Single-arm 6-DOF (position + rotation, no gripper)
        return [
            "eef_delta_pos_x",
            "eef_delta_pos_y",
            "eef_delta_pos_z",  # 0-2: End-effector position
            "eef_delta_rot_rx",
            "eef_delta_rot_ry",
            "eef_delta_rot_rz",  # 3-5: End-effector rotation
        ]
    if action_dim == 4:
        # Simplified 4-DOF (position + gripper)
        return [
            "eef_delta_pos_x",
            "eef_delta_pos_y",
            "eef_delta_pos_z",  # 0-2: End-effector position
            "gripper_action",  # 3: Gripper
        ]
    # Fail if action dimension is not supported
    raise ValueError(f"Action dimension {action_dim} not supported")


# ==============================================================================
# MULTIPROCESSING WORKERS & METADATA HANDLING
# ==============================================================================

def _process_shard_worker(shard_id: int, demo_keys_chunk: list, dataset_path: str, env_name: str, features: dict, image_keys: list, fps: int, tmp_dir: str) -> tuple:
    """Worker generates a mini LeRobot Dataset (a Shard) with resumability and safeguards."""
    shard_dir = Path(tmp_dir) / f"shard_{shard_id}"
    hf_data_path = shard_dir / "hf_data"
    
    # --- RESUMABILITY CHECK ---
    if hf_data_path.exists() and (shard_dir / "videos").exists():
        video_paths = [str(p) for p in shard_dir.rglob("*.mp4")]
        if len(video_paths) > 0:
            try:
                # Actually try to load it! If it's an empty/corrupt folder from a previous crash, this fails.
                from datasets import load_from_disk
                _test_ds = load_from_disk(str(hf_data_path))
                if len(_test_ds) > 0:
                    return shard_id, str(shard_dir), video_paths
            except Exception:
                pass # The folder is corrupted. We will fall through and nuke it below!

    # If it's a partial/corrupted run, nuke it and start fresh
    if shard_dir.exists():
        shutil.rmtree(shard_dir)

    dataset = LeRobotDataset.create(
        repo_id=f"shard_{shard_id}",
        fps=fps,
        root=str(shard_dir),
        robot_type="robomimic",
        features=features,
        use_videos=True, 
    )

    expected_total_frames = 0

    with h5py.File(dataset_path, "r", libver="latest", swmr=True) as f:
        for demo_key in demo_keys_chunk:
            demo_grp = f[f"data/{demo_key}"]
            actions = demo_grp["actions"][()]
            num_frames = len(actions)
            expected_total_frames += num_frames

            dones = demo_grp["dones"][()] if "dones" in demo_grp else np.zeros(num_frames, dtype=bool)
            if "dones" not in demo_grp: dones[-1] = True

            rewards = demo_grp["rewards"][()] if "rewards" in demo_grp else None
            obs_grp = demo_grp["obs"]

            concatenated_state = None
            if "observation.state" in features:
                expected_low_dim_keys = get_expected_low_dim_keys(env_name)
                state_components = []
                for key in expected_low_dim_keys:
                    if key in obs_grp and len(obs_grp[key].shape) != 4:
                        obs_data = obs_grp[key][()]
                        if obs_data.ndim == 1: obs_data = obs_data[:, np.newaxis]
                        state_components.append(obs_data)
                if state_components:
                    concatenated_state = np.concatenate(state_components, axis=1)

            CHUNK_SIZE = 64
            img_key_map = {k: f"observation.images.{k.replace('_image', '')}" for k in image_keys}

            for start_idx in range(0, num_frames, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, num_frames)
                chunk_images = {}
                for img_key in image_keys:
                    img_data = obs_grp[img_key][start_idx:end_idx]
                    if img_data.dtype != np.uint8:
                        if img_data.max() <= 1.0: img_data = (img_data * 255).clip(0, 255).astype(np.uint8, copy=False)
                        else: img_data = img_data.astype(np.uint8, copy=False)
                    chunk_images[img_key] = img_data

                for i, frame_idx in enumerate(range(start_idx, end_idx)):
                    frame_data = {
                        "action": np.array(actions[frame_idx], dtype=np.float32).reshape(features["action"]["shape"]),
                        "next.done": np.array([dones[frame_idx]], dtype=bool).reshape(features["next.done"]["shape"]),
                    }
                    if rewards is not None and "next.reward" in features:
                        frame_data["next.reward"] = np.array([rewards[frame_idx]], dtype=np.float32).reshape(features["next.reward"]["shape"])
                    if concatenated_state is not None:
                        frame_data["observation.state"] = np.array(concatenated_state[frame_idx], dtype=np.float32).reshape(features["observation.state"]["shape"])
                    for img_key in image_keys:
                        feature_key = img_key_map[img_key]
                        if feature_key in features:
                            frame_data[feature_key] = chunk_images[img_key][i]

                    dataset.add_frame(frame=frame_data, task=env_name)

            dataset.save_episode()

    # --- CORRUPTION SAFEGUARDS ---
    # 1. Strip LeRobot's un-saveable Python functions, then force it to compile to disk
    clean_hf_dataset = dataset.hf_dataset.with_transform(None).with_format(None)
    clean_hf_dataset.save_to_disk(str(hf_data_path))
    
    # 2. Assert that it actually worked and isn't empty
    assert len(clean_hf_dataset) > 0, f"CRITICAL: Shard {shard_id} compiled an empty HuggingFace dataset!"
    
    # 3. Assert that LeRobot didn't drop any frames
    assert len(clean_hf_dataset) == expected_total_frames, f"CRITICAL: Frame mismatch in Shard {shard_id}! Expected {expected_total_frames}, got {len(clean_hf_dataset)}."
    # Explicitly return the video paths to save the Main Thread from searching
    video_paths = [str(p) for p in shard_dir.rglob("*.mp4")]

    return shard_id, str(shard_dir), video_paths

def _run_worker_unpack(args):
    """Helper to unpack tuple args for imap_unordered"""
    return _process_shard_worker(*args)

def update_shard_metadata(batch, ep_offset, frame_offset, img_keys):
    """HuggingFace Dataset mapping function: Safely rewrites all internal Parquet file paths & IDs"""
    batch["episode_index"] = [x + ep_offset for x in batch["episode_index"]]
    batch["frame_index"] = [x + frame_offset for x in batch["frame_index"]]
    batch["index"] = [x + frame_offset for x in batch["index"]]

    for img_key in img_keys:
        clean_key = img_key.replace("_image", "")
        feat_key = f"observation.images.{clean_key}"
        if feat_key in batch:
            for i in range(len(batch[feat_key])):
                old_path = batch[feat_key][i]["path"]
                base_name = os.path.basename(old_path)
                try:
                    old_ep_i = int(base_name.replace("episode_", "").replace(".mp4", ""))
                    new_ep_i = old_ep_i + ep_offset
                    new_name = f"episode_{new_ep_i:06d}.mp4"
                    batch[feat_key][i]["path"] = old_path.replace(base_name, new_name)
                except ValueError:
                    pass
    return batch

# ==============================================================================
# NEW CONVERSION FUNCTION (MAIN CONTROLLER & MERGER)
# ==============================================================================

def convert_robomimic_to_lerobot(
    dataset_path: str,
    output_dir: str,
    repo_id: str | None = None,
    filter_key: str | None = None,
    train_ratio: float = 1.0,
    max_episodes: int | None = None,
    exclude_episodes: list[int] | None = None,
):
    output_dir = Path(output_dir)
    
    # Make the temp folder unique to this specific dataset's filename
    dataset_filename = Path(dataset_path).stem
    tmp_dir = Path(tempfile.gettempdir()) / f"lerobot_shards_{dataset_filename}"
    # We do NOT wipe tmp_dir here anymore! Keeping it allows resumability.

    env_meta = get_env_metadata_from_dataset(dataset_path)
    env_name = env_meta.get("env_name", "RobomimicEnv")

    demo_keys = get_dataset_trajectories(dataset_path, filter_key, max_episodes, exclude_episodes)
    print(f"Found {len(demo_keys)} trajectories to convert.")

    features, image_keys, fps = analyze_dataset_structure(dataset_path, demo_keys)

    # Hardcode cores here if you want, e.g., num_workers = 16
    num_workers = max(1, mp.cpu_count() - 4)
    chunk_size = math.ceil(len(demo_keys) / num_workers)
    chunks = [demo_keys[i:i + chunk_size] for i in range(0, len(demo_keys), chunk_size)]

    tmp_dir.mkdir(parents=True, exist_ok=True)
    worker_args = [(i, chunk, dataset_path, env_name, features, image_keys, fps, str(tmp_dir)) for i, chunk in enumerate(chunks)]

    print(f"\n🚀 Launching {num_workers} workers. Shards are resumable if interrupted...")
    shard_results = {}
    with mp.Pool(processes=num_workers) as pool:
        for shard_id, shard_dir, video_paths in tqdm(pool.imap_unordered(_run_worker_unpack, worker_args), total=len(worker_args), desc="Encoding Shards"):
            shard_results[shard_id] = {"dir": shard_dir, "videos": video_paths}

    ordered_shard_data = [shard_results[i] for i in range(len(chunks))]

    print("\n📦 Merging shards and rewriting metadata paths...")
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    final_data_dir = output_dir / "data"
    final_video_dir = output_dir / "videos" / "chunk-000"
    final_meta_dir = output_dir / "meta"
    
    final_data_dir.mkdir(parents=True, exist_ok=True)
    final_video_dir.mkdir(parents=True, exist_ok=True)
    final_meta_dir.mkdir(parents=True, exist_ok=True)

    global_ep_idx = 0
    global_frame_idx = 0
    all_hf_datasets = []
    all_episodes_meta = []
    all_episodes_stats = []
    base_info = None

    for shard_id, shard_info_dict in enumerate(tqdm(ordered_shard_data, desc="Merging Parquet Files")):
        shard_dir = Path(shard_info_dict["dir"])
        video_paths = shard_info_dict["videos"]

        if shard_id == 0:
            with open(shard_dir / "meta" / "info.json", "r") as f:
                base_info = json.load(f)
            
            # Copy the tasks file from the first shard
            tasks_file = shard_dir / "meta" / "tasks.jsonl"
            if tasks_file.exists():
                shutil.copy(tasks_file, final_meta_dir / "tasks.jsonl")

        with open(shard_dir / "meta" / "info.json", "r") as f:
            shard_info = json.load(f)
        num_episodes = shard_info["total_episodes"]
        num_frames = shard_info["total_frames"]

        episodes_jsonl_path = shard_dir / "meta" / "episodes.jsonl"
        if episodes_jsonl_path.exists():
            with open(episodes_jsonl_path, "r") as f:
                for line in f:
                    ep_meta = json.loads(line)
                    ep_meta["episode_index"] += global_ep_idx
                    ep_meta["data_path"] = "data/train-00000-of-00001.parquet"
                    all_episodes_meta.append(ep_meta)

        # Mathematically merge the true episode statistics
        episodes_stats_path = shard_dir / "meta" / "episodes_stats.jsonl"
        if episodes_stats_path.exists():
            with open(episodes_stats_path, "r") as f:
                for line in f:
                    stats_meta = json.loads(line)
                    if "episode_index" in stats_meta:
                        stats_meta["episode_index"] += global_ep_idx
                    all_episodes_stats.append(stats_meta)

        # Move videos explicitly provided by the worker
        for mp4_file_str in video_paths:
            mp4_file = Path(mp4_file_str)
            base_name = mp4_file.name
            try:
                old_ep_i = int(base_name.replace("episode_", "").replace(".mp4", ""))
                new_ep_i = old_ep_i + global_ep_idx
                new_name = f"episode_{new_ep_i:06d}.mp4"
                
                rel_path = mp4_file.parent.relative_to(shard_dir)
                target_dir = output_dir / rel_path
                target_dir.mkdir(parents=True, exist_ok=True)
                
                shutil.copy(mp4_file, target_dir / new_name)
            except ValueError:
                pass

        # Load the raw HuggingFace Dataset from disk
        hf_data_path = shard_dir / "hf_data"
        if hf_data_path.exists():
            from datasets import load_from_disk
            shard_ds = load_from_disk(str(hf_data_path))
            shard_ds = shard_ds.map(
                update_shard_metadata, 
                batched=True, 
                batch_size=1000, 
                fn_kwargs={"ep_offset": global_ep_idx, "frame_offset": global_frame_idx, "img_keys": image_keys},
                desc=f"Updating Paths (Shard {shard_id})"
            )
            all_hf_datasets.append(shard_ds)

        global_ep_idx += num_episodes
        global_frame_idx += num_frames
        
        # Now that we've successfully merged it, we can safely delete the shard
        shutil.rmtree(shard_dir)

    print("\n💾 Finalizing HuggingFace Dataset Object...")
    final_ds = concatenate_datasets(all_hf_datasets)
    final_ds.to_parquet(final_data_dir / "train-00000-of-00001.parquet")

    if base_info:
        base_info["total_episodes"] = global_ep_idx
        base_info["total_frames"] = global_frame_idx
        if train_ratio < 1.0:
            num_train = int(global_ep_idx * train_ratio)
            base_info["splits"] = {"train": f"0:{num_train}", "test": f"{num_train}:{global_ep_idx}"}
        with open(final_meta_dir / "info.json", "w") as f:
            json.dump(base_info, f, indent=4)

    if all_episodes_meta:
        with open(final_meta_dir / "episodes.jsonl", "w") as f:
            for ep_meta in all_episodes_meta:
                f.write(json.dumps(ep_meta) + "\n")

    # Save the mathematically correct merged statistics
    if all_episodes_stats:
        with open(final_meta_dir / "episodes_stats.jsonl", "w") as f:
            for stats_meta in all_episodes_stats:
                f.write(json.dumps(stats_meta) + "\n")
    else:
        # Fallback if shards didn't have it
        (final_meta_dir / "episodes_stats.jsonl").touch()

    # Create a blank stats.json just in case
    with open(final_meta_dir / "stats.json", "w") as f:
        json.dump({}, f)

    # Final cleanup of the master temp folder
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n✅ Successfully created Parallelized LeRobot dataset at {output_dir}")
    
    # Handle upload if repo_id is provided
    if repo_id:
        print(f"\n🚀 Uploading to HuggingFace Hub: {repo_id}...")
        
        # Define primary upload function (original LeRobot approach)
        def primary_upload():
            try:
                print("Computing dataset statistics (consolidating)...")
                final_repo_id = repo_id if repo_id else Path(output_dir).name
                final_lerobot_dataset = LeRobotDataset(repo_id=final_repo_id, root=str(output_dir))
                final_lerobot_dataset.consolidate()
                
                print(f"Uploading to HuggingFace Hub: {repo_id}...")
                final_lerobot_dataset.push_to_hub()
                print(f"✅ Dataset successfully uploaded to {repo_id}")
                return True
            except Exception as e:
                print(f"\n❌ Primary upload failed: {type(e).__name__}")
                print(f"   Error: {str(e)[:150]}")
                return False
        
        # Try primary method first, fall back to salvage upload
        success = upload_with_retry(
            output_dir=Path(output_dir),
            repo_id=repo_id,
            primary_upload_fn=primary_upload
        )
        
        if success:
            print(f"\n🎉 Dataset upload completed successfully!")
            print(f"   View at: https://huggingface.co/datasets/{repo_id}")
        else:
            print(f"\n⚠️  Dataset upload encountered issues, but files were prepared.")
            print(f"   Try uploading manually or check the dataset structure.")
        
        # Load and return the dataset if we get here
        try:
            final_lerobot_dataset = LeRobotDataset(repo_id=repo_id, root=str(output_dir))
            return final_lerobot_dataset
        except:
            print("Warning: Could not reload dataset after upload")
            return None
    else:
        print(f"\nSkipping upload (repo_id not provided)")
        print(f"Dataset location: {output_dir}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert robomimic HDF5 trajectories to LeRobot format.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to robomimic HDF5 dataset file",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the LeRobot dataset")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace repository ID for uploading the dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="Filter key to select subset of trajectories from the dataset",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="Ratio of trajectories to assign to train split (0.0-1.0). Remaining trajectories assigned to test split.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to convert. Converts all if not specified.",
    )
    parser.add_argument(
        "--exclude-episodes",
        type=int,
        nargs="*",
        default=None,
        help="List of 1-indexed episode numbers to exclude from conversion (e.g., --exclude-episodes 1 3 5)",
    )

    args = parser.parse_args()

    convert_robomimic_to_lerobot(
        args.dataset,
        args.output_dir,
        args.repo_id,
        args.filter_key,
        args.train_ratio,
        args.max_episodes,
        args.exclude_episodes,
    )
