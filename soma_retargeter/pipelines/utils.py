# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, auto
from pathlib import Path

import newton
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.assets.usd as usd_utils


class SourceType(IntEnum):
    """Enumeration of supported source model types."""
    SOMA = auto()


class TargetType(IntEnum):
    """Enumeration of supported target model types."""
    UNITREE_G1 = auto()
    AGILE_ONE = auto()

_SOURCE_TYPE_TO_STR = {
    SourceType.SOMA : "soma"
}
_STR_TO_SOURCE_TYPE = {s : t for t, s in _SOURCE_TYPE_TO_STR.items()}

_TARGET_TYPE_TO_STR = {
    TargetType.UNITREE_G1 : "unitree_g1",
    TargetType.AGILE_ONE : "agile_one"
}
_STR_TO_TARGET_TYPE = {s : t for t, s in _TARGET_TYPE_TO_STR.items()}


def get_source_str_from_type(source: SourceType) -> str:
    """
    Get the string name associated with a given source type.

    Args:
        source (SourceType): The source type enum value.

    Returns:
        str: The string representation of the source type.
    """
    return _SOURCE_TYPE_TO_STR[source]


def get_source_type_from_str(source: str) -> SourceType:
    """
    Convert a string to its corresponding SourceType enum value.

    Args:
        source (str): The string representation of a source.

    Returns:
        SourceType: The corresponding source type enum.

    Raises:
        ValueError: If the provided string does not correspond to a valid source type.
    """
    try:
        return _STR_TO_SOURCE_TYPE[source]
    except KeyError:
        allowed = ", ".join(_STR_TO_SOURCE_TYPE.keys())
        raise ValueError(f"Unknown source type: [{source}]. Allowed values: {allowed}") from None


def get_target_str_from_type(target: TargetType) -> str:
    """
    Get the string name associated with a given target type.

    Args:
        target (TargetType): The target type enum value.

    Returns:
        str: The string representation of the target type.
    """
    return _TARGET_TYPE_TO_STR[target]


def get_target_type_from_str(target: str) -> TargetType:
    """
    Convert a string to its corresponding TargetType enum value.

    Args:
        target (str): The string representation of a target.

    Returns:
        TargetType: The corresponding target type enum.

    Raises:
        ValueError: If the provided string does not correspond to a valid target type.
    """
    try:
        return _STR_TO_TARGET_TYPE[target]
    except KeyError:
        allowed = ", ".join(_STR_TO_TARGET_TYPE.keys())
        raise ValueError(f"Unknown target type: [{target}]. Allowed values: {allowed}") from None


def get_source_model_mesh(source: SourceType, skeleton) -> dict:
    """
    Retrieve model mesh for a given source type.

    Args:
        source (SourceType): The source type for which properties should be retrieved.
        skeleton: The skeleton associated with the source model, used for loading the mesh.

    Returns:
        SkeletalMesh: The skeleton mesh for the given source type.

    Raises:
        ValueError: If the source type is not recognized.
    """
    if source == SourceType.SOMA:
        return usd_utils.load_skeletal_mesh_from_usd(
            str(io_utils.get_config_file('soma', 'soma_base_skel_minimal.usd')),
            skeleton,
            '/OUTPUT/c_geometry_grp',
            '/OUTPUT/c_skeleton_grp/Root')

    raise ValueError(f"Unknown source type {source}.")


def get_retargeter_config(source: SourceType, target: TargetType) -> dict:
    """
    Load the retargeter configuration between a specific source and target.

    Args:
        source (SourceType): The source type.
        target (TargetType): The target type.

    Returns:
        dict: The loaded JSON configuration for the retargeter.

    Raises:
        ValueError: If the source or target type is not supported.
    """
    if source != SourceType.SOMA:
        raise ValueError(f"Unknown source type [{source}] for target [{target}].")

    if target == TargetType.UNITREE_G1:
        config_dir = 'unitree_g1'
        filename = 'soma_to_g1_retargeter_config.json'
    elif target == TargetType.AGILE_ONE:
        config_dir = 'agile_one'
        filename = 'soma_to_ao_triaxial_retargeter_config.json'
    else:
        raise ValueError(f"Unknown target type [{target}].")

    return io_utils.load_json(io_utils.get_config_file(config_dir, filename))


def build_robot_builder(target: TargetType, retargeter_config: dict | None = None):
    """Build a Newton robot builder for a retargeting target."""
    builder = newton.ModelBuilder()
    if target == TargetType.UNITREE_G1:
        if retargeter_config is not None and retargeter_config.get("robot_mjcf_path"):
            mjcf_path = Path(retargeter_config["robot_mjcf_path"]).expanduser()
            if not mjcf_path.exists():
                raise FileNotFoundError(f"[ERROR]: Unitree G1 MJCF not found: {mjcf_path}")
        else:
            mjcf_path = newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml"
        builder.add_mjcf(mjcf_path)
    elif target == TargetType.AGILE_ONE:
        if retargeter_config is None or "robot_mjcf_path" not in retargeter_config:
            raise ValueError("[ERROR]: Agile One target requires robot_mjcf_path in retargeter config.")
        mjcf_path = Path(retargeter_config["robot_mjcf_path"]).expanduser()
        if not mjcf_path.exists():
            raise FileNotFoundError(f"[ERROR]: Agile One MJCF not found: {mjcf_path}")
        builder.add_mjcf(mjcf_path)
    else:
        raise ValueError(f"Unsupported robot type [{target}].")

    return builder
