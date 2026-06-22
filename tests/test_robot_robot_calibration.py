import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_app_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def test_build_robot_robot_pose_pair_uses_source_and_target_semantic_maps(tmp_path: Path):
    build_module = load_app_module("build_robot_robot_pose_pair", "app/build_robot_robot_pose_pair.py")

    source_pose = write_json(
        tmp_path / "g1_forward_reach.json",
        {
            "schema": "robot_pose.v1",
            "robot_type": "unitree_g1",
            "robot_mjcf": "/tmp/g1.xml",
            "robot_body_transforms": {
                "pelvis": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                "left_wrist_yaw_link": [0.4, 0.2, 1.1, 0.0, 0.0, 0.0, 1.0],
            },
        },
    )
    target_pose = write_json(
        tmp_path / "ao_forward_reach.json",
        {
            "schema": "robot_pose.v1",
            "robot_type": "agile_one",
            "robot_mjcf": "/tmp/ao.xml",
            "robot_body_transforms": {
                "pelvis_link": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "left_wrist_pitch_link": [0.5, 0.3, 1.2, 0.0, 0.0, 0.0, 1.0],
            },
        },
    )
    retargeter = write_json(
        tmp_path / "g1_to_ao.json",
        {
            "source_type": "unitree_g1",
            "target_type": "agile_one",
            "source_ik_map": {
                "Hips": {"t_body": "pelvis", "r_body": "pelvis"},
                "LeftHand": {"t_body": "left_wrist_yaw_link", "r_body": "left_wrist_yaw_link"},
            },
            "ik_map": {
                "Hips": {"t_body": "pelvis_link", "r_body": "pelvis_link"},
                "LeftHand": {"t_body": "left_wrist_pitch_link", "r_body": "left_wrist_pitch_link"},
            },
        },
    )

    pair = build_module.build_robot_robot_pose_pair(
        source_pose_path=source_pose,
        target_pose_path=target_pose,
        base_retargeter_config_path=retargeter,
        pose_name="forward_reach",
    )

    assert pair["schema"] == "robot_robot_pose_pair.v1"
    assert pair["pose_name"] == "forward_reach"
    assert pair["source_robot_type"] == "unitree_g1"
    assert pair["target_robot_type"] == "agile_one"
    assert pair["source_ik_targets"]["LeftHand"]["t_body"] == "left_wrist_yaw_link"
    assert pair["target_ik_targets"]["LeftHand"]["t_body"] == "left_wrist_pitch_link"
    np.testing.assert_allclose(pair["source_ik_targets"]["Hips"]["target_position"], [0.0, 0.0, 0.8])
    np.testing.assert_allclose(pair["target_ik_targets"]["Hips"]["target_position"], [0.0, 0.0, 1.0])


def test_collect_robot_robot_observations_are_solver_compatible(tmp_path: Path):
    solve_module = load_app_module("solve_robot_robot_calibration", "app/solve_robot_robot_calibration.py")

    source_pose = write_json(
        tmp_path / "g1_t_pose.json",
        {
            "schema": "robot_pose.v1",
            "robot_type": "unitree_g1",
            "robot_mjcf": "/tmp/g1.xml",
            "robot_body_transforms": {
                "pelvis": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                "left_wrist_yaw_link": [0.3, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
        },
    )
    target_pose = write_json(
        tmp_path / "ao_t_pose.json",
        {
            "schema": "robot_pose.v1",
            "robot_type": "agile_one",
            "robot_mjcf": "/tmp/ao.xml",
            "robot_body_transforms": {
                "pelvis_link": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "left_wrist_pitch_link": [0.6, 0.8, 1.2, 0.0, 0.0, 0.0, 1.0],
            },
        },
    )
    pair = write_json(
        tmp_path / "t_pose_pair.json",
        {
            "schema": "robot_robot_pose_pair.v1",
            "pose_name": "t_pose",
            "source_pose": str(source_pose),
            "target_pose": str(target_pose),
        },
    )
    retargeter_config = {
        "source_ik_map": {
            "Hips": {"t_body": "pelvis", "r_body": "pelvis"},
            "LeftHand": {"t_body": "left_wrist_yaw_link", "r_body": "left_wrist_yaw_link"},
        },
        "ik_map": {
            "Hips": {"t_body": "pelvis_link", "r_body": "pelvis_link"},
            "LeftHand": {"t_body": "left_wrist_pitch_link", "r_body": "left_wrist_pitch_link"},
        },
    }

    observations = solve_module.collect_robot_robot_observations([pair], retargeter_config, root_name="Hips")

    assert sorted(observations) == ["Hips", "LeftHand"]
    assert observations["LeftHand"][0]["source_pose"] == str(source_pose.resolve())
    np.testing.assert_allclose(observations["LeftHand"][0]["human"][:3], [0.3, 0.4, 1.0])
    np.testing.assert_allclose(observations["LeftHand"][0]["robot_position"], [0.6, 0.8, 1.2])
    np.testing.assert_allclose(observations["LeftHand"][0]["human_root"][:3], [0.0, 0.0, 0.8])
    np.testing.assert_allclose(observations["LeftHand"][0]["robot_root_position"], [0.0, 0.0, 1.0])


def test_robot_pose_editor_builds_pose_json_with_semantic_targets(tmp_path: Path):
    editor_module = load_app_module("robot_pose_editor", "app/robot_pose_editor.py")

    pose = editor_module.build_robot_pose_sample(
        robot_type="unitree_g1",
        robot_mjcf=Path("/tmp/g1.xml"),
        pose_name="t_pose",
        robot_q=np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.25], dtype=np.float32),
        robot_joints={"left_elbow_joint": 0.25},
        robot_body_names=["pelvis", "left_wrist_yaw_link"],
        robot_body_q=np.array(
            [
                [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                [0.4, 0.2, 1.1, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        semantic_map={
            "Hips": {"t_body": "pelvis", "r_body": "pelvis"},
            "LeftHand": {"t_body": "left_wrist_yaw_link", "r_body": "left_wrist_yaw_link"},
        },
        base_retargeter_config=Path("/tmp/g1_to_ao.json"),
    )

    assert pose["schema"] == "robot_pose.v1"
    assert pose["pose_name"] == "t_pose"
    assert pose["robot_type"] == "unitree_g1"
    assert pose["robot_joints"] == {"left_elbow_joint": 0.25}
    assert pose["semantic_targets"]["LeftHand"]["t_body"] == "left_wrist_yaw_link"
    np.testing.assert_allclose(pose["semantic_targets"]["Hips"]["target_position"], [0.0, 0.0, 0.8])


def test_robot_pose_editor_saves_pose_with_user_selected_name(tmp_path: Path):
    editor_module = load_app_module("robot_pose_editor_save_as", "app/robot_pose_editor.py")

    pose = {
        "schema": "robot_pose.v1",
        "pose_name": "old_pose_name",
        "robot_type": "unitree_g1",
    }
    output_path = tmp_path / "g1_forward_reach_pose.json"

    saved_path = editor_module.write_robot_pose_sample_to_path(output_path, pose)
    saved_pose = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path == output_path.resolve()
    assert saved_pose["pose_name"] == "g1_forward_reach_pose"
    assert saved_pose["robot_type"] == "unitree_g1"
