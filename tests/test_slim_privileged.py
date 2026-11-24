"""Slim (CPU-only) install checks for the 7 privileged RoboSuite tasks.

Run with ``uv run pytest tests/test_slim_privileged.py``. Needs only the default
dependencies: no Torch, Ray, Open3D or perception models.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

PRIVILEGED_CONFIGS = {
    "cube_lifting": "env_configs/cube_lifting/franka_robosuite_cube_lifting_privileged.yaml",
    "cube_stack": "env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml",
    "cube_restack": "env_configs/cube_restack/franka_robosuite_cube_restack_privileged.yaml",
    "nut_assembly": "env_configs/nut_assembly/franka_robosuite_nut_assembly_privileged.yaml",
    "spill_wipe": "env_configs/spill_wipe/franka_robosuite_spill_wipe_privileged.yaml",
    "two_arm_lift": "env_configs/two_arm_lift/franka_robosuite_two_arm_lift_privileged.yaml",
    "two_arm_handover": "env_configs/two_arm_handover/two_arm_handover_privileged.yaml",
}
ORACLE_DIR = ROOT / "env_configs/human_oracle_code/robosuite"
ORACLE_CONFIGS = {
    "cube_lifting": "franka_robosuite_cube_lifting_privileged_oracle.yaml",
    "cube_stack": "franka_robosuite_cube_stack_privileged_oracle.yaml",
    "cube_restack": "franka_robosuite_cube_restack_privileged_oracle.yaml",
    "nut_assembly": "franka_robosuite_nut_assembly_privileged_oracle.yaml",
    "spill_wipe": "franka_robosuite_spill_wipe_privileged_oracle.yaml",
    "two_arm_lift": "franka_robosuite_two_arm_lift_privileged_oracle.yaml",
    "two_arm_handover": "two_arm_handover_privileged_oracle.yaml",
}
HEAVY = ("torch", "torchvision", "ray", "open3d", "transformers", "sam3")


def _all_config_paths() -> list[tuple[str, Path]]:
    paths = [(f"{k}-env", ROOT / v) for k, v in PRIVILEGED_CONFIGS.items()]
    paths += [(f"{k}-oracle", ORACLE_DIR / v) for k, v in ORACLE_CONFIGS.items()]
    return paths


def _load_env_factory(path: Path) -> dict:
    from capx.envs.configs.loader import DictLoader

    return DictLoader.load([str(path)])["env"]


def test_launch_path_imports_no_heavy_packages():
    code = (
        "import sys\n"
        "import capx.envs.launch, capx.envs.runner, capx.envs.trial\n"
        "import capx.envs.tasks, capx.envs.simulators, capx.integrations\n"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "assert not bad, f'heavy modules imported: {bad}'\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "MUJOCO_GL": "egl"},
    )
    assert res.returncode == 0, res.stderr[-2000:]


@pytest.mark.parametrize("name,path", _all_config_paths(), ids=lambda v: str(v)[:40])
def test_config_is_privileged_without_render(name, path):
    cfg = _load_env_factory(path)["cfg"]
    # The handover oracle builds its low-level env from an inline _target_,
    # so its flags sit under low_level; everywhere else they sit on cfg.
    low = cfg["low_level"] if isinstance(cfg["low_level"], dict) else cfg
    assert low.get("privileged") is True
    assert low.get("enable_render") is False


@pytest.mark.parametrize("task", list(PRIVILEGED_CONFIGS))
def test_privileged_env_builds_and_resets(task):
    from capx.envs.configs.instantiate import instantiate

    env = instantiate(_load_env_factory(ROOT / PRIVILEGED_CONFIGS[task]))
    try:
        obs, _ = env.reset(seed=0)
        assert obs is not None
        assert env.low_level_env.task_completed() in (True, False)
    finally:
        env.close()
