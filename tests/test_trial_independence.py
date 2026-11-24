"""A trial's outcome must not depend on which trial the worker ran before it.

The robot API objects live as long as the worker's environment and keep their last
IK solution (``cfg``, ``cfg_0``, ``cfg_1``) as the warm start for the next solve.
``CodeExecutionEnvBase.reset`` must clear it.

Run with ``uv run pytest tests/test_trial_independence.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from capx.integrations.base_api import ApiBase
from tests.test_slim_privileged import PRIVILEGED_CONFIGS, ROOT, _load_env_factory


class _FakeApi(ApiBase):
    def __init__(self) -> None:
        super().__init__(env=None)  # type: ignore[arg-type]
        self.cfg = np.ones(8)
        self.cfg_0 = np.ones(8)
        self.cfg_1 = np.ones(8)
        self.other = "kept"

    def functions(self):
        return {}


def test_api_reset_state_clears_ik_warm_starts_only():
    api = _FakeApi()
    api.reset_state()
    assert api.cfg is None and api.cfg_0 is None and api.cfg_1 is None
    assert api.other == "kept"


def test_api_reset_state_ignores_missing_attributes():
    class NoIk(ApiBase):
        def functions(self):
            return {}

    NoIk(env=None).reset_state()  # type: ignore[arg-type]


@pytest.mark.parametrize("task", list(PRIVILEGED_CONFIGS))
def test_env_reset_clears_ik_warm_start(task):
    from capx.envs.configs.instantiate import instantiate

    env = instantiate(_load_env_factory(ROOT / PRIVILEGED_CONFIGS[task]))
    try:
        env.reset(seed=1)
        found = 0
        for api in env._apis.values():
            for attr in ApiBase.IK_WARM_START_ATTRS:
                if hasattr(api, attr):
                    setattr(api, attr, np.ones(8))
                    found += 1
        assert found, f"{task}: no IK warm-start attribute found on its APIs"
        env.reset(seed=2)
        for api in env._apis.values():
            for attr in ApiBase.IK_WARM_START_ATTRS:
                if hasattr(api, attr):
                    assert getattr(api, attr) is None, f"{task}: {attr} survived reset"
    finally:
        env.close()
