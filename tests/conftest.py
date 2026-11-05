import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--env", default=os.environ.get("WM_ENV", "alfworld"),
                     help="alfworld (run in uv_alfworld) or sciworld (run in uv_sciworld)")


@pytest.fixture(scope="session")
def env_name(request):
    return request.config.getoption("--env")


@pytest.fixture(scope="session")
def env(env_name):
    from wm.envs.base import make_env
    e = make_env(env_name)
    yield e
    if hasattr(e, "close"):
        e.close()
