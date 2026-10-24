import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mock_server import MockServer  # noqa: E402


@pytest.fixture(scope="session")
def mock_server():
    server = MockServer()
    server.start()
    yield server
    server.stop()
