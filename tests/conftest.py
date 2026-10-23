import warnings

import pytest
import torch


@pytest.fixture(autouse=True)
def _quiet_and_deterministic():
    torch.manual_seed(0)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
