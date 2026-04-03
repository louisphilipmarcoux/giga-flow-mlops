import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio


@pytest.fixture
def reference_df():
    """Sample reference data for testing."""
    return pd.DataFrame({"text": [
        "I love this product",
        "Terrible experience",
        "Amazing quality",
        "Would not recommend",
        "Best purchase ever",
    ] * 20})  # 100 rows


@pytest.fixture
def live_batch_df():
    """Sample live batch data for testing."""
    return pd.DataFrame({"text": [
        "Great service overall",
        "Very disappointed",
        "Highly recommend this",
        "Awful customer support",
        "Exceeded my expectations",
    ] * 20})  # 100 rows


def test_drift_check_runs_without_error(reference_df, live_batch_df):
    """Test that drift check completes without errors when given valid data."""
    import src.drift_monitor.monitor as monitor

    # Set reference data
    monitor.reference_data = reference_df

    # Run drift check synchronously
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(monitor.run_drift_check(live_batch_df))
    finally:
        loop.close()


def test_drift_check_skips_when_no_reference():
    """Test that drift check gracefully skips when reference data is None."""
    import src.drift_monitor.monitor as monitor

    original = monitor.reference_data
    monitor.reference_data = None

    loop = asyncio.new_event_loop()
    try:
        # Should not raise
        loop.run_until_complete(monitor.run_drift_check(pd.DataFrame({"text": ["test"]})))
    finally:
        monitor.reference_data = original
        loop.close()


def test_batch_size_config():
    """Test that batch size is correctly configured."""
    from src.drift_monitor.monitor import BATCH_SIZE
    assert BATCH_SIZE == 100
    assert isinstance(BATCH_SIZE, int)
