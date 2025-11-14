import pytest
from unittest.mock import MagicMock, patch
import os
import sys

@pytest.fixture(scope="module", autouse=True)
def set_env_vars(request):
    """Sets environment variables needed for the promotion script test."""
    os.environ["MLFLOW_TRACKING_URI"] = request.config.getoption("--mlflow-tracking-uri")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = request.config.getoption("--minio-endpoint-url")
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

# Add the 'scripts' directory to the Python path
# so we can import 'promote_model'
script_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.append(script_dir)

from scripts import promote_model

# Mock run_id, client, and run objects
MOCK_RUN_ID = "new_run_123"

# Mock for a 'Run' object from MLflow
class MockRun:
    def __init__(self, metrics):
        self.data = self._MockRunData(metrics)
    
    class _MockRunData:
        def __init__(self, metrics):
            self.metrics = metrics

# Mock for a 'Version' object from MLflow
class MockVersion:
    def __init__(self, version, run_id):
        self.version = version
        self.run_id = run_id

@pytest.fixture
def mock_mlflow_client():
    """Mocks the MlflowClient."""
    # Create a MagicMock object to simulate the client
    client = MagicMock()
    
    # Simulate a "champion" model (version 1)
    champion_version = MockVersion(version="1", run_id="champ_run_456")
    champion_run = MockRun(metrics={"accuracy": 0.85})
    client.get_registered_model_alias.return_value = champion_version
    
    # Simulate the "new" model (version 2)
    new_version = MockVersion(version="2", run_id=MOCK_RUN_ID)
    client.get_latest_versions.return_value = [new_version]

    # Use a side_effect to return the correct run based on run_id
    def get_run_side_effect(run_id):
        if run_id == MOCK_RUN_ID:
            # New run has 0.90 accuracy
            return MockRun(metrics={"accuracy": 0.90})
        if run_id == "champ_run_456":
            # Champion run has 0.85 accuracy
            return MockRun(metrics={"accuracy": 0.85})
        return None

    client.get_run.side_effect = get_run_side_effect
    return client

@patch('promote_model.MlflowClient')
def test_model_is_promoted(MockClient, mock_mlflow_client):
    """
    Test the case where the new model is BETTER and should be promoted.
    """
    # Set the mock client to be returned
    MockClient.return_value = mock_mlflow_client
    
    # Run the promotion script
    promote_model.promote_model(MOCK_RUN_ID)
    
    # ASSERT: Check that 'set_registered_model_alias' was called
    # This means the script correctly decided to promote the model
    mock_mlflow_client.set_registered_model_alias.assert_called_with(
        name="giga-flow-sentiment",
        alias="champion",
        version="2"
    )

@patch('promote_model.MlflowClient')
def test_model_is_not_promoted(MockClient, mock_mlflow_client):
    """
    Test the case where the new model is WORSE and should NOT be promoted.
    """
    # Modify the mock client for this test case
    # New run only has 0.80 accuracy
    new_run = MockRun(metrics={"accuracy": 0.80})
    mock_mlflow_client.get_run.side_effect = lambda run_id: new_run if run_id == MOCK_RUN_ID else MockRun(metrics={"accuracy": 0.85})

    MockClient.return_value = mock_mlflow_client
    
    # Run the promotion script
    promote_model.promote_model(MOCK_RUN_ID)
    
    # ASSERT: Check that 'set_registered_model_alias' was NOT called
    mock_mlflow_client.set_registered_model_alias.assert_not_called()