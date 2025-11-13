import os
import sys
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
MODEL_NAME = "giga-flow-sentiment"
CHAMPION_ALIAS = "champion"

def promote_model(new_run_id):
    """
    Compares the new model's accuracy to the current champion's.
    Promotes the new model if its accuracy is higher.
    """
    if not new_run_id:
        print("Error: No MLFLOW_RUN_ID specified.")
        sys.exit(1)

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
    
    # Get new model's metrics
    try:
        new_run = client.get_run(new_run_id)
        new_accuracy = new_run.data.metrics.get("accuracy", 0)
        new_version = client.get_latest_versions(MODEL_NAME, stages=None)[0]
        print(f"New Model (Version {new_version.version}): Accuracy = {new_accuracy:.4f}")
    except Exception as e:
        print(f"Error fetching new model data: {e}")
        sys.exit(1)

    # Get current champion's metrics
    current_champion_accuracy = 0.0
    try:
        current_champion = client.get_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS)
        champion_run = client.get_run(current_champion.run_id)
        current_champion_accuracy = champion_run.data.metrics.get("accuracy", 0)
        print(f"Current Champion (Version {current_champion.version}): Accuracy = {current_champion_accuracy:.4f}")
    except Exception:
        print(f"No model currently has the '{CHAMPION_ALIAS}' alias. This new model will become the champion.")

    # The promotion logic
    if new_accuracy > current_champion_accuracy:
        print(f"New model is better. Setting '{CHAMPION_ALIAS}' alias to Version {new_version.version}...")
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=CHAMPION_ALIAS,
            version=new_version.version
        )
        print("Promotion successful.")
    else:
        print("New model is not better than the current champion. No promotion.")

if __name__ == "__main__":
    run_id_from_env = os.getenv("NEW_RUN_ID")
    promote_model(run_id_from_env)