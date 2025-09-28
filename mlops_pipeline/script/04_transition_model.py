# 04_transition_model.py — Dry Beans
from mlflow.tracking import MlflowClient


def transition_model_alias(model_name: str, alias: str, description: str | None = None):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise SystemExit(f"No versions found for model '{model_name}'.")
    latest_version = max(versions, key=lambda mv: int(mv.version))
    version_number = latest_version.version
    print(f"Latest version for {model_name}: v{version_number}")

    if description:
        try:
            client.update_model_version(
                name=model_name, version=version_number, description=description
            )
        except Exception as e:
            print("[WARN] update_model_version failed, setting tag instead:", e)
            client.set_model_version_tag(
                name=model_name, version=version_number, key="description", value=description
            )

    client.set_registered_model_alias(name=model_name, alias=alias, version=version_number)
    print(f"Alias '{alias}' set on {model_name} v{version_number}.")
