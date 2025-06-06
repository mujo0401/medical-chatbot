# training/register_hf_env.py
from dotenv import load_dotenv
load_dotenv()  
import os
import sys
import yaml
import logging
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s │ %(message)s",
)
logger = logging.getLogger("register_hf_env")

if __name__ == "__main__":
    # 1) Read required env vars
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group   = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name   = os.getenv("AZURE_WORKSPACE_NAME")

    missing = []
    if not subscription_id:
        missing.append("AZURE_SUBSCRIPTION_ID")
    if not resource_group:
        missing.append("AZURE_RESOURCE_GROUP")
    if not workspace_name:
        missing.append("AZURE_WORKSPACE_NAME")
    if missing:
        logger.error(f"Missing required Azure env variables: {', '.join(missing)}")
        sys.exit(1)

    # 2) Create a small environment.yml on disk
    env_spec = {
        "name": "hf_env",            # internal name in the YAML (not the AzureML name)
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            "python=3.8",
            {
                "pip": [
                    "datasets>=2.0.0",
                    "transformers>=4.30.0",
                    "torch>=1.13.1",
                ]
            },
        ],
    }

    yaml_path = "hf_environment.yml"
    try:
        with open(yaml_path, "w") as f:
            yaml.safe_dump(env_spec, f, sort_keys=False)
        logger.info(f"Wrote conda YAML to '{yaml_path}'")
    except Exception as e:
        logger.error(f"Failed to write {yaml_path}: {e}")
        sys.exit(1)

    # 3) Instantiate MLClient
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        logger.info("✅ Connected to Azure ML workspace")
    except Exception as e:
        logger.error(f"Failed to create MLClient: {e}")
        sys.exit(1)

    # 4) Define and register the custom Environment
    #
    #    New name must NOT start with “AzureML” or any reserved prefix.
    #    We’ll use “hf-huggingface-pytorch-1-13-cpu” (all lowercase, no periods).
    #
    try:
        hf_env = Environment(
            name="hf-huggingface-pytorch-1-13-cpu",  # <--- new, valid name
            version="1",
            description="HuggingFace CPU env with datasets, transformers, and torch",
            conda_file=yaml_path,
            image="mcr.microsoft.com/azureml/base:latest",  # or any valid Azure ML base image
            tags={"framework": "huggingface", "type": "training"},
        )

        logger.info(f"Registering environment {hf_env.name}:{hf_env.version} ...")
        registered = ml_client.environments.create_or_update(hf_env)
        logger.info(f"✔ Registered environment: {registered.name}:{registered.version}")
    except Exception as e:
        logger.error(f"Failed to register environment: {e}")
        sys.exit(1)

    logger.info(
        "All done. You can now reference 'hf-huggingface-pytorch-1-13-cpu:1' in your training code."
    )
