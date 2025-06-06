# utils/azure_utils.py

import logging
import pickle
from pathlib import Path
from typing import Optional, Union

from azure.ai.ml import MLClient
from azure.ai.ml.entities import CommandJob, Environment
from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential

from config import (
    AZURE_SUBSCRIPTION_ID,
    AZURE_RESOURCE_GROUP,
    AZURE_WORKSPACE_NAME,
    UPLOADS_DIR,
    MODELS_DIR
)

logger = logging.getLogger(__name__)


def get_ml_client() -> Optional[MLClient]:
    """
    Create and return an Azure MLClient if all required environment variables are set.
    Returns None if any required config is missing or client initialization fails.
    """
    if not (AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP and AZURE_WORKSPACE_NAME):
        logger.warning("Azure ML configuration is incomplete. "
                       "Skipping Azure MLClient initialization.")
        return None

    try:
        creds = DefaultAzureCredential()
        client = MLClient(
            credential=creds,
            subscription_id=AZURE_SUBSCRIPTION_ID,
            resource_group_name=AZURE_RESOURCE_GROUP,
            workspace_name=AZURE_WORKSPACE_NAME
        )
        logger.info("Azure MLClient initialized successfully.")
        return client
    except AzureError as e:
        logger.error(f"Failed to initialize Azure MLClient: {e}")
        return None


def submit_training_job(
    ml_client: MLClient,
    training_data_obj: object,
    compute_target: str,
    training_id: str,
    base_model_name: str = "microsoft/DialoGPT-medium",
    script_dir: Union[str, Path] = Path("."),
    environment_name: str = "medical-env",
    conda_file: str = "environment.yml",
    image: str = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04"
) -> Optional[str]:
    """
    Serialize `training_data_obj`, write a training script, and submit an Azure Command job.
    Returns the Azure job name on success, or None on failure.
    """
    if not ml_client:
        logger.error("MLClient is None. Cannot submit Azure training job.")
        return None

    try:
        # Prepare paths
        MODELS_DIR.mkdir(exist_ok=True)
        temp_data_path = Path("training_data.pkl")
        temp_script_path = Path("azure_train_script.py")

        # Serialize the training dataset or object
        with open(temp_data_path, "wb") as f:
            pickle.dump(training_data_obj, f)
        logger.info(f"Training data serialized to {temp_data_path}.")

        # Write the training script content
        script_content = f"""
import pickle
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

def train():
    # Load serialized dataset
    with open("{temp_data_path.name}", "rb") as f:
        ds = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained("{base_model_name}")
    model = AutoModelForCausalLM.from_pretrained("{base_model_name}")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator
    )
    trainer.train()
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    print("Azure training job completed successfully.")

if __name__ == "__main__":
    train()
"""
        with open(temp_script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        logger.info(f"Training script written to {temp_script_path}.")

        # Define Azure environment
        env = Environment(
            name=environment_name,
            conda_file=conda_file,
            image=image
        )

        # Create and submit the Command job
        job = CommandJob(
           code=str(script_dir),
           command=f"python {temp_script_path.name}",
           environment=env,
           compute=compute_target,
           display_name=f"medical-chatbot-training-{training_id}"
        )

        submitted_job = ml_client.jobs.create_or_update(job)
        logger.info(f"Submitted Azure ML job: {submitted_job.name}")
        return submitted_job.name

    except Exception as e:
        logger.error(f"Error submitting Azure training job: {e}")
        return None
