
import pickle
import sys
import os
import logging
from pathlib import Path
from datasets import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    try:
        logger.info("Starting Azure ML training...")
        logger.info(f"Training ID: 718ae459-60b3-4ab8-a148-220481ed7c6e")
        logger.info(f"Compute Target: gpu-cluster")

        # Load chunks metadata
        metadata_file = "chunks_metadata.pkl"
        logger.info(f"Loading metadata from: {metadata_file}")

        with open(metadata_file, "rb") as f:
            chunks = pickle.load(f)

        logger.info(f"Loaded {len(chunks)} chunks for training")

        # Convert to HF Dataset
        dataset = Dataset.from_dict({'text': chunks})
        logger.info(f"Created dataset with {len(dataset)} samples")

        # Initialize and run local trainer (adapted for Azure environment)
        try:
            from training.local_model_trainer import LocalModelTrainer
            trainer = LocalModelTrainer()

            logger.info("Starting model training...")
            model_dir = trainer.train_model(
                dataset,
                "718ae459-60b3-4ab8-a148-220481ed7c6e",
                lambda training, progress, message: logger.info(f"Progress: {progress}% - {message}")
            )

            logger.info(f"Training completed! Model saved to: {model_dir}")
            logger.info("Azure ML training job finished successfully")

        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

    except Exception as e:
        logger.error(f"Azure ML job failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
