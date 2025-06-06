# local_model_trainer.py

import pickle
import logging
from pathlib import Path
from typing import List, Callable, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from datasets import Dataset
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LocalModelTrainer:
    """
    Local model training & inference for a fine-tuned DialoGPT-medium.
    - During Azure ML training, saves under outputs/local_model_<training_id>/ so Azure persists it.
    - For local inference, checks backend/models/local_model_<training_id>/ to see if a fine-tuned checkpoint is available.
    """

    def __init__(self, training_id: Optional[str] = None):
        # Base model name for fallback
        self.base_model_name = "microsoft/DialoGPT-medium"
        # Max sequence length
        self.max_length = 512
        self.tokenizer = None
        self.model = None
        # Device (GPU or CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Sentence-Transformers for embeddings (optional)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # training_id indicates which fine-tuned folder to look for
        self.training_id = training_id

    def is_available(self) -> bool:
        """
        Return True if a fine-tuned checkpoint exists under backend/models/local_model_<training_id>/.
        """
        if not self.training_id:
            return False

        from config import MODELS_DIR
        fine_tuned_dir = MODELS_DIR / f"local_model_{self.training_id}"
        if not fine_tuned_dir.is_dir():
            return False

        # Check for at least one model file (.bin or .safetensors)
        has_bin = any(fine_tuned_dir.glob("*.bin"))
        has_safe = any(fine_tuned_dir.glob("*.safetensors"))
        return has_bin or has_safe

    def _load_model_and_tokenizer(self):
        """
        Load tokenizer + model:
        - If a fine-tuned checkpoint exists under backend/models/local_model_<training_id>/, load from there.
        - Otherwise, load the base microsoft/DialoGPT-medium.
        """
        from config import MODELS_DIR

        if self.tokenizer is None or self.model is None:
            try:
                if self.is_available():
                    fine_tuned_dir = MODELS_DIR / f"local_model_{self.training_id}"
                    logger.info(f"Loading fine-tuned model from: {fine_tuned_dir}")

                    self.tokenizer = AutoTokenizer.from_pretrained(str(fine_tuned_dir))
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(fine_tuned_dir),
                        torch_dtype=torch.float32
                    )
                else:
                    logger.info(f"Fine-tuned model not found. Loading base model: {self.base_model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token

                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )

                self.model.to(self.device)
                logger.info(f"Loaded model on {self.device}")

            except Exception as e:
                logger.error(f"Failed to load model/tokenizer: {e}")
                raise

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Tokenize + filter the Hugging Face Dataset.
        Assumes dataset has a "text" column (strings).
        """
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )

        def filter_short_sequences(example):
            return len(example["input_ids"]) >= 10

        filtered_dataset = tokenized_dataset.filter(filter_short_sequences)
        logger.info(f"Preprocessed dataset: {len(filtered_dataset)} examples")
        return filtered_dataset

    def train_model(
        self,
        metadata_path: str,
        training_id: str,
        progress_callback: Optional[Callable[[bool, int, str], None]] = None
    ) -> str:
        """
        Fine-tune DialoGPT on the raw-text chunks at metadata_path, then save under outputs/local_model_<training_id>/.

        Args:
            metadata_path (str): Path to a pickle file containing a List[str] of chunks.
            training_id (str):   Unique identifier; used to name the output folder.
            progress_callback:   Optional function to report (success_flag, percent, message).

        Returns:
            Path (string) to the folder under "outputs/" where the model was saved.
        """
        # Set training_id so is_available() knows what folder to look for after saving
        self.training_id = training_id

        try:
            # 1) Loading raw‐text chunks
            if progress_callback:
                progress_callback(True, 10, "Loading raw text chunks from pickle.")

            meta_file = Path(metadata_path)
            if not meta_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            with open(meta_file, "rb") as f:
                chunk_text_list: List[str] = pickle.load(f)

            if not isinstance(chunk_text_list, list) or len(chunk_text_list) == 0:
                raise ValueError("Loaded metadata was not a nonempty list of strings.")

            logger.info(f"Loaded {len(chunk_text_list)} raw-text chunks from {metadata_path}")

            # 2) Build HF Dataset from strings
            hf_dicts = [{"text": txt} for txt in chunk_text_list]
            hf_dataset = Dataset.from_list(hf_dicts)
            logger.info("Created Hugging Face Dataset from raw-text chunks")

            # 3) Preprocess (tokenize + filter)
            if progress_callback:
                progress_callback(True, 20, "Preprocessing dataset (tokenization).")
            processed_dataset = self._preprocess_dataset(hf_dataset)
            if len(processed_dataset) == 0:
                raise ValueError("No valid training examples after preprocessing.")

            # 4) Split into train/test if large enough
            if len(processed_dataset) > 100:
                split = processed_dataset.train_test_split(test_size=0.1)
                train_dataset = split["train"]
                eval_dataset = split["test"]
                logger.info(f"Dataset split: {len(train_dataset)} train / {len(eval_dataset)} eval")
            else:
                train_dataset = processed_dataset
                eval_dataset = None
                logger.info(f"Using full dataset ({len(train_dataset)}) for training; no eval split.")

            # 5) Load base model + tokenizer (so we fine-tune from base, not from a previous checkpoint)
            if progress_callback:
                progress_callback(True, 30, "Loading base model and tokenizer.")
            # Temporarily clear training_id so is_available() returns False
            self.training_id = None
            self._load_model_and_tokenizer()

            # 6) Build output directory under "outputs/"
            output_dir = Path("outputs") / f"local_model_{training_id}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 7) Configure training args
            if progress_callback:
                progress_callback(True, 40, "Setting up training arguments.")
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                overwrite_output_dir=True,
                num_train_epochs=2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=min(100, len(train_dataset) // 4),
                logging_steps=10,
                save_steps=100,
                evaluation_strategy="steps" if eval_dataset else "no",
                eval_steps=100 if eval_dataset else None,
                save_total_limit=2,
                prediction_loss_only=True,
                fp16=False,
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to=None,
            )

            # 8) Create the Trainer
            if progress_callback:
                progress_callback(True, 50, "Initializing Trainer.")
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
            )

            # 9) (Optional) Per-step progress callback
            if progress_callback:
                total_steps = (len(train_dataset) // training_args.per_device_train_batch_size) * training_args.num_train_epochs

                class ProgressCallbackTrainer(TrainerCallback):
                    def __init__(self, cb_fn, total_steps: int):
                        self.cb_fn = cb_fn
                        self.total_steps = total_steps

                    def on_train_begin(self, args, state, control, **kwargs):
                        if self.cb_fn:
                            self.cb_fn(True, 55, "Training begun")
                        return control

                    def on_step_end(self, args, state, control, **kwargs):
                        if self.cb_fn and state.global_step > 0:
                            prog = min(
                                95,
                                55 + int((state.global_step / self.total_steps) * 40)
                            )
                            self.cb_fn(True, prog, f"Step {state.global_step}/{self.total_steps}")
                        return control

                trainer.add_callback(ProgressCallbackTrainer(progress_callback, total_steps))

            # 10) Train
            if progress_callback:
                progress_callback(True, 60, "Starting training.")
            trainer.train()

            # 11) Save the fine-tuned model & tokenizer under outputs/
            if progress_callback:
                progress_callback(True, 95, "Saving fine-tuned model and tokenizer.")
            trainer.save_model(str(output_dir))
            self.tokenizer.save_pretrained(str(output_dir))

            # 12) After saving, set training_id so is_available() picks up the local folder
            self.training_id = training_id
            if progress_callback:
                progress_callback(True, 100, "Training completed successfully!")
            logger.info(f"Model training completed. Saved to {output_dir}")

            return str(output_dir)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if progress_callback:
                progress_callback(False, 0, f"Training failed: {str(e)}")
            raise

    def generate_with_local_model(self, messages: List[dict], session_id: str) -> str:
        """
        Generate a response with the (fine-tuned) DialoGPT model.
        `messages` should be a list of dicts: [{"role":"user","content":"Hello"}, …].
        """
        try:
            # Load the fine-tuned model if available; otherwise fallback
            self._load_model_and_tokenizer()

            # Build a single prompt from the last 10 messages
            prompt = ""
            for msg in messages[-10:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
                elif role == "system":
                    prompt += f"System: {content}\n"

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_return_sequences=1
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

        except Exception as e:
            logger.error(f"Failed to generate with local model: {e}")
            raise
