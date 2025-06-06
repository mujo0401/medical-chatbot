
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import pickle

def train_model():
    # Load data
    with open('training_data.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        dataloader_num_workers=0
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained('./trained_model')
    tokenizer.save_pretrained('./trained_model')
    
    print("Training completed successfully!")

if __name__ == "__main__":
    train_model()
