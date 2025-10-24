#!/usr/bin/env python3
"""
Mobile LLM Lab - Training Script
Fine-tunes Hugging Face models with configurable parameters.
"""

import argparse
import os
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Hugging Face models")
    parser.add_argument("--model_name", type=str, required=True, help="Name for your fine-tuned model")
    parser.add_argument("--base_model", type=str, required=True, help="Base model from Hugging Face Hub")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file or HF dataset name")
    parser.add_argument("--task_type", type=str, default="causal_lm", choices=["causal_lm", "classification"],
                        help="Type of task: causal_lm or classification")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for classification")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--hf_username", type=str, default=None, help="Hugging Face username")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")

    return parser.parse_args()


def load_and_prepare_dataset(dataset_path, tokenizer, max_length, task_type):
    """Load and tokenize dataset from various formats."""
    print(f"Loading dataset from: {dataset_path}")

    # Determine dataset format
    if os.path.isfile(dataset_path):
        ext = Path(dataset_path).suffix.lower()
        if ext == ".txt":
            dataset = load_dataset("text", data_files={"train": dataset_path})
        elif ext == ".csv":
            dataset = load_dataset("csv", data_files={"train": dataset_path})
        elif ext == ".json":
            dataset = load_dataset("json", data_files={"train": dataset_path})
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    else:
        # Try loading from Hugging Face Hub
        dataset = load_dataset(dataset_path)

    print(f"Dataset loaded: {dataset}")

    # Tokenize dataset
    def tokenize_function(examples):
        if task_type == "causal_lm":
            # For causal language modeling
            result = tokenizer(
                examples["text"] if "text" in examples else examples["content"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result
        else:
            # For classification
            result = tokenizer(
                examples["text"] if "text" in examples else examples["content"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            if "label" in examples:
                result["labels"] = examples["label"]
            return result

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset


def create_model_repo_if_needed(model_name, hf_username, token):
    """Check if model exists on HF Hub, create if not."""
    if not hf_username:
        print("No HF username provided, skipping repo creation check")
        return None

    api = HfApi(token=token)
    repo_id = f"{hf_username}/{model_name}"

    try:
        # Check if repo exists
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"Model repo '{repo_id}' already exists, will update it")
    except Exception:
        # Repo doesn't exist, create it
        print(f"Creating new model repo: {repo_id}")
        create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)

    return repo_id


def main():
    args = parse_args()

    # Get Hugging Face token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and args.push_to_hub:
        print("WARNING: HF_TOKEN not found in environment. Cannot push to hub.")
        args.push_to_hub = False

    # Create output directory
    output_dir = f"models/{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Save training config
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print(f"Mobile LLM Lab - Training {args.model_name}")
    print("=" * 80)
    print(f"Base Model: {args.base_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Task Type: {args.task_type}")
    print(f"Output Dir: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print("=" * 80)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)

    # Add padding token if not present (needed for some models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"\nLoading base model: {args.base_model}")
    if args.task_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(args.base_model, token=hf_token)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=args.num_labels,
            token=hf_token
        )

    # Load and prepare dataset
    print("\nPreparing dataset...")
    tokenized_dataset = load_and_prepare_dataset(
        args.dataset,
        tokenizer,
        args.max_length,
        args.task_type
    )

    # Data collator
    if args.task_type == "causal_lm":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        push_to_hub=args.push_to_hub,
        hub_token=hf_token,
        hub_model_id=f"{args.hf_username}/{args.model_name}" if args.hf_username else None,
        report_to="none",  # Disable wandb/tensorboard
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Start training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    # Create model repo and push if requested
    if args.push_to_hub and args.hf_username:
        print("\nCreating/updating Hugging Face repo...")
        repo_id = create_model_repo_if_needed(args.model_name, args.hf_username, hf_token)

        print(f"\nPushing model to Hugging Face Hub: {repo_id}")
        trainer.push_to_hub(commit_message=f"Training completed at {datetime.now().isoformat()}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Model saved to: {output_dir}")
    if args.push_to_hub and args.hf_username:
        print(f"Model pushed to: https://huggingface.co/{repo_id}")
    print("=" * 80)


if __name__ == "__main__":
    main()
