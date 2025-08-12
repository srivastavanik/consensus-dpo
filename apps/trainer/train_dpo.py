from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import datasets as hf_datasets
import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer


@dataclass
class TrainArgs:
    model_name: str
    beta: float = 0.2
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_seq_len: int = 8192
    epochs: int = 1
    per_device_batch: int = 1
    grad_accum_steps: int = 64
    output_dir: str = "./data/runs/dpo"


def load_pairs_dataset(path: str) -> hf_datasets.Dataset:
    ds = hf_datasets.load_dataset("json", data_files=path, split="train")
    return ds


def get_tokenizer_and_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


def main() -> None:
    pairs_path = os.environ.get("DPO_PAIRS_PATH", "./data/pairs.v1.jsonl")
    model_name = os.environ.get("STUDENT_MODEL", "gpt2")
    args = TrainArgs(model_name=model_name)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "consensus-dpo"))

    with mlflow.start_run(run_name="dpo"):
        mlflow.log_params({
            "beta": args.beta,
            "lr": args.learning_rate,
            "wd": args.weight_decay,
            "max_seq_len": args.max_seq_len,
            "epochs": args.epochs,
            "grad_accum": args.grad_accum_steps,
        })

        ds = load_pairs_dataset(pairs_path)
        tokenizer, model = get_tokenizer_and_model(args.model_name)
        collator = None

        from transformers import TrainingArguments

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_batch,
            gradient_accumulation_steps=args.grad_accum_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.epochs,
            logging_steps=50,
            save_steps=1000,
            report_to=["none"],
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            beta=args.beta,
            train_dataset=ds,
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
            max_prompt_length=min(1024, args.max_seq_len // 2),
            max_target_length=min(1024, args.max_seq_len // 2),
        )

        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "main":
    main()


