import json, pathlib, torch, logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import DPOTrainer, DPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dpo_data(path: str):
    data = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            data.append({
                "prompt": obj["prompt"],
                "chosen": obj["chosen"],
                "rejected": obj["rejected"],
            })
    return Dataset.from_list(data)

def train_dpo(model_name: str, train_data_path: str, output_dir: str, learning_rate: float = 1e-5, epochs: int = 3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    dpo_dataset = load_dpo_data(train_data_path)
    
    # Split into train/test (80/20)
    dpo_dataset = dpo_dataset.train_test_split(test_size=0.2, seed=42)
    
    dpo_config = DPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
    )
    
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        tokenizer=tokenizer,
        train_dataset=dpo_dataset["train"],
        eval_dataset=dpo_dataset["test"],
        beta=0.1,
    )
    
    logger.info("Starting DPO training...")
    train_result = dpo_trainer.train()
    logger.info(f"Training completed. Final loss: {train_result.metrics['eval_loss']:.4f}")
    
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training summary
    summary = {
        "model_name": model_name,
        "train_samples": len(dpo_dataset["train"]),
        "eval_samples": len(dpo_dataset["test"]),
        "final_eval_loss": train_result.metrics.get("eval_loss", 0),
        "epochs": epochs,
    }
    with open(pathlib.Path(output_dir)/"training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return train_result
