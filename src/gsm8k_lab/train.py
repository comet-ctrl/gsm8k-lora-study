"""Completion-only supervised LoRA training with Hugging Face Trainer."""

import math

from .prompts import completion, messages


def encode_example(tokenizer, row, max_length, prompt="standard"):
    conversation = messages(row["question"], prompt)
    prefix = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        conversation + [{"role": "assistant", "content": completion(row["answer"])}],
        tokenize=True, add_generation_prompt=False,
    )
    if full[:len(prefix)] != prefix:
        raise ValueError("Chat template does not preserve the assistant prefix")
    # Reject rather than train on a clipped rationale or missing final answer.
    if len(full) > max_length:
        return None
    if len(full) <= len(prefix):
        raise ValueError("Example contains no assistant tokens")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):]}


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        width = max(len(feature["input_ids"]) for feature in features)
        return {key: torch.tensor([
            feature[key] + [pad] * (width - len(feature[key])) for feature in features
        ]) for key, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))}


def train(model, tokenizer, rows, output, config):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    tokenizer.padding_side = "right"
    encoded = [encode_example(tokenizer, row, config["max_length"], config["prompt"]) for row in rows]
    features = [row for row in encoded if row is not None]
    if not features:
        raise ValueError("All training rows exceed max_length; increase the limit")
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=config["rank"], lora_alpha=2 * config["rank"], lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    ))
    model.config.use_cache = False
    model.enable_input_require_grads()
    step_budget = config["max_steps"] if config["max_steps"] > 0 else (
        math.ceil(len(features) / (config["batch_size"] * config["accumulation"])) * config["epochs"]
    )
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["accumulation"], num_train_epochs=config["epochs"],
        max_steps=config["max_steps"], learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine", warmup_ratio=0.0 if step_budget == 1 else 0.05,
        seed=config["seed"], data_seed=config["seed"],
        bf16=model.dtype == torch.bfloat16, fp16=False, use_cpu=model.device.type == "cpu",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no", logging_strategy="steps", logging_steps=1, report_to=[],
        remove_unused_columns=False, dataloader_pin_memory=model.device.type == "cuda",
    )
    trainer = Trainer(model=model, args=args, train_dataset=features,
                      data_collator=CompletionCollator(tokenizer.pad_token_id))
    result = trainer.train()
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    return {"used_rows": len(features), "skipped_long_rows": len(rows) - len(features),
            "used_row_ids": [row["id"] for row, encoding in zip(rows, encoded) if encoding is not None],
            "optimizer_steps": result.global_step, "training_loss": result.training_loss}
