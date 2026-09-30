"""Lazy model loading and metadata without usernames or machine paths."""

import importlib.metadata
import json
from pathlib import Path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def dependencies() -> dict:
    result = {}
    for name in ("torch", "transformers", "peft", "datasets", "accelerate"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def precision(device: str, dtype: str):
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; install a CUDA PyTorch build or use --device cpu")
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    if dtype == "bfloat16" and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 requires a supported CUDA device; use float32")
    return device, getattr(torch, dtype)


def load_model(model_id: str, revision: str, device: str, dtype: str, adapter: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    device, torch_dtype = precision(device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("The model tokenizer must provide a chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch_dtype, trust_remote_code=False,
        attn_implementation="eager",
    ).to(device)
    if adapter:
        metadata_path = Path(adapter) / "experiment.json"
        if not metadata_path.is_file():
            raise ValueError("Adapter must be produced by gsm8k-lab train (missing experiment.json)")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        resolved = getattr(model.config, "_commit_hash", None)
        if metadata["model"] != model_id or metadata["model_revision"] != resolved:
            raise ValueError("Adapter base model/revision does not match the selected model")
        model = PeftModel.from_pretrained(model, adapter)
    return model, tokenizer


def new_output(path: str) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
