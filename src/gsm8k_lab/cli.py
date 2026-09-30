"""Command-line entry point. Model downloads happen only for train/evaluate."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from .answers import score, summarize
from .data import DATASET, load_rows
from .prompts import PROMPTS
from .runtime import MODEL, dependencies, load_model, new_output, write_json


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return parsed


def remote_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError("Use a public Hugging Face owner/model ID")
    return value


def parser():
    root = argparse.ArgumentParser(description="Compare prompting and LoRA on GSM8K.")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Score synthetic responses; no downloads or ML libraries")
    compare = commands.add_parser("compare", help="Compare compatible evaluation summaries")
    compare.add_argument("results", nargs="+", help="Paths to summary.json files")
    for name in ("train", "evaluate"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--model", type=remote_id, default=MODEL)
        cmd.add_argument("--revision", default="main", help="Model revision; resolved to a commit before loading")
        cmd.add_argument("--dataset-revision", default="main")
        cmd.add_argument("--seed", type=int, default=42)
        cmd.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        cmd.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
        cmd.add_argument("--prompt", choices=PROMPTS, default="standard")
        cmd.add_argument("--limit", type=positive, default=1000 if name == "train" else 100)
        cmd.add_argument("--batch-size", type=positive, default=1)
        cmd.add_argument("--output", required=True, help="New directory; existing directories are never overwritten")
        if name == "train":
            cmd.add_argument("--rank", type=positive, default=8)
            cmd.add_argument("--epochs", type=positive, default=1)
            cmd.add_argument("--learning-rate", type=float, default=0.0002)
            cmd.add_argument("--accumulation", type=positive, default=8)
            cmd.add_argument("--max-length", type=positive, default=1024)
            cmd.add_argument("--max-steps", type=positive, default=None, help="Optional smoke-run optimizer step cap")
        else:
            cmd.add_argument("--split", choices=("dev", "test"), default="dev")
            cmd.add_argument("--shots", type=int, choices=(0, 3), default=0)
            cmd.add_argument("--adapter", help="Local adapter folder from a completed training run")
            cmd.add_argument("--max-new-tokens", type=positive, default=512)
    return root


def compare(paths):
    summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    reference = summaries[0]["comparison_key"]
    if any(item["comparison_key"] != reference for item in summaries):
        raise ValueError("Runs differ in model/revision, dataset/rows, seed, precision, or generation length")
    print("run | numeric accuracy | boxed accuracy | box rate | invalid rate")
    for index, item in enumerate(summaries, 1):
        metrics = item["metrics"]
        label = f"{index}: {item['kind']}, {item['prompt']}, {item['shots']}-shot"
        print(label + " | " + " | ".join(f"{metrics[key]:.1%}" for key in ("correct", "strict_correct", "boxed", "invalid")))


def run(args):
    if args.command == "demo":
        examples = [(r"3 + 4 = 7. \boxed{7}", "7"), ("The answer is 7.", "7"),
                    (r"\boxed{8}", "7"), ("I do not know.", "7")]
        print(json.dumps({"description": "Synthetic scoring demo, not model benchmark results",
                          "metrics": summarize([score(text, target) for text, target in examples])}, indent=2))
        return
    if args.command == "compare":
        compare(args.results)
        return
    if Path(args.output).exists():
        raise ValueError("Output directory exists; choose a new run name")
    if args.command == "train" and not 0 < args.learning_rate < 1:
        raise ValueError("learning-rate must be between 0 and 1")
    from huggingface_hub import HfApi
    from transformers import set_seed
    from .evaluate import evaluate
    from .train import train

    api = HfApi()
    model_info = api.model_info(args.model, revision=args.revision)
    if model_info.private:
        raise ValueError("This project accepts public model repositories only")
    model_revision = model_info.sha
    dataset_revision = api.dataset_info(DATASET, revision=args.dataset_revision).sha
    set_seed(args.seed)
    split = "train" if args.command == "train" else args.split
    rows, data_info = load_rows(split, args.limit, args.seed, dataset_revision)
    model, tokenizer = load_model(args.model, model_revision, args.device, args.dtype,
                                  getattr(args, "adapter", None))
    output = new_output(args.output)
    metadata = {"schema_version": 1, "model": args.model, "model_revision": model_revision,
                "data": data_info, "versions": dependencies(), "seed": args.seed,
                "device": model.device.type, "dtype": str(model.dtype), "prompt": args.prompt}
    if args.command == "train":
        settings = {key: getattr(args, key) for key in (
            "rank", "epochs", "learning_rate", "accumulation", "max_length", "batch_size", "seed", "prompt")}
        settings["max_steps"] = args.max_steps or -1
        # Write configuration before training so even an interrupted run is identifiable.
        write_json(output / "config.json", {**metadata, "training": settings})
        result = train(model, tokenizer, rows, output, settings)
        metadata.update(training=settings, metrics=result)
        write_json(output / "adapter" / "experiment.json", metadata)
        write_json(output / "summary.json", metadata)
        print(json.dumps(result, indent=2))
    else:
        examples = load_rows("shots", 3, args.seed, dataset_revision)[0] if args.shots else []
        records, metrics = evaluate(model, tokenizer, rows, args.prompt, examples,
                                    args.max_new_tokens, args.batch_size)
        adapter_hash = None
        if args.adapter:
            adapter_hash = hashlib.sha256((Path(args.adapter) / "adapter_model.safetensors").read_bytes()).hexdigest()
        summary = {**metadata, "kind": "lora" if args.adapter else "base", "adapter_sha256": adapter_hash,
                   "shots": args.shots, "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
                   "metrics": metrics, "comparison_key": {
                       "model": args.model, "revision": model_revision, "data": data_info,
                       "dtype": str(model.dtype), "device": model.device.type,
                       "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
                   }}
        write_json(output / "summary.json", summary)
        write_json(output / "predictions.json", records)
        print(json.dumps(metrics, indent=2))


def main():
    root = parser()
    args = root.parse_args()
    try:
        run(args)
    except ImportError as exc:
        root.exit(1, f"Missing ML dependency: {exc.name}. Install with: pip install -e '.[ml]'\n")
    except (ValueError, OSError, KeyError) as exc:
        root.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
