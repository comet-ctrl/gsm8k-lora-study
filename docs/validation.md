# Validation and limitations

Local validation on 2026-09-29 used Python 3.12 and the pinned ML dependencies on Windows CPU.
The public Qwen tokenizer and GSM8K loader were also checked with three training examples;
all preserved the assistant prefix and fit within 1,024 tokens. The checked upstream commits were:

- Model: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Dataset: `740312add88f781978c0658806c59bc2815b9866`

Standard-library tests cover exact-number scoring, malformed answers, prompt escapes, disjoint data
pools, nested training samples, incompatible result comparisons, and output overwrite protection.

An opt-in CPU integration test creates a tiny random Qwen2 model and synthetic tokenizer locally.
It checks prompt/padding masks, performs an optimizer step on LoRA weights, saves and reloads the
adapter, checks finite parameters, and generates and scores two synthetic responses. No model or
dataset download is required for this test.

PowerShell:

```powershell
$env:GSM8K_RUN_ML_TESTS = '1'
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
python -m unittest discover -s tests -v
```

macOS/Linux:

```bash
GSM8K_RUN_ML_TESTS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m unittest discover -s tests -v
```

The tiny-model test does not validate full 1.5B benchmarking or CUDA training. Historical notebook
scores are not presented as new results. Quantized training, distributed training, automatic checkpoint
recovery, and adapter merging are not implemented.

Training targets Qwen-style projection names and requires a compatible chat template. Unrelated
architectures may need code changes. If a run fails, keep its directory for diagnosis and retry with
a new output directory.
