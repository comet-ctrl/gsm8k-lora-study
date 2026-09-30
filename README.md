# GSM8K Reasoning Lab

**Does a small language model solve math problems better after LoRA fine-tuning?**

Train and compare [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
on [GSM8K](https://huggingface.co/datasets/openai/gsm8k), a collection of grade-school math problems.
Try the base model, train a small adapter, and compare zero-shot, three-shot, and alternative prompts.

Start with the [quickstart notebook](notebooks/quickstart.ipynb) or the commands below.
The notebook calls the same package as the command line, so experiments do not depend on hidden cell state.

## Install

Use Python 3.11–3.13 (tested with 3.12).

```bash
git clone https://github.com/comet-ctrl/open-source.git
cd open-source
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install and try the download-free scoring demo:

```bash
python -m pip install -e .
gsm8k-lab demo
```

The demo scores four synthetic responses. **It does not run a model or report benchmark accuracy.**

For real experiments, install PyTorch 2.8.0 for your machine using the
[official installation instructions](https://pytorch.org/get-started/previous-versions/#v280), then:

```bash
python -m pip install -e ".[ml]"
```

Model commands download the model and dataset on first use. CPU execution is supported but slow.
A CUDA GPU with ample free memory is recommended; **4 GB VRAM is not a validated training configuration**.
Quantization and CPU offloading are not implemented. The integration test uses a tiny random model on CPU.

## Run an experiment

Every command writes to a **new** directory. Existing results are never overwritten.

**1. Evaluate the base model on development data.**

```bash
gsm8k-lab evaluate --split dev --limit 100 --output runs/base-dev
```

**2. Train an adapter on 1,000 examples.**

```bash
gsm8k-lab train --limit 1000 --output runs/lora-1k
```

Defaults: rank 8, alpha 16, one epoch, learning rate 0.0002, microbatch 1, accumulation 8,
sequence limit 1,024. Training masks prompt tokens out of the loss. Overlength examples are skipped
and counted rather than silently cut short.

**3. Evaluate the adapter on the same examples.**

```bash
gsm8k-lab evaluate --adapter runs/lora-1k/adapter --split dev --limit 100 --output runs/lora-dev
gsm8k-lab compare runs/base-dev/summary.json runs/lora-dev/summary.json
```

**4. Try three-shot prompting or a different instruction.**

```bash
gsm8k-lab evaluate --adapter runs/lora-1k/adapter --shots 3 --output runs/lora-3shot-dev
gsm8k-lab evaluate --adapter runs/lora-1k/adapter --prompt check --output runs/lora-check-dev
```

Prompts: `standard`, `check`, `concise`. To study training size, repeat training with `--limit 2000`
and a new output directory, keeping the other settings fixed. The 1k sample is a subset of the 2k sample.
More examples at fixed epochs also mean more optimizer steps; this is not a compute-matched comparison.

**5. Freeze your choices, then evaluate on the test split.**

```bash
gsm8k-lab evaluate --split test --output runs/base-test
gsm8k-lab evaluate --adapter runs/lora-1k/adapter --split test --output runs/lora-test
gsm8k-lab compare runs/base-test/summary.json runs/lora-test/summary.json
```

Choose settings using development data, then assess the final configuration on the held-out test set.
For a short training smoke run, add `--limit 8 --max-steps 1`; this still downloads the full model.
Use `gsm8k-lab train --help` or `gsm8k-lab evaluate --help` for more options.

## Read the results

| Metric | Meaning |
| --- | --- |
| Numeric accuracy (`correct`) | Correct boxed answer, or final numeric token when no box was supplied |
| Boxed accuracy (`strict_correct`) | Correct number inside `\boxed{...}` |
| Box rate (`boxed`) | Final box contains a supported number |
| Invalid rate (`invalid`) | No supported numeric answer could be extracted |

Scoring compares decimals and fractions exactly. Empty answers never match. The fallback can select an
incidental final number; inspect `predictions.json` as well as aggregate accuracy. Units, currency symbols,
and percentages inside answers are deliberately not normalized away.

Run summaries record model/dataset commit revisions, key dependency versions, settings, seed, and row IDs.
Reuse saved commits with `--revision COMMIT` and `--dataset-revision COMMIT`. Hardware and library differences
can still change results. `compare` rejects incompatible samples and generation settings.

**No benchmark improvement is claimed yet.** See [validation](docs/validation.md) and the
[experiment protocol](docs/protocol.md).

## Notebook and tests

```bash
python -m pip install -e ".[notebook]"
jupyter lab notebooks/quickstart.ipynb
python -m unittest discover -s tests -v
```

Install the ML extra and set `GSM8K_RUN_ML_TESTS=1` to include the CPU adapter round-trip test.
CI runs both test layers.

`src/gsm8k_lab/` contains the package; `notebooks/` contains the clean walkthrough; `tests/` contains
scoring, experiment, and model checks. Run outputs, downloaded data, weights, credentials, and notebook
checkpoints are ignored by Git. Review artifacts before sharing; `.gitignore` is not a content scanner.
See [sharing guidance](docs/sharing.md) and [sources](docs/sources.md).
