# GSM8K LoRA Study

### Can a small language model get better at math?

I tested **Qwen2.5-1.5B-Instruct** on GSM8K, a dataset of math word problems. I used LoRA to train
a small set of extra model weights, then compared prompts with no examples (zero-shot) and three
examples (three-shot). The best recorded setup answered **47 out of 100 questions correctly**,
compared with **36 out of 100** for the base model.

**Just browsing? [Open the notebook →](math_reasoning.ipynb)—no setup needed.**

![Historical accuracy comparison across five model and prompting configurations](assets/results.png)

## What I learned

- The trained model with three examples answered **11 more questions correctly** than the base model.
- It fixed **26** wrong answers but got **15** previously correct answers wrong.
- Giving the base model three examples did not help here: **34%**, versus **36%** without examples.
- A clear, well-formatted answer could still contain reasoning mistakes.

The notebook shows one answer that improved and one that got worse, explains the limits of the study,
and includes code for trying a new experiment.

## About these results

All five setups used the same first 100 test questions.

Several training settings changed between the 1,000-example and 2,000-example runs, so more training data
may not be the only reason their scores differ. The best setup also changed both training and prompting.
A separate prompt comparison had a Python escape bug, so its scores are left out.
See the [results notes](results/README.md) for details.

The new experiment code keeps training, prompt examples, and development questions separate. It also
uses updated prompts and stricter answer checks, so new scores may differ from the saved results.

## Run it yourself (optional)

Use Python 3.12 and clone the repository:

```bash
git clone https://github.com/comet-ctrl/gsm8k-lora-study.git
cd gsm8k-lora-study
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate`
on macOS/Linux. To recreate the chart from the saved results, install JupyterLab and the plotting library:

```bash
python -m pip install matplotlib==3.10.6 jupyterlab
jupyter lab math_reasoning.ipynb
```

**Run All** recreates the results chart and runs basic checks. Model downloads and training are off by default.

To train and test the model yourself, install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

If you use an NVIDIA GPU, install the matching [PyTorch 2.8.0 CUDA version](https://pytorch.org/get-started/previous-versions/#v280)
before installing those packages. Then set `RUN_EXPERIMENTS = True` in the notebook and run it again.
Training can be demanding: a CPU will be slow, and training on a 4 GB GPU has not been tested.

## What's included

- **[Notebook — math_reasoning.ipynb](math_reasoning.ipynb):** the approach, results, example mistakes, and code for new experiments.
- **[Results table — historical_summary.csv](results/historical_summary.csv):** correct-answer counts from the earlier runs.
- **[Results notes — results/README.md](results/README.md):** where the numbers came from and what they can tell us.
- **[Required packages — requirements.txt](requirements.txt):** what to install to run the full experiment.

The original experiments used supplied learning code and Hugging Face libraries. This version uses
newly written helpers and builds on [Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
[GSM8K](https://huggingface.co/datasets/openai/gsm8k), and [LoRA](https://arxiv.org/abs/2106.09685).
Original assignment files and private details are not included in this version. New experiment files go
in `runs/`, which Git ignores. Before sharing your notebook, clear its outputs and check for private details.
Files removed from this version may still exist in older Git commits.
