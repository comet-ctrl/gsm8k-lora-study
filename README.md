# Git practice and ML coursework

A personal learning repository containing a pull-request exercise and an M146 machine-learning notebook.

## Start here

| File | What it contains |
| --- | --- |
| [M146 notebook](Tuning_Param_M146_Project.ipynb) | Qwen2.5-1.5B-Instruct experiments: LoRA fine-tuning on GSM8K, few-shot prompting, and error analysis. Includes saved outputs and written discussion. |
| [index.html](index.html) | A small page used to practice Git and pull requests. Open it in your browser. |
| [navbar.html](navbar.html) | A standalone navigation markup exercise; it is not connected to the page. |

The Git practice files originate from [hiteshpw1/open-source](https://github.com/hiteshpw1/open-source).

## Read or run the notebook

**[Open in Google Colab](https://colab.research.google.com/github/comet-ctrl/open-source/blob/main/Tuning_Param_M146_Project.ipynb)**

You can read the saved results on GitHub without running anything. To rerun the experiment, open it in Colab, choose a GPU runtime, and run the cells in order. The setup cells install dependencies, and later cells download the model and dataset and train adapters. Runtime and memory requirements depend on the GPU; the notebook records earlier out-of-memory adjustments.

Generated results and adapters are written to `m146_outputs/` in the notebook runtime. They are not included in this repository; download them before ending a temporary Colab session.

## Recorded results

These are the notebook's saved results on the first **100 GSM8K test questions**, not a fresh reproduction or a full benchmark:

| Experiment | Correct answers |
| --- | --- |
| Base model | 36 / 100 |
| LoRA with 1,000 training examples | 41 / 100 |
| LoRA with 2,000 training examples | 44 / 100 |
| Base model with three examples in the prompt | 34 / 100 |
| LoRA 2k with three examples in the prompt | 47 / 100 |

The 2k experiment also changes rank, alpha, epochs, and learning rate, so the difference cannot be attributed to training-set size alone.

## Status and limitations

The practice exercise and coursework record are complete. A clean, reproducible training run has **not** been verified during this repository cleanup.

- Setup installs unpinned dependencies; saved logs include dependency conflicts and optional-library warnings.
- In section B.12, the five prompt variants use `\boxed` in ordinary Python strings, where `\b` becomes a backspace character. Before rerunning this section, use raw strings or escape the backslash. Its saved scores and conclusions should be revisited after that correction.
- The original notebook and outputs are preserved as a coursework record.

## Download with Git

```sh
git clone https://github.com/comet-ctrl/open-source.git
cd open-source
```
