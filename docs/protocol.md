# Experiment protocol

The dataset's training split is partitioned by original row index:

- Rows 0–199: development pool.
- Rows 200–202: fixed three-shot demonstrations.
- Rows 203 onward: fine-tuning pool.
- The original test split: final evaluation only.

Pools are shuffled with a local Python RNG using seed 42 by default; demonstrations retain their order.
Selected original row IDs are saved. Training, development, and demonstrations do not overlap.
Test IDs belong to a different source split.

When studying training changes, keep model/dataset commits, evaluation rows, seed, precision, prompt,
shot count, and generation budget fixed. `compare` permits intentional prompt and shot changes and
labels them; it cannot establish causality. Nested 1k/2k samples use the same training defaults, but
fixed epochs imply different compute. Check skipped overlength examples for actual training size.

Generation is greedy, with up to 512 new tokens. The full prompt is preserved; exceeding context limits
fails instead of silently truncating. Only continuation tokens are scored. Training uses assistant
reasoning and the final answer; prompt and padding labels are -100.

Choose settings on development data and evaluate the final choice on held-out test data. A 100-example
score is a small-sample measurement. Report sample size, inspect errors, and repeat across seeds before
making broad claims. Top-level ML dependencies are pinned; transitive dependencies and hardware can
still change results. Hugging Face revisions are resolved to commit hashes before loading.
