# Recorded results

`historical_summary.csv` contains aggregate counts verified against five saved evaluation files.
All five runs evaluated the same first 100 GSM8K test questions. No raw submission files or full model
responses are included. These figures predate the newly written notebook rerun code.

The zero-shot base and three-shot 2k adapter records were also matched by question and index:
21 remained correct, 26 became correct, 15 became incorrect, and 38 remained incorrect.
The notebook discusses two verified cases using paraphrases of public dataset problems.

The 1k adapter used rank 8, alpha 16, one epoch, and learning rate 0.0002. The 2k adapter used rank 16,
alpha 32, two epochs, and learning rate 0.0001. Training-size effects are therefore confounded with
other changes. These exploratory results used the original numeric scorer, including an unboxed
final-number fallback. A later prompt sweep affected by a Python escape bug is excluded.

The optional fresh experiment uses revised sampling, prompts, and scoring. Its outputs are saved
separately under ignored `runs/` directories and must not be presented as reproductions of this CSV.
