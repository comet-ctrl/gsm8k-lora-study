"""One model per process; deterministic greedy decoding."""

from .answers import score, summarize
from .prompts import messages


def evaluate(model, tokenizer, rows, prompt="standard", examples=None, max_new_tokens=512, batch_size=1):
    import torch
    tokenizer.padding_side = "left"
    model.eval()
    records = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        rendered = [tokenizer.apply_chat_template(
            messages(row["question"], prompt, examples), tokenize=False, add_generation_prompt=True,
        ) for row in batch]
        tokens = tokenizer(rendered, padding=True, add_special_tokens=False,
                           return_token_type_ids=False, return_tensors="pt").to(model.device)
        context = getattr(model.config, "max_position_embeddings", None)
        if context and tokens.input_ids.shape[1] + max_new_tokens > context:
            raise ValueError("Prompt plus response exceeds model context; reduce shots or response length")
        with torch.inference_mode():
            generated = model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
        responses = tokenizer.batch_decode(generated[:, tokens.input_ids.shape[1]:], skip_special_tokens=True)
        for row, response in zip(batch, responses):
            expected = row["answer"].rsplit("####", 1)[-1].strip()
            records.append({"id": row["id"], "response": response, "expected": expected,
                            **score(response, expected)})
    return records, summarize(records)
