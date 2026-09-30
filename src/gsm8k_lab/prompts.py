"""Fresh prompt templates, shared by training and inference."""

PROMPTS = {
    "standard": r"Solve the math problem. Explain your calculation briefly and finish with \boxed{answer}.",
    "check": r"Solve the math problem and check your arithmetic. Finish with \boxed{answer}.",
    "concise": r"Solve the math problem using only the necessary calculations. Finish with \boxed{answer}.",
}


def completion(answer: str) -> str:
    reasoning, separator, final = answer.rpartition("####")
    if not separator or not final.strip():
        raise ValueError("Expected a GSM8K answer with a #### delimiter")
    return reasoning.strip() + "\n" + r"\boxed{" + final.strip() + "}"


def messages(question: str, prompt: str, examples: list[dict] | None = None) -> list[dict]:
    result = [{"role": "system", "content": PROMPTS[prompt]}]
    for example in examples or []:
        result.extend([
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": completion(example["answer"])},
        ])
    return result + [{"role": "user", "content": question}]
