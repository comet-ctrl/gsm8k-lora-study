"""Opt-in CPU integration: random tiny Qwen, synthetic data, no downloads."""

import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("GSM8K_RUN_ML_TESTS") == "1", "Set GSM8K_RUN_ML_TESTS=1 with ML extras installed")
class ModelIntegrationTests(unittest.TestCase):
    def test_train_save_reload_and_evaluate(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, set_seed
        from peft import PeftModel
        from gsm8k_lab.evaluate import evaluate
        from gsm8k_lab.train import CompletionCollator, encode_example, train

        torch.set_num_threads(2)
        set_seed(42)
        backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "1": 3,
                                      "2": 4, "3": 5, "4": 6, "assistant": 7}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
        tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% if m['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
        config = Qwen2Config(vocab_size=8, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
                             pad_token_id=1, eos_token_id=2, bos_token_id=2, attention_dropout=0)
        model = Qwen2ForCausalLM(config)
        initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        rows = [{"id": 0, "question": "1 + 1?", "answer": "1 + 1 = 2 #### 2"},
                {"id": 1, "question": "2 + 2?", "answer": "2 + 2 = 4 #### 4"}]
        encoded = encode_example(tokenizer, rows[0], 256)
        self.assertIn(-100, encoded["labels"])
        first_target = next(i for i, token in enumerate(encoded["labels"]) if token != -100)
        self.assertGreater(first_target, 0)
        self.assertEqual(encoded["labels"][first_target:], encoded["input_ids"][first_target:])
        self.assertIsNone(encode_example(tokenizer, rows[0], 4))
        short = {key: value[:-1] for key, value in encoded.items()}
        padded = CompletionCollator(1)([encoded, short])
        self.assertEqual(padded["labels"][1, -1].item(), -100)
        self.assertEqual(padded["attention_mask"][1, -1].item(), 0)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            settings = dict(max_length=256, prompt="standard", rank=2, batch_size=1, accumulation=2,
                            epochs=1, max_steps=-1, learning_rate=0.01, seed=42)
            result = train(model, tokenizer, rows, output, settings)
            self.assertEqual(result["optimizer_steps"], 1)
            self.assertTrue(torch.isfinite(torch.tensor(result["training_loss"])))
            self.assertEqual(result["used_rows"], 2)
            base = Qwen2ForCausalLM(config)
            base.load_state_dict(initial_state)
            restored = PeftModel.from_pretrained(base, output / "adapter")
            adapted = [param for name, param in restored.named_parameters() if "lora_B" in name]
            self.assertTrue(any(torch.count_nonzero(param).item() > 0 for param in adapted))
            model.eval()
            restored.eval()
            probe = torch.tensor([encoded["input_ids"]])
            with torch.inference_mode():
                torch.testing.assert_close(model(probe).logits, restored(probe).logits)
            records, metrics = evaluate(restored, tokenizer, rows, max_new_tokens=4, batch_size=2)
            self.assertEqual(metrics["total"], 2)
            self.assertEqual(len(records), 2)
            for parameter in restored.parameters():
                self.assertTrue(torch.isfinite(parameter).all())


if __name__ == "__main__":
    unittest.main()
