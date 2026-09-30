import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from gsm8k_lab.answers import boxed_content, number, score, summarize
from gsm8k_lab.cli import compare, parser
from gsm8k_lab.data import select_indices
from gsm8k_lab.prompts import PROMPTS, completion, messages
from gsm8k_lab.runtime import new_output


class ScoringTests(unittest.TestCase):
    def test_exact_numeric_forms(self):
        for text in ("0.5", "1/2", r"\frac{1}{2}", r"\dfrac{2}{4}", "5e-1"):
            with self.subTest(text=text):
                self.assertEqual(number(text), Fraction(1, 2))
        self.assertEqual(number("1,200.5"), Fraction(2401, 2))
        self.assertEqual(number("-3/2"), Fraction(-3, 2))

    def test_invalid_not_coerced(self):
        for text in (None, "", "1,5", "50%", "1/0", "nan", "2+3", "__import__('os')", "$5"):
            with self.subTest(text=text):
                self.assertIsNone(number(text))
        self.assertTrue(score("No answer", "0")["invalid"])
        self.assertFalse(score(r"\boxed{}", "0")["correct"])

    def test_boxes_and_fallback(self):
        self.assertEqual(boxed_content(r"\boxed{4} then \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertTrue(score(r"\boxed{\frac{1}{2}}", "0.5")["strict_correct"])
        self.assertFalse(score(r"\boxed{7", "7")["correct"])
        self.assertTrue(score("The answer is 7.", "7")["correct"])
        self.assertFalse(score("The answer is 7.", "7")["strict_correct"])
        self.assertFalse(score("50%", "50")["correct"])
        self.assertFalse(score(r"\boxed{wrong} 7", "7")["correct"])

    def test_metrics_and_invalid_reference(self):
        result = summarize([score(r"\boxed{7}", "7"), score("7", "7"), score("no", "7")])
        self.assertEqual(result["correct"], 2 / 3)
        self.assertEqual(result["strict_correct"], 1 / 3)
        with self.assertRaises(ValueError):
            score("", "")
        with self.assertRaises(ValueError):
            summarize([])


class ExperimentTests(unittest.TestCase):
    def test_splits_are_disjoint_and_nested(self):
        train = set(select_indices(7473, "train", 2000, 42))
        dev = set(select_indices(7473, "dev", 200, 42))
        shots = set(select_indices(7473, "shots", 3, 42))
        self.assertFalse(train & dev or train & shots or dev & shots)
        self.assertTrue(set(select_indices(7473, "train", 1000, 42)) <= train)
        self.assertEqual(select_indices(7473, "dev", 100, 42), select_indices(7473, "dev", 100, 42))
        with self.assertRaises(ValueError):
            select_indices(20, "train", 1, 42)

    def test_prompts_have_no_escape_corruption(self):
        for prompt in PROMPTS.values():
            self.assertIn(r"\boxed{answer}", prompt)
            self.assertNotIn("\b", prompt)
        self.assertEqual(completion("Add two. #### 2"), "Add two.\n" + r"\boxed{2}")
        roles = [entry["role"] for entry in messages("q", "standard", [{"question": "q0", "answer": "r #### 1"}])]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_output_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                new_output(folder)

    def test_compare_rejects_different_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / "a.json", Path(folder) / "b.json"]
            for index, path in enumerate(paths):
                path.write_text(json.dumps({"comparison_key": {"rows": [index]}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                compare(paths)

    def test_cli_defaults(self):
        args = parser().parse_args(["evaluate", "--output", "runs/test"])
        self.assertEqual(args.split, "dev")
        self.assertEqual(args.model, "Qwen/Qwen2.5-1.5B-Instruct")


if __name__ == "__main__":
    unittest.main()
