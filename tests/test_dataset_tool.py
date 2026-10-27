import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "scripts" / "dataset_tool.py"
SPEC = importlib.util.spec_from_file_location("dataset_tool", TOOL_PATH)
TOOL = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(TOOL)


class DatasetToolTest(unittest.TestCase):
    def make_samples(self):
        rows = []
        for account in ("a", "b"):
            for index, views in enumerate((200, 400, 1000, 2000, 5000, 9000)):
                rows.append(
                    {
                        "sample_id": f"{account}-{index}",
                        "account_id": account,
                        "transcript": f"这是账号 {account} 的合成测试文稿 {index}，不代表真实预测证据。",
                        "views": views,
                        "likes": max(1, views // 20),
                        "comments": max(0, views // 200),
                        "shares": max(0, views // 300),
                        "account_baseline_views": 2000,
                    }
                )
        return rows

    def test_build_freeze_and_evaluate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_path = root / "samples.jsonl"
            with raw_path.open("w", encoding="utf-8") as handle:
                for row in self.make_samples():
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            run_dir = root / "run"
            manifest = TOOL.build_dataset(raw_path, run_dir, 1, 7)
            self.assertEqual(manifest["blind_count"], 3)
            blind_rows = TOOL.read_jsonl(run_dir / "blind" / "blind_inputs.jsonl")
            for row in blind_rows:
                self.assertEqual(set(row), {"blind_id", "transcript"})

            labels = {
                row["blind_id"]: row
                for row in TOOL.read_jsonl(run_dir / "vault" / "sealed_labels.jsonl")
            }
            predictions = []
            for blind_id, label in labels.items():
                actual = label["actual_bucket"]
                probabilities = {bucket: 0.05 for bucket in TOOL.BUCKETS}
                probabilities[actual] = 0.90
                predictions.append(
                    {
                        "blind_id": blind_id,
                        "rubric_version": "test-v0",
                        "probabilities": probabilities,
                        "predicted_percentile": label["actual_percentile"],
                    }
                )
            predictions_path = root / "predictions.jsonl"
            TOOL.write_jsonl(predictions_path, predictions)
            frozen_path = root / "frozen.json"
            manifest_path = run_dir / "blind" / "manifest.json"
            TOOL.freeze_predictions(predictions_path, manifest_path, frozen_path)
            report = TOOL.evaluate_predictions(
                frozen_path,
                run_dir / "vault" / "sealed_labels.jsonl",
                manifest_path,
                root / "evaluation.json",
            )
            self.assertEqual(report["bucket_accuracy"], 1.0)
            self.assertEqual(report["percentile_mae"], 0.0)

            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            frozen["predictions"][0]["predicted_percentile"] = 0
            frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "modified after freeze"):
                TOOL.evaluate_predictions(
                    frozen_path,
                    run_dir / "vault" / "sealed_labels.jsonl",
                    manifest_path,
                    root / "tampered.json",
                )

    def test_likes_can_be_the_uniform_outcome(self):
        rows = self.make_samples()
        for row in rows:
            row["outcome_metric"] = "likes"
            row["account_baseline_likes"] = 100
            row.pop("account_baseline_views")
        labelled = TOOL.label_samples(rows, 0.65, 1.55)
        self.assertEqual({row["_outcome_metric"] for row in labelled}, {"likes"})
        self.assertEqual({row["_bucket"] for row in labelled}, {"low", "medium", "high"})


if __name__ == "__main__":
    unittest.main()
