from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_lab.cli import main
from rag_lab.config import Settings
from rag_lab.evaluate import DEFAULT_DATASET, load_dataset, retrieval_metrics, summarize


class EvaluationTests(unittest.TestCase):
    def test_checked_in_dataset_has_answerable_and_abstention_cases(self):
        rows = load_dataset(DEFAULT_DATASET)
        self.assertEqual(len(rows), 14)
        self.assertEqual(sum(r["unanswerable"] for r in rows), 2)
        self.assertEqual(len({r["id"] for r in rows}), 14)

    def test_duplicate_ids_bad_sources_and_empty_dataset_are_rejected(self):
        valid = {"id": "q", "question": "What?", "unanswerable": False, "sources": ["sun.txt"],
                 "expected_answer": "fusion", "gold_quote": "nuclear fusion"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            for rows in ([], [valid, valid], [{**valid, "sources": ["../secret"]}],
                         [{**valid, "gold_quote": []}], [{**valid, "unanswerable": True}]):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(ValueError):
                    load_dataset(path)

    def test_unanswerable_cases_do_not_inflate_recall(self):
        answerable = {"sources": ["sun.txt"], "gold_quote": "fusion"}
        unanswerable = {"sources": [], "gold_quote": None}
        matches = [{"source": "sun.txt", "text": "fusion"}]
        rows = [
            {"unanswerable": False, "v": {str(k): retrieval_metrics(answerable, matches, k) for k in (1, 3, 5)}},
            {"unanswerable": True, "v": {str(k): retrieval_metrics(unanswerable, matches, k) for k in (1, 3, 5)}},
        ]
        summary = summarize(rows, "v")
        self.assertEqual(summary["document_recall@1"], 1)
        self.assertEqual(summary["unanswerable_returned_match_rate"], 1)

    def test_cli_saves_a_parseable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports/metrics.json"
            with patch("rag_lab.cli.Settings.from_env", return_value=Settings()), patch("rag_lab.evaluate.evaluate_retrieval", return_value={"cases": 14}), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["evaluate", "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text()), {"cases": 14})
