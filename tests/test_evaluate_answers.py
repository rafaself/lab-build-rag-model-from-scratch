from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_lab.cli import main
from rag_lab.config import Settings
from rag_lab.evaluate import checked_sources
from rag_lab.evaluate_answers import audit_citations, evaluate_answers, grade_answer, summarize_answers

CASE = {"question": "What powers the Sun?", "expected_answer": "Nuclear fusion.",
        "gold_quote": "The Sun is powered by nuclear fusion.", "unanswerable": False}
CLAIM = {"source_id": 1, "text": "The Sun is powered by nuclear fusion.", "quote": "powered by nuclear fusion"}
ANSWER = {"status": "answered", "claims": [CLAIM],
          "sources": [{"id": 1, "source": "sun.txt", "text": "The Sun is powered by nuclear fusion."}]}


class AnswerEvaluationTests(unittest.TestCase):
    def test_judge_separates_reference_accuracy_from_context_faithfulness(self):
        verdict = {"correct": False, "faithful": [True], "reason": "Incomplete answer."}
        with patch("rag_lab.evaluate_answers.generate_structured_response", return_value=verdict) as generate:
            result = grade_answer(CASE, ANSWER, Settings(), judge_model="judge")
        self.assertEqual(result, verdict)
        messages = generate.call_args.args[0]
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["reference_answer"], CASE["expected_answer"])
        self.assertEqual(payload["retrieved_evidence"][0]["text"], ANSWER["sources"][0]["text"])
        self.assertEqual(generate.call_args.kwargs["model"], "judge")

    def test_judge_malformed_verdicts_are_errors_not_passes(self):
        for bad in ({"correct": "yes", "faithful": [True], "reason": "x"},
                    {"correct": True, "faithful": [], "reason": "x"},
                    {"correct": True, "faithful": [1], "reason": "x"}):
            with patch("rag_lab.evaluate_answers.generate_structured_response", return_value=bad):
                with self.assertRaisesRegex(RuntimeError, "invalid verdict"):
                    grade_answer(CASE, ANSWER, Settings())

    def test_private_answer_cannot_be_graded_as_successful_abstention(self):
        with patch("rag_lab.evaluate_answers.generate_structured_response", return_value={"correct": True, "faithful": [True], "reason": "x"}):
            verdict = grade_answer({**CASE, "unanswerable": True}, ANSWER, Settings())
        self.assertFalse(verdict["correct"])

    def test_abstentions_errors_and_empty_denominators_are_reported_honestly(self):
        rows = [
            {"unanswerable": False, "answer": ANSWER, "verdict": {"correct": True, "faithful": [True]}, "citations": audit_citations(ANSWER, ["sun.txt"])},
            {"unanswerable": False, "answer": {"status": "abstained"}, "verdict": {"correct": False, "faithful": []}},
            {"unanswerable": False, "error_stage": "generation"},
            {"unanswerable": True, "answer": {"status": "abstained"}, "verdict": {"correct": True, "faithful": []}},
        ]
        summary = summarize_answers(rows)
        self.assertEqual(summary["accuracy_all_cases"], .5)
        self.assertEqual(summary["answerable_accuracy"], 1 / 3)
        self.assertEqual(summary["unanswerable_abstention_rate"], 1)
        self.assertEqual(summary["judged_claims"], 1)
        self.assertEqual(summary["claim_faithfulness"], 1)
        self.assertEqual(summary["generation_errors"], 1)
        empty = summarize_answers([])
        self.assertIsNone(empty["accuracy_all_cases"])
        self.assertIsNone(empty["claim_faithfulness"])

    def test_quote_validity_and_expected_document_are_different_metrics(self):
        self.assertEqual(audit_citations(ANSWER, ["other.txt"]),
                         {"claims": 1, "valid_source_and_quote": 1, "expected_source_citations": 0})
        bad = {**ANSWER, "claims": [{**CLAIM, "quote": "fabricated evidence quote"}]}
        self.assertEqual(audit_citations(bad, ["sun.txt"])["valid_source_and_quote"], 0)

    def test_failed_generation_and_failed_judgment_remain_in_report_without_raw_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "cases.jsonl"
            dataset.write_text("".join(json.dumps({**CASE, "id": str(i), "sources": ["sun.txt"]}) + "\n" for i in range(3)))
            with patch("rag_lab.evaluate_answers.checked_sources", return_value={}), patch("rag_lab.evaluate_answers.validate_indexed_sources", return_value=[]), patch("rag_lab.evaluate_answers.model_digests", return_value={}), patch("rag_lab.evaluate_answers.ask_question", side_effect=[ANSWER, RuntimeError("fake-private-value"), ANSWER]), patch("rag_lab.evaluate_answers.grade_answer", side_effect=[{"correct": True, "faithful": [True]}, RuntimeError("fake-private-value")]):
                report = evaluate_answers(Settings(), dataset=dataset)
        self.assertEqual(report["summary"]["cases"], 3)
        self.assertEqual(report["summary"]["accuracy_all_cases"], 1 / 3)
        self.assertEqual(report["summary"]["generation_errors"], 1)
        self.assertEqual(report["summary"]["judge_errors"], 1)
        self.assertEqual(report["summary"]["judged_claims"], 1)
        self.assertNotIn("fake-private-value", json.dumps(report))

    def test_reference_source_symlinks_cannot_escape_the_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            root.mkdir()
            outside = Path(directory) / "outside.txt"
            outside.write_text("private data")
            (root / "sun.txt").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "inside the corpus"):
                checked_sources([{**CASE, "sources": ["sun.txt"]}], root)

    def test_cli_saves_answer_report_and_passes_model_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch("rag_lab.cli.Settings.from_env", return_value=Settings()), patch("rag_lab.evaluate_answers.evaluate_answers", return_value={"summary": {}}) as evaluate, redirect_stdout(io.StringIO()):
                self.assertEqual(main(["evaluate-answers", "--judge-model", "judge", "--output", str(output)]), 0)
            self.assertEqual(evaluate.call_args.kwargs["judge_model"], "judge")
            self.assertEqual(json.loads(output.read_text()), {"summary": {}})
