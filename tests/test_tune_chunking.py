import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from sqlalchemy.schema import CreateSchema, DropSchema

from rag_lab.config import Settings
from rag_lab.tune_chunking import tune_chunking
from rag_lab.evaluate import DEFAULT_DATASET, load_dataset


def evaluation(hit, false_matches=0, rank1=0.5):
    return {"improved": {"evidence_hit@3": hit, "unanswerable_returned_match_rate": false_matches,
                         "evidence_hit@1": rank1}}


class ChunkTuningTests(unittest.TestCase):
    def engine(self):
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value.execute.return_value.mappings.return_value.all.return_value = []
        return engine

    def test_selection_uses_evidence_quality_and_cleans_benchmark_schemas(self):
        engine = self.engine()
        with patch("rag_lab.tune_chunking.create_db_engine", return_value=engine), patch("rag_lab.tune_chunking.ensure_population_schema"), patch("rag_lab.tune_chunking.populate_database", return_value={"chunks": 20}) as populate, patch("rag_lab.tune_chunking.evaluate_retrieval", side_effect=[evaluation(0.5), evaluation(0.8, 0.5), evaluation(0.8, 0)]):
            result = tune_chunking(Settings(), variants=((256, 32), (384, 64)))
        self.assertEqual(result["selected"], {"chunk_size": 384, "overlap": 64})
        self.assertEqual(len(result["variants"]), 2)
        isolated_settings = populate.call_args_list[0].args[0]
        self.assertTrue(isolated_settings.postgres_schema.startswith("rag_chunk_tune_"))
        self.assertIsNone(Settings().postgres_schema)
        statements = [call.args[0] for call in engine.begin.return_value.__enter__.return_value.execute.call_args_list]
        self.assertEqual(sum(isinstance(statement, CreateSchema) for statement in statements), 2)
        self.assertEqual(sum(isinstance(statement, DropSchema) for statement in statements), 2)

    def test_failed_ingestion_still_drops_only_its_temporary_schema(self):
        engine = self.engine()
        with patch("rag_lab.tune_chunking.create_db_engine", return_value=engine), patch("rag_lab.tune_chunking.ensure_population_schema"), patch("rag_lab.tune_chunking.evaluate_retrieval", return_value=evaluation(0.5)), patch("rag_lab.tune_chunking.populate_database", side_effect=RuntimeError("embedding unavailable")):
            with self.assertRaisesRegex(RuntimeError, "embedding unavailable"):
                tune_chunking(Settings(), variants=((256, 32),))
        statements = [call.args[0] for call in engine.begin.return_value.__enter__.return_value.execute.call_args_list]
        created = next(statement for statement in statements if isinstance(statement, CreateSchema))
        dropped = next(statement for statement in statements if isinstance(statement, DropSchema))
        self.assertEqual(created.element, dropped.element)
        self.assertTrue(dropped.element.startswith("rag_chunk_tune_"))

    def test_answerable_only_dataset_has_no_false_match_metric(self):
        engine = self.engine()
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "one-case.jsonl"
            dataset.write_text(json.dumps(load_dataset(DEFAULT_DATASET)[0]) + "\n")
            with patch("rag_lab.tune_chunking.create_db_engine", return_value=engine), patch("rag_lab.tune_chunking.ensure_population_schema"), patch("rag_lab.tune_chunking.populate_database", return_value={"chunks": 20}), patch("rag_lab.tune_chunking.evaluate_retrieval", return_value=evaluation(1, None)):
                result = tune_chunking(Settings(), dataset=dataset, variants=((512, 64),))
        self.assertEqual(result["selected"], {"chunk_size": 512, "overlap": 64})
