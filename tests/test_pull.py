from contextlib import redirect_stdout
import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch
import uuid

from sqlalchemy import func, select
from sqlalchemy.schema import CreateSchema, DropSchema

from rag_lab.cli import main
from rag_lab.config import Settings
from rag_lab.database import create_db_engine
from rag_lab.populate import chunks, documents, metadata
from rag_lab.pull import pull_chunks


class PullTests(unittest.TestCase):
    def test_invalid_query_or_limit_does_not_connect(self):
        with patch("rag_lab.pull.create_db_engine") as factory:
            for query, count in [(" \n", 5), ("query", 0), ("query", -1), ("query", 1001)]:
                with self.assertRaises(ValueError):
                    pull_chunks(query, Settings(), top_k=count)
            factory.assert_not_called()

    def engine(self, dimensions):
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value.scalars.return_value = dimensions
        return engine

    def test_empty_model_has_no_matches_and_does_not_request_embeddings(self):
        engine = self.engine([])
        with patch("rag_lab.pull.create_db_engine", return_value=engine), patch("rag_lab.pull.embed_texts") as embed:
            self.assertEqual(pull_chunks("query", Settings()), [])
        embed.assert_not_called()
        engine.dispose.assert_called_once()

    def test_mixed_stored_dimensions_fail_before_embedding(self):
        engine = self.engine([2, 3])
        with patch("rag_lab.pull.create_db_engine", return_value=engine), patch("rag_lab.pull.embed_texts") as embed:
            with self.assertRaisesRegex(RuntimeError, "mixed dimensions"):
                pull_chunks("query", Settings())
        embed.assert_not_called()
        engine.dispose.assert_called_once()

    def test_invalid_or_incompatible_query_vectors_close_the_engine(self):
        for vectors in ([], [[]], [[0., 0.]], [[float("nan"), 1.]], [[1., 2., 3.]], [[1., 2.], [3., 4.]]):
            engine = self.engine([2])
            with self.subTest(vectors=vectors), patch("rag_lab.pull.create_db_engine", return_value=engine), patch("rag_lab.pull.embed_texts", return_value=vectors):
                with self.assertRaises(RuntimeError):
                    pull_chunks("query", Settings())
            engine.dispose.assert_called_once()

    def test_cli_prints_retrieved_text_and_metadata_as_json(self):
        result = [{"source": "fixture.txt", "chunk_index": 1, "similarity": 0.9, "text": "matching text"}]
        output = io.StringIO()
        with patch("rag_lab.cli.Settings.from_env", return_value=Settings()), patch("rag_lab.pull.pull_chunks", return_value=result) as pull, redirect_stdout(output):
            self.assertEqual(main(["pull", "planets", "--top-k", "2", "--model", "fixture"]), 0)
        pull.assert_called_once_with("planets", Settings(), top_k=2, model="fixture",
                                     min_similarity=None, candidate_k=None, deduplicate=True, rerank=True)
        self.assertEqual(json.loads(output.getvalue()), {"query": "planets", "model": "fixture", "matches": result})


@unittest.skipUnless(os.environ.get("RAG_TEST_POSTGRES") == "1", "Set RAG_TEST_POSTGRES=1 for isolated PostgreSQL tests.")
class PostgreSQLPullTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_db_engine(Settings.from_env())
        self.schema = "rag_pull_test_" + uuid.uuid4().hex
        with self.engine.begin() as connection:
            connection.execute(CreateSchema(self.schema))
        self.addCleanup(self.cleanup_schema)
        self.scoped_engine = self.engine.execution_options(schema_translate_map={None: self.schema})
        with self.scoped_engine.begin() as connection:
            metadata.create_all(connection)
            for source, model, vector in [
                ("nearest.txt", "fixture", [1., 0.]),
                ("middle.txt", "fixture", [1., 1.]),
                ("opposite.txt", "fixture", [-1., 0.]),
                ("other-model.txt", "other", [1., 0., 0.]),
            ]:
                connection.execute(documents.insert().values(
                    source=source, model=model, content_sha256="fixture", chunk_size=300, overlap=50,
                    dimensions=len(vector), chunk_count=1,
                ))
                connection.execute(chunks.insert().values(
                    source=source, model=model, chunk_index=0, content="Text from " + source, embedding=vector,
                ))

    def cleanup_schema(self):
        try:
            with self.engine.begin() as connection:
                connection.execute(DropSchema(self.schema, cascade=True))
        finally:
            self.engine.dispose()

    def test_cosine_ranking_limit_model_isolation_and_no_writes(self):
        with patch("rag_lab.pull.create_db_engine", return_value=self.scoped_engine), patch("rag_lab.pull.embed_texts", return_value=[[1., 0.]]) as embed:
            matches = pull_chunks("query", Settings(), model="fixture", top_k=2)
            self.assertEqual([match["source"] for match in matches], ["nearest.txt", "middle.txt"])
            self.assertAlmostEqual(matches[0]["similarity"], 1.)
            self.assertAlmostEqual(matches[1]["similarity"], 2 ** -0.5)
            self.assertEqual(matches[0]["text"], "Text from nearest.txt")
            embed.assert_called_once_with(["query"], Settings(), model="fixture")
            matches = pull_chunks("query", Settings(), model="fixture", top_k=10)
            self.assertEqual(len(matches), 3)
            self.assertAlmostEqual(matches[-1]["similarity"], -1.)
            embed.reset_mock()
            self.assertEqual(pull_chunks("query", Settings(), model="missing' OR 1=1 --"), [])
            embed.assert_not_called()
        with self.scoped_engine.connect() as connection:
            self.assertEqual(connection.scalar(select(func.count()).select_from(documents)), 4)
            self.assertEqual(connection.scalar(select(func.count()).select_from(chunks)), 4)

    def test_dimension_mismatch_has_clear_error(self):
        with patch("rag_lab.pull.create_db_engine", return_value=self.scoped_engine), patch("rag_lab.pull.embed_texts", return_value=[[1., 0., 0.]]):
            with self.assertRaisesRegex(RuntimeError, "different dimensions"):
                pull_chunks("query", Settings(), model="fixture")

    def test_threshold_and_deduplication_apply_to_real_vector_results(self):
        with self.scoped_engine.begin() as connection:
            connection.execute(documents.insert().values(
                source="nearest-copy.txt", model="fixture", content_sha256="fixture", chunk_size=300,
                overlap=50, dimensions=2, chunk_count=1,
            ))
            connection.execute(chunks.insert().values(
                source="nearest-copy.txt", model="fixture", chunk_index=0,
                content="Text from nearest.txt", embedding=[1., 0.],
            ))
        with patch("rag_lab.pull.create_db_engine", return_value=self.scoped_engine), patch("rag_lab.pull.embed_texts", return_value=[[1., 0.]]):
            matches = pull_chunks("query", Settings(), model="fixture", top_k=5, min_similarity=0.8)
            self.assertEqual(len(matches), 1)
            self.assertAlmostEqual(matches[0]["similarity"], 1.)
            self.assertEqual(len(pull_chunks("query", Settings(), model="fixture", top_k=5, min_similarity=0.8, deduplicate=False)), 2)


if __name__ == "__main__":
    unittest.main()
