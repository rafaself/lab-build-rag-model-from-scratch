from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from sqlalchemy import Text, cast, event, select
from sqlalchemy.schema import CreateSchema, DropSchema

from rag_lab.cli import build_parser
from rag_lab.config import Settings
from rag_lab.database import create_db_engine
from rag_lab.populate import chunk_text, chunks, documents, embed_chunks, populate_database, ensure_population_schema


class PopulationTests(unittest.TestCase):
    def test_chunk_overlap_and_no_redundant_tail(self):
        self.assertEqual(chunk_text("one two three four five six seven", 4, 1),
                         ["one two three four", "four five six seven"])
        self.assertEqual(chunk_text("one\n two", 4, 0), ["one two"])
        self.assertEqual(chunk_text(" \n ", 4, 1), [])
        for size, overlap in [(0, 0), (3, -1), (3, 3)]:
            with self.assertRaises(ValueError):
                chunk_text("text", size, overlap)

    def test_batches_preserve_chunk_order(self):
        with patch("rag_lab.populate.embed_texts", side_effect=[[[1., 0.], [0., 1.]], [[1., 1.]]]) as embed:
            result = embed_chunks(["first", "second", "third"], Settings(), "test-model", 2)
        self.assertEqual(result, [[1., 0.], [0., 1.], [1., 1.]])
        self.assertEqual([call.args[0] for call in embed.call_args_list], [["first", "second"], ["third"]])

    def test_invalid_vectors_are_rejected_before_writing(self):
        for response in ([], [[]], [[0., 0.]], [[float("nan")]], [[float("inf")]]):
            with self.subTest(response=response), patch("rag_lab.populate.embed_texts", return_value=response):
                with self.assertRaises(RuntimeError):
                    embed_chunks(["text"], Settings(), "test-model", 16)
        with patch("rag_lab.populate.embed_texts", side_effect=[[[1., 2.]], [[1., 2., 3.]]]):
            with self.assertRaisesRegex(RuntimeError, "dimensions"):
                embed_chunks(["first", "second"], Settings(), "test-model", 1)

    def test_invalid_input_does_not_connect(self):
        with tempfile.TemporaryDirectory() as directory, patch("rag_lab.populate.create_db_engine") as engine:
            with self.assertRaisesRegex(ValueError, "No .txt"):
                populate_database(Settings(), input_dir=Path(directory))
            with self.assertRaisesRegex(ValueError, "overlap"):
                populate_database(Settings(), input_dir=Path(directory), chunk_size=300, overlap=300)
            engine.assert_not_called()

    def test_cli_population_options(self):
        args = build_parser().parse_args(["populate", "--limit", "1", "--overlap", "0", "--force"])
        self.assertEqual((args.limit, args.overlap, args.force), (1, 0, True))

    def test_file_symlinks_are_not_sent_to_the_model(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "private.txt"
            secret.write_text("private fixture")
            (Path(directory) / "linked.txt").symlink_to(secret)
            with patch("rag_lab.populate.create_db_engine") as engine, patch("rag_lab.populate.embed_texts") as embed:
                with self.assertRaisesRegex(ValueError, "No .txt"):
                    populate_database(Settings(), input_dir=Path(directory))
            engine.assert_not_called()
            embed.assert_not_called()


@unittest.skipUnless(os.environ.get("RAG_TEST_POSTGRES") == "1", "Set RAG_TEST_POSTGRES=1 for isolated PostgreSQL tests.")
class PostgreSQLPopulationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_db_engine(Settings.from_env())
        self.schema = "rag_test_" + uuid.uuid4().hex
        with self.engine.begin() as connection:
            connection.execute(CreateSchema(self.schema))
        self.scoped_engine = self.engine.execution_options(schema_translate_map={None: self.schema})
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self.cleanup_schema)
        self.path = Path(self.directory.name) / "article.txt"
        self.path.write_text("one two three four five six seven", encoding="utf-8")

    def cleanup_schema(self):
        try:
            with self.engine.begin() as connection:
                connection.execute(DropSchema(self.schema, cascade=True))
        finally:
            self.engine.dispose()

    def populate(self, **overrides):
        options = dict(input_dir=Path(self.directory.name), chunk_size=4, overlap=1, batch_size=1, model="fixture", strategy="words")
        options.update(overrides)
        with patch("rag_lab.populate.create_db_engine", return_value=self.scoped_engine):
            return populate_database(Settings(), **options)

    @contextmanager
    def fixture_embeddings(self):
        # Synthetic vectors exercise real PostgreSQL storage, without model downloads.
        with patch("rag_lab.populate.embed_texts", side_effect=lambda texts, *a, **kw: [[1., 2., 3.] for _ in texts]) as embed:
            yield embed

    def snapshot(self):
        with self.scoped_engine.connect() as connection:
            docs = [dict(row) for row in connection.execute(select(documents)).mappings()]
            stored = connection.execute(select(chunks.c.content, cast(chunks.c.embedding, Text)).order_by(chunks.c.chunk_index)).all()
            return docs, [(content, json.loads(vector)) for content, vector in stored]

    def test_insert_resume_replace_force_and_separate_models(self):
        with self.fixture_embeddings() as embed:
            first = self.populate()
            self.assertEqual((first["populated"], first["chunks"]), (1, 2))
            docs, stored = self.snapshot()
            self.assertEqual(docs[0]["dimensions"], 3)
            self.assertEqual(stored, [("one two three four", [1., 2., 3.]), ("four five six seven", [1., 2., 3.])])
            embed.reset_mock()
            self.assertEqual(self.populate()["skipped"], 1)
            embed.assert_not_called()
            self.path.write_text("changed shorter article", encoding="utf-8")
            self.assertEqual(self.populate()["chunks"], 1)
            self.assertEqual(len(self.snapshot()[1]), 1)
            self.assertEqual(self.populate(force=True)["populated"], 1)
            self.assertEqual(self.populate(model="another-model")["populated"], 1)
            self.assertEqual(len(self.snapshot()[0]), 2)
            self.path.write_text(" \n ", encoding="utf-8")
            self.assertEqual(self.populate()["empty"], 1)
            self.assertEqual(len(self.snapshot()[1]), 2)

    def test_embedding_failure_keeps_old_rows_and_completed_documents(self):
        with self.fixture_embeddings():
            self.populate()
        original = self.snapshot()
        self.path.write_text("updated article", encoding="utf-8")
        with patch("rag_lab.populate.embed_texts", side_effect=RuntimeError("Ollama unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Ollama unavailable"):
                self.populate()
        self.assertEqual(self.snapshot(), original)
        other = self.path.with_name("z_second.txt")
        other.write_text("second article", encoding="utf-8")
        with patch("rag_lab.populate.embed_texts", side_effect=[[[1., 2., 3.]], RuntimeError("second failed")]):
            with self.assertRaisesRegex(RuntimeError, "second failed"):
                self.populate()
        self.assertEqual(self.snapshot()[1], [("updated article", [1., 2., 3.])])

    def test_insert_failure_rolls_back_document_and_chunk_replacement(self):
        with self.fixture_embeddings():
            self.populate()
            original = self.snapshot()
            self.path.write_text("replacement article", encoding="utf-8")

            def fail_chunk_insert(_conn, _cursor, statement, _parameters, _context, _executemany):
                if statement.startswith("INSERT INTO") and ".rag_chunks" in statement:
                    raise RuntimeError("Simulated insertion failure")

            event.listen(self.scoped_engine, "before_cursor_execute", fail_chunk_insert)
            try:
                with self.assertRaisesRegex(RuntimeError, "insertion failure"):
                    self.populate()
            finally:
                event.remove(self.scoped_engine, "before_cursor_execute", fail_chunk_insert)
            self.assertEqual(self.snapshot(), original)

    def test_token_chunk_offsets_headings_and_source_metadata_round_trip(self):
        from test_chunking import word_tokenizer
        content = "# Stars\n\nFusion provides energy.\n## Core\n\nHydrogen turns into helium."
        self.path.write_text(content, encoding="utf-8")
        original_metadata = {"filename": "article.txt", "page_id": "42", "title": "Stellar physics",
                             "url": "https://example.com/stars", "custom_field": {"edition": 2}}
        (self.path.parent / "metadata.json").write_text(json.dumps([original_metadata]))
        with self.fixture_embeddings() as embed:
            self.populate(strategy="tokens", chunk_size=12, overlap=2, tokenizer=word_tokenizer())
            with self.scoped_engine.connect() as connection:
                stored = connection.execute(select(chunks).order_by(chunks.c.chunk_index)).mappings().all()
                document = connection.execute(select(documents)).mappings().one()
                self.assertEqual(document["source_metadata"], original_metadata)
                self.assertEqual(document["chunking_config"]["strategy"], "tokens")
                self.assertEqual(stored[-1]["heading_path"], ["Stars", "Core"])
                for row in stored:
                    self.assertEqual(row["content"], content[row["start_char"]:row["end_char"]])
                    self.assertLessEqual(row["token_count"], 12)
            embed.reset_mock()
            self.assertEqual(self.populate(strategy="tokens", chunk_size=12, overlap=2, tokenizer=word_tokenizer())["skipped"], 1)
            embed.assert_not_called()

    def test_schema_upgrade_preserves_old_rows(self):
        with self.fixture_embeddings():
            self.populate()
        before = self.snapshot()[1]
        with self.scoped_engine.begin() as connection:
            schema = self.engine.dialect.identifier_preparer.quote_schema(self.schema)
            from sqlalchemy import text
            connection.execute(text(f"ALTER TABLE {schema}.rag_documents DROP COLUMN chunking_config, DROP COLUMN source_metadata"))
            connection.execute(text(f"ALTER TABLE {schema}.rag_chunks DROP COLUMN start_char, DROP COLUMN end_char, DROP COLUMN token_count, DROP COLUMN heading_path"))
        ensure_population_schema(self.scoped_engine)
        ensure_population_schema(self.scoped_engine)
        self.assertEqual(self.snapshot()[1], before)
        with self.scoped_engine.connect() as connection:
            row = connection.execute(select(documents)).mappings().one()
            self.assertEqual(row["chunking_config"], {})

    def test_strategy_change_reembeds_an_unchanged_markdown_file(self):
        from test_chunking import word_tokenizer
        markdown = self.path.with_suffix(".md")
        self.path.rename(markdown)
        with self.fixture_embeddings() as embed:
            self.populate()
            embed.reset_mock()
            result = self.populate(strategy="tokens", tokenizer=word_tokenizer())
            self.assertEqual(result["populated"], 1)
            self.assertTrue(embed.called)

    def test_crlf_sources_keep_original_newlines_offsets_and_hash(self):
        import hashlib
        from test_chunking import word_tokenizer
        original = b"# Stars\r\n\r\nFusion provides energy.\r\n"
        self.path.write_bytes(original)
        with self.fixture_embeddings():
            self.populate(strategy="tokens", chunk_size=12, overlap=2, tokenizer=word_tokenizer())
        with self.scoped_engine.connect() as connection:
            document = connection.execute(select(documents)).mappings().one()
            row = connection.execute(select(chunks)).mappings().one()
        self.assertEqual(document["content_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(row["content"], original.decode()[row["start_char"]:row["end_char"]])
        self.assertIn("\r\n\r\n", row["content"])


if __name__ == "__main__":
    unittest.main()
