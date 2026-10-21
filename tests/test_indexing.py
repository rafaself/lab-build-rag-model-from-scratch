import unittest

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

from rag_lab.indexing import candidate_statement, hnsw_definition, index_name
from rag_lab.populate import chunks


class IndexingTests(unittest.TestCase):
    def test_partial_expression_index_is_model_specific_and_safely_quoted(self):
        before = set(chunks.indexes)
        model = "model'; DROP TABLE rag_chunks; --"
        definition = hnsw_definition(model, 1024, 'schema"quoted')
        sql = str(CreateIndex(definition).compile(dialect=postgresql.dialect()))
        self.assertIn("USING hnsw", sql)
        self.assertIn("vector_cosine_ops", sql)
        self.assertIn("VECTOR(1024)", sql)
        self.assertIn("model''; DROP TABLE", sql)
        self.assertIn('"schema""quoted"', sql)
        self.assertEqual(chunks.indexes, before)
        self.assertNotEqual(index_name("a", 1024), index_name("b", 1024))

    def test_metadata_joins_happen_after_materialized_vector_limit(self):
        statement = candidate_statement("fixture", [1., 0.], 40)
        sql = str(statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
        self.assertIn("AS MATERIALIZED", sql)
        inner = sql[:sql.index("\n SELECT")]
        self.assertNotIn("rag_documents", inner)
        self.assertIn("LIMIT 40", inner)
        self.assertIn("ORDER BY distance", inner)

    def test_exact_mode_disables_approximate_operator_ordering(self):
        statement = candidate_statement("fixture", [1., 0.], 40, exact=True)
        sql = str(statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
        self.assertIn(" + 0", sql)
        self.assertEqual(sql.count("<=>"), 1)

    def test_unsupported_index_dimensions_have_a_clear_error(self):
        for dimensions in (0, 2001):
            with self.assertRaisesRegex(ValueError, "2000"):
                hnsw_definition("fixture", dimensions, "public")
