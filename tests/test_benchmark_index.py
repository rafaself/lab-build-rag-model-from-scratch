import unittest
from unittest.mock import patch

from rag_lab.benchmark_index import benchmark_indexes, neighbor_recall, percentile
from rag_lab.config import Settings


class IndexBenchmarkTests(unittest.TestCase):
    def test_neighbor_recall_uses_chunk_identity_not_just_document_names(self):
        exact = [{"source": "article", "chunk_index": 0}, {"source": "article", "chunk_index": 1}]
        approximate = [{"source": "article", "chunk_index": 0}, {"source": "article", "chunk_index": 2}]
        self.assertEqual(neighbor_recall(exact, approximate, 2), 0.5)
        self.assertEqual(neighbor_recall(exact, approximate, 1), 1.)

    def test_p95_uses_nearest_rank(self):
        self.assertEqual(percentile(list(range(1, 21)), .95), 19)

    def test_invalid_scale_or_repeat_count_fails_before_database(self):
        with patch("rag_lab.benchmark_index.create_db_engine") as factory:
            for options in ({"sizes": []}, {"sizes": [1]}, {"sizes": [100001]}, {"repeats": 0}):
                with self.assertRaises(ValueError):
                    benchmark_indexes(Settings(), **options)
        factory.assert_not_called()
