import unittest
from unittest.mock import patch

from rag_lab.config import Settings
from rag_lab.evaluate import retrieval_metrics
from rag_lab.pull import pull_chunks
from rag_lab.retrieval import bm25_scores, rank_candidates


def match(name, text, similarity):
    return {"source": name, "chunk_index": 0, "text": text, "similarity": similarity}


class RetrievalQualityTests(unittest.TestCase):
    def test_threshold_is_applied_to_raw_cosine_before_reranking(self):
        rows = [match("low", "Kepler Tycho Prague", 0.2), match("high", "astronomy", 0.7)]
        result = rank_candidates(rows, "Kepler Tycho Prague", top_k=5, min_similarity=0.3)
        self.assertEqual([r["source"] for r in result], ["high"])

    def test_reranking_can_promote_a_lexically_relevant_candidate(self):
        rows = [match("generic", "stars and planets in space", 0.61),
                match("specific", "Kepler assisted Tycho Brahe in Prague", 0.6)]
        ranked = rank_candidates(rows, "Who did Kepler assist in Prague?", top_k=2)
        self.assertEqual(ranked[0]["source"], "specific")
        self.assertEqual(rank_candidates(rows, "Who did Kepler assist in Prague?", top_k=2, rerank=False)[0]["source"], "generic")

    def test_exact_and_near_duplicates_are_removed_without_mutating_input(self):
        text = " ".join(f"word{i}" for i in range(100))
        rows = [match("a", text, 0.9), match("b", text.upper(), 0.85),
                match("c", text + " extra", 0.8), match("d", "distinct document content", 0.7)]
        self.assertEqual([r["source"] for r in rank_candidates(rows, "query", top_k=5)], ["a", "d"])
        self.assertEqual(len(rank_candidates(rows, "query", top_k=5, deduplicate=False)), 4)
        self.assertNotIn("rerank_score", rows[0])

    def test_bm25_handles_empty_queries_unicode_and_zero_length_candidates(self):
        self.assertEqual(bm25_scores("the and", ["", "stars"]), [0., 0.])
        self.assertGreater(bm25_scores("天文", ["天文", "stars"])[0], 0)
        self.assertEqual(rank_candidates([], "query", top_k=3), [])

    def test_bad_retrieval_limits_fail_before_database(self):
        with patch("rag_lab.pull.create_db_engine") as engine:
            for options in ({"min_similarity": float("nan")}, {"min_similarity": 1.1},
                            {"candidate_k": 2}, {"candidate_k": 1001}, {"search_mode": "invalid"},
                            {"ef_search": 0}, {"ef_search": 1001}):
                with self.assertRaises(ValueError):
                    pull_chunks("query", Settings(), **options)
        engine.assert_not_called()

    def test_document_recall_and_evidence_hit_are_different_metrics(self):
        case = {"sources": ["sun.txt", "gravity.txt"], "gold_quote": "nuclear fusion"}
        matches = [match("/corpus/sun.txt", "unrelated solar passage", 0.9)]
        metrics = retrieval_metrics(case, matches, 3)
        self.assertEqual(metrics["document_recall"], 0.5)
        self.assertFalse(metrics["evidence_hit"])
        matches.append(match("/corpus/sun.txt", "powered by nuclear fusion", 0.8))
        self.assertTrue(retrieval_metrics(case, matches, 3)["evidence_hit"])
