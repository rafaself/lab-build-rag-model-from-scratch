import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from rag_lab.corpus import CorpusConfig, WikipediaClient, generate_corpus, retry_after_seconds


def response(status, body=None, headers=None):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers or {})
    result._content = json.dumps(body or {}).encode()
    return result


class WikipediaRequestTests(unittest.TestCase):
    def config(self, directory):
        return CorpusConfig(Path(directory), "https://example.com/lab", request_delay=0)

    def test_connection_failure_does_not_reuse_prior_retry_after(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            with WikipediaClient(config) as client, patch.object(client.session, "get") as get:
                get.side_effect = [response(429, headers={"Retry-After": "7"}), requests.ConnectionError("temporary failure"), response(200, {"ok": True})]
                with patch("rag_lab.corpus.time.sleep") as sleep, patch("rag_lab.corpus.random.uniform", return_value=0):
                    self.assertEqual(client.request({}), {"ok": True})
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [7, 0, 4, 0, 0])

    def test_permanent_http_error_is_not_retried_and_session_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("rag_lab.corpus.requests.Session") as session_factory, patch("rag_lab.corpus.time.sleep"):
                session = session_factory.return_value
                session.get.return_value = response(400)
                with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                    with WikipediaClient(self.config(directory)) as client:
                        client.request({})
                session.get.assert_called_once()
                session.close.assert_called_once()

    def test_pagination_collects_unique_articles(self):
        with tempfile.TemporaryDirectory() as directory, WikipediaClient(self.config(directory)) as client:
            first = {"pageid": 1, "title": "First"}
            second = {"pageid": 2, "title": "Second"}
            with patch.object(client, "request", side_effect=[
                {"query": {"search": [first]}, "continue": {"sroffset": 1}},
                {"query": {"search": [first, second]}},
            ]):
                self.assertEqual(client.search(), [first, second])

    def test_retry_after_accepts_seconds_and_dates(self):
        self.assertEqual(retry_after_seconds("8"), 8)
        self.assertEqual(retry_after_seconds("Wed, 01 Jan 2020 00:00:00 GMT"), 0)
        self.assertIsNone(retry_after_seconds("invalid"))


class CorpusResumeTests(unittest.TestCase):
    def test_resume_preserves_existing_articles_and_recovers_failed_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "1_saved.txt").write_text("original text")
            (output / "metadata.json").write_text(json.dumps([
                {"page_id": "1", "filename": "1_saved.txt", "word_count": 2},
                {"page_id": "2", "filename": "2_missing.txt", "word_count": 10},
            ]))
            (output / "failed_articles.json").write_text(json.dumps([{"page_id": "2", "error": "old failure"}]))
            config = CorpusConfig(output, "https://example.com/lab", num_articles=3)
            search = [{"pageid": i, "title": f"Article {i}"} for i in range(1, 4)]
            with patch("rag_lab.corpus.WikipediaClient") as factory:
                client = factory.return_value.__enter__.return_value
                client.search.return_value = search
                client.fetch.side_effect = [{"title": "Recovered", "extract": "three recovered words"}, RuntimeError("temporary failure")]
                result = generate_corpus(config)
                self.assertEqual((result["saved"], result["skipped"], result["failed"], result["total_words"]), (1, 1, 1, 5))
                self.assertEqual((output / "1_saved.txt").read_text(), "original text")
                self.assertEqual(client.fetch.call_args_list[0].args, ("2",))
                failures = json.loads((output / "failed_articles.json").read_text())
                self.assertEqual([item["page_id"] for item in failures], ["3"])
                client.fetch.side_effect = None
                client.fetch.return_value = {"title": "Recovered again", "extract": "now saved"}
                result = generate_corpus(config)
                self.assertEqual((result["saved"], result["skipped"], result["failed"]), (1, 2, 0))
                self.assertEqual(json.loads((output / "failed_articles.json").read_text()), [])
                records = json.loads((output / "metadata.json").read_text())
                self.assertEqual({item["page_id"] for item in records}, {"1", "2", "3"})

    def test_corrupt_metadata_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_text("invalid json")
            with self.assertRaisesRegex(RuntimeError, "inspect the file"):
                generate_corpus(CorpusConfig(Path(directory), "https://example.com/lab"))
            self.assertEqual(path.read_text(), "invalid json")


if __name__ == "__main__":
    unittest.main()
