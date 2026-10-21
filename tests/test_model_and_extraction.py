from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ollama import ResponseError

from rag_lab.config import Settings
from rag_lab.extraction import extract_text
from rag_lab.ollama_client import embed_texts, generate_answer, generate_structured_response


class ModelTests(unittest.TestCase):
    def test_embedding_batches_use_the_embedding_model_and_close_the_client(self):
        with patch("rag_lab.ollama_client.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.embed.return_value = SimpleNamespace(embeddings=[[1.0, 0.0], [0.0, 1.0]])
            self.assertEqual(len(embed_texts(["one", "two"], Settings())), 2)
            self.assertEqual(client.embed.call_args.kwargs["model"], "qwen3-embedding:0.6b")
            self.assertFalse(client.embed.call_args.kwargs["truncate"])
            client.chat.assert_not_called()
            factory.return_value.__exit__.assert_called_once()

    def test_chat_uses_the_chat_model_and_returns_only_the_answer(self):
        with patch("rag_lab.ollama_client.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content="answer", thinking="reasoning"))
            self.assertEqual(generate_answer("question", Settings()), "answer")
            self.assertEqual(client.chat.call_args.kwargs["model"], "custom_qwen")
            client.embed.assert_not_called()

    def test_missing_final_answer_has_an_actionable_error(self):
        with patch("rag_lab.ollama_client.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=""))
            with self.assertRaisesRegex(RuntimeError, "no final answer"):
                generate_answer("question", Settings())

    def test_missing_model_has_an_actionable_error(self):
        with patch("rag_lab.ollama_client.Client") as factory:
            factory.return_value.__enter__.return_value.embed.side_effect = ResponseError("not found", 404)
            with self.assertRaisesRegex(RuntimeError, "ollama pull"):
                embed_texts(["one"], Settings())

    def test_empty_input_does_not_call_ollama(self):
        with patch("rag_lab.ollama_client.Client") as factory:
            for texts in ([], [""], "not a batch"):
                with self.assertRaises(ValueError):
                    embed_texts(texts, Settings())
            factory.assert_not_called()

    def test_structured_generation_uses_schema_and_closes_client(self):
        with patch("rag_lab.ollama_client.Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.return_value = SimpleNamespace(done_reason="stop", message=SimpleNamespace(content='{"claims": []}'))
            schema = {"type": "object"}
            self.assertEqual(generate_structured_response([{"role": "user", "content": "q"}], schema, Settings()), {"claims": []})
            self.assertEqual(client.chat.call_args.kwargs["format"], schema)
            self.assertFalse(client.chat.call_args.kwargs["think"])
            factory.return_value.__exit__.assert_called_once()

    def test_truncated_or_invalid_json_is_not_published(self):
        for reason, content in [("length", '{"claims": []}'), ("stop", "not json"), ("stop", "[]")]:
            with patch("rag_lab.ollama_client.Client") as factory:
                client = factory.return_value.__enter__.return_value
                client.chat.return_value = SimpleNamespace(done_reason=reason, message=SimpleNamespace(content=content))
                with self.assertRaises(RuntimeError):
                    generate_structured_response([], {}, Settings())


class ExtractionTests(unittest.TestCase):
    def test_text_order_is_preserved_and_non_text_blocks_are_skipped(self):
        with patch("rag_lab.extraction.convert_document", return_value={"texts": [
            {"orig": "First paragraph"}, {}, {"orig": "Second paragraph"}, {"orig": ""},
        ]}):
            self.assertEqual(extract_text("example.docx"), "First paragraph\nSecond paragraph")


if __name__ == "__main__":
    unittest.main()
