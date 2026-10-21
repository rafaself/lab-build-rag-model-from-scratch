import unittest

from tokenizers import Tokenizer, models, pre_tokenizers

from rag_lab.chunking import sections, token_chunks


def word_tokenizer():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer


def byte_tokenizer():
    vocabulary = {character: i for i, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocabulary, []))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    return tokenizer


class TokenChunkingTests(unittest.TestCase):
    def test_exact_source_slices_and_token_bounds(self):
        content = "First paragraph has several words.\n\nSecond paragraph also has several words.\nLast sentence."
        tokenizer = word_tokenizer()
        result = token_chunks(content, 6, 1, tokenizer=tokenizer)
        self.assertGreater(len(result), 1)
        covered = set()
        for chunk in result:
            self.assertEqual(chunk["content"], content[chunk["start_char"]:chunk["end_char"]])
            self.assertLessEqual(chunk["token_count"], 6)
            self.assertEqual(chunk["token_count"], len(tokenizer.encode(chunk["content"], add_special_tokens=False).ids))
            covered.update(range(chunk["start_char"], chunk["end_char"]))
        self.assertTrue(all(index in covered for index, character in enumerate(content) if not character.isspace()))

    def test_heading_hierarchy_and_fenced_code(self):
        text = "Intro.\n# Astronomy\nBody.\n## Stars\nMore.\n```python\n# not a heading\n```\n## Planets\nEnd."
        result = token_chunks(text, 100, 10, tokenizer=word_tokenizer())
        self.assertEqual([row["heading_path"] for row in result], [[], ["Astronomy"], ["Astronomy", "Stars"], ["Astronomy", "Planets"]])
        self.assertIn("# not a heading", result[2]["content"])
        self.assertTrue(result[2]["content"].startswith("## Stars"))

    def test_wikipedia_sections_do_not_share_overlap(self):
        text = "== Core ==\nFusion happens here.\n=== Reactions ===\nHydrogen fuses.\n== Surface ==\nLight escapes."
        result = token_chunks(text, 100, 10, tokenizer=word_tokenizer())
        self.assertEqual([r["heading_path"] for r in result], [["Core"], ["Core", "Reactions"], ["Surface"]])
        self.assertLessEqual(result[0]["end_char"], result[1]["start_char"])
        self.assertLessEqual(result[1]["end_char"], result[2]["start_char"])
        self.assertFalse(text[result[0]["end_char"]:result[1]["start_char"]].strip())

    def test_unicode_multibyte_tokens_do_not_corrupt_or_lose_characters(self):
        text = "宇宙🛰️星空。\n\nCafé et étoiles 🌍. " * 3
        result = token_chunks(text, 16, 0, tokenizer=byte_tokenizer())
        covered = set()
        for row in result:
            self.assertNotIn("\ufffd", row["content"])
            self.assertEqual(row["content"], text[row["start_char"]:row["end_char"]])
            self.assertLessEqual(row["token_count"], 16)
            covered.update(range(row["start_char"], row["end_char"]))
        self.assertTrue(all(index in covered for index, character in enumerate(text) if not character.isspace()))

    def test_natural_boundary_and_overlap_preserve_progress(self):
        text = "One two three four five.\n\nSix seven eight nine ten eleven twelve."
        result = token_chunks(text, 6, 2, tokenizer=word_tokenizer())
        self.assertTrue(result[0]["content"].endswith("five."))
        self.assertLess(result[1]["start_char"], result[0]["end_char"])
        self.assertEqual(sorted(set(row["start_char"] for row in result)), [row["start_char"] for row in result])

    def test_empty_and_invalid_input_does_not_load_tokenizer(self):
        self.assertEqual(token_chunks(" \n ", 10, 1), [])
        for size, overlap in [(0, 0), (8193, 0), (10, 10), (10, -1)]:
            with self.assertRaises(ValueError):
                token_chunks("text", size, overlap)

    def test_heading_only_sections_are_preserved_without_separate_vectors(self):
        text = "== Stars ==\n\n=== Core ===\nFusion provides energy.\n== References ==\n"
        result = token_chunks(text, 100, 10, tokenizer=word_tokenizer())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["heading_path"], ["Stars", "Core"])
        self.assertEqual(result[0]["content"], text.rstrip())
