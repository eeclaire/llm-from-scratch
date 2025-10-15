import unittest
from pathlib import Path
from src.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup that reads the file and builds the tokenizer vocabulary."""
        # Get the path relative to the test directory
        test_dir = Path(__file__).parent
        filepath = test_dir.parent / "the_verdict.txt"

        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cls.tokenizer = Tokenizer()
        cls.tokenizer.build_vocab(raw_text)

    def test_basic_encode_decode(self):
        """
        Test encoding of a specific sample text for which all the words are available
        in the training corpus.
        """
        sample_text = """
            "It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride.
        """

        embedding = self.tokenizer.encode(sample_text)

        expected = [
            1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108,
            754, 793, 7
        ]

        self.assertEqual(embedding, expected)

        reverted_text = self.tokenizer.decode(embedding)
        print(f"Reverted text: {reverted_text}")

    def test_encode_decode_with_unknowns(self):
        """Test basic encoding and decoding with unknown words."""
        text1 = "Hello, do you like tea?"
        text2 = "In the sunlit terraces of the palace."
        text = " <|eot|> ".join((text1, text2))

        encoded = self.tokenizer.encode(text)
        expected_encoded = [
            1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7
        ]
        self.assertEqual(encoded, expected_encoded)

        decoded = self.tokenizer.decode(encoded)
        expected_decoded = (
            "<|unk|>, do you like tea? <|eot|> In the sunlit terraces of the <|unk|>."
        )
        self.assertEqual(decoded, expected_decoded)

