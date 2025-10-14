import re

EncoderType = dict[str, int]
DecoderType = dict[int, str]


class Tokenizer:
    def __init__(self):
        self.encoder = EncoderType()
        self.decoder = DecoderType()

    def build_vocab(self, text: str):
        words = self.tokenize(text)
        self.encoder: EncoderType = self.__build_vocabulary(set(words))
        self.decoder: DecoderType = {i: s for s, i in self.encoder.items()}

    @staticmethod
    def __build_vocabulary(words: set[str]) -> dict[str, int]:
        vocabulary = {}
        words = sorted(words)
        for i, word in enumerate(words):
            vocabulary[word] = i

        return vocabulary

    @staticmethod
    def tokenize(text: str) -> list[str]:
        result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        return [r.strip() for r in result if r.strip()]

    def encode(self, text: str) -> list[int]:
        words = self.tokenize(text)
        return [self.encoder[word] for word in words]

    def decode(self, embeddings: list[int]) -> str:
        words = [self.decoder[word] for word in embeddings]
        text = " ".join(words)
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)