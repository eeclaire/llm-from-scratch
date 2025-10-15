
from tokenizer import Tokenizer


def read_text(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = Tokenizer()
    tokenizer.build_vocab(raw_text)

    '''
    sample_text = """
    "It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride.
    """

    embedding = tokenizer.encode(sample_text)
    print(embedding)
    assert embedding == [
        1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108,
        754, 793, 7
    ]
    reverted_text = tokenizer.decode(embedding)
    print(reverted_text)
    '''

    #
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|eot|> ".join((text1, text2))
    print(text)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))


if __name__ == '__main__':
    read_text("../the_verdict.txt")
