"""Training data for the laboratory (the starting dataset from the lab sheet)."""

START, END = "<START>", "<END>"

RAW_SENTENCES = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def tokenise(sentence):
    """Lower-case and split into word tokens (one word = one token)."""
    return sentence.lower().split()


def training_data():
    """List of tokenised sentences (without START/END; the models add them)."""
    return [tokenise(s) for s in RAW_SENTENCES]
