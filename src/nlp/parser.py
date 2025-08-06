
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt', quiet=True)

def parse_prompt(prompt: str) -> dict:
    tokens = word_tokenize(prompt)
    sentences = sent_tokenize(prompt)
    return {
        "token_count": len(tokens),
        "sentence_count": len(sentences)
    }