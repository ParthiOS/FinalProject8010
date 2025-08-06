import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

def get_prompt_complexity(prompt: str) -> int:
    tokens = word_tokenize(prompt)
    return len(tokens)