import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

if __name__ == "__main__":
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    # Test sentences
    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

    for text in sentences:
        print(f"\nOriginal: {text}")
        print("Simple:", simple_tokenizer.tokenize(text))
        print("Regex :", regex_tokenizer.tokenize(text))

    # Dataset test
    dataset_path = "../data/UD_English-EWT/en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)
    sample_text = raw_text[:500]

    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample: {sample_text[:100]}...")

    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print("SimpleTokenizer Output:", simple_tokens[:20])

    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print("RegexTokenizer Output:", regex_tokens[:20])
