import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
# nltk.download('stopwords')

def clean_text(text):
    # Replace newlines with spaces
    text = text.replace('\n', ' ')

    # Tokenize the text into words
    words = word_tokenize(text)

    print(words)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words_without_stopwords = [word for word in words if word.lower() not in stop_words]

    # Remove words that are only numbers
    words_cleaned = [word for word in words_without_stopwords if not word.isdigit()]

    # Join the words back into a cleaned sentence
    cleaned_text = ' '.join(words_cleaned)

    # Preserve certain punctuations
    preserved_punctuations = ['"', "'", "-"]  # Add any other punctuations you want to preserve
    for punctuation in preserved_punctuations:
        cleaned_text = cleaned_text.replace(f" {punctuation} ", f"{punctuation} ")

    return cleaned_text

# Example usage:
text = """
NLTK is a powerful library for \"natural language processing\".
It provides tools for text processing tasks like tokenization,
removing stop words, and more!
"""

cleaned_text = clean_text(text)
print(cleaned_text)

