import json
import re
import nltk
from nltk.tokenize import word_tokenize
import pyidaungsu as pds
from gensim.models import Word2Vec

nltk.download('punkt')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    words = word_tokenize(text)
    return words

def preprocess_myanmar_text(text):
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    words = pds.tokenize(text)  # Tokenize by lswords for Myanmar using pyidaungsu
    return words

# Load the JSON dataset
with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Preprocess the texts
preprocessed_data = []
for item in data:
    myanglish_text = item['myanglish']
    myanmar_text = item['myanmar']
    
    myanglish_words = preprocess_text(myanglish_text)
    myanmar_words = preprocess_myanmar_text(myanmar_text)
    
    preprocessed_data.append({
        'myanglish': myanglish_words,
        'myanmar': myanmar_words
    })

# Example output
for item in preprocessed_data:
    print(f"Myanglish: {item['myanglish']}")
    print(f"Myanmar: {item['myanmar']}")
    print()


# Save the preprocessed data to a JSON file
with open('preprocessed_data.json', 'w', encoding='utf-8') as f:
    json.dump(preprocessed_data, f, ensure_ascii=False, indent=4)


# Extract tokenized sentences for Word2Vec training
myanglish_sentences = [item['myanglish'] for item in preprocessed_data]
myanmar_sentences = [item['myanmar'] for item in preprocessed_data]

# Combine both Myanglish and Myanmar tokenized sentences
all_sentences = myanglish_sentences + myanmar_sentences

# Train Word2Vec model
model = Word2Vec(sentences=all_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Save the model
model.save("word2vec_model.model")


# Test the model
test_word = 'kya'  # Replace with a word from your dataset
if test_word in model.wv:
    print(f"Vector for '{test_word}': {model.wv[test_word]}")
else:
    print(f"'{test_word}' not in vocabulary")