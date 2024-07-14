import json
import re
import nltk
from nltk.tokenize import word_tokenize
import pyidaungsu as pds
from gensim.models import Word2Vec
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# # Example output
# for item in preprocessed_data:
#     print(f"Myanglish: {item['myanglish']}")
#     print(f"Myanmar: {item['myanmar']}")
#     print()


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



# # Test the model
# test_word = 'kya'  # Replace with a word from your dataset
# if test_word in model.wv:
#     print(f"Vector for '{test_word}': {model.wv[test_word]}")
# else:
#     print(f"'{test_word}' not in vocabulary")
# Create the embedding matrices

myanglish_embedding_matrix = []
myanmar_embedding_matrix = []

for word_list in myanglish_sentences:
    for word in word_list:
        if word in model.wv:
            myanglish_embedding_matrix.append(model.wv[word])
        else:
            myanglish_embedding_matrix.append(np.zeros(100))

for word_list in myanmar_sentences:
    for word in word_list:
        if word in model.wv:
            myanmar_embedding_matrix.append(model.wv[word])
        else:
            myanmar_embedding_matrix.append(np.zeros(100))

myanglish_embedding_matrix = np.array(myanglish_embedding_matrix)
myanmar_embedding_matrix = np.array(myanmar_embedding_matrix)


class TranslationDataset(Dataset):
    def __init__(self, myanglish_sequences, myanmar_sequences):
        self.myanglish_sequences = myanglish_sequences
        self.myanmar_sequences = myanmar_sequences

    def __len__(self):
        return len(self.myanglish_sequences)

    def __getitem__(self, idx):
        return {
            'myanglish': torch.tensor(self.myanglish_sequences[idx], dtype=torch.long),
            'myanmar': torch.tensor(self.myanmar_sequences[idx], dtype=torch.long)
        }

# Create the dataset
dataset = TranslationDataset(myanglish_embedding_matrix, myanmar_embedding_matrix)

# Create the dataloaders
BATCH_SIZE = 64
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)




