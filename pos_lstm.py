import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
import random
import nltk
from nltk.corpus import brown

class BiLSTMModel:
    def __init__(self, vocab_size, num_tags, max_seq_length):
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.max_seq_length = max_seq_length
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=128, input_length=self.max_seq_length, mask_zero=True),
            Bidirectional(LSTM(units=64, return_sequences=True)),
            TimeDistributed(Dense(self.num_tags, activation='softmax'))
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.1):
        print("Training Bi-LSTM model...")
        self.model.fit(X_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=validation_split)

    def evaluate(self, X_test, y_test):
        print("\nEvaluating Bi-LSTM model...")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def predict(self, sentence, word_to_int, int_to_tag, max_seq_length):
        sentence_lower = [word.lower() for word in sentence]
        sentence_int = [word_to_int.get(word, word_to_int.get("UNK", 0)) for word in sentence_lower] # Use 0 for UNK if UNK exists, else just 0
        
        # Pad sequence
        sentence_padded = pad_sequences([sentence_int], maxlen=max_seq_length, padding='post')

        # Predict
        predictions = self.model.predict(sentence_padded)

        # Get the tag with the highest probability for each word (excluding padding)
        predicted_tag_indices = np.argmax(predictions[0], axis=1)
        
        # Map indices back to tags, only for the original words (ignore padding)
        predicted_tags = [int_to_tag[idx] for idx in predicted_tag_indices[:len(sentence)]]

        return predicted_tags

# --- Data Preparation ---
def prepare_data(corpus_sentences, test_split=0.2):
    # Create word and tag vocabularies
    all_words = []
    all_tags = []
    for sentence in corpus_sentences:
        words = [word.lower() for word, tag in sentence]
        tags = [tag for word, tag in sentence]
        all_words.extend(words)
        all_tags.extend(tags)

    # Build word-to-integer and tag-to-integer mappings
    word_counts = defaultdict(int)
    for word in all_words:
        word_counts[word] += 1
    
    # Add an "UNK" token for words not seen during training
    word_vocab = ["PAD", "UNK"] + sorted([word for word, count in word_counts.items()])
    word_to_int = {word: i for i, word in enumerate(word_vocab)}
    int_to_word = {i: word for word, i in word_to_int.items()}

    # Get unique tags
    tag_vocab = sorted(list(set(all_tags)))
    tag_to_int = {tag: i for i, tag in enumerate(tag_vocab)}
    int_to_tag = {i: tag for tag, i in tag_to_int.items()}

    print(f"Word vocabulary size: {len(word_vocab)}")
    print(f"Tag vocabulary size: {len(tag_vocab)}")

    # Convert sentences to sequences of integers
    X = [] # Word sequences
    y = [] # Tag sequences
    for sentence in corpus_sentences:
        words_int = [word_to_int.get(word.lower(), word_to_int["UNK"]) for word, tag in sentence]
        tags_int = [tag_to_int[tag] for word, tag in sentence]
        X.append(words_int)
        y.append(tags_int)

    # Find max sequence length
    max_seq_length = max(len(seq) for seq in X)
    print(f"Maximum sequence length: {max_seq_length}")

    # Pad sequences
    X_padded = pad_sequences(X, maxlen=max_seq_length, padding='post', value=word_to_int["PAD"])
    y_padded = pad_sequences(y, maxlen=max_seq_length, padding='post', value=tag_to_int[tag_vocab[0]]) # Pad with the first tag's int, masking handles this

    # Split data into training and testing sets
    combined = list(zip(X_padded, y_padded))
    random.shuffle(combined)
    split_index = int(len(combined) * (1 - test_split))
    train_data = combined[:split_index]
    test_data = combined[split_index:]

    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return (X_train, y_train), (X_test, y_test), word_to_int, int_to_word, tag_to_int, int_to_tag, max_seq_length, corpus_sentences[split_index:] # Return raw test sentences too for HMM eval