import nltk
from nltk.corpus import brown
from pos_hmm import HMM
from pos_lstm import BiLSTMModel, prepare_data
import numpy as np
from collections import defaultdict

# Download NLTK Brown Corpus
try:
    nltk.data.find('corpora/brown')
except nltk.downloader.DownloadError:
    nltk.download('brown')
except nltk.downloader.DownloadNLTKDataError:
     nltk.download('brown')

# --- Data Loading and Preparation ---
corpus_sentences = brown.tagged_sents(categories='news') # Using 'news' category for a smaller subset

# Shuffle sentences before splitting
import random
random.shuffle(corpus_sentences)

# Split data for HMM and NN
hmm_train_data = corpus_sentences[:int(len(corpus_sentences)*0.8)]
hmm_test_data = corpus_sentences[int(len(corpus_sentences)*0.8):]

# Prepare data for Neural Network
(X_train_nn, y_train_nn), (X_test_nn, y_test_nn), word_to_int, int_to_word, tag_to_int, int_to_tag, max_seq_length, raw_test_sentences_nn = prepare_data(corpus_sentences, test_split=0.2)

print(f"HMM Training sentences: {len(hmm_train_data)}")
print(f"HMM Test sentences: {len(hmm_test_data)}")
print(f"NN Training sentences (after padding): {len(X_train_nn)}")
print(f"NN Test sentences (after padding): {len(X_test_nn)}")


# --- Train and Evaluate HMM ---
print("\n--- Training HMM ---")
hmm_model = HMM()
hmm_model.train(hmm_train_data)

print("\n--- Evaluating HMM ---")
# Evaluate HMM on the raw test sentences
correct_hmm = 0
total_hmm = 0

# Need to handle potential OOV words and smoothing in evaluation
for sentence_tags in hmm_test_data:
    sentence = [word for word, tag in sentence_tags]
    actual_tags = [tag for word, tag in sentence_tags]

    # HMM prediction
    predicted_tags_hmm = hmm_model.viterbi(sentence)

    # Compare tags
    for actual_tag, predicted_tag in zip(actual_tags, predicted_tags_hmm):
        if actual_tag == predicted_tag:
            correct_hmm += 1
        total_hmm += 1

hmm_accuracy = correct_hmm / total_hmm if total_hmm > 0 else 0
print(f"HMM Test Accuracy: {hmm_accuracy:.4f}")


# --- Train and Evaluate Neural Network ---
print("\n--- Training Neural Network (Bi-LSTM) ---")
# Get vocab size from the mapping created in prepare_data
nn_vocab_size = len(word_to_int)
nn_num_tags = len(tag_to_int)

lstm_model = BiLSTMModel(vocab_size=nn_vocab_size, num_tags=nn_num_tags, max_seq_length=max_seq_length)
lstm_model.model.summary() # Print model summary

# Train the NN model
lstm_model.train(X_train_nn, y_train_nn, epochs=5, batch_size=64, validation_split=0.1) # Reduced epochs for faster run

# Evaluate the NN model
lstm_accuracy = lstm_model.evaluate(X_test_nn, y_test_nn)


# --- Comparison ---
print("\n--- Model Comparison ---")
print(f"HMM Test Accuracy: {hmm_accuracy:.4f}")
print(f"Bi-LSTM Test Accuracy: {lstm_accuracy:.4f}")

# --- Example Predictions ---
print("\n--- Example Predictions ---")
sample_sentence_1 = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]
sample_sentence_2 = ["This", "is", "a", "test", "sentence", "."]
sample_sentence_3 = ["Apple", "buys", "new", "company"] # Contains potential proper noun/verb ambiguity

print(f"\nSentence: {' '.join(sample_sentence_1)}")
print(f"HMM Prediction: {hmm_model.viterbi(sample_sentence_1)}")
print(f"Bi-LSTM Prediction: {lstm_model.predict(sample_sentence_1, word_to_int, int_to_tag, max_seq_length)}")

print(f"\nSentence: {' '.join(sample_sentence_2)}")
print(f"HMM Prediction: {hmm_model.viterbi(sample_sentence_2)}")
print(f"Bi-LSTM Prediction: {lstm_model.predict(sample_sentence_2, word_to_int, int_to_tag, max_seq_length)}")

print(f"\nSentence: {' '.join(sample_sentence_3)}")
print(f"HMM Prediction: {hmm_model.viterbi(sample_sentence_3)}")
print(f"Bi-LSTM Prediction: {lstm_model.predict(sample_sentence_3, word_to_int, int_to_tag, max_seq_length)}")