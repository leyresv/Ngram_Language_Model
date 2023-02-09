import math
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

cur_file_path = os.path.dirname(os.path.realpath(__file__))

class TrigramLM:

    def __init__(self):
        self.unigram_counts = None
        self.bigram_counts = None
        self.trigram_counts = None

    @staticmethod
    def count_ngrams(sentences, n, bos_token="<BOS>", eos_token="<EOS>"):
        ngram_counts = {}
        for sentence in sentences:
            sentence = [bos_token] * n + sentence #+ [eos_token]
            for i in range(len(sentence) - n + 1):
                ngram = tuple(sentence[i:i + n])
                if ngram in ngram_counts:
                    ngram_counts[ngram] += 1
                else:
                    ngram_counts[ngram] = 1
        return ngram_counts

    def train(self, train_data, vocab):
        self.unigram_counts = vocab
        self.bigram_counts = self.count_ngrams(train_data, 2)
        self.trigram_counts = self.count_ngrams(train_data, 3)

    def compute_trigram_probability(self, bigram, next_word, eos_token="<EOS>", mode="interpolation", options=None):
        if options is None:
            options = {"smoothing": 1.0, "backoff": 0.4, "interpolation": [0.8, 0.15, 0.05]}
        k = options[mode]
        vocab_list = list(self.unigram_counts.keys()) #+ [eos_token]
        vocab_size = len(vocab_list)
        corpus_size = sum(self.unigram_counts.values())

        numerator = self.trigram_counts.get(tuple(bigram) + (next_word,), 0)
        denominator = self.bigram_counts.get(bigram, 0)

        # Smoothing
        if mode == "smoothing":
            probability = (numerator + k) / (denominator + k * vocab_size)

        # Backoff
        elif mode == "backoff":
            probability = 1
            # If trigram probability is 0, get bigram probability
            if numerator == 0 or denominator == 0:
                probability *= k
                unigram = bigram[-1]
                numerator = self.bigram_counts.get(tuple(unigram) + (next_word,), 0)
                denominator = self.unigram_counts.get(unigram, 0)
                # If bigram probability is 0, get unigram probability
                if numerator == 0 or denominator == 0:
                    probability *= k
                    numerator = self.unigram_counts.get(next_word, 0)
                    denominator = corpus_size
            probability *= (numerator / denominator)

        # Interpolation
        else:
            if numerator == 0 or denominator == 0:
                trigram_prob = 0
            else:
                trigram_prob = numerator / denominator
            unigram = bigram[-1]
            bigram_prob_numerator = self.bigram_counts.get(tuple(unigram) + (next_word,), 0)
            bigram_prob_denominator = self.unigram_counts.get(unigram, 0)
            if bigram_prob_numerator == 0 or bigram_prob_denominator == 0:
                bigram_prob = 0
            else:
                bigram_prob = bigram_prob_numerator / bigram_prob_denominator
            unigram_prob = self.unigram_counts.get(next_word, 0) / corpus_size
            probability = k[0] * trigram_prob + k[1] * bigram_prob + k[2] * unigram_prob

        return probability

    def compute_trigram_probabilities(self, bigram, eos_token="<EOS>", mode="interpolation", options=None):
        if options is None:
            options = {"smoothing": 1.0, "backoff": 0.4, "interpolation": [0.8, 0.15, 0.05]}
        vocab_list = list(self.unigram_counts.keys()) #+ [eos_token]
        probabilities = {}

        for word in vocab_list:
            probabilities[word] = self.compute_trigram_probability(bigram, word, mode=mode, options=options)

        return sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    def evaluate_sentence_perplexity(self, sentence, bos_token="<BOS>", eos_token="<EOS>", n=2):
        sentence = [bos_token] * n + sentence #+ [eos_token]
        sentence = tuple(sentence)
        sent_len = len(sentence)
        sent_prob = 0.0
        # For each n-gram in the sentence:
        for i in range(sent_len - n):
            ngram = sentence[i:i + n]
            next_word = sentence[i + n]
            # Calculate the probability of the next word being the specified one
            prob = self.compute_trigram_probability(ngram, next_word)
            sent_prob += np.log(prob)

        perplexity = (1 / np.exp(sent_prob)) ** (1 / float(sent_len))
        if math.isinf(perplexity):
            perplexity = None
        return perplexity




