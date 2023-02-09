import os
import json
import sys
import pickle
from spacy.lang.es import Spanish

cur_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(cur_file_path, "../data")
sys.path.append(os.path.join(cur_file_path, '../..'))

class NextWordPedictor:

    def __init__(self):
        self.trigram_lm = pickle.load(open(os.path.join(cur_file_path, "../../models/cervantes_trigram_lm.pkl"), "rb"))
        self.vocab = json.load(open(os.path.join(cur_file_path, "../../data/cervantes_vocab_counts.json")))

    def preprocess_input(self, sentence, unk_token="<UNK>", bos_token="<BOS>", n=2):
        nlp = Spanish()
        tok_sent = nlp(sentence)
        sent = [tok.text if tok.text in self.vocab else unk_token for tok in tok_sent]
        return [bos_token] * n + sent

    def predict(self, input, k=10, starts_with=None, unk_token="<UNK>", n=2):
        prev_words = tuple(self.preprocess_input(input))
        prev_bigram = prev_words[-n:]
        probs = self.trigram_lm.compute_trigram_probabilities(prev_bigram)
        if starts_with != "NO":
            best_k_words = []
            for word, prob in probs:
                if word.startswith(starts_with):
                    best_k_words.append(word)
                    if len(best_k_words) == k:
                        break
        else:
            best_k_words = probs[:k]
            best_k_words = [w[0] for w in best_k_words]
            if unk_token in best_k_words:
                best_k_words.remove(unk_token)
                best_k_words.append(probs[k][0])
        return best_k_words


if __name__ == "__main__":
    print("Welcome to this next word predictor for Spanish. Loading...")
    next_word_predictor = NextWordPedictor()
    prev_words = input("Please, insert your sentence in Spanish ('Q' to quit): ")
    starts_with = input("Insert the first letter of the next word ('NO' otherwise): ")
    k = int(input("How many suggestions do you want? Insert number: "))
    while prev_words != "Q":
        suggestions = next_word_predictor.predict(prev_words, k, starts_with=starts_with)
        print(f"The best {k} suggestions to continue your sentence: ")
        for sug in suggestions:
            print(sug)
        print()
        prev_words = input("Please, insert your sentence in Spanish ('Q' to quit): ")
        starts_with = input("Insert the first letter of the next word ('NO' otherwise): ")
        k = int(input("How many suggestions do you want? Insert number: "))